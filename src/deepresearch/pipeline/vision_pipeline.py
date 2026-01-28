"""Vision research pipeline for CV experiments."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from deepresearch.core.config import (
    APIConfig,
    PipelineConfig,
    PipelineStage,
    ProviderConfig,
    ProviderType,
)
from deepresearch.core.exceptions import PipelineError
from deepresearch.core.state import SessionState, StateManager, create_session
from deepresearch.modules.analysis.analyzer import StatisticalAnalyzer
from deepresearch.modules.analysis.validator import HypothesisValidator
from deepresearch.modules.experiment.api_manager import APIManager
from deepresearch.modules.experiment.vision_designer import VisionExperimentDesigner
from deepresearch.modules.figures.result_figures import ResultFigureGenerator
from deepresearch.modules.ideation.idea_generator import IdeaGenerator
from deepresearch.modules.literature.novelty_checker import NoveltyChecker
from deepresearch.modules.literature.searcher import LiteratureSearcher
from deepresearch.modules.vision.executor import VisionExperimentExecutor
from deepresearch.modules.writing.section_writer import SectionWriter
from deepresearch.pipeline.stage import StageResult


@dataclass
class VisionPipelineOutput:
    """Output from vision pipeline."""

    session_id: str
    research_topic: str
    stage_results: dict[str, StageResult]
    paper_path: Path | None
    figure_paths: list[Path]
    total_cost_usd: float
    success: bool
    error: str | None = None
    metrics_summary: dict | None = None


class VisionResearchPipeline:
    """Pipeline specialized for computer vision research."""

    def __init__(
        self,
        config: PipelineConfig,
        datasets: list[str] | None = None,
    ) -> None:
        self.config = config
        self.datasets = datasets or ["mnist", "cifar10"]
        self.state_manager = StateManager(config.session_dir)

        # Initialize API manager
        self.api_manager = self._create_api_manager()

        # Initialize modules
        self.literature_searcher = LiteratureSearcher()
        self.novelty_checker = NoveltyChecker(self.api_manager)
        self.idea_generator = IdeaGenerator(self.api_manager)
        self.experiment_designer = VisionExperimentDesigner(self.api_manager)
        self.experiment_executor = VisionExperimentExecutor(
            self.api_manager,
            config.results_dir / "checkpoints",
            config.checkpoint_interval,
        )
        self.analyzer = StatisticalAnalyzer()
        self.validator = HypothesisValidator(self.api_manager)
        self.figure_gen = ResultFigureGenerator(config.output_dir / "figures")
        self.section_writer = SectionWriter(
            self.api_manager,
            config.output_dir,
            config.output_format,
        )

    def _create_api_manager(self) -> APIManager:
        """Create API manager with vision-capable models."""
        if not self.config.api.providers:
            self.config.api.providers = {
                ProviderType.OPENAI: ProviderConfig(
                    provider_type=ProviderType.OPENAI,
                    model="gpt-4o-mini",
                    requests_per_minute=500,
                    cost_per_1k_input=0.00015,
                    cost_per_1k_output=0.0006,
                ),
                ProviderType.ANTHROPIC: ProviderConfig(
                    provider_type=ProviderType.ANTHROPIC,
                    model="claude-3-5-sonnet-20241022",
                    requests_per_minute=60,
                ),
                ProviderType.GOOGLE: ProviderConfig(
                    provider_type=ProviderType.GOOGLE,
                    model="gemini-1.5-flash",
                    requests_per_minute=60,
                ),
            }
        return APIManager(self.config.api)

    async def run(
        self,
        research_topic: str | None = None,
        session_id: str | None = None,
        progress_callback: Callable[[PipelineStage, str], None] | None = None,
    ) -> VisionPipelineOutput:
        """Run the vision research pipeline."""
        # Load or create session
        if session_id and self.state_manager.exists(session_id):
            state = self.state_manager.load(session_id)
        else:
            if not research_topic:
                raise PipelineError("Research topic required for new session")
            state = create_session(research_topic, self.config)

        stage_results: dict[str, StageResult] = {}
        total_cost = 0.0
        metrics_summary = None

        try:
            # Stage 1: Literature Search (CV papers)
            if not state.is_stage_complete(PipelineStage.LITERATURE_SEARCH):
                result = await self._run_literature_search(state, progress_callback)
                stage_results["literature_search"] = result
                total_cost += result.cost_usd
                if not result.success:
                    raise PipelineError(result.error or "Literature search failed")
                state.mark_stage_complete(PipelineStage.LITERATURE_SEARCH)
                self.state_manager.save(state)

            # Stage 2: Idea Generation
            if not state.is_stage_complete(PipelineStage.IDEA_GENERATION):
                result = await self._run_idea_generation(state, progress_callback)
                stage_results["idea_generation"] = result
                total_cost += result.cost_usd
                if not result.success:
                    raise PipelineError(result.error or "Idea generation failed")
                state.mark_stage_complete(PipelineStage.IDEA_GENERATION)
                self.state_manager.save(state)

            # Stage 3: Novelty Check
            if not state.is_stage_complete(PipelineStage.NOVELTY_CHECK):
                result = await self._run_novelty_check(state, progress_callback)
                stage_results["novelty_check"] = result
                total_cost += result.cost_usd
                state.mark_stage_complete(PipelineStage.NOVELTY_CHECK)
                self.state_manager.save(state)

            # Stage 4: Vision Experiment Design
            if not state.is_stage_complete(PipelineStage.EXPERIMENT_DESIGN):
                result = await self._run_experiment_design(state, progress_callback)
                stage_results["experiment_design"] = result
                total_cost += result.cost_usd
                if not result.success:
                    raise PipelineError(result.error or "Experiment design failed")
                state.mark_stage_complete(PipelineStage.EXPERIMENT_DESIGN)
                self.state_manager.save(state)

            # Stage 5: Vision Experiment Execution
            if not state.is_stage_complete(PipelineStage.EXPERIMENT_EXECUTION):
                result = await self._run_experiments(state, progress_callback)
                stage_results["experiment_execution"] = result
                total_cost += result.cost_usd
                if not result.success:
                    raise PipelineError(result.error or "Experiment execution failed")
                state.mark_stage_complete(PipelineStage.EXPERIMENT_EXECUTION)
                self.state_manager.save(state)

                # Extract metrics summary
                metrics_summary = {
                    exp_id: {
                        "accuracy": r.metrics.get("accuracy", 0),
                        "status": r.status,
                    }
                    for exp_id, r in state.experiment_results.items()
                }

            # Stage 6: Analysis
            if not state.is_stage_complete(PipelineStage.ANALYSIS):
                result = await self._run_analysis(state, progress_callback)
                stage_results["analysis"] = result
                total_cost += result.cost_usd
                state.mark_stage_complete(PipelineStage.ANALYSIS)
                self.state_manager.save(state)

            # Stage 7: Figure Generation
            if not state.is_stage_complete(PipelineStage.FIGURE_GENERATION):
                result = await self._run_figure_generation(state, progress_callback)
                stage_results["figure_generation"] = result
                state.mark_stage_complete(PipelineStage.FIGURE_GENERATION)
                self.state_manager.save(state)

            # Stage 8: Paper Writing
            if not state.is_stage_complete(PipelineStage.PAPER_WRITING):
                result = await self._run_paper_writing(state, progress_callback)
                stage_results["paper_writing"] = result
                total_cost += result.cost_usd
                state.mark_stage_complete(PipelineStage.PAPER_WRITING)
                self.state_manager.save(state)

            paper_path = self.config.output_dir / f"{state.session_id}_paper"
            paper_path = paper_path.with_suffix(
                ".tex" if self.config.output_format == "latex" else ".md"
            )

            return VisionPipelineOutput(
                session_id=state.session_id,
                research_topic=state.research_topic,
                stage_results=stage_results,
                paper_path=paper_path if paper_path.exists() else None,
                figure_paths=[Path(f) for f in state.figures],
                total_cost_usd=total_cost,
                success=True,
                metrics_summary=metrics_summary,
            )

        except Exception as e:
            self.state_manager.save(state)
            return VisionPipelineOutput(
                session_id=state.session_id,
                research_topic=state.research_topic,
                stage_results=stage_results,
                paper_path=None,
                figure_paths=[],
                total_cost_usd=total_cost,
                success=False,
                error=str(e),
            )

        finally:
            await self.api_manager.close()

    async def _run_literature_search(
        self,
        state: SessionState,
        progress_callback: Callable[[PipelineStage, str], None] | None,
    ) -> StageResult:
        """Search for CV papers."""
        start_time = datetime.now()

        if progress_callback:
            progress_callback(PipelineStage.LITERATURE_SEARCH, "Searching CV literature...")

        try:
            # Add "computer vision" and "image classification" to the search
            cv_topic = f"{state.research_topic} computer vision image classification"
            papers = await self.literature_searcher.search(
                cv_topic,
                max_results=self.config.literature.max_papers,
            )
            state.papers = papers

            return StageResult(
                stage=PipelineStage.LITERATURE_SEARCH,
                success=True,
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                outputs={"paper_count": len(papers)},
            )
        except Exception as e:
            return StageResult(
                stage=PipelineStage.LITERATURE_SEARCH,
                success=False,
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                error=str(e),
            )

    async def _run_idea_generation(
        self,
        state: SessionState,
        progress_callback: Callable[[PipelineStage, str], None] | None,
    ) -> StageResult:
        """Generate CV research ideas."""
        start_time = datetime.now()

        if progress_callback:
            progress_callback(PipelineStage.IDEA_GENERATION, "Generating vision research ideas...")

        try:
            ideas = await self.idea_generator.generate_ideas(
                state.research_topic + " (computer vision, image classification)",
                state.papers,
                num_ideas=5,
            )
            best_idea = await self.idea_generator.select_best_idea(ideas)
            state.research_idea = best_idea

            return StageResult(
                stage=PipelineStage.IDEA_GENERATION,
                success=True,
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                outputs={"idea_title": best_idea.title},
            )
        except Exception as e:
            return StageResult(
                stage=PipelineStage.IDEA_GENERATION,
                success=False,
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                error=str(e),
            )

    async def _run_novelty_check(
        self,
        state: SessionState,
        progress_callback: Callable[[PipelineStage, str], None] | None,
    ) -> StageResult:
        """Check novelty."""
        start_time = datetime.now()

        if progress_callback:
            progress_callback(PipelineStage.NOVELTY_CHECK, "Checking novelty...")

        try:
            if state.research_idea:
                novelty = await self.novelty_checker.check_novelty(
                    state.research_idea, state.papers
                )
                state.novelty_score = novelty.overall_score

            return StageResult(
                stage=PipelineStage.NOVELTY_CHECK,
                success=True,
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                outputs={"novelty_score": state.novelty_score},
            )
        except Exception as e:
            return StageResult(
                stage=PipelineStage.NOVELTY_CHECK,
                success=False,
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                error=str(e),
            )

    async def _run_experiment_design(
        self,
        state: SessionState,
        progress_callback: Callable[[PipelineStage, str], None] | None,
    ) -> StageResult:
        """Design vision experiments."""
        start_time = datetime.now()

        if progress_callback:
            progress_callback(PipelineStage.EXPERIMENT_DESIGN, "Designing vision experiments...")

        try:
            if not state.research_idea:
                raise PipelineError("No research idea")

            plan = await self.experiment_designer.design_experiments(
                state.research_idea,
                budget_usd=self.config.api.total_budget_usd / 2,
                preferred_datasets=self.datasets,
            )
            state.experiment_plan = plan

            return StageResult(
                stage=PipelineStage.EXPERIMENT_DESIGN,
                success=True,
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                outputs={
                    "experiment_count": len(plan.experiments),
                    "datasets": self.datasets,
                },
            )
        except Exception as e:
            return StageResult(
                stage=PipelineStage.EXPERIMENT_DESIGN,
                success=False,
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                error=str(e),
            )

    async def _run_experiments(
        self,
        state: SessionState,
        progress_callback: Callable[[PipelineStage, str], None] | None,
    ) -> StageResult:
        """Run vision experiments."""
        start_time = datetime.now()

        if progress_callback:
            progress_callback(PipelineStage.EXPERIMENT_EXECUTION, "Running vision experiments...")

        try:
            if not state.experiment_plan:
                raise PipelineError("No experiment plan")

            def exp_progress(exp_id: str, current: int, total: int) -> None:
                if progress_callback:
                    progress_callback(
                        PipelineStage.EXPERIMENT_EXECUTION,
                        f"{exp_id}: {current}/{total} images",
                    )

            results = await self.experiment_executor.execute_plan(
                state.experiment_plan,
                progress_callback=exp_progress,
            )
            state.experiment_results = results

            total_cost = sum(r.cost_usd for r in results.values())
            completed = sum(1 for r in results.values() if r.status == "completed")

            return StageResult(
                stage=PipelineStage.EXPERIMENT_EXECUTION,
                success=True,
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                outputs={"completed": completed, "total": len(results)},
                cost_usd=total_cost,
            )
        except Exception as e:
            return StageResult(
                stage=PipelineStage.EXPERIMENT_EXECUTION,
                success=False,
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                error=str(e),
            )

    async def _run_analysis(
        self,
        state: SessionState,
        progress_callback: Callable[[PipelineStage, str], None] | None,
    ) -> StageResult:
        """Analyze results."""
        start_time = datetime.now()

        if progress_callback:
            progress_callback(PipelineStage.ANALYSIS, "Analyzing results...")

        try:
            if state.experiment_plan and state.experiment_results:
                validation = await self.validator.validate(
                    state.experiment_plan.hypothesis,
                    state.experiment_plan,
                    state.experiment_results,
                )
                state.validation_result = validation

            return StageResult(
                stage=PipelineStage.ANALYSIS,
                success=True,
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                outputs={
                    "outcome": state.validation_result.outcome.value if state.validation_result else "unknown"
                },
            )
        except Exception as e:
            return StageResult(
                stage=PipelineStage.ANALYSIS,
                success=False,
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                error=str(e),
            )

    async def _run_figure_generation(
        self,
        state: SessionState,
        progress_callback: Callable[[PipelineStage, str], None] | None,
    ) -> StageResult:
        """Generate figures."""
        start_time = datetime.now()

        if progress_callback:
            progress_callback(PipelineStage.FIGURE_GENERATION, "Generating figures...")

        try:
            figures = []
            if state.experiment_results and state.validation_result:
                figs = self.figure_gen.generate_all_figures(
                    state.experiment_results,
                    state.validation_result.statistical_comparisons,
                    primary_metric="accuracy",
                )
                figures.extend(figs)

            state.figures = [str(f) for f in figures]

            return StageResult(
                stage=PipelineStage.FIGURE_GENERATION,
                success=True,
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                outputs={"figure_count": len(figures)},
            )
        except Exception as e:
            return StageResult(
                stage=PipelineStage.FIGURE_GENERATION,
                success=False,
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                error=str(e),
            )

    async def _run_paper_writing(
        self,
        state: SessionState,
        progress_callback: Callable[[PipelineStage, str], None] | None,
    ) -> StageResult:
        """Write paper."""
        start_time = datetime.now()

        if progress_callback:
            progress_callback(PipelineStage.PAPER_WRITING, "Writing paper...")

        try:
            if state.research_idea and state.validation_result:
                paper = await self.section_writer.write_full_paper(
                    state.research_idea,
                    state.experiment_results,
                    state.validation_result,
                    state.papers,
                    state.figures,
                )
                self.section_writer.save_paper(paper, f"{state.session_id}_paper")

            return StageResult(
                stage=PipelineStage.PAPER_WRITING,
                success=True,
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )
        except Exception as e:
            return StageResult(
                stage=PipelineStage.PAPER_WRITING,
                success=False,
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                error=str(e),
            )
