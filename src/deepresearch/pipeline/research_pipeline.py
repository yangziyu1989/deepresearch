"""Main research pipeline orchestrator."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from deepresearch.core.config import (
    APIConfig,
    PipelineConfig,
    PipelineStage,
    ProviderConfig,
    ProviderType,
)
from deepresearch.core.exceptions import PipelineError
from deepresearch.core.state import (
    SessionState,
    StateManager,
    create_session,
)
from deepresearch.modules.analysis.analyzer import StatisticalAnalyzer
from deepresearch.modules.analysis.validator import HypothesisValidator
from deepresearch.modules.experiment.api_manager import APIManager
from deepresearch.modules.experiment.designer import ExperimentDesigner
from deepresearch.modules.experiment.executor import ExperimentExecutor
from deepresearch.modules.figures.method_figures import MethodFigureGenerator
from deepresearch.modules.figures.result_figures import ResultFigureGenerator
from deepresearch.modules.ideation.idea_generator import IdeaGenerator
from deepresearch.modules.literature.novelty_checker import NoveltyChecker
from deepresearch.modules.literature.searcher import LiteratureSearcher
from deepresearch.modules.writing.section_writer import SectionWriter
from deepresearch.pipeline.stage import StageResult


@dataclass
class PipelineOutput:
    """Complete pipeline output."""

    session_id: str
    research_topic: str
    stage_results: dict[str, StageResult]
    paper_path: Path | None
    figure_paths: list[Path]
    total_cost_usd: float
    success: bool
    error: str | None = None


class ResearchPipeline:
    """Main orchestrator for the research pipeline."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.state_manager = StateManager(config.session_dir)

        # Initialize API manager
        self.api_manager = self._create_api_manager()

        # Initialize modules
        self.literature_searcher = LiteratureSearcher()
        self.novelty_checker = NoveltyChecker(self.api_manager)
        self.idea_generator = IdeaGenerator(self.api_manager)
        self.experiment_designer = ExperimentDesigner(self.api_manager)
        self.experiment_executor = ExperimentExecutor(
            self.api_manager,
            config.results_dir / "checkpoints",
            config.checkpoint_interval,
        )
        self.analyzer = StatisticalAnalyzer()
        self.validator = HypothesisValidator(self.api_manager)
        self.result_figure_gen = ResultFigureGenerator(config.output_dir / "figures")
        self.method_figure_gen = MethodFigureGenerator(
            self.api_manager,
            config.output_dir / "figures",
        )
        self.section_writer = SectionWriter(
            self.api_manager,
            config.output_dir,
            config.output_format,
        )

    def _create_api_manager(self) -> APIManager:
        """Create API manager from config."""
        # Ensure default providers are configured
        if not self.config.api.providers:
            self.config.api.providers = {
                ProviderType.ANTHROPIC: ProviderConfig(
                    provider_type=ProviderType.ANTHROPIC,
                    model="claude-3-5-sonnet-20241022",
                    requests_per_minute=60,
                ),
                ProviderType.OPENAI: ProviderConfig(
                    provider_type=ProviderType.OPENAI,
                    model="gpt-4o",
                    requests_per_minute=500,
                ),
            }
        return APIManager(self.config.api)

    async def run(
        self,
        research_topic: str | None = None,
        session_id: str | None = None,
        progress_callback: Callable[[PipelineStage, str], None] | None = None,
    ) -> PipelineOutput:
        """Run the complete research pipeline."""
        # Load or create session
        if session_id and self.state_manager.exists(session_id):
            state = self.state_manager.load(session_id)
        else:
            if not research_topic:
                raise PipelineError("Research topic required for new session")
            state = create_session(research_topic, self.config)

        stage_results: dict[str, StageResult] = {}
        total_cost = 0.0

        try:
            # Stage 1: Literature Search
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
                if not result.success:
                    raise PipelineError(result.error or "Novelty check failed")
                state.mark_stage_complete(PipelineStage.NOVELTY_CHECK)
                self.state_manager.save(state)

            # Stage 4: Experiment Design
            if not state.is_stage_complete(PipelineStage.EXPERIMENT_DESIGN):
                result = await self._run_experiment_design(state, progress_callback)
                stage_results["experiment_design"] = result
                total_cost += result.cost_usd
                if not result.success:
                    raise PipelineError(result.error or "Experiment design failed")
                state.mark_stage_complete(PipelineStage.EXPERIMENT_DESIGN)
                self.state_manager.save(state)

            # Stage 5: Experiment Execution
            if not state.is_stage_complete(PipelineStage.EXPERIMENT_EXECUTION):
                result = await self._run_experiments(state, progress_callback)
                stage_results["experiment_execution"] = result
                total_cost += result.cost_usd
                if not result.success:
                    raise PipelineError(result.error or "Experiment execution failed")
                state.mark_stage_complete(PipelineStage.EXPERIMENT_EXECUTION)
                self.state_manager.save(state)

            # Stage 6: Analysis
            if not state.is_stage_complete(PipelineStage.ANALYSIS):
                result = await self._run_analysis(state, progress_callback)
                stage_results["analysis"] = result
                total_cost += result.cost_usd
                if not result.success:
                    raise PipelineError(result.error or "Analysis failed")
                state.mark_stage_complete(PipelineStage.ANALYSIS)
                self.state_manager.save(state)

            # Stage 7: Figure Generation
            if not state.is_stage_complete(PipelineStage.FIGURE_GENERATION):
                result = await self._run_figure_generation(state, progress_callback)
                stage_results["figure_generation"] = result
                total_cost += result.cost_usd
                if not result.success:
                    raise PipelineError(result.error or "Figure generation failed")
                state.mark_stage_complete(PipelineStage.FIGURE_GENERATION)
                self.state_manager.save(state)

            # Stage 8: Paper Writing
            if not state.is_stage_complete(PipelineStage.PAPER_WRITING):
                result = await self._run_paper_writing(state, progress_callback)
                stage_results["paper_writing"] = result
                total_cost += result.cost_usd
                if not result.success:
                    raise PipelineError(result.error or "Paper writing failed")
                state.mark_stage_complete(PipelineStage.PAPER_WRITING)
                self.state_manager.save(state)

            # Get output paths
            paper_path = self.config.output_dir / f"{state.session_id}_paper"
            paper_path = paper_path.with_suffix(
                ".tex" if self.config.output_format == "latex" else ".md"
            )

            return PipelineOutput(
                session_id=state.session_id,
                research_topic=state.research_topic,
                stage_results=stage_results,
                paper_path=paper_path if paper_path.exists() else None,
                figure_paths=[Path(f) for f in state.figures],
                total_cost_usd=total_cost,
                success=True,
            )

        except Exception as e:
            self.state_manager.save(state)
            return PipelineOutput(
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
        """Run literature search stage."""
        start_time = datetime.now()

        if progress_callback:
            progress_callback(PipelineStage.LITERATURE_SEARCH, "Searching literature...")

        try:
            papers = await self.literature_searcher.search(
                state.research_topic,
                max_results=self.config.literature.max_papers,
                sources=self.config.literature.search_sources,
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
        """Run idea generation stage."""
        start_time = datetime.now()

        if progress_callback:
            progress_callback(PipelineStage.IDEA_GENERATION, "Generating research ideas...")

        try:
            ideas = await self.idea_generator.generate_ideas(
                state.research_topic,
                state.papers,
                num_ideas=5,
            )

            # Select best idea
            best_idea = await self.idea_generator.select_best_idea(ideas)
            state.research_idea = best_idea

            cost = self.api_manager.get_cost_summary()["total_cost_usd"]

            return StageResult(
                stage=PipelineStage.IDEA_GENERATION,
                success=True,
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                outputs={"idea_title": best_idea.title, "ideas_generated": len(ideas)},
                cost_usd=cost,
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
        """Run novelty check stage."""
        start_time = datetime.now()

        if progress_callback:
            progress_callback(PipelineStage.NOVELTY_CHECK, "Checking novelty...")

        try:
            if not state.research_idea:
                raise PipelineError("No research idea to check")

            novelty_score = await self.novelty_checker.check_novelty(
                state.research_idea,
                state.papers,
            )

            state.novelty_score = novelty_score.overall_score
            state.research_idea.novelty_score = novelty_score.overall_score

            # Refine idea if not novel enough
            if not self.novelty_checker.is_sufficiently_novel(novelty_score):
                if progress_callback:
                    progress_callback(PipelineStage.NOVELTY_CHECK, "Refining idea for novelty...")

                state.research_idea = await self.idea_generator.refine_idea(
                    state.research_idea,
                    novelty_score.explanation,
                    [],
                    novelty_score.suggestions,
                )

            cost = self.api_manager.get_cost_summary()["total_cost_usd"]

            return StageResult(
                stage=PipelineStage.NOVELTY_CHECK,
                success=True,
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                outputs={"novelty_score": novelty_score.overall_score},
                cost_usd=cost,
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
        """Run experiment design stage."""
        start_time = datetime.now()

        if progress_callback:
            progress_callback(PipelineStage.EXPERIMENT_DESIGN, "Designing experiments...")

        try:
            if not state.research_idea:
                raise PipelineError("No research idea for experiment design")

            plan = await self.experiment_designer.design_experiments(
                state.research_idea,
                budget_usd=self.config.api.total_budget_usd / 2,
            )

            state.experiment_plan = plan

            cost = self.api_manager.get_cost_summary()["total_cost_usd"]

            return StageResult(
                stage=PipelineStage.EXPERIMENT_DESIGN,
                success=True,
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                outputs={
                    "experiment_count": len(plan.experiments),
                    "estimated_cost": plan.estimated_cost_usd,
                },
                cost_usd=cost,
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
        """Run experiment execution stage."""
        start_time = datetime.now()

        if progress_callback:
            progress_callback(PipelineStage.EXPERIMENT_EXECUTION, "Running experiments...")

        try:
            if not state.experiment_plan:
                raise PipelineError("No experiment plan")

            def exp_progress(exp_id: str, current: int, total: int) -> None:
                if progress_callback:
                    progress_callback(
                        PipelineStage.EXPERIMENT_EXECUTION,
                        f"{exp_id}: {current}/{total}",
                    )

            results = await self.experiment_executor.execute_plan(
                state.experiment_plan,
                progress_callback=exp_progress,
            )

            state.experiment_results = results

            # Calculate total cost
            total_exp_cost = sum(r.cost_usd for r in results.values())
            state.total_cost_usd += total_exp_cost

            return StageResult(
                stage=PipelineStage.EXPERIMENT_EXECUTION,
                success=True,
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                outputs={
                    "completed": sum(1 for r in results.values() if r.status == "completed"),
                    "failed": sum(1 for r in results.values() if r.status == "failed"),
                },
                cost_usd=total_exp_cost,
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
        """Run analysis stage."""
        start_time = datetime.now()

        if progress_callback:
            progress_callback(PipelineStage.ANALYSIS, "Analyzing results...")

        try:
            if not state.experiment_plan or not state.experiment_results:
                raise PipelineError("No experiment results to analyze")

            validation = await self.validator.validate(
                state.experiment_plan.hypothesis,
                state.experiment_plan,
                state.experiment_results,
            )

            state.validation_result = validation

            cost = self.api_manager.get_cost_summary()["total_cost_usd"]

            return StageResult(
                stage=PipelineStage.ANALYSIS,
                success=True,
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                outputs={
                    "outcome": validation.outcome.value,
                    "confidence": validation.confidence,
                },
                cost_usd=cost,
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
        """Run figure generation stage."""
        start_time = datetime.now()

        if progress_callback:
            progress_callback(PipelineStage.FIGURE_GENERATION, "Generating figures...")

        try:
            figures = []

            # Generate result figures
            if state.experiment_results and state.validation_result:
                result_figs = self.result_figure_gen.generate_all_figures(
                    state.experiment_results,
                    state.validation_result.statistical_comparisons,
                    primary_metric="accuracy",
                )
                figures.extend(result_figs)

            # Generate method diagram
            if state.research_idea and self.config.tikz_enabled:
                try:
                    method_fig = await self.method_figure_gen.generate_method_diagram(
                        state.research_idea,
                        filename=f"{state.session_id}_method",
                    )
                    figures.append(method_fig)
                except Exception:
                    pass  # TikZ might not be available

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
        """Run paper writing stage."""
        start_time = datetime.now()

        if progress_callback:
            progress_callback(PipelineStage.PAPER_WRITING, "Writing paper...")

        try:
            if not state.research_idea or not state.validation_result:
                raise PipelineError("Missing required state for paper writing")

            paper = await self.section_writer.write_full_paper(
                state.research_idea,
                state.experiment_results,
                state.validation_result,
                state.papers,
                state.figures,
            )

            # Save paper
            paper_path = self.section_writer.save_paper(
                paper,
                filename=f"{state.session_id}_paper",
            )

            state.paper_sections = {
                "abstract": paper.abstract,
                "introduction": paper.introduction,
                "methodology": paper.methodology,
                "results": paper.results,
                "conclusion": paper.conclusion,
            }

            cost = self.api_manager.get_cost_summary()["total_cost_usd"]

            return StageResult(
                stage=PipelineStage.PAPER_WRITING,
                success=True,
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                outputs={"paper_path": str(paper_path)},
                cost_usd=cost,
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

    def list_sessions(self) -> list[str]:
        """List all available sessions."""
        return self.state_manager.list_sessions()

    def load_session(self, session_id: str) -> SessionState:
        """Load a session by ID."""
        return self.state_manager.load(session_id)
