"""Pipeline stage definitions and execution."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from deepresearch.core.config import PipelineStage
from deepresearch.core.state import SessionState


@dataclass
class StageResult:
    """Result of a pipeline stage execution."""

    stage: PipelineStage
    success: bool
    start_time: str
    end_time: str
    duration_seconds: float
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    cost_usd: float = 0.0


class PipelineStageRunner(ABC):
    """Abstract base class for pipeline stage runners."""

    @property
    @abstractmethod
    def stage(self) -> PipelineStage:
        """The pipeline stage this runner handles."""
        ...

    @property
    @abstractmethod
    def dependencies(self) -> list[PipelineStage]:
        """Stages that must complete before this one."""
        ...

    @abstractmethod
    async def run(
        self,
        state: SessionState,
        progress_callback: Callable[[str], None] | None = None,
    ) -> StageResult:
        """Execute the stage and return results."""
        ...

    def can_run(self, state: SessionState) -> bool:
        """Check if all dependencies are satisfied."""
        for dep in self.dependencies:
            if not state.is_stage_complete(dep):
                return False
        return True


class StageExecutor:
    """Executes pipeline stages with dependency management."""

    def __init__(self) -> None:
        self._runners: dict[PipelineStage, PipelineStageRunner] = {}

    def register(self, runner: PipelineStageRunner) -> None:
        """Register a stage runner."""
        self._runners[runner.stage] = runner

    def get_execution_order(
        self,
        stages: list[PipelineStage],
    ) -> list[PipelineStage]:
        """Determine execution order respecting dependencies."""
        # Topological sort
        visited: set[PipelineStage] = set()
        order: list[PipelineStage] = []

        def visit(stage: PipelineStage) -> None:
            if stage in visited:
                return
            visited.add(stage)
            runner = self._runners.get(stage)
            if runner:
                for dep in runner.dependencies:
                    visit(dep)
            order.append(stage)

        for stage in stages:
            visit(stage)

        return order

    async def execute(
        self,
        stage: PipelineStage,
        state: SessionState,
        progress_callback: Callable[[str], None] | None = None,
    ) -> StageResult:
        """Execute a single stage."""
        runner = self._runners.get(stage)
        if not runner:
            return StageResult(
                stage=stage,
                success=False,
                start_time=datetime.now().isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=0,
                error=f"No runner registered for stage: {stage}",
            )

        if not runner.can_run(state):
            missing = [
                dep.value for dep in runner.dependencies
                if not state.is_stage_complete(dep)
            ]
            return StageResult(
                stage=stage,
                success=False,
                start_time=datetime.now().isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=0,
                error=f"Dependencies not satisfied: {missing}",
            )

        return await runner.run(state, progress_callback)

    async def execute_pipeline(
        self,
        stages: list[PipelineStage],
        state: SessionState,
        progress_callback: Callable[[PipelineStage, str], None] | None = None,
    ) -> list[StageResult]:
        """Execute multiple stages in dependency order."""
        order = self.get_execution_order(stages)
        results: list[StageResult] = []

        for stage in order:
            if state.is_stage_complete(stage):
                continue

            if progress_callback:
                progress_callback(stage, "Starting...")

            result = await self.execute(
                stage,
                state,
                lambda msg: progress_callback(stage, msg) if progress_callback else None,
            )
            results.append(result)

            if result.success:
                state.mark_stage_complete(stage)
            else:
                # Stop on failure
                break

        return results
