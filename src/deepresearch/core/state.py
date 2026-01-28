"""Session state management for DeepResearch."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from deepresearch.core.config import (
    ExperimentPlan,
    HypothesisOutcome,
    PipelineConfig,
    PipelineStage,
    ResearchIdea,
    ValidationResult,
)
from deepresearch.core.exceptions import CheckpointError


@dataclass
class Paper:
    """Represents a retrieved paper."""

    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    year: int
    source: str  # arxiv, semantic_scholar
    url: str
    citations: int = 0
    embedding: list[float] | None = None


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""

    experiment_id: str
    status: str  # pending, running, completed, failed
    metrics: dict[str, float] = field(default_factory=dict)
    raw_outputs: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    cost_usd: float = 0.0


@dataclass
class SessionState:
    """Complete state of a research session."""

    session_id: str
    research_topic: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Pipeline progress
    current_stage: PipelineStage = PipelineStage.LITERATURE_SEARCH
    completed_stages: list[str] = field(default_factory=list)

    # Literature search results
    papers: list[Paper] = field(default_factory=list)

    # Research idea
    research_idea: ResearchIdea | None = None
    novelty_score: float = 0.0

    # Experiment tracking
    experiment_plan: ExperimentPlan | None = None
    experiment_results: dict[str, ExperimentResult] = field(default_factory=dict)

    # Analysis results
    validation_result: ValidationResult | None = None

    # Generated outputs
    figures: list[str] = field(default_factory=list)  # Paths to generated figures
    paper_sections: dict[str, str] = field(default_factory=dict)  # Section name -> content

    # Cost tracking
    total_cost_usd: float = 0.0
    api_calls: int = 0

    def mark_stage_complete(self, stage: PipelineStage) -> None:
        """Mark a pipeline stage as complete."""
        if stage.value not in self.completed_stages:
            self.completed_stages.append(stage.value)
        self.updated_at = datetime.now().isoformat()

    def is_stage_complete(self, stage: PipelineStage) -> bool:
        """Check if a pipeline stage is complete."""
        return stage.value in self.completed_stages

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary for serialization."""
        data = asdict(self)
        # Convert enums to strings
        data["current_stage"] = self.current_stage.value
        if self.validation_result:
            data["validation_result"]["outcome"] = self.validation_result.outcome.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionState":
        """Create state from dictionary."""
        # Convert stage string back to enum
        data["current_stage"] = PipelineStage(data["current_stage"])

        # Reconstruct nested dataclasses
        if data.get("papers"):
            data["papers"] = [Paper(**p) for p in data["papers"]]

        if data.get("research_idea"):
            data["research_idea"] = ResearchIdea(**data["research_idea"])

        if data.get("experiment_plan"):
            from deepresearch.core.config import ExperimentConfig, ProviderType
            plan_data = data["experiment_plan"]
            experiments = []
            for exp in plan_data.get("experiments", []):
                exp["provider"] = ProviderType(exp["provider"])
                experiments.append(ExperimentConfig(**exp))
            plan_data["experiments"] = experiments
            data["experiment_plan"] = ExperimentPlan(**plan_data)

        if data.get("experiment_results"):
            data["experiment_results"] = {
                k: ExperimentResult(**v) for k, v in data["experiment_results"].items()
            }

        if data.get("validation_result"):
            from deepresearch.core.config import StatisticalComparison
            vr = data["validation_result"]
            vr["outcome"] = HypothesisOutcome(vr["outcome"])
            vr["statistical_comparisons"] = [
                StatisticalComparison(**sc) for sc in vr.get("statistical_comparisons", [])
            ]
            data["validation_result"] = ValidationResult(**vr)

        return cls(**data)


class StateManager:
    """Manages saving and loading session state."""

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def _get_state_path(self, session_id: str) -> Path:
        """Get the path for a session state file."""
        return self.session_dir / f"{session_id}.json"

    def save(self, state: SessionState) -> Path:
        """Save session state to disk."""
        state.updated_at = datetime.now().isoformat()
        state_path = self._get_state_path(state.session_id)

        try:
            with open(state_path, "w") as f:
                json.dump(state.to_dict(), f, indent=2)
            return state_path
        except Exception as e:
            raise CheckpointError(f"Failed to save session state: {e}")

    def load(self, session_id: str) -> SessionState:
        """Load session state from disk."""
        state_path = self._get_state_path(session_id)

        if not state_path.exists():
            raise CheckpointError(f"Session not found: {session_id}")

        try:
            with open(state_path) as f:
                data = json.load(f)
            return SessionState.from_dict(data)
        except json.JSONDecodeError as e:
            raise CheckpointError(f"Invalid session file: {e}")
        except Exception as e:
            raise CheckpointError(f"Failed to load session state: {e}")

    def exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        return self._get_state_path(session_id).exists()

    def list_sessions(self) -> list[str]:
        """List all available sessions."""
        return [p.stem for p in self.session_dir.glob("*.json")]

    def delete(self, session_id: str) -> None:
        """Delete a session."""
        state_path = self._get_state_path(session_id)
        if state_path.exists():
            state_path.unlink()


def create_session(
    research_topic: str,
    config: PipelineConfig,
) -> SessionState:
    """Create a new research session."""
    import uuid

    session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    return SessionState(
        session_id=session_id,
        research_topic=research_topic,
    )
