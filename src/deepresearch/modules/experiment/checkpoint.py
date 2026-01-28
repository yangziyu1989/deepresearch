"""Checkpoint management for experiment execution."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from deepresearch.core.exceptions import CheckpointError


@dataclass
class ExperimentCheckpoint:
    """Checkpoint state for an experiment."""

    experiment_id: str
    total_samples: int
    completed_samples: int = 0
    iterator_position: int = 0
    partial_results: list[dict[str, Any]] = field(default_factory=list)
    metrics_accumulator: dict[str, list[float]] = field(default_factory=dict)
    status: str = "running"  # running, completed, failed
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentCheckpoint":
        """Create from dictionary."""
        return cls(**data)


class CheckpointManager:
    """Manages experiment checkpoints for resumable execution."""

    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_checkpoint_path(self, experiment_id: str) -> Path:
        """Get the path for an experiment checkpoint."""
        return self.checkpoint_dir / f"{experiment_id}.checkpoint.json"

    def save(self, checkpoint: ExperimentCheckpoint) -> Path:
        """Save a checkpoint to disk."""
        checkpoint.updated_at = datetime.now().isoformat()
        checkpoint_path = self._get_checkpoint_path(checkpoint.experiment_id)

        try:
            # Write to temp file first, then rename for atomicity
            temp_path = checkpoint_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(checkpoint.to_dict(), f, indent=2)
            temp_path.rename(checkpoint_path)
            return checkpoint_path
        except Exception as e:
            raise CheckpointError(f"Failed to save checkpoint: {e}")

    def load(self, experiment_id: str) -> ExperimentCheckpoint | None:
        """Load a checkpoint from disk, returns None if not found."""
        checkpoint_path = self._get_checkpoint_path(experiment_id)

        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path) as f:
                data = json.load(f)
            return ExperimentCheckpoint.from_dict(data)
        except json.JSONDecodeError as e:
            raise CheckpointError(f"Invalid checkpoint file: {e}")
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint: {e}")

    def exists(self, experiment_id: str) -> bool:
        """Check if a checkpoint exists."""
        return self._get_checkpoint_path(experiment_id).exists()

    def delete(self, experiment_id: str) -> None:
        """Delete a checkpoint."""
        checkpoint_path = self._get_checkpoint_path(experiment_id)
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    def list_checkpoints(self) -> list[str]:
        """List all checkpoint experiment IDs."""
        return [
            p.stem.replace(".checkpoint", "")
            for p in self.checkpoint_dir.glob("*.checkpoint.json")
        ]

    def cleanup_completed(self) -> int:
        """Delete checkpoints for completed experiments. Returns count deleted."""
        deleted = 0
        for experiment_id in self.list_checkpoints():
            checkpoint = self.load(experiment_id)
            if checkpoint and checkpoint.status == "completed":
                self.delete(experiment_id)
                deleted += 1
        return deleted


@dataclass
class SessionCheckpoint:
    """Higher-level checkpoint for entire research sessions."""

    session_id: str
    experiment_checkpoints: dict[str, str] = field(
        default_factory=dict
    )  # experiment_id -> status
    current_experiment: str | None = None
    completed_experiments: list[str] = field(default_factory=list)
    failed_experiments: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class SessionCheckpointManager:
    """Manages session-level checkpoints."""

    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_manager = CheckpointManager(checkpoint_dir / "experiments")

    def _get_session_path(self, session_id: str) -> Path:
        """Get the path for a session checkpoint."""
        return self.checkpoint_dir / f"{session_id}.session.json"

    def save_session(self, checkpoint: SessionCheckpoint) -> Path:
        """Save a session checkpoint."""
        checkpoint.updated_at = datetime.now().isoformat()
        session_path = self._get_session_path(checkpoint.session_id)

        try:
            temp_path = session_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(asdict(checkpoint), f, indent=2)
            temp_path.rename(session_path)
            return session_path
        except Exception as e:
            raise CheckpointError(f"Failed to save session checkpoint: {e}")

    def load_session(self, session_id: str) -> SessionCheckpoint | None:
        """Load a session checkpoint."""
        session_path = self._get_session_path(session_id)

        if not session_path.exists():
            return None

        try:
            with open(session_path) as f:
                data = json.load(f)
            return SessionCheckpoint(**data)
        except Exception as e:
            raise CheckpointError(f"Failed to load session checkpoint: {e}")

    def save_experiment(self, checkpoint: ExperimentCheckpoint) -> Path:
        """Save an experiment checkpoint."""
        return self.experiment_manager.save(checkpoint)

    def load_experiment(self, experiment_id: str) -> ExperimentCheckpoint | None:
        """Load an experiment checkpoint."""
        return self.experiment_manager.load(experiment_id)
