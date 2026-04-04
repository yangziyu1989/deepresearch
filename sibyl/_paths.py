"""Central path resolution for the Sibyl system."""
from __future__ import annotations
from pathlib import Path
import os


def sibyl_root() -> Path:
    """Return the Sibyl system root (repo checkout).

    Uses SIBYL_ROOT env var if set, otherwise derives from this file's location.
    """
    env = os.environ.get("SIBYL_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parent.parent


def system_data_dir() -> Path:
    """Return ~/.sibyl/ for cross-project persistent data."""
    return Path.home() / ".sibyl"


def prompts_dir() -> Path:
    """Return the directory containing prompt templates."""
    return Path(__file__).resolve().parent / "prompts"


def global_config_path() -> Path:
    """Return ~/.sibyl/config.yaml path."""
    return system_data_dir() / "config.yaml"
