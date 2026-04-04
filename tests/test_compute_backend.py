"""Tests for compute backend."""
from unittest.mock import patch, MagicMock
from sibyl.compute.base import ComputeBackend
from sibyl.compute.runpod_backend import RunPodBackend
from sibyl.compute import get_backend
from sibyl.config import Config


def test_backend_type():
    cfg = Config()
    backend = RunPodBackend(cfg)
    assert backend.backend_type == "runpod"


def test_project_dir():
    cfg = Config()
    backend = RunPodBackend(cfg)
    assert backend.project_dir("my_project") == "/workspace/projects/my_project"


def test_env_cmd():
    cfg = Config()
    backend = RunPodBackend(cfg)
    cmd = backend.env_cmd("my_project")
    assert "cd" in cmd
    assert "my_project" in cmd


def test_gpu_poll_script():
    cfg = Config()
    backend = RunPodBackend(cfg)
    script = backend.gpu_poll_script(
        candidate_gpu_ids=[0, 1],
        threshold_mb=2000,
        poll_interval_sec=30,
        max_polls=10,
        marker_file="/tmp/gpu_marker.json",
    )
    assert "nvidia-smi" in script
    assert "gpu_marker.json" in script
    assert "THRESHOLD=2000" in script


def test_gpu_poll_script_aggressive():
    cfg = Config()
    backend = RunPodBackend(cfg)
    script = backend.gpu_poll_script(
        candidate_gpu_ids=[0],
        threshold_mb=2000,
        poll_interval_sec=30,
        max_polls=5,
        marker_file="/tmp/m.json",
        aggressive_mode=True,
        aggressive_threshold_pct=30,
    )
    assert "AGGRESSIVE=true" in script
    assert "AGG_PCT=30" in script


def test_experiment_monitor_script():
    cfg = Config()
    backend = RunPodBackend(cfg)
    script = backend.experiment_monitor_script(
        project_dir="/workspace/projects/test",
        task_ids=["train_baseline", "train_ablation"],
        poll_interval_sec=30,
        timeout_minutes=120,
        marker_file="/tmp/monitor.json",
        workspace_path="/local/ws",
    )
    assert "train_baseline" in script
    assert "train_ablation" in script
    assert "TIMEOUT_SEC" in script
    assert "_DONE" in script
    assert ".pid" in script


def test_get_backend():
    cfg = Config()
    backend = get_backend(cfg)
    assert isinstance(backend, RunPodBackend)
    assert backend.backend_type == "runpod"


def test_create_pod_no_api_key():
    import os
    import sys
    # Clear env var BEFORE constructing the backend so __init__ sees no key
    old_env = os.environ.pop("RUNPOD_API_KEY", None)
    # Mock the runpod module so the import inside create_pod succeeds
    mock_runpod = MagicMock()
    had_runpod = "runpod" in sys.modules
    prev_mod = sys.modules.get("runpod")
    sys.modules["runpod"] = mock_runpod
    try:
        cfg = Config()
        cfg.runpod_api_key = ""
        backend = RunPodBackend(cfg)
        import pytest
        with pytest.raises(ValueError, match="RUNPOD_API_KEY"):
            backend.create_pod("test-pod")
    finally:
        if old_env is not None:
            os.environ["RUNPOD_API_KEY"] = old_env
        if had_runpod:
            sys.modules["runpod"] = prev_mod
        else:
            sys.modules.pop("runpod", None)


def test_abstract_methods():
    """Verify ComputeBackend can't be instantiated."""
    import pytest
    with pytest.raises(TypeError):
        ComputeBackend()
