"""Tests for experiment task helpers and launcher utilities."""
from pathlib import Path

from tao.experiment_launcher import stage_workspace_bundle
from tao.experiment_tasks import (
    choose_task_script,
    pending_phase_tasks,
    phase_task_ids,
    resolve_dataset_info,
    resolve_model_id,
    summarize_phase,
)
from tao.gpu_scheduler import save_gpu_progress


def _write_plan(workspace: Path) -> None:
    plan_dir = workspace / "plan"
    plan_dir.mkdir(parents=True)
    (plan_dir / "task_plan.json").write_text(
        """
{
  "tasks": [
    {"id": "pilot_dense", "type": "pilot", "code_template": "train_peft_baseline.py", "depends_on": []},
    {"id": "pilot_routed", "type": "pilot", "code_template": "train_routed_peft.py", "depends_on": ["pilot_dense"]},
    {"id": "full_routed", "type": "full", "code_template": "train_routed_peft.py", "depends_on": ["pilot_routed"]}
  ]
}
""".strip(),
        encoding="utf-8",
    )


def test_resolve_aliases():
    assert resolve_model_id("Qwen2.5-7B-Instruct") == "Qwen/Qwen2.5-7B-Instruct"
    assert resolve_dataset_info("LongAlpaca-12k pilot subset")["dataset_id"] == "Yukang/LongAlpaca-12k"


def test_phase_task_ids_respect_dependencies(tmp_path):
    _write_plan(tmp_path)
    assert phase_task_ids(tmp_path, "pilot") == ["pilot_dense", "pilot_routed"]
    assert phase_task_ids(tmp_path, "full") == ["full_routed"]


def test_pending_phase_tasks_filters_completed(tmp_path):
    _write_plan(tmp_path)
    save_gpu_progress(tmp_path, {"running": {}, "completed": ["pilot_dense"]})
    pending = pending_phase_tasks(tmp_path, "pilot")
    assert [task["id"] for task in pending] == ["pilot_routed"]


def test_choose_task_script():
    assert choose_task_script({"code_template": "train_routed_peft.py --route_fraction 0.25"}) == "scripts/train_routed_peft.py"
    assert choose_task_script({"code_template": "train_peft_baseline.py"}) == "scripts/train_peft_baseline.py"


def test_summarize_phase(tmp_path):
    _write_plan(tmp_path)
    save_gpu_progress(tmp_path, {"running": {}, "completed": ["pilot_dense"]})
    summary = summarize_phase(tmp_path, "pilot")
    assert summary["total"] == 2
    assert summary["completed"] == 1
    assert summary["pending"] == 1


def test_stage_workspace_bundle_includes_runtime_and_workspace(tmp_path):
    (tmp_path / "topic.txt").write_text("test topic", encoding="utf-8")
    _write_plan(tmp_path)
    bundle = Path(stage_workspace_bundle(tmp_path))
    try:
        assert (bundle / "tao").exists()
        assert (bundle / "scripts").exists()
        assert (bundle / "pyproject.toml").exists()
        assert (bundle / "topic.txt").read_text(encoding="utf-8") == "test topic"
        assert (bundle / "plan" / "task_plan.json").exists()
    finally:
        import shutil
        shutil.rmtree(bundle, ignore_errors=True)
