"""Task-plan helpers for runnable LLM fine-tuning experiments."""
from __future__ import annotations

import json
from pathlib import Path

from tao.gpu_scheduler import load_gpu_progress, load_task_plan, topological_sort


MODEL_ALIASES = {
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
}

DATASET_ALIASES = {
    "LongAlpaca-12k": {"dataset_id": "Yukang/LongAlpaca-12k", "split": "train"},
    "LongAlpaca-12k pilot subset": {"dataset_id": "Yukang/LongAlpaca-12k", "split": "train"},
    "LongAlpaca-12k full set": {"dataset_id": "Yukang/LongAlpaca-12k", "split": "train"},
}


def resolve_model_id(name: str) -> str:
    """Resolve a workspace-facing model alias to a concrete model repo id."""
    return MODEL_ALIASES.get(name, name)


def resolve_dataset_info(name: str) -> dict:
    """Resolve a workspace-facing dataset alias to dataset metadata."""
    if name in DATASET_ALIASES:
        return dict(DATASET_ALIASES[name])
    return {"dataset_id": name, "split": "train"}


def load_task(workspace_root: str | Path, task_id: str) -> dict:
    """Load a single task definition by id."""
    plan = load_task_plan(workspace_root)
    for task in plan.get("tasks", []):
        if task.get("id") == task_id:
            return task
    raise KeyError(f"Task '{task_id}' not found in task_plan.json")


def phase_task_ids(workspace_root: str | Path, phase: str) -> list[str]:
    """Return ordered task ids for a plan phase."""
    plan = load_task_plan(workspace_root)
    if phase == "pilot":
        ids = plan.get("pilot_tasks") or [
            t["id"] for t in plan.get("tasks", []) if t.get("type") == "pilot"
        ]
    elif phase == "full":
        ids = plan.get("full_tasks") or [
            t["id"] for t in plan.get("tasks", []) if t.get("type") == "full"
        ]
    else:
        raise ValueError(f"Unsupported phase '{phase}'")

    order = topological_sort(plan.get("tasks", [])) if plan.get("tasks") else []
    order_map = {task_id: idx for idx, task_id in enumerate(order)}
    return sorted(ids, key=lambda task_id: order_map.get(task_id, len(order)))


def pending_phase_tasks(workspace_root: str | Path, phase: str) -> list[dict]:
    """Return phase tasks that are not yet completed."""
    completed = set(load_gpu_progress(workspace_root).get("completed", []))
    return [
        load_task(workspace_root, task_id)
        for task_id in phase_task_ids(workspace_root, phase)
        if task_id not in completed
    ]


def choose_task_script(task: dict) -> str:
    """Map a task code template to the concrete training script."""
    template = str(task.get("code_template", "")).lower()
    if "routed" in template:
        return "scripts/train_routed_peft.py"
    return "scripts/train_peft_baseline.py"


def summarize_phase(workspace_root: str | Path, phase: str) -> dict:
    """Return a compact phase summary for logs and CLI output."""
    all_ids = phase_task_ids(workspace_root, phase)
    progress = load_gpu_progress(workspace_root)
    completed = set(progress.get("completed", []))
    return {
        "phase": phase,
        "total": len(all_ids),
        "completed": len([task_id for task_id in all_ids if task_id in completed]),
        "pending": len([task_id for task_id in all_ids if task_id not in completed]),
        "task_ids": all_ids,
    }


def write_phase_summary(workspace_root: str | Path, phase: str) -> Path:
    """Persist a phase summary into the workspace."""
    out = Path(workspace_root) / "exp" / f"{phase}_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summarize_phase(workspace_root, phase), indent=2), encoding="utf-8")
    return out
