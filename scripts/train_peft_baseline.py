#!/usr/bin/env python3
"""Dense PEFT baseline trainer for Tao task plans."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tao.experiment_tasks import load_task
from tao.llm_experiment import run_training_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one dense PEFT experiment task")
    parser.add_argument("--workspace", required=True, help="Workspace root")
    parser.add_argument("--task-id", required=True, help="Task id in plan/task_plan.json")
    args = parser.parse_args()

    task = load_task(args.workspace, args.task_id)
    run_training_task(task, args.workspace, routed=False)


if __name__ == "__main__":
    main()
