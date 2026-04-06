# Stop Research Pipeline

**Usage:** `/stop <workspace_path>`

Gracefully pause the research pipeline at the next safe checkpoint.

## Steps

1. Set `stop_requested: true` in workspace `status.json`.
2. The orchestration loop will stop at the next safe checkpoint.
3. Running experiments will continue to completion on RunPod.
