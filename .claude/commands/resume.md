# Resume Paused Pipeline

**Usage:** `/resume <workspace_path>`

Resume a pipeline that was explicitly stopped with `/stop`.

## Steps

1. Clear any pause/stop markers in workspace `status.json`.
2. Resume the orchestration loop from the current stage.
