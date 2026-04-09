# TODOS

## Wire up CHECKPOINT_DIRS for crash recovery

**What:** Implement checkpoint save/restore using the `CHECKPOINT_DIRS` mapping in `tao/orchestration/constants.py`.

**Why:** The mapping is defined (maps stages to workspace directories) but nothing in `tao/` imports or reads it. If the pipeline crashes mid-`writing_assets`, there's no way to restore partial progress. As pipelines run for hours on RunPod GPUs, crash recovery saves both time and money.

**Context:** `CHECKPOINT_DIRS` maps `writing_assets` → `writing/figures`, `writing_sections` → `writing/sections`, etc. The checkpoint reader should: (1) detect which stages have completed artifacts, (2) allow the orchestrator to skip completed stages on restart. Also needs to include `writing/tables` alongside `writing/figures` for the `writing_assets` stage.

**Depends on:** Nothing. Can be implemented independently.

## Writing revision loop should reach writing_assets

**What:** Add a state machine path from `writing_final_review` back to `writing_assets` when the reviewer flags figure/table defects.

**Why:** Currently, a low score in `writing_final_review` can only loop back to `writing_integrate` (state_machine.py:58-62). If the reviewer identifies that a figure is misleading, a table has wrong numbers, or the method diagram is unclear, the state machine can't regenerate those assets. The defect persists through all revision rounds.

**Context:** The fix would check the `writing_final_review` result text for specific keywords (e.g., "REGENERATE_ASSETS") and route to `writing_assets` instead of `writing_integrate`. This also means the figure/table generators need to handle partial regeneration (only regenerate flagged assets, not all of them).

**Depends on:** Checkpoint system (above) would make this safer, since regenerating assets while preserving good ones requires knowing what exists.
