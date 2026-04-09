# Autonomy Level System: Human-in-Loop vs Full Auto

**Date:** 2026-04-09
**Status:** Design complete, ready for implementation

## Problem

The Tao research pipeline currently runs either fully autonomously (making suboptimal choices at creative/strategic decision points) or requires ad-hoc human interruption. There's no structured way to involve humans at key moments while keeping the pipeline running smoothly.

## Design

### Binary Autonomy Levels

| Mode | Name | Workspace Suffix | Behavior |
|:----:|------|:----------------:|----------|
| **A0** | Full Auto | `_A0` | Never pause. Auto-picks best-scoring option at every decision point. |
| **A1** | Human-in-Loop | `_A1` | Pause at decision points. Present choices via AskUserQuestion (single/multiple choice). Wait for human before proceeding. |

### Configuration

```yaml
# config.yaml
autonomy_level: 0  # 0=full auto, 1=human-in-loop
```

### Workspace Naming

```
{topic_slug}_{random_id}_A{level}/
```

Examples:
```
early_exit_selfspeculative_decoding_59208_A0/
early_exit_selfspeculative_decoding_59208_A1/
```

The `_A{level}` suffix is appended during `tao init` and visible in all file paths, logs, and status output.

## Implementation

### Decision Dataclass

```python
@dataclass
class Option:
    value: str          # machine-readable key
    label: str          # human-readable label
    description: str    # explanation shown to user
    score: float = 0.0  # auto-ranking score (higher = better)

@dataclass
class Decision:
    stage: str                    # pipeline stage (e.g., "idea_validation_decision")
    question: str                 # e.g., "Which figure candidate to use?"
    options: List[Option]         # sorted by score (best first)
    multi_select: bool = False    # single or multiple choice
    context: str = ""             # extra context shown to user
```

### Core Decide Function

```python
# In orchestrate.py or lifecycle.py
async def decide(self, decision: Decision) -> str:
    """Route decision based on autonomy level."""
    # Log the decision point
    self._log_decision_start(decision)
    
    if self.config.autonomy_level == 0:
        # A0: auto-pick best option by score
        chosen = decision.options[0].value
        self._log_decision_end(decision, chosen, chosen_by="auto")
        return chosen
    else:
        # A1: pause and ask human via AskUserQuestion
        chosen = await self.ask_human(decision)
        self._log_decision_end(decision, chosen, chosen_by="human")
        return chosen
```

### AskUserQuestion Integration (A1 mode)

When `decide()` is called in A1 mode, it formats the `Decision` into Claude Code's `AskUserQuestion` format:

```python
async def ask_human(self, decision: Decision) -> str:
    """Present decision to user via AskUserQuestion interface."""
    # Maps to Claude Code's native choice UI:
    # - Single select for most decisions
    # - Multi select for "which experiments to run" type decisions
    # - User can always pick "Other" and type free text
    
    question = {
        "question": decision.question,
        "header": decision.stage[:12],  # max 12 chars
        "options": [
            {"label": opt.label, "description": opt.description}
            for opt in decision.options
        ],
        "multiSelect": decision.multi_select,
    }
    
    # Block until user responds
    response = await ask_user_question([question])
    return response
```

### Decision Points in Pipeline

Every stage where the pipeline currently makes a choice becomes a `decide()` call:

| Stage | Decision | Options (typical) |
|-------|----------|-------------------|
| `idea_validation_decision` | Pivot, refine, or advance? | advance, refine, pivot |
| `experiment_decision` | Proceed to writing or iterate? | proceed, iterate, pivot |
| `quality_gate` | Pass or do another iteration? | pass, iterate |
| `pilot_experiments` | Go/no-go after pilot? | full_go, partial_go, no_go |
| PaperBanana figure gen | Which candidate? | candidate_0, candidate_1, ... |
| Paper framing | Which narrative angle? | framing_a, framing_b, framing_c |
| Result debate | Accept conclusions? | accept, re_run, add_experiments |

### Decision Log

Every decision is logged to `workspace/logs/decisions.jsonl`:

```json
{
  "timestamp": "2026-04-09T12:34:56",
  "stage": "pilot_experiments",
  "question": "Pilot failed. Pivot to layer skipping or train exit heads?",
  "options": [
    {"value": "pivot_layer_skip", "label": "Pivot to layer skipping", "score": 0.85},
    {"value": "train_exit_heads", "label": "Train cheap exit heads", "score": 0.60},
    {"value": "retry_with_fix", "label": "Retry with different params", "score": 0.40}
  ],
  "chosen": "pivot_layer_skip",
  "chosen_by": "auto",
  "score": 0.85,
  "reasoning": "Layer skipping has stronger literature support",
  "user_notes": null
}
```

For A1 (human), `chosen_by` = `"human"` and `user_notes` captures any free-text the user typed.

### Log Purposes

1. **Reproducibility** — A0 runs can be audited to see what was auto-chosen
2. **Learning** — compare human vs auto decisions across projects to improve auto-picker
3. **Resume** — if a session crashes, decisions log shows where to restart

## CLI Integration

### Init with autonomy level
```bash
tao init "research topic" --autonomy 1    # human-in-loop
tao init "research topic" --autonomy 0    # full auto (default)
tao init "research topic"                 # reads from config.yaml
```

### Status shows autonomy level
```bash
tao status .
# Output:
# {
#   "stage": "experiment_cycle",
#   "iteration": 0,
#   "autonomy_level": 1,
#   "pending_decisions": 0,
#   ...
# }
```

### Review decisions
```bash
tao decisions .              # show all decisions made
tao decisions . --pending    # show decisions waiting for human (A1 only)
```

## Migration

Existing workspaces without `_A{level}` suffix default to A0 (full auto) behavior. No breaking changes.

## Future Extensions

- **A0 with review** — auto-proceed but flag low-confidence decisions for post-hoc review
- **Per-stage override** — `autonomy_overrides: {quality_gate: 1, pilot_experiments: 1}` to mix auto/human per stage
- **Learning from decisions** — use decision log to train a scorer that better matches human preferences
