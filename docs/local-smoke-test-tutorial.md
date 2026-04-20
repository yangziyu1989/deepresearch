# Local Smoke Test Tutorial (uv + Python 3.11+)

This tutorial shows how to run DeepResearch's **local no-cloud smoke test** end to end.

Scope:
- No RunPod pods
- No external paid compute
- Verifies the orchestrator/state-machine pipeline path locally

## 1) Prerequisites

- macOS/Linux shell
- `uv` installed
- Python **3.11+** available to `uv`

Check uv:

```bash
uv --version
```

## 2) Create/Recreate venv with Python 3.11+

From repo root:

```bash
cd /Users/zzyang/GitHub/deepresearch
rm -rf .venv .venv*
uv venv --python 3.11 .venv
```

Verify interpreter version:

```bash
source .venv/bin/activate
python --version
which python
```

Expected:
- Python version is `3.11.x` (or above)
- Python path points to `.../deepresearch/.venv/bin/python`

## 3) Install project dependencies into this venv

Use `uv pip` to avoid mixing global/conda environment:

```bash
uv pip install --python .venv/bin/python -e ".[dev]"
```

### Command simplification notes (from our discussion)

- Why we sometimes use `./.venv/bin/python ...` explicitly:
	It guarantees the command runs in the project venv and avoids accidentally using system/conda Python.

- If you want shorter commands for daily use:

```bash
source .venv/bin/activate
python -m tao.demo
```

- Alternative without manual activate:

```bash
uv run python -m tao.demo
```

- About `uv pip install -e ".[dev]"`:
	It is fine **if your target venv is already active**. If not active, prefer the explicit form below for safety:

```bash
uv pip install --python .venv/bin/python -e ".[dev]"
```

## 4) Run local smoke test

```bash
./.venv/bin/python -m tao.demo
```

Expected key output:
- `Stages visited: 20`
- `Final stage: done`
- `All checks passed: True`

## 5) What this smoke test validates

`tao.demo` performs a dry-run for the full pipeline and checks:
- workspace initialization
- stage transitions via state machine
- action generation across stages
- final completion status and basic artifacts

Code reference:
- `tao/demo.py`

## 6) If something fails

1. Ensure command uses the venv interpreter:

```bash
./.venv/bin/python --version
```

2. Reinstall dependencies in this exact venv:

```bash
uv pip install --python .venv/bin/python -e ".[dev]"
```

3. Re-run smoke test:

```bash
./.venv/bin/python -m tao.demo
```

## 7) Optional next checks

Run unit tests (still local, no cloud):

```bash
./.venv/bin/python -m pytest tests/ -v
```

Quick CLI sanity check:

```bash
./.venv/bin/python -m tao.cli --help
```

## 8) Anti-mix environment self-check (recommended before running commands)

Run these 3 commands in repo root:

```bash
which python
python --version
python -c "import sys; print(sys.executable)"
```

Pass criteria:
- Python version is `3.11.x` (or above)
- Interpreter path points to this project venv:
	`/Users/zzyang/GitHub/deepresearch/.venv/bin/python`

If not matched, use one of these fixes:

```bash
# Option A: activate venv for current shell
source .venv/bin/activate

# Option B: bypass shell state and run through uv
uv run python -m tao.demo

# Option C: always target the venv interpreter explicitly
uv pip install --python .venv/bin/python -e ".[dev]"
./.venv/bin/python -m tao.demo
```

---

## Verified run on this machine (2026-04-16)

Command:

```bash
./.venv/bin/python -m tao.demo
```

Observed result:
- 20 stages visited
- final stage is `done`
- all checks passed
