# EvoScientist vs. Tao: Deep Comparison

## Architecture Philosophy

| | **EvoScientist** | **Tao** |
|---|---|---|
| **Framework** | LangChain + LangGraph + DeepAgents | Custom Python, zero framework dependencies |
| **Pipeline control** | Prompt-guided (LLM follows instructions) | Deterministic state machine (code-enforced) |
| **Agent communication** | LangGraph in-memory message passing | File-based IPC via workspace (JSON + Markdown) |
| **Pipeline stages** | 7 stages (prompt-described) | 20 stages (code-defined with transitions) |
| **Agent count** | 6 sub-agents | 35 agent definitions |

## What EvoScientist Does Better

### 1. Multi-channel messaging hub (10 platforms)

Telegram, Discord, Slack, WeChat, Feishu, DingTalk, QQ, Signal, Email, iMessage. Tao has CLI only. This is a genuine product advantage for real-time monitoring and collaboration.

### 2. Broad LLM provider support (15+ providers, 100+ models)

EvoScientist has a unified model factory supporting OpenAI, Google, NVIDIA, DeepSeek, Ollama, etc. Tao is Anthropic-only. This matters for cost optimization and model-specific strengths.

### 3. Adaptive tool selection middleware

When >26 tools exist, an LLM dynamically filters which tools are relevant per-turn. Smart token-saving mechanism. Tao has no equivalent.

### 4. MCP integration with agent-specific tool routing

`expose_to` field in MCP config routes specific tools to specific agents. Tao doesn't use MCP for agent tooling.

### 5. Interactive onboarding wizard

11-step Rich TUI wizard walks users through setup. Tao requires manual YAML editing.

### 6. Skill ecosystem (installable skills)

Built-in `find-skills` and `skill-creator` tools enable community extension. Tao's skills are project-internal.

## What Tao Does Better

### 1. Deterministic state machine vs. prompt-guided pipeline

This is the **biggest architectural difference**. Tao's state machine guarantees transitions: pivot/refine loops, quality gates, experiment monitoring, and writing revision rounds are all code-enforced. EvoScientist trusts the LLM to follow a 7-stage description in the system prompt -- there's no mechanism preventing it from skipping stages or getting stuck.

### 2. Cloud GPU compute (RunPod)

Tao has full pod lifecycle management (create, wait, upload, execute, download, terminate), GPU scheduling with topological sort, and cost controls. EvoScientist runs everything locally in a sandbox -- no remote compute at all.

### 3. Multi-agent debate (6 specialized perspectives + synthesis)

Tao's idea/result debates use 6 domain-specific agents (innovator, pragmatist, theoretical, contrarian, interdisciplinary, empiricist) with a dedicated synthesizer. EvoScientist has an "Idea Tournament" stage but delegates it as a single prompt concept -- no structured multi-perspective debate.

### 4. Dual-loop self-improvement

- **Inner loop**: Reflection -> quality trajectory -> action plan -> threshold adjustment
- **Outer loop**: Cross-project evolution -> agent-specific overlays -> lesson injection

EvoScientist has per-workspace memory persistence (`evo-memory`) but no cross-project evolution or agent-specific lesson overlays.

### 5. 7-stage writing pipeline

Outline -> assets -> parallel sections -> integration -> teaser -> final review -> LaTeX. Assets (figures/tables) generated before text. EvoScientist has a single writing-agent that produces Markdown.

### 6. Self-healing with circuit breaker

Auto-fix (zero-LLM mechanical repairs) -> skill-based repair -> circuit breaker (after N attempts, stop trying). EvoScientist has no error recovery system.

### 7. Crash recovery & atomic writes

`experiment_state.json` tracks task lifecycle (pending -> running -> done | dead). Atomic tmp-file swaps prevent corruption. EvoScientist has SQLite session persistence but no experiment crash recovery.

### 8. GPU-aware task scheduling

DAG-based topological sort, greedy GPU allocation, dependency tracking. EvoScientist has no GPU scheduling.

## Ideas Worth Stealing from EvoScientist

| Feature | Effort | Value |
|---------|--------|-------|
| **Messaging channels** (Telegram/Feishu/Slack notifications) | Medium | High -- real-time monitoring without SSH |
| **Multi-provider LLM factory** | Medium | High -- use cheaper models for light tasks, specialized models for others |
| **Adaptive tool selection** | Low | Medium -- reduce token waste when agents have many tools |
| **Interactive onboarding wizard** | Low | Medium -- better UX for new users |
| **Human-in-the-loop interrupt** | Low | Medium -- structured pause points for dataset/parameter questions |
| **Sandboxed local execution** | Low | Low -- useful for testing before deploying to RunPod |

## Summary

**EvoScientist** is a well-packaged **LLM agent product** -- great UX (multi-channel, TUI, onboarding), broad model support, and easy installation. But it's architecturally shallow: prompt-guided pipeline, local-only compute, no quality gates, no self-healing.

**Tao** is a **research production system** -- deterministic state machine, cloud GPU orchestration, multi-agent debates, dual-loop evolution, crash recovery. It trades polish for depth.

The main takeaway: **EvoScientist validates that multi-channel notifications and broad LLM support are table stakes for this kind of system.** Those are the two features most worth adding to Tao. The core pipeline architecture (state machine, GPU scheduling, self-healing, evolution) is significantly more robust in Tao.
