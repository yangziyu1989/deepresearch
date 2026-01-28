# DeepResearch

Automated AI Research Pipeline that conducts literature research, proposes novel ideas, designs and executes experiments, analyzes results, and generates papers.

## Features

- **Literature Search**: Search arXiv and Semantic Scholar for related papers
- **Novelty Checking**: Embedding-based similarity + LLM-powered methodology comparison
- **Idea Generation**: LLM-based research idea generation with feasibility/novelty/impact scoring
- **Experiment Design**: Automatic experiment plan generation with baselines and ablations
- **Experiment Execution**: Parallel execution with checkpointing and resume support
- **Statistical Analysis**: t-tests, Wilcoxon tests, effect sizes, confidence intervals
- **Figure Generation**: Matplotlib result plots + TikZ architecture diagrams
- **Paper Writing**: LaTeX/Markdown generation for all paper sections

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/deepresearch.git
cd deepresearch

# Install the package
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

## Configuration

Set up API keys (at least one required):

```bash
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
```

Optional: For TikZ figure generation, install LaTeX:

```bash
# macOS
brew install --cask mactex

# Ubuntu/Debian
sudo apt-get install texlive-full
```

## Usage

### Run Vision Research (MNIST, CIFAR-10)

```bash
# Run vision experiments on MNIST and CIFAR-10
deepresearch vision "zero-shot image classification with multimodal LLMs" --datasets mnist,cifar10 --samples 100

# Run on specific dataset
deepresearch vision "few-shot learning for image classification" --datasets cifar10 --budget 10
```

### Run NLP Pipeline

```bash
deepresearch run "Improve chain-of-thought reasoning with self-consistency" --budget 50
```

### Search Literature

```bash
deepresearch search "transformer attention mechanisms" --max 20
```

### Resume a Session

```bash
deepresearch run --resume SESSION_ID "topic"
```

### List Sessions

```bash
deepresearch list-sessions
```

### Show Session Details

```bash
deepresearch show SESSION_ID
```

### Check Total Cost

```bash
deepresearch cost
```

## Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RESEARCH PIPELINE                                │
│                                                                          │
│  [Research Topic] → Literature Search → Idea Generation → Novelty Check  │
│                              ↓                                           │
│  Paper Writing ← Figure Generation ← Analysis ← Experiment Execution     │
│                              ↓                                           │
│                    [Output: Paper + Figures]                             │
└─────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
deepresearch/
├── pyproject.toml
├── configs/
│   ├── config.yaml              # Main configuration
│   ├── api_providers.yaml       # API pricing and rate limits
│   └── datasets.yaml            # Dataset configurations
├── src/deepresearch/
│   ├── core/                    # Config, state, exceptions
│   ├── providers/               # OpenAI, Anthropic, Google providers
│   ├── modules/
│   │   ├── literature/          # Search and novelty checking
│   │   ├── ideation/            # Idea generation
│   │   ├── experiment/          # Design, execution, checkpointing
│   │   ├── analysis/            # Statistical analysis
│   │   ├── figures/             # Matplotlib and TikZ figures
│   │   └── writing/             # Paper section generation
│   ├── pipeline/                # Main orchestrator
│   └── cli/                     # Command-line interface
├── data/
│   ├── sessions/                # Saved research sessions
│   ├── results/                 # Experiment results
│   └── outputs/                 # Generated papers and figures
└── tests/
```

## Supported AI Providers

| Provider | Models | Features |
|----------|--------|----------|
| OpenAI | gpt-4o, gpt-4o-mini | Generation, Embeddings |
| Anthropic | claude-3-5-sonnet, claude-3-haiku | Generation |
| Google | gemini-1.5-pro, gemini-1.5-flash | Generation, Embeddings |

## Supported Datasets

### Vision (NEW)
- **MNIST**: Handwritten digit classification (10 classes)
- **CIFAR-10**: Natural image classification (10 classes)
- **CIFAR-100**: Fine-grained classification (100 classes)

### NLP
- **Math Reasoning**: GSM8K, MATH
- **General Knowledge**: MMLU
- **Commonsense**: HellaSwag, WinoGrande
- **Code Generation**: HumanEval, MBPP
- **Question Answering**: TriviaQA, Natural Questions

## Vision Experiment Types

| Type | Description |
|------|-------------|
| `zero_shot` | Direct classification with class names |
| `zero_shot_detailed` | Classification with detailed instructions |
| `few_shot_1` | 1-shot learning (1 example per class) |
| `few_shot_5` | 5-shot learning (5 examples per class) |
| `chain_of_thought` | Step-by-step reasoning before classification |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Type checking
mypy src/deepresearch

# Linting
ruff check src/deepresearch
```

## License

MIT
