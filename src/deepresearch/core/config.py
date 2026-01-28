"""Configuration dataclasses for DeepResearch."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ProviderType(str, Enum):
    """Supported AI provider types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class HypothesisOutcome(str, Enum):
    """Outcome of hypothesis validation."""

    SUPPORTED = "supported"
    PARTIAL = "partial"
    NOT_SUPPORTED = "not_supported"


class PipelineStage(str, Enum):
    """Pipeline execution stages."""

    LITERATURE_SEARCH = "literature_search"
    NOVELTY_CHECK = "novelty_check"
    IDEA_GENERATION = "idea_generation"
    EXPERIMENT_DESIGN = "experiment_design"
    EXPERIMENT_EXECUTION = "experiment_execution"
    ANALYSIS = "analysis"
    FIGURE_GENERATION = "figure_generation"
    PAPER_WRITING = "paper_writing"


class ProviderConfig(BaseModel):
    """Configuration for an AI provider."""

    provider_type: ProviderType
    model: str
    api_key: str | None = None  # Can be loaded from env
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    max_retries: int = 3
    timeout: float = 60.0

    model_config = {"extra": "forbid"}


class APIConfig(BaseModel):
    """Configuration for all API providers."""

    default_provider: ProviderType = ProviderType.ANTHROPIC
    providers: dict[ProviderType, ProviderConfig] = Field(default_factory=dict)
    total_budget_usd: float = 100.0
    parallel_requests: int = 5

    model_config = {"extra": "forbid"}


class LiteratureConfig(BaseModel):
    """Configuration for literature search."""

    max_papers: int = 50
    search_sources: list[str] = Field(default_factory=lambda: ["arxiv", "semantic_scholar"])
    embedding_model: str = "text-embedding-3-small"
    similarity_threshold: float = 0.85

    model_config = {"extra": "forbid"}


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    experiment_id: str
    name: str
    description: str
    provider: ProviderType
    model: str
    dataset: str
    num_samples: int
    parameters: dict[str, Any] = field(default_factory=dict)
    metrics: list[str] = field(default_factory=list)
    baseline: bool = False


@dataclass
class ExperimentPlan:
    """Complete experiment plan with multiple experiments."""

    hypothesis: str
    experiments: list[ExperimentConfig]
    execution_order: list[list[str]]  # Groups of experiment IDs to run in parallel
    estimated_cost_usd: float
    ablations: list[str] = field(default_factory=list)


@dataclass
class ResearchIdea:
    """A research idea with novelty assessment."""

    title: str
    description: str
    methodology: str
    key_contributions: list[str]
    hypothesis: str
    novelty_score: float = 0.0
    feasibility_score: float = 0.0
    impact_score: float = 0.0
    related_papers: list[str] = field(default_factory=list)


@dataclass
class StatisticalComparison:
    """Result of a statistical comparison between methods."""

    method_a: str
    method_b: str
    metric: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    p_value: float
    effect_size: float
    significant: bool


@dataclass
class ValidationResult:
    """Result of hypothesis validation."""

    outcome: HypothesisOutcome
    statistical_comparisons: list[StatisticalComparison]
    key_findings: list[str]
    suggested_followups: list[str]
    confidence: float = 0.0


class PipelineConfig(BaseModel):
    """Configuration for the research pipeline."""

    research_topic: str
    output_dir: Path = Path("data/outputs")
    session_dir: Path = Path("data/sessions")
    results_dir: Path = Path("data/results")

    # Stage toggles
    skip_literature_search: bool = False
    skip_novelty_check: bool = False

    # Execution settings
    checkpoint_interval: int = 10  # Save checkpoint every N samples
    max_concurrent_experiments: int = 3

    # Literature settings
    literature: LiteratureConfig = Field(default_factory=LiteratureConfig)

    # API settings
    api: APIConfig = Field(default_factory=APIConfig)

    # Figure settings
    figure_format: str = "pdf"  # pdf, png, svg
    tikz_enabled: bool = True

    # Writing settings
    output_format: str = "latex"  # latex, markdown

    model_config = {"extra": "forbid"}


def load_config(config_path: Path) -> PipelineConfig:
    """Load pipeline configuration from YAML file."""
    import yaml

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return PipelineConfig(**data)
