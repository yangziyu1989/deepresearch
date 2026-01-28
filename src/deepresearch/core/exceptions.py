"""Custom exceptions for DeepResearch."""

from typing import Any


class DeepResearchError(Exception):
    """Base exception for all DeepResearch errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ProviderError(DeepResearchError):
    """Error related to AI provider operations."""

    def __init__(
        self,
        message: str,
        provider: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.provider = provider


class RateLimitError(ProviderError):
    """Rate limit exceeded for a provider."""

    def __init__(
        self,
        provider: str,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        message = f"Rate limit exceeded for {provider}"
        if retry_after:
            message += f", retry after {retry_after}s"
        super().__init__(message, provider, details)
        self.retry_after = retry_after


class LiteratureSearchError(DeepResearchError):
    """Error during literature search operations."""

    def __init__(
        self,
        message: str,
        source: str | None = None,
        query: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.source = source
        self.query = query


class NoveltyCheckError(DeepResearchError):
    """Error during novelty checking operations."""

    pass


class ExperimentError(DeepResearchError):
    """Error during experiment execution."""

    def __init__(
        self,
        message: str,
        experiment_id: str | None = None,
        stage: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.experiment_id = experiment_id
        self.stage = stage


class PipelineError(DeepResearchError):
    """Error in the research pipeline orchestration."""

    def __init__(
        self,
        message: str,
        stage: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.stage = stage


class ConfigurationError(DeepResearchError):
    """Error in configuration loading or validation."""

    pass


class CheckpointError(DeepResearchError):
    """Error during checkpoint save/load operations."""

    pass


class FigureGenerationError(DeepResearchError):
    """Error during figure generation."""

    def __init__(
        self,
        message: str,
        figure_type: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.figure_type = figure_type


class WritingError(DeepResearchError):
    """Error during paper section writing."""

    def __init__(
        self,
        message: str,
        section: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.section = section
