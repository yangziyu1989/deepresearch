"""Base provider interface for AI providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class Message:
    """A message in a conversation."""

    role: str  # system, user, assistant
    content: str


@dataclass
class GenerationRequest:
    """Request for text generation."""

    messages: list[Message]
    temperature: float = 0.7
    max_tokens: int = 4096
    stop_sequences: list[str] | None = None
    json_mode: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResponse:
    """Response from text generation."""

    content: str
    finish_reason: str  # stop, length, error
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResponse:
    """Response from embedding generation."""

    embeddings: list[list[float]]
    input_tokens: int
    cost_usd: float
    model: str


class BaseProvider(ABC):
    """Abstract base class for AI providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        ...

    @property
    @abstractmethod
    def model(self) -> str:
        """Current model being used."""
        ...

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a response for the given request."""
        ...

    @abstractmethod
    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[str]:
        """Generate a streaming response."""
        ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        """Generate embeddings for texts."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close any open connections."""
        ...

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for token usage. Override in subclasses."""
        return 0.0
