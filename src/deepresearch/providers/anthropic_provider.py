"""Anthropic provider implementation."""

import os
import time
from typing import AsyncIterator

from anthropic import AsyncAnthropic

from deepresearch.core.config import ProviderConfig
from deepresearch.core.exceptions import ProviderError, RateLimitError
from deepresearch.providers.base import (
    BaseProvider,
    EmbeddingResponse,
    GenerationRequest,
    GenerationResponse,
    Message,
)


class AnthropicProvider(BaseProvider):
    """Anthropic Claude API provider."""

    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
        "claude-3-5-haiku-20241022": {"input": 0.001, "output": 0.005},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    }

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderError("Anthropic API key not found", provider="anthropic")

        self.client = AsyncAnthropic(
            api_key=api_key,
            timeout=config.timeout,
            max_retries=0,  # We handle retries ourselves
        )
        self._model = config.model

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return self._model

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for token usage."""
        pricing = self.PRICING.get(self._model, {"input": 0.003, "output": 0.015})
        return (
            input_tokens * pricing["input"] / 1000
            + output_tokens * pricing["output"] / 1000
        )

    def _extract_system_message(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict]]:
        """Extract system message and format remaining messages for Anthropic."""
        system_content = None
        formatted = []

        for m in messages:
            if m.role == "system":
                system_content = m.content
            else:
                formatted.append({"role": m.role, "content": m.content})

        return system_content, formatted

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a response using Anthropic."""
        start_time = time.time()

        try:
            system_content, messages = self._extract_system_message(request.messages)

            kwargs = {
                "model": self._model,
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }

            if system_content:
                kwargs["system"] = system_content

            if request.stop_sequences:
                kwargs["stop_sequences"] = request.stop_sequences

            response = await self.client.messages.create(**kwargs)

            content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text

            finish_reason = response.stop_reason or "stop"
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            latency_ms = (time.time() - start_time) * 1000
            cost = self.calculate_cost(input_tokens, output_tokens)

            return GenerationResponse(
                content=content,
                finish_reason=finish_reason,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                model=self._model,
                latency_ms=latency_ms,
            )

        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                raise RateLimitError(provider="anthropic", retry_after=60.0)
            raise ProviderError(
                f"Anthropic generation failed: {e}", provider="anthropic"
            )

    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[str]:
        """Generate a streaming response."""
        try:
            system_content, messages = self._extract_system_message(request.messages)

            kwargs = {
                "model": self._model,
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }

            if system_content:
                kwargs["system"] = system_content

            if request.stop_sequences:
                kwargs["stop_sequences"] = request.stop_sequences

            async with self.client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                raise RateLimitError(provider="anthropic", retry_after=60.0)
            raise ProviderError(
                f"Anthropic streaming failed: {e}", provider="anthropic"
            )

    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        """Generate embeddings - Anthropic doesn't support embeddings directly.

        Falls back to using a text similarity approach or raises an error.
        """
        raise ProviderError(
            "Anthropic does not support embeddings. Use OpenAI for embeddings.",
            provider="anthropic",
        )

    async def close(self) -> None:
        """Close the client."""
        await self.client.close()
