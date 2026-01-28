"""OpenAI provider implementation."""

import os
import time
from typing import AsyncIterator

from openai import AsyncOpenAI

from deepresearch.core.config import ProviderConfig
from deepresearch.core.exceptions import ProviderError, RateLimitError
from deepresearch.providers.base import (
    BaseProvider,
    EmbeddingResponse,
    GenerationRequest,
    GenerationResponse,
    Message,
)


class OpenAIProvider(BaseProvider):
    """OpenAI API provider."""

    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
        "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
    }

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("OpenAI API key not found", provider="openai")

        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=config.timeout,
            max_retries=0,  # We handle retries ourselves
        )
        self._model = config.model
        self._embedding_model = "text-embedding-3-small"

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for token usage."""
        pricing = self.PRICING.get(self._model, {"input": 0.0, "output": 0.0})
        return (
            input_tokens * pricing["input"] / 1000
            + output_tokens * pricing["output"] / 1000
        )

    def _format_messages(self, messages: list[Message]) -> list[dict]:
        """Format messages for OpenAI API."""
        return [{"role": m.role, "content": m.content} for m in messages]

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a response using OpenAI."""
        start_time = time.time()

        try:
            kwargs = {
                "model": self._model,
                "messages": self._format_messages(request.messages),
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }

            if request.stop_sequences:
                kwargs["stop"] = request.stop_sequences

            if request.json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = await self.client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason or "stop"
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

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
            if "rate_limit" in error_msg.lower():
                raise RateLimitError(provider="openai", retry_after=60.0)
            raise ProviderError(f"OpenAI generation failed: {e}", provider="openai")

    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[str]:
        """Generate a streaming response."""
        try:
            kwargs = {
                "model": self._model,
                "messages": self._format_messages(request.messages),
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": True,
            }

            if request.stop_sequences:
                kwargs["stop"] = request.stop_sequences

            stream = await self.client.chat.completions.create(**kwargs)

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower():
                raise RateLimitError(provider="openai", retry_after=60.0)
            raise ProviderError(f"OpenAI streaming failed: {e}", provider="openai")

    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        """Generate embeddings using OpenAI."""
        try:
            response = await self.client.embeddings.create(
                model=self._embedding_model,
                input=texts,
            )

            embeddings = [item.embedding for item in response.data]
            input_tokens = response.usage.prompt_tokens if response.usage else 0

            pricing = self.PRICING.get(
                self._embedding_model, {"input": 0.0, "output": 0.0}
            )
            cost = input_tokens * pricing["input"] / 1000

            return EmbeddingResponse(
                embeddings=embeddings,
                input_tokens=input_tokens,
                cost_usd=cost,
                model=self._embedding_model,
            )

        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower():
                raise RateLimitError(provider="openai", retry_after=60.0)
            raise ProviderError(f"OpenAI embedding failed: {e}", provider="openai")

    async def close(self) -> None:
        """Close the client."""
        await self.client.close()
