"""Google Gemini provider implementation."""

import os
import time
from typing import AsyncIterator

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from deepresearch.core.config import ProviderConfig
from deepresearch.core.exceptions import ProviderError, RateLimitError
from deepresearch.providers.base import (
    BaseProvider,
    EmbeddingResponse,
    GenerationRequest,
    GenerationResponse,
    Message,
)


class GoogleProvider(BaseProvider):
    """Google Gemini API provider."""

    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        "gemini-pro": {"input": 0.00025, "output": 0.0005},
        "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
        "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
        "gemini-2.0-flash-exp": {"input": 0.0, "output": 0.0},  # Free during preview
    }

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        api_key = config.api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ProviderError("Google API key not found", provider="google")

        genai.configure(api_key=api_key)
        self._model = config.model
        self._client = genai.GenerativeModel(self._model)
        self._embedding_model = "models/text-embedding-004"

    @property
    def name(self) -> str:
        return "google"

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

    def _format_messages(self, messages: list[Message]) -> tuple[str | None, list[dict]]:
        """Format messages for Gemini API."""
        system_instruction = None
        history = []

        for m in messages:
            if m.role == "system":
                system_instruction = m.content
            elif m.role == "user":
                history.append({"role": "user", "parts": [m.content]})
            elif m.role == "assistant":
                history.append({"role": "model", "parts": [m.content]})

        return system_instruction, history

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a response using Google Gemini."""
        start_time = time.time()

        try:
            system_instruction, history = self._format_messages(request.messages)

            # Create model with system instruction if provided
            if system_instruction:
                model = genai.GenerativeModel(
                    self._model,
                    system_instruction=system_instruction,
                )
            else:
                model = self._client

            generation_config = GenerationConfig(
                temperature=request.temperature,
                max_output_tokens=request.max_tokens,
                stop_sequences=request.stop_sequences,
            )

            # Get the last user message for single-turn or use chat for multi-turn
            if len(history) == 1:
                response = await model.generate_content_async(
                    history[0]["parts"][0],
                    generation_config=generation_config,
                )
            else:
                chat = model.start_chat(history=history[:-1])
                response = await chat.send_message_async(
                    history[-1]["parts"][0],
                    generation_config=generation_config,
                )

            content = response.text if response.text else ""
            finish_reason = "stop"

            # Token counting (approximate if not available)
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, "usage_metadata"):
                input_tokens = getattr(
                    response.usage_metadata, "prompt_token_count", 0
                )
                output_tokens = getattr(
                    response.usage_metadata, "candidates_token_count", 0
                )

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
            if "quota" in error_msg.lower() or "rate" in error_msg.lower():
                raise RateLimitError(provider="google", retry_after=60.0)
            raise ProviderError(f"Google generation failed: {e}", provider="google")

    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[str]:
        """Generate a streaming response."""
        try:
            system_instruction, history = self._format_messages(request.messages)

            if system_instruction:
                model = genai.GenerativeModel(
                    self._model,
                    system_instruction=system_instruction,
                )
            else:
                model = self._client

            generation_config = GenerationConfig(
                temperature=request.temperature,
                max_output_tokens=request.max_tokens,
                stop_sequences=request.stop_sequences,
            )

            if len(history) == 1:
                response = await model.generate_content_async(
                    history[0]["parts"][0],
                    generation_config=generation_config,
                    stream=True,
                )
            else:
                chat = model.start_chat(history=history[:-1])
                response = await chat.send_message_async(
                    history[-1]["parts"][0],
                    generation_config=generation_config,
                    stream=True,
                )

            async for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower() or "rate" in error_msg.lower():
                raise RateLimitError(provider="google", retry_after=60.0)
            raise ProviderError(f"Google streaming failed: {e}", provider="google")

    async def embed(self, texts: list[str]) -> EmbeddingResponse:
        """Generate embeddings using Google."""
        try:
            embeddings = []
            total_tokens = 0

            for text in texts:
                result = genai.embed_content(
                    model=self._embedding_model,
                    content=text,
                    task_type="retrieval_document",
                )
                embeddings.append(result["embedding"])
                # Approximate token count
                total_tokens += len(text.split()) * 1.3

            return EmbeddingResponse(
                embeddings=embeddings,
                input_tokens=int(total_tokens),
                cost_usd=0.0,  # Google embeddings are free
                model=self._embedding_model,
            )

        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower() or "rate" in error_msg.lower():
                raise RateLimitError(provider="google", retry_after=60.0)
            raise ProviderError(f"Google embedding failed: {e}", provider="google")

    async def close(self) -> None:
        """Close the client - no cleanup needed for Google client."""
        pass
