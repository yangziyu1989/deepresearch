"""Unified API manager with rate limiting and cost tracking."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from deepresearch.core.config import APIConfig, ProviderConfig, ProviderType
from deepresearch.core.exceptions import ProviderError, RateLimitError
from deepresearch.providers.base import (
    BaseProvider,
    GenerationRequest,
    GenerationResponse,
)
from deepresearch.providers.registry import ProviderRegistry


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""

    capacity: float
    tokens: float
    refill_rate: float  # tokens per second
    last_refill: float = field(default_factory=time.time)

    def consume(self, tokens: float = 1.0) -> bool:
        """Try to consume tokens, return True if successful."""
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def _refill(self) -> None:
        """Refill tokens based on time elapsed."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def wait_time(self, tokens: float = 1.0) -> float:
        """Calculate wait time needed for given tokens."""
        self._refill()
        if self.tokens >= tokens:
            return 0.0
        needed = tokens - self.tokens
        return needed / self.refill_rate


@dataclass
class CostTracker:
    """Tracks API costs across providers."""

    total_cost: float = 0.0
    costs_by_provider: dict[str, float] = field(default_factory=dict)
    requests_by_provider: dict[str, int] = field(default_factory=dict)
    tokens_by_provider: dict[str, dict[str, int]] = field(default_factory=dict)
    budget_limit: float = 100.0

    def add_cost(
        self, provider: str, cost: float, input_tokens: int, output_tokens: int
    ) -> None:
        """Record a cost."""
        self.total_cost += cost
        self.costs_by_provider[provider] = (
            self.costs_by_provider.get(provider, 0.0) + cost
        )
        self.requests_by_provider[provider] = (
            self.requests_by_provider.get(provider, 0) + 1
        )

        if provider not in self.tokens_by_provider:
            self.tokens_by_provider[provider] = {"input": 0, "output": 0}
        self.tokens_by_provider[provider]["input"] += input_tokens
        self.tokens_by_provider[provider]["output"] += output_tokens

    def is_over_budget(self) -> bool:
        """Check if total cost exceeds budget."""
        return self.total_cost >= self.budget_limit

    def remaining_budget(self) -> float:
        """Get remaining budget."""
        return max(0.0, self.budget_limit - self.total_cost)

    def summary(self) -> dict[str, Any]:
        """Get cost summary."""
        return {
            "total_cost_usd": round(self.total_cost, 4),
            "remaining_budget_usd": round(self.remaining_budget(), 4),
            "costs_by_provider": {
                k: round(v, 4) for k, v in self.costs_by_provider.items()
            },
            "requests_by_provider": self.requests_by_provider,
            "tokens_by_provider": self.tokens_by_provider,
        }


class APIManager:
    """Unified API manager with rate limiting and cost tracking."""

    def __init__(self, config: APIConfig) -> None:
        self.config = config
        self.cost_tracker = CostTracker(budget_limit=config.total_budget_usd)

        # Initialize rate limiters for each provider
        self._rate_limiters: dict[ProviderType, TokenBucket] = {}
        self._providers: dict[ProviderType, BaseProvider] = {}
        self._semaphore = asyncio.Semaphore(config.parallel_requests)
        self._lock = asyncio.Lock()

        for provider_type, provider_config in config.providers.items():
            # Create token bucket based on requests per minute
            rpm = provider_config.requests_per_minute
            self._rate_limiters[provider_type] = TokenBucket(
                capacity=rpm,
                tokens=rpm,
                refill_rate=rpm / 60.0,  # Convert to per-second
            )

    def _get_provider(self, provider_type: ProviderType) -> BaseProvider:
        """Get or create a provider instance."""
        if provider_type not in self._providers:
            config = self.config.providers.get(provider_type)
            if not config:
                raise ProviderError(
                    f"Provider {provider_type} not configured",
                    provider=provider_type.value,
                )
            self._providers[provider_type] = ProviderRegistry.create(config)
        return self._providers[provider_type]

    async def _wait_for_rate_limit(self, provider_type: ProviderType) -> None:
        """Wait if rate limited."""
        bucket = self._rate_limiters.get(provider_type)
        if bucket:
            wait_time = bucket.wait_time()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            bucket.consume()

    async def generate(
        self,
        request: GenerationRequest,
        provider_type: ProviderType | None = None,
        max_retries: int = 3,
        progress_callback: Callable[[str], None] | None = None,
    ) -> GenerationResponse:
        """Generate a response with rate limiting and retries."""
        provider_type = provider_type or self.config.default_provider

        # Check budget
        if self.cost_tracker.is_over_budget():
            raise ProviderError(
                "Budget exceeded",
                provider=provider_type.value,
                details={"remaining": self.cost_tracker.remaining_budget()},
            )

        async with self._semaphore:
            provider = self._get_provider(provider_type)
            last_error: Exception | None = None

            for attempt in range(max_retries):
                try:
                    # Wait for rate limit
                    await self._wait_for_rate_limit(provider_type)

                    if progress_callback:
                        progress_callback(
                            f"Calling {provider_type.value} (attempt {attempt + 1})"
                        )

                    # Make the request
                    response = await provider.generate(request)

                    # Track costs
                    async with self._lock:
                        self.cost_tracker.add_cost(
                            provider_type.value,
                            response.cost_usd,
                            response.input_tokens,
                            response.output_tokens,
                        )

                    return response

                except RateLimitError as e:
                    last_error = e
                    wait_time = e.retry_after or (2**attempt * 10)
                    if progress_callback:
                        progress_callback(
                            f"Rate limited, waiting {wait_time}s..."
                        )
                    await asyncio.sleep(wait_time)

                except ProviderError as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait_time = 2**attempt
                        await asyncio.sleep(wait_time)
                    else:
                        raise

            # If we get here, all retries failed
            if last_error:
                raise last_error
            raise ProviderError(
                "All retries failed", provider=provider_type.value
            )

    async def generate_batch(
        self,
        requests: list[GenerationRequest],
        provider_type: ProviderType | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[GenerationResponse]:
        """Generate responses for multiple requests in parallel."""
        provider_type = provider_type or self.config.default_provider

        async def process_one(idx: int, request: GenerationRequest) -> GenerationResponse:
            response = await self.generate(request, provider_type)
            if progress_callback:
                progress_callback(idx + 1, len(requests))
            return response

        tasks = [process_one(i, req) for i, req in enumerate(requests)]
        return await asyncio.gather(*tasks)

    def get_cost_summary(self) -> dict[str, Any]:
        """Get current cost summary."""
        return self.cost_tracker.summary()

    async def close(self) -> None:
        """Close all provider connections."""
        for provider in self._providers.values():
            await provider.close()
        self._providers.clear()


def create_api_manager(config: APIConfig) -> APIManager:
    """Create an API manager from configuration."""
    return APIManager(config)
