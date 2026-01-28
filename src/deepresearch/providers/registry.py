"""Provider registry and factory for AI providers."""

from typing import Type

from deepresearch.core.config import ProviderConfig, ProviderType
from deepresearch.core.exceptions import ProviderError
from deepresearch.providers.anthropic_provider import AnthropicProvider
from deepresearch.providers.base import BaseProvider
from deepresearch.providers.google_provider import GoogleProvider
from deepresearch.providers.openai_provider import OpenAIProvider


class ProviderRegistry:
    """Registry for AI provider implementations."""

    _providers: dict[ProviderType, Type[BaseProvider]] = {
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.ANTHROPIC: AnthropicProvider,
        ProviderType.GOOGLE: GoogleProvider,
    }

    _instances: dict[str, BaseProvider] = {}

    @classmethod
    def register(
        cls, provider_type: ProviderType, provider_class: Type[BaseProvider]
    ) -> None:
        """Register a new provider implementation."""
        cls._providers[provider_type] = provider_class

    @classmethod
    def create(cls, config: ProviderConfig) -> BaseProvider:
        """Create a provider instance from configuration."""
        provider_class = cls._providers.get(config.provider_type)
        if not provider_class:
            raise ProviderError(
                f"Unknown provider type: {config.provider_type}",
                provider=str(config.provider_type),
            )

        # Create cache key from config
        cache_key = f"{config.provider_type.value}:{config.model}"

        # Return cached instance if exists
        if cache_key in cls._instances:
            return cls._instances[cache_key]

        # Create and cache new instance
        instance = provider_class(config)
        cls._instances[cache_key] = instance
        return instance

    @classmethod
    def get(cls, provider_type: ProviderType, model: str) -> BaseProvider | None:
        """Get a cached provider instance."""
        cache_key = f"{provider_type.value}:{model}"
        return cls._instances.get(cache_key)

    @classmethod
    async def close_all(cls) -> None:
        """Close all provider instances."""
        for instance in cls._instances.values():
            await instance.close()
        cls._instances.clear()


def get_provider(config: ProviderConfig) -> BaseProvider:
    """Convenience function to get a provider instance."""
    return ProviderRegistry.create(config)
