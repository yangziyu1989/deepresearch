"""Tests for provider module."""

import pytest

from deepresearch.core.config import ProviderConfig, ProviderType
from deepresearch.providers.base import GenerationRequest, GenerationResponse, Message


class TestGenerationRequest:
    """Test GenerationRequest dataclass."""

    def test_creation(self):
        request = GenerationRequest(
            messages=[
                Message(role="system", content="You are helpful."),
                Message(role="user", content="Hello"),
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        assert len(request.messages) == 2
        assert request.temperature == 0.7

    def test_json_mode(self):
        request = GenerationRequest(
            messages=[Message(role="user", content="Return JSON")],
            json_mode=True,
        )
        assert request.json_mode is True


class TestGenerationResponse:
    """Test GenerationResponse dataclass."""

    def test_creation(self):
        response = GenerationResponse(
            content="Hello!",
            finish_reason="stop",
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.001,
            model="gpt-4o",
            latency_ms=500.0,
        )
        assert response.content == "Hello!"
        assert response.cost_usd == 0.001


class TestProviderConfig:
    """Test ProviderConfig."""

    def test_openai_config(self):
        config = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            model="gpt-4o",
            requests_per_minute=500,
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.015,
        )
        assert config.provider_type == ProviderType.OPENAI
        assert config.model == "gpt-4o"

    def test_anthropic_config(self):
        config = ProviderConfig(
            provider_type=ProviderType.ANTHROPIC,
            model="claude-3-5-sonnet-20241022",
            requests_per_minute=60,
        )
        assert config.provider_type == ProviderType.ANTHROPIC

    def test_google_config(self):
        config = ProviderConfig(
            provider_type=ProviderType.GOOGLE,
            model="gemini-1.5-pro",
            requests_per_minute=60,
        )
        assert config.provider_type == ProviderType.GOOGLE


# Note: Integration tests for actual API calls would require API keys
# and should be run separately with pytest markers
class TestProviderIntegration:
    """Integration tests for providers (requires API keys)."""

    @pytest.mark.skip(reason="Requires API key")
    async def test_openai_generate(self):
        """Test OpenAI generation."""
        from deepresearch.providers.openai_provider import OpenAIProvider

        config = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            model="gpt-4o-mini",
        )
        provider = OpenAIProvider(config)

        request = GenerationRequest(
            messages=[Message(role="user", content="Say hello")],
            max_tokens=10,
        )

        response = await provider.generate(request)
        assert response.content
        await provider.close()

    @pytest.mark.skip(reason="Requires API key")
    async def test_anthropic_generate(self):
        """Test Anthropic generation."""
        from deepresearch.providers.anthropic_provider import AnthropicProvider

        config = ProviderConfig(
            provider_type=ProviderType.ANTHROPIC,
            model="claude-3-haiku-20240307",
        )
        provider = AnthropicProvider(config)

        request = GenerationRequest(
            messages=[Message(role="user", content="Say hello")],
            max_tokens=10,
        )

        response = await provider.generate(request)
        assert response.content
        await provider.close()
