"""Tests for core module."""

import json
import tempfile
from pathlib import Path

import pytest

from deepresearch.core.config import (
    APIConfig,
    ExperimentConfig,
    ExperimentPlan,
    HypothesisOutcome,
    PipelineConfig,
    PipelineStage,
    ProviderConfig,
    ProviderType,
    ResearchIdea,
    StatisticalComparison,
    ValidationResult,
)
from deepresearch.core.exceptions import (
    CheckpointError,
    DeepResearchError,
    ProviderError,
    RateLimitError,
)
from deepresearch.core.state import (
    ExperimentResult,
    Paper,
    SessionState,
    StateManager,
    create_session,
)


class TestExceptions:
    """Test custom exceptions."""

    def test_base_exception(self):
        exc = DeepResearchError("test error", details={"key": "value"})
        assert exc.message == "test error"
        assert exc.details == {"key": "value"}

    def test_provider_error(self):
        exc = ProviderError("api failed", provider="openai")
        assert exc.provider == "openai"
        assert "api failed" in str(exc)

    def test_rate_limit_error(self):
        exc = RateLimitError(provider="anthropic", retry_after=60.0)
        assert exc.provider == "anthropic"
        assert exc.retry_after == 60.0
        assert "60" in str(exc)


class TestConfig:
    """Test configuration classes."""

    def test_provider_config(self):
        config = ProviderConfig(
            provider_type=ProviderType.OPENAI,
            model="gpt-4o",
            requests_per_minute=100,
        )
        assert config.provider_type == ProviderType.OPENAI
        assert config.model == "gpt-4o"
        assert config.requests_per_minute == 100

    def test_api_config(self):
        config = APIConfig(
            default_provider=ProviderType.ANTHROPIC,
            total_budget_usd=50.0,
        )
        assert config.default_provider == ProviderType.ANTHROPIC
        assert config.total_budget_usd == 50.0

    def test_experiment_config(self):
        config = ExperimentConfig(
            experiment_id="exp_001",
            name="Test Experiment",
            description="A test",
            provider=ProviderType.OPENAI,
            model="gpt-4o",
            dataset="gsm8k",
            num_samples=100,
        )
        assert config.experiment_id == "exp_001"
        assert config.num_samples == 100

    def test_research_idea(self):
        idea = ResearchIdea(
            title="Test Idea",
            description="A test idea",
            methodology="Test methodology",
            key_contributions=["Contribution 1"],
            hypothesis="Test hypothesis",
            novelty_score=0.8,
        )
        assert idea.title == "Test Idea"
        assert idea.novelty_score == 0.8


class TestState:
    """Test session state management."""

    def test_session_state_creation(self):
        state = SessionState(
            session_id="test_123",
            research_topic="Test topic",
        )
        assert state.session_id == "test_123"
        assert state.current_stage == PipelineStage.LITERATURE_SEARCH

    def test_mark_stage_complete(self):
        state = SessionState(
            session_id="test_123",
            research_topic="Test topic",
        )
        state.mark_stage_complete(PipelineStage.LITERATURE_SEARCH)
        assert state.is_stage_complete(PipelineStage.LITERATURE_SEARCH)
        assert not state.is_stage_complete(PipelineStage.NOVELTY_CHECK)

    def test_state_serialization(self):
        state = SessionState(
            session_id="test_123",
            research_topic="Test topic",
        )
        state.papers = [
            Paper(
                paper_id="p1",
                title="Test Paper",
                abstract="Abstract",
                authors=["Author 1"],
                year=2024,
                source="arxiv",
                url="http://example.com",
            )
        ]
        state.mark_stage_complete(PipelineStage.LITERATURE_SEARCH)

        # Serialize and deserialize
        data = state.to_dict()
        assert data["session_id"] == "test_123"
        assert len(data["papers"]) == 1

        restored = SessionState.from_dict(data)
        assert restored.session_id == state.session_id
        assert len(restored.papers) == 1
        assert restored.is_stage_complete(PipelineStage.LITERATURE_SEARCH)

    def test_state_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StateManager(Path(tmpdir))

            state = SessionState(
                session_id="test_123",
                research_topic="Test topic",
            )

            # Save
            path = manager.save(state)
            assert path.exists()

            # Load
            loaded = manager.load("test_123")
            assert loaded.session_id == state.session_id

            # List
            sessions = manager.list_sessions()
            assert "test_123" in sessions

            # Delete
            manager.delete("test_123")
            assert not manager.exists("test_123")

    def test_create_session(self):
        config = PipelineConfig(research_topic="Test")
        state = create_session("Test topic", config)
        assert state.research_topic == "Test topic"
        assert state.session_id is not None


class TestExperimentResult:
    """Test experiment result dataclass."""

    def test_experiment_result(self):
        result = ExperimentResult(
            experiment_id="exp_001",
            status="completed",
            metrics={"accuracy": 0.85},
            cost_usd=1.5,
        )
        assert result.experiment_id == "exp_001"
        assert result.metrics["accuracy"] == 0.85


class TestValidationResult:
    """Test validation result dataclass."""

    def test_validation_result(self):
        comparison = StatisticalComparison(
            method_a="MethodA",
            method_b="MethodB",
            metric="accuracy",
            mean_a=0.85,
            mean_b=0.80,
            std_a=0.05,
            std_b=0.06,
            p_value=0.03,
            effect_size=0.8,
            significant=True,
        )

        result = ValidationResult(
            outcome=HypothesisOutcome.SUPPORTED,
            statistical_comparisons=[comparison],
            key_findings=["Finding 1"],
            suggested_followups=["Followup 1"],
            confidence=0.9,
        )

        assert result.outcome == HypothesisOutcome.SUPPORTED
        assert len(result.statistical_comparisons) == 1
        assert result.confidence == 0.9
