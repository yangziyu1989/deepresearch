"""Tests for experiment module."""

import tempfile
from pathlib import Path

import pytest

from deepresearch.modules.experiment.checkpoint import (
    CheckpointManager,
    ExperimentCheckpoint,
)
from deepresearch.modules.experiment.api_manager import TokenBucket, CostTracker


class TestTokenBucket:
    """Test token bucket rate limiter."""

    def test_initial_state(self):
        bucket = TokenBucket(capacity=10, tokens=10, refill_rate=1.0)
        assert bucket.tokens == 10
        assert bucket.capacity == 10

    def test_consume(self):
        bucket = TokenBucket(capacity=10, tokens=10, refill_rate=1.0)
        assert bucket.consume(1) is True
        assert bucket.tokens == 9

    def test_consume_insufficient(self):
        bucket = TokenBucket(capacity=10, tokens=5, refill_rate=1.0)
        assert bucket.consume(10) is False
        assert bucket.tokens == pytest.approx(5, abs=0.1)

    def test_wait_time(self):
        bucket = TokenBucket(capacity=10, tokens=0, refill_rate=1.0)
        wait = bucket.wait_time(5)
        assert wait == pytest.approx(5.0, rel=0.1)


class TestCostTracker:
    """Test cost tracking."""

    def test_add_cost(self):
        tracker = CostTracker(budget_limit=100.0)
        tracker.add_cost("openai", 1.5, 1000, 500)

        assert tracker.total_cost == 1.5
        assert tracker.costs_by_provider["openai"] == 1.5
        assert tracker.requests_by_provider["openai"] == 1
        assert tracker.tokens_by_provider["openai"]["input"] == 1000
        assert tracker.tokens_by_provider["openai"]["output"] == 500

    def test_budget_check(self):
        tracker = CostTracker(budget_limit=10.0)
        tracker.add_cost("openai", 5.0, 1000, 500)
        assert not tracker.is_over_budget()

        tracker.add_cost("openai", 6.0, 1000, 500)
        assert tracker.is_over_budget()

    def test_remaining_budget(self):
        tracker = CostTracker(budget_limit=100.0)
        tracker.add_cost("openai", 30.0, 1000, 500)
        assert tracker.remaining_budget() == 70.0

    def test_summary(self):
        tracker = CostTracker(budget_limit=100.0)
        tracker.add_cost("openai", 10.0, 1000, 500)
        tracker.add_cost("anthropic", 5.0, 500, 200)

        summary = tracker.summary()
        assert summary["total_cost_usd"] == 15.0
        assert "openai" in summary["costs_by_provider"]
        assert "anthropic" in summary["costs_by_provider"]


class TestCheckpointManager:
    """Test checkpoint management."""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))

            checkpoint = ExperimentCheckpoint(
                experiment_id="exp_001",
                total_samples=100,
                completed_samples=50,
                iterator_position=50,
            )

            # Save
            path = manager.save(checkpoint)
            assert path.exists()

            # Load
            loaded = manager.load("exp_001")
            assert loaded is not None
            assert loaded.experiment_id == "exp_001"
            assert loaded.completed_samples == 50

    def test_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))

            assert not manager.exists("exp_001")

            checkpoint = ExperimentCheckpoint(
                experiment_id="exp_001",
                total_samples=100,
            )
            manager.save(checkpoint)

            assert manager.exists("exp_001")

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))

            checkpoint = ExperimentCheckpoint(
                experiment_id="exp_001",
                total_samples=100,
            )
            manager.save(checkpoint)
            assert manager.exists("exp_001")

            manager.delete("exp_001")
            assert not manager.exists("exp_001")

    def test_list_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))

            for i in range(3):
                checkpoint = ExperimentCheckpoint(
                    experiment_id=f"exp_{i:03d}",
                    total_samples=100,
                )
                manager.save(checkpoint)

            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) == 3

    def test_cleanup_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(Path(tmpdir))

            # Create completed and running checkpoints
            completed = ExperimentCheckpoint(
                experiment_id="exp_completed",
                total_samples=100,
                status="completed",
            )
            running = ExperimentCheckpoint(
                experiment_id="exp_running",
                total_samples=100,
                status="running",
            )
            manager.save(completed)
            manager.save(running)

            # Cleanup
            deleted = manager.cleanup_completed()
            assert deleted == 1
            assert not manager.exists("exp_completed")
            assert manager.exists("exp_running")
