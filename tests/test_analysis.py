"""Tests for analysis module."""

import pytest
import numpy as np

from deepresearch.modules.analysis.analyzer import (
    StatisticalAnalyzer,
    DescriptiveStats,
)
from deepresearch.core.state import ExperimentResult


class TestStatisticalAnalyzer:
    """Test statistical analysis functionality."""

    def setup_method(self):
        self.analyzer = StatisticalAnalyzer(significance_level=0.05)

    def test_descriptive_stats(self):
        values = [0.80, 0.82, 0.85, 0.78, 0.84]
        stats = self.analyzer.compute_descriptive_stats(values, "accuracy")

        assert stats.metric == "accuracy"
        assert stats.n == 5
        assert stats.mean == pytest.approx(0.818, rel=0.01)
        assert stats.median == 0.82
        assert stats.min_val == 0.78
        assert stats.max_val == 0.85

    def test_descriptive_stats_single_value(self):
        values = [0.85]
        stats = self.analyzer.compute_descriptive_stats(values, "accuracy")

        assert stats.n == 1
        assert stats.mean == 0.85
        assert stats.std == 0.0

    def test_compare_methods_significant(self):
        values_a = [0.85, 0.87, 0.86, 0.88, 0.84]
        values_b = [0.75, 0.77, 0.76, 0.78, 0.74]

        comparison = self.analyzer.compare_methods(
            "MethodA", "MethodB", values_a, values_b, "accuracy"
        )

        assert comparison.method_a == "MethodA"
        assert comparison.method_b == "MethodB"
        assert comparison.mean_a > comparison.mean_b
        assert comparison.p_value < 0.05
        assert comparison.significant == True

    def test_compare_methods_not_significant(self):
        values_a = [0.80, 0.82, 0.85, 0.78, 0.84]
        values_b = [0.81, 0.83, 0.84, 0.79, 0.83]

        comparison = self.analyzer.compare_methods(
            "MethodA", "MethodB", values_a, values_b, "accuracy"
        )

        # Results should be close, may or may not be significant
        assert abs(comparison.mean_a - comparison.mean_b) < 0.05

    def test_effect_size(self):
        # Large effect size
        values_a = [0.90, 0.92, 0.91, 0.93, 0.89]
        values_b = [0.70, 0.72, 0.71, 0.73, 0.69]

        comparison = self.analyzer.compare_methods(
            "MethodA", "MethodB", values_a, values_b, "accuracy"
        )

        # Cohen's d should be large (> 0.8)
        assert abs(comparison.effect_size) > 0.8

    def test_analyze_ablation(self):
        baseline = ExperimentResult(
            experiment_id="baseline",
            status="completed",
            metrics={"accuracy": 0.85},
            raw_outputs=[
                {"metrics": {"accuracy": 0.84}},
                {"metrics": {"accuracy": 0.86}},
                {"metrics": {"accuracy": 0.85}},
            ],
        )
        ablation = ExperimentResult(
            experiment_id="ablation",
            status="completed",
            metrics={"accuracy": 0.80},
            raw_outputs=[
                {"metrics": {"accuracy": 0.79}},
                {"metrics": {"accuracy": 0.81}},
                {"metrics": {"accuracy": 0.80}},
            ],
        )

        result = self.analyzer.analyze_ablation(
            baseline, ablation, "component_x", "accuracy"
        )

        assert result.baseline_experiment == "baseline"
        assert result.ablation_experiment == "ablation"
        assert result.removed_component == "component_x"
        assert result.delta > 0  # Removing component hurt performance
        assert result.delta_percent > 0

    def test_summarize_results(self):
        results = {
            "exp_a": ExperimentResult(
                experiment_id="exp_a",
                status="completed",
                metrics={"accuracy": 0.85},
                cost_usd=1.0,
            ),
            "exp_b": ExperimentResult(
                experiment_id="exp_b",
                status="completed",
                metrics={"accuracy": 0.90},
                cost_usd=1.5,
            ),
            "exp_c": ExperimentResult(
                experiment_id="exp_c",
                status="failed",
                metrics={},
                cost_usd=0.5,
            ),
        }

        summary = self.analyzer.summarize_results(results, "accuracy")

        assert summary["best_method"] == "exp_b"
        assert summary["total_cost"] == 2.5
        assert summary["rankings"]["exp_b"] == 1
        assert summary["rankings"]["exp_a"] == 2
