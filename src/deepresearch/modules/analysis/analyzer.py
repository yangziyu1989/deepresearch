"""Statistical analysis of experiment results."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from deepresearch.core.config import StatisticalComparison
from deepresearch.core.state import ExperimentResult


@dataclass
class DescriptiveStats:
    """Descriptive statistics for a metric."""

    metric: str
    n: int
    mean: float
    std: float
    median: float
    min_val: float
    max_val: float
    ci_lower: float  # 95% CI
    ci_upper: float


@dataclass
class AblationResult:
    """Result of an ablation study."""

    baseline_experiment: str
    ablation_experiment: str
    removed_component: str
    metric: str
    baseline_mean: float
    ablation_mean: float
    delta: float
    delta_percent: float
    significant: bool
    p_value: float


class StatisticalAnalyzer:
    """Performs statistical analysis on experiment results."""

    def __init__(self, significance_level: float = 0.05) -> None:
        self.significance_level = significance_level

    def compute_descriptive_stats(
        self,
        values: list[float],
        metric: str,
    ) -> DescriptiveStats:
        """Compute descriptive statistics for a list of values."""
        arr = np.array(values)
        n = len(arr)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        median = float(np.median(arr))

        # 95% confidence interval
        if n > 1:
            se = std / np.sqrt(n)
            ci_margin = stats.t.ppf(0.975, n - 1) * se
            ci_lower = mean - ci_margin
            ci_upper = mean + ci_margin
        else:
            ci_lower = ci_upper = mean

        return DescriptiveStats(
            metric=metric,
            n=n,
            mean=mean,
            std=std,
            median=median,
            min_val=float(np.min(arr)),
            max_val=float(np.max(arr)),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )

    def compare_methods(
        self,
        method_a_name: str,
        method_b_name: str,
        values_a: list[float],
        values_b: list[float],
        metric: str,
        paired: bool = False,
    ) -> StatisticalComparison:
        """Compare two methods using appropriate statistical test."""
        arr_a = np.array(values_a)
        arr_b = np.array(values_b)

        mean_a = float(np.mean(arr_a))
        mean_b = float(np.mean(arr_b))
        std_a = float(np.std(arr_a, ddof=1)) if len(arr_a) > 1 else 0.0
        std_b = float(np.std(arr_b, ddof=1)) if len(arr_b) > 1 else 0.0

        # Choose test based on data characteristics
        if paired and len(arr_a) == len(arr_b):
            # Paired t-test or Wilcoxon signed-rank
            if self._is_normal(arr_a - arr_b):
                stat, p_value = stats.ttest_rel(arr_a, arr_b)
            else:
                stat, p_value = stats.wilcoxon(arr_a, arr_b)
        else:
            # Independent samples
            if self._is_normal(arr_a) and self._is_normal(arr_b):
                stat, p_value = stats.ttest_ind(arr_a, arr_b)
            else:
                stat, p_value = stats.mannwhitneyu(arr_a, arr_b, alternative='two-sided')

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
        effect_size = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

        return StatisticalComparison(
            method_a=method_a_name,
            method_b=method_b_name,
            metric=metric,
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b,
            p_value=float(p_value),
            effect_size=float(effect_size),
            significant=p_value < self.significance_level,
        )

    def _is_normal(self, data: np.ndarray) -> bool:
        """Check if data is normally distributed using Shapiro-Wilk test."""
        if len(data) < 3:
            return True  # Assume normal for very small samples
        if len(data) > 5000:
            data = np.random.choice(data, 5000, replace=False)
        try:
            stat, p = stats.shapiro(data)
            return p > 0.05
        except Exception:
            return True

    def analyze_ablation(
        self,
        baseline_result: ExperimentResult,
        ablation_result: ExperimentResult,
        component_name: str,
        metric: str,
    ) -> AblationResult:
        """Analyze the effect of removing a component."""
        # Extract per-sample metrics if available
        baseline_values = [
            r.get("metrics", {}).get(metric, 0.0)
            for r in baseline_result.raw_outputs
        ]
        ablation_values = [
            r.get("metrics", {}).get(metric, 0.0)
            for r in ablation_result.raw_outputs
        ]

        # Fall back to aggregate metrics if per-sample not available
        if not baseline_values:
            baseline_values = [baseline_result.metrics.get(metric, 0.0)]
        if not ablation_values:
            ablation_values = [ablation_result.metrics.get(metric, 0.0)]

        baseline_mean = float(np.mean(baseline_values))
        ablation_mean = float(np.mean(ablation_values))
        delta = baseline_mean - ablation_mean
        delta_percent = (delta / baseline_mean * 100) if baseline_mean != 0 else 0.0

        # Statistical test
        if len(baseline_values) > 1 and len(ablation_values) > 1:
            _, p_value = stats.ttest_ind(baseline_values, ablation_values)
        else:
            p_value = 1.0

        return AblationResult(
            baseline_experiment=baseline_result.experiment_id,
            ablation_experiment=ablation_result.experiment_id,
            removed_component=component_name,
            metric=metric,
            baseline_mean=baseline_mean,
            ablation_mean=ablation_mean,
            delta=delta,
            delta_percent=delta_percent,
            significant=p_value < self.significance_level,
            p_value=float(p_value),
        )

    def summarize_results(
        self,
        results: dict[str, ExperimentResult],
        primary_metric: str,
    ) -> dict[str, Any]:
        """Generate a summary of all experiment results."""
        summary = {
            "experiments": {},
            "rankings": {},
            "best_method": None,
            "total_cost": 0.0,
        }

        for exp_id, result in results.items():
            if result.status != "completed":
                continue

            metric_value = result.metrics.get(primary_metric, 0.0)
            summary["experiments"][exp_id] = {
                "metric": metric_value,
                "cost": result.cost_usd,
                "status": result.status,
            }
            summary["total_cost"] += result.cost_usd

        # Rank by primary metric
        ranked = sorted(
            summary["experiments"].items(),
            key=lambda x: x[1]["metric"],
            reverse=True,
        )
        summary["rankings"] = {exp_id: rank + 1 for rank, (exp_id, _) in enumerate(ranked)}

        if ranked:
            summary["best_method"] = ranked[0][0]

        return summary
