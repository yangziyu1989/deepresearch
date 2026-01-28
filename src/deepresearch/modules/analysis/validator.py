"""Hypothesis validation module."""

import json
from dataclasses import dataclass
from typing import Any

from deepresearch.core.config import (
    ExperimentPlan,
    HypothesisOutcome,
    StatisticalComparison,
    ValidationResult,
)
from deepresearch.core.state import ExperimentResult
from deepresearch.modules.analysis.analyzer import StatisticalAnalyzer
from deepresearch.modules.experiment.api_manager import APIManager
from deepresearch.providers.base import GenerationRequest, Message


VALIDATION_PROMPT = """You are a research analyst. Evaluate whether the experimental results support the hypothesis.

Hypothesis: {hypothesis}

Experiment Results Summary:
{results_summary}

Statistical Comparisons:
{comparisons}

Based on these results, determine:
1. Is the hypothesis supported, partially supported, or not supported?
2. What are the key findings?
3. What are potential follow-up studies?
4. How confident should we be in these conclusions?

Output a JSON object:
{{
    "outcome": "supported" | "partial" | "not_supported",
    "confidence": 0.0-1.0,
    "key_findings": ["finding 1", "finding 2"],
    "evidence_for": ["evidence supporting the hypothesis"],
    "evidence_against": ["evidence against the hypothesis"],
    "limitations": ["limitation 1", "limitation 2"],
    "suggested_followups": ["followup study 1", "followup study 2"],
    "explanation": "detailed explanation of the verdict"
}}
"""


class HypothesisValidator:
    """Validates hypotheses based on experiment results."""

    def __init__(self, api_manager: APIManager) -> None:
        self.api_manager = api_manager
        self.analyzer = StatisticalAnalyzer()

    async def validate(
        self,
        hypothesis: str,
        plan: ExperimentPlan,
        results: dict[str, ExperimentResult],
    ) -> ValidationResult:
        """Validate a hypothesis based on experiment results."""
        # Compute statistical comparisons
        comparisons = self._compute_comparisons(plan, results)

        # Generate results summary
        results_summary = self._format_results_summary(plan, results)
        comparisons_text = self._format_comparisons(comparisons)

        # Use LLM for overall validation
        prompt = VALIDATION_PROMPT.format(
            hypothesis=hypothesis,
            results_summary=results_summary,
            comparisons=comparisons_text,
        )

        request = GenerationRequest(
            messages=[
                Message(
                    role="system",
                    content="You are a research analyst. Evaluate experimental evidence objectively. Output only valid JSON.",
                ),
                Message(role="user", content=prompt),
            ],
            temperature=0.3,
            max_tokens=2048,
            json_mode=True,
        )

        response = await self.api_manager.generate(request)

        try:
            data = json.loads(response.content)
            outcome_str = data.get("outcome", "partial")
            outcome_map = {
                "supported": HypothesisOutcome.SUPPORTED,
                "partial": HypothesisOutcome.PARTIAL,
                "not_supported": HypothesisOutcome.NOT_SUPPORTED,
            }
            outcome = outcome_map.get(outcome_str, HypothesisOutcome.PARTIAL)

            return ValidationResult(
                outcome=outcome,
                statistical_comparisons=comparisons,
                key_findings=data.get("key_findings", []),
                suggested_followups=data.get("suggested_followups", []),
                confidence=data.get("confidence", 0.5),
            )
        except json.JSONDecodeError:
            # Return partial support with low confidence if parsing fails
            return ValidationResult(
                outcome=HypothesisOutcome.PARTIAL,
                statistical_comparisons=comparisons,
                key_findings=["Unable to fully analyze results"],
                suggested_followups=["Repeat experiments with more samples"],
                confidence=0.3,
            )

    def _compute_comparisons(
        self,
        plan: ExperimentPlan,
        results: dict[str, ExperimentResult],
    ) -> list[StatisticalComparison]:
        """Compute statistical comparisons between experiments."""
        comparisons = []

        # Find baseline experiments
        baselines = [e for e in plan.experiments if e.baseline]
        main_experiments = [e for e in plan.experiments if not e.baseline]

        for baseline in baselines:
            baseline_result = results.get(baseline.experiment_id)
            if not baseline_result or baseline_result.status != "completed":
                continue

            for main_exp in main_experiments:
                main_result = results.get(main_exp.experiment_id)
                if not main_result or main_result.status != "completed":
                    continue

                # Compare on shared metrics
                shared_metrics = set(baseline.metrics) & set(main_exp.metrics)
                for metric in shared_metrics:
                    # Extract per-sample values
                    baseline_values = [
                        r.get("metrics", {}).get(metric, 0.0)
                        for r in baseline_result.raw_outputs
                    ]
                    main_values = [
                        r.get("metrics", {}).get(metric, 0.0)
                        for r in main_result.raw_outputs
                    ]

                    # Fall back to aggregate if no per-sample data
                    if not baseline_values:
                        baseline_values = [baseline_result.metrics.get(metric, 0.0)]
                    if not main_values:
                        main_values = [main_result.metrics.get(metric, 0.0)]

                    comparison = self.analyzer.compare_methods(
                        method_a_name=main_exp.experiment_id,
                        method_b_name=baseline.experiment_id,
                        values_a=main_values,
                        values_b=baseline_values,
                        metric=metric,
                    )
                    comparisons.append(comparison)

        return comparisons

    def _format_results_summary(
        self,
        plan: ExperimentPlan,
        results: dict[str, ExperimentResult],
    ) -> str:
        """Format results summary for LLM."""
        lines = []
        for exp in plan.experiments:
            result = results.get(exp.experiment_id)
            if not result:
                lines.append(f"- {exp.name}: No results")
                continue

            status = result.status
            if status == "completed":
                metrics_str = ", ".join(
                    f"{k}={v:.3f}" for k, v in result.metrics.items()
                )
                lines.append(f"- {exp.name} ({'baseline' if exp.baseline else 'main'}): {metrics_str}")
            else:
                lines.append(f"- {exp.name}: {status}")

        return "\n".join(lines)

    def _format_comparisons(
        self,
        comparisons: list[StatisticalComparison],
    ) -> str:
        """Format statistical comparisons for LLM."""
        if not comparisons:
            return "No statistical comparisons available."

        lines = []
        for c in comparisons:
            sig_marker = "*" if c.significant else ""
            lines.append(
                f"- {c.method_a} vs {c.method_b} ({c.metric}): "
                f"{c.mean_a:.3f} vs {c.mean_b:.3f}, "
                f"p={c.p_value:.4f}{sig_marker}, d={c.effect_size:.2f}"
            )

        return "\n".join(lines)

    def check_success_criteria(
        self,
        results: dict[str, ExperimentResult],
        criteria: dict[str, float],
    ) -> dict[str, bool]:
        """Check if experiments meet success criteria."""
        met_criteria = {}

        for exp_id, result in results.items():
            if result.status != "completed":
                met_criteria[exp_id] = False
                continue

            all_met = True
            for metric, threshold in criteria.items():
                if result.metrics.get(metric, 0.0) < threshold:
                    all_met = False
                    break
            met_criteria[exp_id] = all_met

        return met_criteria
