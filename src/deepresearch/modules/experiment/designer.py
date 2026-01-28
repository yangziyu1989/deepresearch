"""Experiment design module."""

import json
from dataclasses import dataclass, field

from deepresearch.core.config import (
    ExperimentConfig,
    ExperimentPlan,
    ProviderType,
    ResearchIdea,
)
from deepresearch.core.exceptions import ExperimentError
from deepresearch.providers.base import GenerationRequest, Message
from deepresearch.modules.experiment.api_manager import APIManager


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""

    name: str
    description: str
    num_samples: int
    task_type: str  # classification, generation, qa, etc.
    metrics: list[str]


# Common AI research datasets
DATASETS = {
    "gsm8k": DatasetConfig(
        name="gsm8k",
        description="Grade school math word problems",
        num_samples=1319,
        task_type="math_reasoning",
        metrics=["accuracy", "solve_rate"],
    ),
    "mmlu": DatasetConfig(
        name="mmlu",
        description="Massive Multitask Language Understanding",
        num_samples=14042,
        task_type="multiple_choice",
        metrics=["accuracy"],
    ),
    "humaneval": DatasetConfig(
        name="humaneval",
        description="Code generation benchmark",
        num_samples=164,
        task_type="code_generation",
        metrics=["pass@1", "pass@10"],
    ),
    "triviaqa": DatasetConfig(
        name="triviaqa",
        description="Reading comprehension QA",
        num_samples=9960,
        task_type="qa",
        metrics=["exact_match", "f1"],
    ),
    "hellaswag": DatasetConfig(
        name="hellaswag",
        description="Commonsense reasoning",
        num_samples=10042,
        task_type="multiple_choice",
        metrics=["accuracy"],
    ),
}


DESIGN_PROMPT = """You are an AI research experiment designer. Given a research hypothesis, design a comprehensive experiment plan.

Research Idea:
Title: {title}
Description: {description}
Methodology: {methodology}
Hypothesis: {hypothesis}

Design experiments to test this hypothesis. Consider:
1. What baselines are needed for comparison?
2. What ablations would test specific components?
3. What datasets are appropriate?
4. What metrics should be measured?
5. What is a reasonable sample size?

Available datasets: {datasets}
Available providers: openai (gpt-4o), anthropic (claude-3-5-sonnet), google (gemini-pro)

Output a JSON object with this structure:
{{
    "experiments": [
        {{
            "experiment_id": "unique_id",
            "name": "human readable name",
            "description": "what this experiment tests",
            "experiment_type": "baseline|main|ablation",
            "provider": "openai|anthropic|google",
            "model": "model name",
            "dataset": "dataset name",
            "num_samples": number,
            "parameters": {{"key": "value"}},
            "metrics": ["metric1", "metric2"]
        }}
    ],
    "execution_order": [["exp1", "exp2"], ["exp3"]],
    "estimated_cost_usd": number,
    "rationale": "explanation of the experimental design"
}}

The execution_order groups experiments that can run in parallel. Dependencies should be in separate groups.
"""


class ExperimentDesigner:
    """Designs experiments based on research ideas."""

    def __init__(self, api_manager: APIManager) -> None:
        self.api_manager = api_manager
        self.datasets = DATASETS

    async def design_experiments(
        self,
        research_idea: ResearchIdea,
        max_experiments: int = 10,
        budget_usd: float = 50.0,
    ) -> ExperimentPlan:
        """Design an experiment plan for the research idea."""
        # Format available datasets
        datasets_str = "\n".join(
            f"- {name}: {cfg.description} ({cfg.num_samples} samples, metrics: {cfg.metrics})"
            for name, cfg in self.datasets.items()
        )

        prompt = DESIGN_PROMPT.format(
            title=research_idea.title,
            description=research_idea.description,
            methodology=research_idea.methodology,
            hypothesis=research_idea.hypothesis,
            datasets=datasets_str,
        )

        request = GenerationRequest(
            messages=[
                Message(role="system", content="You are an AI research experiment designer. Output only valid JSON."),
                Message(role="user", content=prompt),
            ],
            temperature=0.7,
            max_tokens=4096,
            json_mode=True,
        )

        response = await self.api_manager.generate(request)

        try:
            design = json.loads(response.content)
        except json.JSONDecodeError as e:
            raise ExperimentError(f"Failed to parse experiment design: {e}")

        # Convert to ExperimentConfig objects
        experiments = []
        for exp in design.get("experiments", [])[:max_experiments]:
            provider_str = exp.get("provider", "anthropic")
            try:
                provider = ProviderType(provider_str)
            except ValueError:
                provider = ProviderType.ANTHROPIC

            config = ExperimentConfig(
                experiment_id=exp["experiment_id"],
                name=exp["name"],
                description=exp.get("description", ""),
                provider=provider,
                model=exp.get("model", "claude-3-5-sonnet-20241022"),
                dataset=exp.get("dataset", "gsm8k"),
                num_samples=min(exp.get("num_samples", 100), 500),  # Cap samples
                parameters=exp.get("parameters", {}),
                metrics=exp.get("metrics", ["accuracy"]),
                baseline=exp.get("experiment_type") == "baseline",
            )
            experiments.append(config)

        # Build execution order
        execution_order = design.get("execution_order", [[e.experiment_id for e in experiments]])

        # Estimate cost
        estimated_cost = min(design.get("estimated_cost_usd", 10.0), budget_usd)

        # Extract ablations
        ablations = [
            exp["experiment_id"]
            for exp in design.get("experiments", [])
            if exp.get("experiment_type") == "ablation"
        ]

        return ExperimentPlan(
            hypothesis=research_idea.hypothesis,
            experiments=experiments,
            execution_order=execution_order,
            estimated_cost_usd=estimated_cost,
            ablations=ablations,
        )

    def validate_plan(self, plan: ExperimentPlan) -> list[str]:
        """Validate an experiment plan, return list of issues."""
        issues = []

        # Check for at least one baseline
        baselines = [e for e in plan.experiments if e.baseline]
        if not baselines:
            issues.append("No baseline experiment defined")

        # Check dataset validity
        for exp in plan.experiments:
            if exp.dataset not in self.datasets:
                issues.append(f"Unknown dataset: {exp.dataset}")

        # Check execution order references valid experiments
        exp_ids = {e.experiment_id for e in plan.experiments}
        for group in plan.execution_order:
            for exp_id in group:
                if exp_id not in exp_ids:
                    issues.append(f"Execution order references unknown experiment: {exp_id}")

        return issues

    def estimate_cost(self, plan: ExperimentPlan) -> float:
        """Estimate the total cost of an experiment plan."""
        total = 0.0

        # Rough estimates per provider per 1K tokens
        costs = {
            ProviderType.OPENAI: {"input": 0.005, "output": 0.015},
            ProviderType.ANTHROPIC: {"input": 0.003, "output": 0.015},
            ProviderType.GOOGLE: {"input": 0.00025, "output": 0.0005},
        }

        for exp in plan.experiments:
            # Estimate tokens per sample (rough)
            tokens_per_sample = 500  # input
            output_per_sample = 200  # output

            provider_cost = costs.get(
                exp.provider, {"input": 0.003, "output": 0.015}
            )

            exp_cost = (
                exp.num_samples * tokens_per_sample * provider_cost["input"] / 1000
                + exp.num_samples * output_per_sample * provider_cost["output"] / 1000
            )
            total += exp_cost

        return round(total, 2)
