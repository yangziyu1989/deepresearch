"""Experiment design for vision tasks."""

import json
from dataclasses import dataclass

from deepresearch.core.config import (
    ExperimentConfig,
    ExperimentPlan,
    ProviderType,
    ResearchIdea,
)
from deepresearch.modules.experiment.api_manager import APIManager
from deepresearch.providers.base import GenerationRequest, Message


# Vision datasets for experiment design
VISION_DATASETS = {
    "mnist": {
        "name": "MNIST",
        "description": "Handwritten digit classification (0-9)",
        "num_classes": 10,
        "test_size": 10000,
        "metrics": ["accuracy", "per_class_accuracy"],
    },
    "cifar10": {
        "name": "CIFAR-10",
        "description": "10-class natural image classification",
        "num_classes": 10,
        "test_size": 10000,
        "metrics": ["accuracy", "top_3_accuracy", "f1_macro"],
    },
    "cifar100": {
        "name": "CIFAR-100",
        "description": "100-class fine-grained classification",
        "num_classes": 100,
        "test_size": 10000,
        "metrics": ["accuracy", "top_5_accuracy"],
    },
}


VISION_DESIGN_PROMPT = """You are an AI research experiment designer specializing in computer vision.

Research Idea:
Title: {title}
Description: {description}
Methodology: {methodology}
Hypothesis: {hypothesis}

Design vision experiments to test this hypothesis using multimodal LLMs (GPT-4V, Claude Vision, Gemini Vision).

Available datasets:
{datasets}

Available experiment types:
- zero_shot: Direct classification without examples
- zero_shot_detailed: Classification with detailed instructions
- few_shot_1: 1-shot learning (1 example per class)
- few_shot_5: 5-shot learning (5 examples per class)
- chain_of_thought: Step-by-step reasoning before classification

Available providers:
- openai: GPT-4o, GPT-4o-mini (vision capable)
- anthropic: Claude 3.5 Sonnet (vision capable)
- google: Gemini 1.5 Pro/Flash (vision capable)

Design experiments considering:
1. What baselines are needed? (e.g., simple zero-shot)
2. What is the main method being tested?
3. What ablations would validate specific components?
4. Sample sizes should be reasonable (50-200 for quick tests, 500-1000 for full evaluation)

Output a JSON object:
{{
    "experiments": [
        {{
            "experiment_id": "unique_id",
            "name": "human readable name",
            "description": "what this experiment tests",
            "experiment_type": "baseline|main|ablation",
            "provider": "openai|anthropic|google",
            "model": "gpt-4o-mini|claude-3-5-sonnet-20241022|gemini-1.5-flash",
            "dataset": "mnist|cifar10|cifar100",
            "num_samples": number,
            "parameters": {{
                "prompt_type": "zero_shot|zero_shot_detailed|few_shot|chain_of_thought",
                "shots_per_class": 0,
                "temperature": 0.0
            }},
            "metrics": ["accuracy", "per_class_accuracy"]
        }}
    ],
    "execution_order": [["exp1", "exp2"], ["exp3"]],
    "estimated_cost_usd": number,
    "rationale": "explanation of the experimental design"
}}
"""


class VisionExperimentDesigner:
    """Designs experiments for vision tasks."""

    def __init__(self, api_manager: APIManager) -> None:
        self.api_manager = api_manager
        self.datasets = VISION_DATASETS

    async def design_experiments(
        self,
        research_idea: ResearchIdea,
        max_experiments: int = 8,
        budget_usd: float = 20.0,
        preferred_datasets: list[str] | None = None,
    ) -> ExperimentPlan:
        """Design a vision experiment plan."""
        preferred_datasets = preferred_datasets or ["mnist", "cifar10"]

        # Format datasets for prompt
        datasets_str = "\n".join(
            f"- {name}: {cfg['description']} ({cfg['num_classes']} classes, {cfg['test_size']} test samples)"
            for name, cfg in self.datasets.items()
            if name in preferred_datasets or not preferred_datasets
        )

        prompt = VISION_DESIGN_PROMPT.format(
            title=research_idea.title,
            description=research_idea.description,
            methodology=research_idea.methodology,
            hypothesis=research_idea.hypothesis,
            datasets=datasets_str,
        )

        request = GenerationRequest(
            messages=[
                Message(role="system", content="You are a vision AI research experiment designer. Output only valid JSON."),
                Message(role="user", content=prompt),
            ],
            temperature=0.7,
            max_tokens=4096,
            json_mode=True,
        )

        response = await self.api_manager.generate(request)

        try:
            design = json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback to default experiments
            return self._create_default_plan(research_idea, preferred_datasets)

        # Convert to ExperimentConfig objects
        experiments = []
        for exp in design.get("experiments", [])[:max_experiments]:
            provider_str = exp.get("provider", "openai")
            try:
                provider = ProviderType(provider_str)
            except ValueError:
                provider = ProviderType.OPENAI

            # Ensure dataset is valid
            dataset = exp.get("dataset", "mnist")
            if dataset not in self.datasets:
                dataset = preferred_datasets[0] if preferred_datasets else "mnist"

            config = ExperimentConfig(
                experiment_id=exp["experiment_id"],
                name=exp["name"],
                description=exp.get("description", ""),
                provider=provider,
                model=exp.get("model", "gpt-4o-mini"),
                dataset=dataset,
                num_samples=min(exp.get("num_samples", 100), 500),
                parameters=exp.get("parameters", {"prompt_type": "zero_shot"}),
                metrics=exp.get("metrics", ["accuracy"]),
                baseline=exp.get("experiment_type") == "baseline",
            )
            experiments.append(config)

        # Build execution order
        execution_order = design.get("execution_order", [[e.experiment_id for e in experiments]])

        # Estimate cost (vision API calls are more expensive)
        estimated_cost = min(design.get("estimated_cost_usd", 5.0), budget_usd)

        return ExperimentPlan(
            hypothesis=research_idea.hypothesis,
            experiments=experiments,
            execution_order=execution_order,
            estimated_cost_usd=estimated_cost,
            ablations=[
                exp["experiment_id"]
                for exp in design.get("experiments", [])
                if exp.get("experiment_type") == "ablation"
            ],
        )

    def _create_default_plan(
        self,
        research_idea: ResearchIdea,
        datasets: list[str],
    ) -> ExperimentPlan:
        """Create a default experiment plan if LLM design fails."""
        experiments = []
        dataset = datasets[0] if datasets else "mnist"

        # Baseline: zero-shot with GPT-4o-mini
        experiments.append(ExperimentConfig(
            experiment_id="baseline_zero_shot",
            name="Zero-shot Baseline",
            description="Simple zero-shot classification baseline",
            provider=ProviderType.OPENAI,
            model="gpt-4o-mini",
            dataset=dataset,
            num_samples=100,
            parameters={"prompt_type": "zero_shot", "temperature": 0.0},
            metrics=["accuracy", "per_class_accuracy"],
            baseline=True,
        ))

        # Main: detailed prompt
        experiments.append(ExperimentConfig(
            experiment_id="main_detailed",
            name="Detailed Prompt",
            description="Classification with detailed instructions",
            provider=ProviderType.OPENAI,
            model="gpt-4o-mini",
            dataset=dataset,
            num_samples=100,
            parameters={"prompt_type": "zero_shot_detailed", "temperature": 0.0},
            metrics=["accuracy", "per_class_accuracy"],
            baseline=False,
        ))

        # Ablation: chain-of-thought
        experiments.append(ExperimentConfig(
            experiment_id="ablation_cot",
            name="Chain-of-Thought",
            description="Step-by-step reasoning",
            provider=ProviderType.OPENAI,
            model="gpt-4o-mini",
            dataset=dataset,
            num_samples=100,
            parameters={"prompt_type": "chain_of_thought", "temperature": 0.0},
            metrics=["accuracy", "per_class_accuracy"],
            baseline=False,
        ))

        return ExperimentPlan(
            hypothesis=research_idea.hypothesis,
            experiments=experiments,
            execution_order=[["baseline_zero_shot"], ["main_detailed", "ablation_cot"]],
            estimated_cost_usd=5.0,
            ablations=["ablation_cot"],
        )

    def estimate_cost(self, plan: ExperimentPlan) -> float:
        """Estimate cost for vision experiments."""
        total = 0.0

        # Cost per image (rough estimates)
        cost_per_image = {
            ProviderType.OPENAI: 0.002,  # GPT-4o-mini with low detail
            ProviderType.ANTHROPIC: 0.003,
            ProviderType.GOOGLE: 0.001,
        }

        for exp in plan.experiments:
            provider_cost = cost_per_image.get(exp.provider, 0.002)
            total += exp.num_samples * provider_cost

        return round(total, 2)
