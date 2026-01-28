"""Vision experiment executor using multimodal LLMs."""

import asyncio
import base64
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from deepresearch.core.config import ExperimentConfig, ExperimentPlan, ProviderType
from deepresearch.core.state import ExperimentResult
from deepresearch.modules.experiment.api_manager import APIManager
from deepresearch.modules.experiment.checkpoint import CheckpointManager, ExperimentCheckpoint
from deepresearch.modules.vision.datasets import VisionDatasetLoader, VisionSample
from deepresearch.modules.vision.metrics import (
    ClassificationMetrics,
    PredictionResult,
    VisionMetricsCalculator,
)


@dataclass
class VisionExperimentConfig(ExperimentConfig):
    """Extended config for vision experiments."""

    task_type: str = "classification"  # classification, zero-shot, few-shot
    shots_per_class: int = 0  # For few-shot learning
    prompt_template: str = ""
    include_class_names: bool = True
    image_detail: str = "low"  # low, high, auto (for OpenAI)


# Prompt templates for different experiment types
PROMPT_TEMPLATES = {
    "zero_shot_simple": """What is in this image? Answer with just the class name.
Classes: {class_names}
Answer:""",

    "zero_shot_detailed": """You are an image classifier. Analyze this image and classify it into one of the following categories.

Available classes: {class_names}

Look at the image carefully and respond with ONLY the class name, nothing else.""",

    "few_shot": """You are an image classifier. I will show you examples of each class, then ask you to classify a new image.

{examples}

Now classify this new image. Respond with ONLY the class name from: {class_names}
Answer:""",

    "chain_of_thought": """You are an image classifier. Analyze this image step by step.

Available classes: {class_names}

1. First, describe what you see in the image
2. Identify key features that distinguish it
3. Match these features to the most likely class
4. State your final classification

Final answer (class name only):""",

    "confidence_scoring": """Classify this image into one of these classes: {class_names}

Respond in JSON format:
{{"class": "class_name", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}""",
}


class VisionExperimentExecutor:
    """Executes vision experiments using multimodal LLMs."""

    def __init__(
        self,
        api_manager: APIManager,
        checkpoint_dir: Path,
        checkpoint_interval: int = 10,
    ) -> None:
        self.api_manager = api_manager
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        self.dataset_loader = VisionDatasetLoader()

    async def execute_plan(
        self,
        plan: ExperimentPlan,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, ExperimentResult]:
        """Execute a vision experiment plan."""
        results = {}

        for group in plan.execution_order:
            tasks = []
            for exp_id in group:
                exp = next(
                    (e for e in plan.experiments if e.experiment_id == exp_id),
                    None
                )
                if exp:
                    tasks.append(self.execute_experiment(exp, progress_callback))

            group_results = await asyncio.gather(*tasks, return_exceptions=True)

            for exp_id, result in zip(group, group_results):
                if isinstance(result, Exception):
                    results[exp_id] = ExperimentResult(
                        experiment_id=exp_id,
                        status="failed",
                        error=str(result),
                    )
                else:
                    results[exp_id] = result

        return results

    async def execute_experiment(
        self,
        config: ExperimentConfig,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> ExperimentResult:
        """Execute a single vision experiment."""
        # Check for existing checkpoint
        checkpoint = self.checkpoint_manager.load(config.experiment_id)

        if checkpoint and checkpoint.status == "completed":
            return self._checkpoint_to_result(checkpoint, config)

        # Load dataset
        samples = self.dataset_loader.load_samples(
            config.dataset,
            config.num_samples,
            split="test",
        )
        class_names = self.dataset_loader.get_class_names(config.dataset)
        metrics_calc = VisionMetricsCalculator(class_names)

        # Get prompt template
        prompt_type = config.parameters.get("prompt_type", "zero_shot_simple")
        prompt_template = config.parameters.get(
            "prompt_template",
            PROMPT_TEMPLATES.get(prompt_type, PROMPT_TEMPLATES["zero_shot_simple"])
        )

        # Load few-shot examples if needed
        few_shot_examples = None
        shots_per_class = config.parameters.get("shots_per_class", 0)
        if shots_per_class > 0:
            few_shot_examples = self.dataset_loader.load_few_shot_examples(
                config.dataset,
                shots_per_class=shots_per_class,
            )

        # Initialize or resume checkpoint
        if checkpoint:
            start_idx = checkpoint.iterator_position
            predictions = [PredictionResult(**p) for p in checkpoint.partial_results]
        else:
            start_idx = 0
            predictions = []
            checkpoint = ExperimentCheckpoint(
                experiment_id=config.experiment_id,
                total_samples=len(samples),
            )

        result = ExperimentResult(
            experiment_id=config.experiment_id,
            status="running",
            start_time=datetime.now().isoformat(),
        )

        try:
            for idx in range(start_idx, len(samples)):
                sample = samples[idx]

                # Run prediction
                pred_result = await self._classify_image(
                    sample=sample,
                    class_names=class_names,
                    prompt_template=prompt_template,
                    provider=config.provider,
                    few_shot_examples=few_shot_examples,
                    metrics_calc=metrics_calc,
                    parameters=config.parameters,
                )

                predictions.append(pred_result)
                result.cost_usd += config.parameters.get("cost_per_call", 0.001)

                # Update checkpoint periodically
                if (idx + 1) % self.checkpoint_interval == 0:
                    checkpoint.completed_samples = idx + 1
                    checkpoint.iterator_position = idx + 1
                    checkpoint.partial_results = [
                        {
                            "sample_id": p.sample_id,
                            "true_label": p.true_label,
                            "true_label_name": p.true_label_name,
                            "predicted_label": p.predicted_label,
                            "predicted_label_name": p.predicted_label_name,
                            "correct": p.correct,
                            "raw_response": p.raw_response[:500],  # Truncate
                            "error": p.error,
                        }
                        for p in predictions
                    ]
                    self.checkpoint_manager.save(checkpoint)

                if progress_callback:
                    progress_callback(config.experiment_id, idx + 1, len(samples))

            # Calculate final metrics
            metrics = metrics_calc.calculate_metrics(predictions)

            result.metrics = {
                "accuracy": metrics.accuracy,
                "top_3_accuracy": metrics.top_k_accuracy.get(3, metrics.accuracy),
                "precision_macro": metrics.precision_macro,
                "recall_macro": metrics.recall_macro,
                "f1_macro": metrics.f1_macro,
            }

            result.status = "completed"
            result.end_time = datetime.now().isoformat()
            result.raw_outputs = [
                {
                    "sample_id": p.sample_id,
                    "true_label": p.true_label,
                    "predicted_label": p.predicted_label,
                    "correct": p.correct,
                    "metrics": {"accuracy": 1.0 if p.correct else 0.0},
                }
                for p in predictions
            ]

            # Mark checkpoint complete
            checkpoint.status = "completed"
            checkpoint.completed_samples = len(samples)
            self.checkpoint_manager.save(checkpoint)

        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            result.end_time = datetime.now().isoformat()

            checkpoint.status = "failed"
            checkpoint.error = str(e)
            self.checkpoint_manager.save(checkpoint)

        return result

    async def _classify_image(
        self,
        sample: VisionSample,
        class_names: list[str],
        prompt_template: str,
        provider: ProviderType,
        few_shot_examples: dict[int, list[VisionSample]] | None,
        metrics_calc: VisionMetricsCalculator,
        parameters: dict[str, Any],
    ) -> PredictionResult:
        """Classify a single image using multimodal LLM."""
        try:
            # Build prompt
            if few_shot_examples and "{examples}" in prompt_template:
                examples_text = self._format_few_shot_examples(few_shot_examples, class_names)
                prompt = prompt_template.format(
                    class_names=", ".join(class_names),
                    examples=examples_text,
                )
            else:
                prompt = prompt_template.format(class_names=", ".join(class_names))

            # Make API call with image
            response = await self._call_vision_api(
                prompt=prompt,
                image=sample,
                provider=provider,
                parameters=parameters,
            )

            # Parse response
            pred_idx, pred_name, confidence = metrics_calc.parse_classification_response(
                response, class_names
            )

            correct = pred_idx == sample.label if pred_idx is not None else False

            return PredictionResult(
                sample_id=sample.sample_id,
                true_label=sample.label,
                true_label_name=sample.label_name,
                predicted_label=pred_idx,
                predicted_label_name=pred_name,
                confidence=confidence,
                correct=correct,
                raw_response=response,
            )

        except Exception as e:
            return PredictionResult(
                sample_id=sample.sample_id,
                true_label=sample.label,
                true_label_name=sample.label_name,
                predicted_label=None,
                predicted_label_name=None,
                correct=False,
                error=str(e),
            )

    async def _call_vision_api(
        self,
        prompt: str,
        image: VisionSample,
        provider: ProviderType,
        parameters: dict[str, Any],
    ) -> str:
        """Call the vision API with an image."""
        image_b64 = image.to_base64("PNG")
        image_url = f"data:image/png;base64,{image_b64}"

        if provider == ProviderType.OPENAI:
            return await self._call_openai_vision(prompt, image_url, parameters)
        elif provider == ProviderType.ANTHROPIC:
            return await self._call_anthropic_vision(prompt, image_b64, parameters)
        elif provider == ProviderType.GOOGLE:
            return await self._call_google_vision(prompt, image, parameters)
        else:
            raise ValueError(f"Unsupported provider for vision: {provider}")

    async def _call_openai_vision(
        self,
        prompt: str,
        image_url: str,
        parameters: dict[str, Any],
    ) -> str:
        """Call OpenAI vision API."""
        from openai import AsyncOpenAI
        import os

        client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        response = await client.chat.completions.create(
            model=parameters.get("model", "gpt-4o-mini"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": parameters.get("image_detail", "low"),
                            },
                        },
                    ],
                }
            ],
            max_tokens=parameters.get("max_tokens", 100),
            temperature=parameters.get("temperature", 0.0),
        )

        return response.choices[0].message.content or ""

    async def _call_anthropic_vision(
        self,
        prompt: str,
        image_b64: str,
        parameters: dict[str, Any],
    ) -> str:
        """Call Anthropic vision API."""
        from anthropic import AsyncAnthropic
        import os

        client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        response = await client.messages.create(
            model=parameters.get("model", "claude-3-5-sonnet-20241022"),
            max_tokens=parameters.get("max_tokens", 100),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        return response.content[0].text if response.content else ""

    async def _call_google_vision(
        self,
        prompt: str,
        image: VisionSample,
        parameters: dict[str, Any],
    ) -> str:
        """Call Google Gemini vision API."""
        import google.generativeai as genai
        import os

        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(parameters.get("model", "gemini-1.5-flash"))

        response = await model.generate_content_async([prompt, image.image])
        return response.text if response.text else ""

    def _format_few_shot_examples(
        self,
        examples: dict[int, list[VisionSample]],
        class_names: list[str],
    ) -> str:
        """Format few-shot examples for the prompt."""
        lines = []
        for class_idx, samples in examples.items():
            class_name = class_names[class_idx]
            lines.append(f"Examples of '{class_name}':")
            for i, sample in enumerate(samples):
                lines.append(f"  [Image {i+1} of {class_name}]")
        return "\n".join(lines)

    def _checkpoint_to_result(
        self,
        checkpoint: ExperimentCheckpoint,
        config: ExperimentConfig,
    ) -> ExperimentResult:
        """Convert checkpoint to ExperimentResult."""
        predictions = [PredictionResult(**p) for p in checkpoint.partial_results]
        num_correct = sum(1 for p in predictions if p.correct)

        return ExperimentResult(
            experiment_id=checkpoint.experiment_id,
            status="completed",
            metrics={
                "accuracy": num_correct / len(predictions) if predictions else 0.0,
            },
            raw_outputs=checkpoint.partial_results,
        )
