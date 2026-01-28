"""Experiment execution with checkpointing and parallelism."""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from deepresearch.core.config import ExperimentConfig, ExperimentPlan, ProviderType
from deepresearch.core.exceptions import ExperimentError
from deepresearch.core.state import ExperimentResult
from deepresearch.modules.experiment.api_manager import APIManager
from deepresearch.modules.experiment.checkpoint import (
    CheckpointManager,
    ExperimentCheckpoint,
)
from deepresearch.providers.base import GenerationRequest, Message


@dataclass
class Sample:
    """A single data sample for evaluation."""

    sample_id: str
    input_text: str
    expected_output: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleResult:
    """Result of running a sample."""

    sample_id: str
    input_text: str
    output_text: str
    expected_output: str | None
    metrics: dict[str, float]
    latency_ms: float
    cost_usd: float
    error: str | None = None


class DatasetLoader:
    """Loads datasets for experiments."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or Path("data/datasets")

    async def load_samples(
        self,
        dataset: str,
        num_samples: int,
        seed: int = 42,
    ) -> list[Sample]:
        """Load samples from a dataset."""
        # For now, generate synthetic samples
        # In production, this would load from actual datasets
        samples = []

        if dataset == "gsm8k":
            samples = self._generate_math_samples(num_samples)
        elif dataset == "mmlu":
            samples = self._generate_mcq_samples(num_samples)
        elif dataset == "humaneval":
            samples = self._generate_code_samples(num_samples)
        else:
            # Generic samples
            samples = self._generate_generic_samples(dataset, num_samples)

        return samples[:num_samples]

    def _generate_math_samples(self, n: int) -> list[Sample]:
        """Generate math problem samples."""
        problems = [
            ("If John has 5 apples and gives 2 to Mary, how many does he have?", "3"),
            ("A train travels 60 miles per hour. How far does it go in 2.5 hours?", "150"),
            ("If a shirt costs $25 and is 20% off, what is the sale price?", "20"),
        ]
        samples = []
        for i in range(n):
            problem, answer = problems[i % len(problems)]
            samples.append(Sample(
                sample_id=f"gsm8k_{i}",
                input_text=problem,
                expected_output=answer,
                metadata={"task": "math"},
            ))
        return samples

    def _generate_mcq_samples(self, n: int) -> list[Sample]:
        """Generate multiple choice samples."""
        questions = [
            ("What is the capital of France? A) London B) Paris C) Berlin D) Madrid", "B"),
            ("Which planet is closest to the Sun? A) Venus B) Earth C) Mercury D) Mars", "C"),
        ]
        samples = []
        for i in range(n):
            q, a = questions[i % len(questions)]
            samples.append(Sample(
                sample_id=f"mmlu_{i}",
                input_text=q,
                expected_output=a,
                metadata={"task": "mcq"},
            ))
        return samples

    def _generate_code_samples(self, n: int) -> list[Sample]:
        """Generate code generation samples."""
        prompts = [
            ("Write a function that returns the sum of two numbers.", "def add(a, b):\n    return a + b"),
            ("Write a function that checks if a number is prime.", "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"),
        ]
        samples = []
        for i in range(n):
            prompt, code = prompts[i % len(prompts)]
            samples.append(Sample(
                sample_id=f"humaneval_{i}",
                input_text=prompt,
                expected_output=code,
                metadata={"task": "code"},
            ))
        return samples

    def _generate_generic_samples(self, dataset: str, n: int) -> list[Sample]:
        """Generate generic test samples."""
        return [
            Sample(
                sample_id=f"{dataset}_{i}",
                input_text=f"Sample input {i} for {dataset}",
                expected_output=f"Expected output {i}",
            )
            for i in range(n)
        ]


class MetricCalculator:
    """Calculates metrics for experiment results."""

    def calculate(
        self,
        output: str,
        expected: str | None,
        task_type: str,
    ) -> dict[str, float]:
        """Calculate metrics based on task type."""
        metrics = {}

        if task_type == "math_reasoning" or task_type == "math":
            metrics["accuracy"] = self._check_math_answer(output, expected)
        elif task_type == "multiple_choice" or task_type == "mcq":
            metrics["accuracy"] = self._check_mcq_answer(output, expected)
        elif task_type == "code_generation" or task_type == "code":
            metrics["pass@1"] = self._check_code_output(output, expected)
        elif task_type == "qa":
            metrics["exact_match"] = self._exact_match(output, expected)
            metrics["f1"] = self._f1_score(output, expected)
        else:
            metrics["exact_match"] = self._exact_match(output, expected)

        return metrics

    def _check_math_answer(self, output: str, expected: str | None) -> float:
        """Check if math answer is correct."""
        if not expected:
            return 0.0
        # Extract numbers from output
        import re
        numbers = re.findall(r'-?\d+\.?\d*', output)
        expected_num = re.findall(r'-?\d+\.?\d*', expected)
        if numbers and expected_num:
            try:
                return 1.0 if float(numbers[-1]) == float(expected_num[-1]) else 0.0
            except ValueError:
                return 0.0
        return 0.0

    def _check_mcq_answer(self, output: str, expected: str | None) -> float:
        """Check if MCQ answer is correct."""
        if not expected:
            return 0.0
        output_upper = output.strip().upper()
        expected_upper = expected.strip().upper()
        # Check if the expected letter appears in the output
        return 1.0 if expected_upper in output_upper else 0.0

    def _check_code_output(self, output: str, expected: str | None) -> float:
        """Basic code output check."""
        if not expected:
            return 0.0
        # Very basic check - in production would run tests
        return 1.0 if "def " in output else 0.0

    def _exact_match(self, output: str, expected: str | None) -> float:
        """Exact string match."""
        if not expected:
            return 0.0
        return 1.0 if output.strip().lower() == expected.strip().lower() else 0.0

    def _f1_score(self, output: str, expected: str | None) -> float:
        """Token-level F1 score."""
        if not expected:
            return 0.0
        output_tokens = set(output.lower().split())
        expected_tokens = set(expected.lower().split())
        if not output_tokens or not expected_tokens:
            return 0.0
        common = output_tokens & expected_tokens
        precision = len(common) / len(output_tokens)
        recall = len(common) / len(expected_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


class ExperimentExecutor:
    """Executes experiments with checkpointing support."""

    def __init__(
        self,
        api_manager: APIManager,
        checkpoint_dir: Path,
        checkpoint_interval: int = 10,
    ) -> None:
        self.api_manager = api_manager
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        self.dataset_loader = DatasetLoader()
        self.metric_calculator = MetricCalculator()

    async def execute_plan(
        self,
        plan: ExperimentPlan,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, ExperimentResult]:
        """Execute an entire experiment plan."""
        results = {}

        for group in plan.execution_order:
            # Execute experiments in this group in parallel
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
        """Execute a single experiment with checkpointing."""
        # Check for existing checkpoint
        checkpoint = self.checkpoint_manager.load(config.experiment_id)

        if checkpoint and checkpoint.status == "completed":
            # Already done
            return self._checkpoint_to_result(checkpoint, config)

        # Load samples
        samples = await self.dataset_loader.load_samples(
            config.dataset,
            config.num_samples,
        )

        # Initialize or resume checkpoint
        if checkpoint:
            start_idx = checkpoint.iterator_position
            partial_results = checkpoint.partial_results
            metrics_accumulator = checkpoint.metrics_accumulator
        else:
            start_idx = 0
            partial_results = []
            metrics_accumulator = {m: [] for m in config.metrics}
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

                # Run sample
                sample_result = await self._run_sample(sample, config)

                # Accumulate results
                partial_results.append({
                    "sample_id": sample_result.sample_id,
                    "output": sample_result.output_text,
                    "metrics": sample_result.metrics,
                    "latency_ms": sample_result.latency_ms,
                    "cost_usd": sample_result.cost_usd,
                })

                for metric, value in sample_result.metrics.items():
                    if metric in metrics_accumulator:
                        metrics_accumulator[metric].append(value)

                result.cost_usd += sample_result.cost_usd

                # Update checkpoint periodically
                if (idx + 1) % self.checkpoint_interval == 0:
                    checkpoint.completed_samples = idx + 1
                    checkpoint.iterator_position = idx + 1
                    checkpoint.partial_results = partial_results
                    checkpoint.metrics_accumulator = metrics_accumulator
                    self.checkpoint_manager.save(checkpoint)

                # Progress callback
                if progress_callback:
                    progress_callback(config.experiment_id, idx + 1, len(samples))

            # Compute final metrics
            for metric, values in metrics_accumulator.items():
                if values:
                    result.metrics[metric] = sum(values) / len(values)

            result.status = "completed"
            result.end_time = datetime.now().isoformat()
            result.raw_outputs = partial_results

            # Mark checkpoint complete and clean up
            checkpoint.status = "completed"
            checkpoint.completed_samples = len(samples)
            self.checkpoint_manager.save(checkpoint)

        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            result.end_time = datetime.now().isoformat()

            # Save checkpoint for potential resume
            checkpoint.status = "failed"
            checkpoint.error = str(e)
            self.checkpoint_manager.save(checkpoint)

        return result

    async def _run_sample(
        self,
        sample: Sample,
        config: ExperimentConfig,
    ) -> SampleResult:
        """Run a single sample through the model."""
        # Build prompt based on experiment parameters
        system_prompt = config.parameters.get(
            "system_prompt",
            "You are a helpful AI assistant. Answer concisely."
        )

        prompt = config.parameters.get("prompt_template", "{input}")
        formatted_prompt = prompt.format(input=sample.input_text)

        request = GenerationRequest(
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=formatted_prompt),
            ],
            temperature=config.parameters.get("temperature", 0.0),
            max_tokens=config.parameters.get("max_tokens", 1024),
        )

        response = await self.api_manager.generate(request, config.provider)

        # Calculate metrics
        task_type = sample.metadata.get("task", "generic")
        metrics = self.metric_calculator.calculate(
            response.content,
            sample.expected_output,
            task_type,
        )

        return SampleResult(
            sample_id=sample.sample_id,
            input_text=sample.input_text,
            output_text=response.content,
            expected_output=sample.expected_output,
            metrics=metrics,
            latency_ms=response.latency_ms,
            cost_usd=response.cost_usd,
        )

    def _checkpoint_to_result(
        self,
        checkpoint: ExperimentCheckpoint,
        config: ExperimentConfig,
    ) -> ExperimentResult:
        """Convert a completed checkpoint to ExperimentResult."""
        metrics = {}
        for metric, values in checkpoint.metrics_accumulator.items():
            if values:
                metrics[metric] = sum(values) / len(values)

        total_cost = sum(r.get("cost_usd", 0) for r in checkpoint.partial_results)

        return ExperimentResult(
            experiment_id=checkpoint.experiment_id,
            status="completed",
            metrics=metrics,
            raw_outputs=checkpoint.partial_results,
            cost_usd=total_cost,
        )

    def resume_experiment(self, experiment_id: str) -> ExperimentCheckpoint | None:
        """Check if an experiment can be resumed."""
        return self.checkpoint_manager.load(experiment_id)
