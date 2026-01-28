"""Vision metrics calculator."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ClassificationMetrics:
    """Metrics for classification tasks."""

    accuracy: float
    top_k_accuracy: dict[int, float]  # k -> accuracy
    per_class_accuracy: dict[str, float]  # class_name -> accuracy
    confusion_matrix: list[list[int]]
    precision_macro: float
    recall_macro: float
    f1_macro: float
    num_samples: int
    num_correct: int


@dataclass
class PredictionResult:
    """Result of a single prediction."""

    sample_id: str
    true_label: int
    true_label_name: str
    predicted_label: int | None
    predicted_label_name: str | None
    confidence: float | None = None
    top_k_predictions: list[tuple[int, str, float]] = field(default_factory=list)
    correct: bool = False
    raw_response: str = ""
    error: str | None = None


class VisionMetricsCalculator:
    """Calculates metrics for vision experiments."""

    def __init__(self, class_names: list[str]) -> None:
        self.class_names = class_names
        self.num_classes = len(class_names)

    def calculate_metrics(
        self,
        predictions: list[PredictionResult],
        top_k_values: list[int] | None = None,
    ) -> ClassificationMetrics:
        """Calculate classification metrics from predictions."""
        top_k_values = top_k_values or [1, 3, 5]

        # Filter valid predictions
        valid_preds = [p for p in predictions if p.predicted_label is not None]

        if not valid_preds:
            return ClassificationMetrics(
                accuracy=0.0,
                top_k_accuracy={k: 0.0 for k in top_k_values},
                per_class_accuracy={name: 0.0 for name in self.class_names},
                confusion_matrix=[[0] * self.num_classes for _ in range(self.num_classes)],
                precision_macro=0.0,
                recall_macro=0.0,
                f1_macro=0.0,
                num_samples=len(predictions),
                num_correct=0,
            )

        # Basic accuracy
        num_correct = sum(1 for p in valid_preds if p.correct)
        accuracy = num_correct / len(valid_preds)

        # Top-k accuracy
        top_k_accuracy = {}
        for k in top_k_values:
            if k == 1:
                top_k_accuracy[k] = accuracy
            else:
                correct_at_k = 0
                for p in valid_preds:
                    if p.top_k_predictions:
                        top_k_labels = [pred[0] for pred in p.top_k_predictions[:k]]
                        if p.true_label in top_k_labels:
                            correct_at_k += 1
                    elif p.correct:
                        correct_at_k += 1
                top_k_accuracy[k] = correct_at_k / len(valid_preds)

        # Per-class accuracy
        per_class_correct: dict[int, int] = {i: 0 for i in range(self.num_classes)}
        per_class_total: dict[int, int] = {i: 0 for i in range(self.num_classes)}

        for p in valid_preds:
            per_class_total[p.true_label] += 1
            if p.correct:
                per_class_correct[p.true_label] += 1

        per_class_accuracy = {}
        for i, name in enumerate(self.class_names):
            if per_class_total[i] > 0:
                per_class_accuracy[name] = per_class_correct[i] / per_class_total[i]
            else:
                per_class_accuracy[name] = 0.0

        # Confusion matrix
        confusion_matrix = [[0] * self.num_classes for _ in range(self.num_classes)]
        for p in valid_preds:
            if p.predicted_label is not None:
                confusion_matrix[p.true_label][p.predicted_label] += 1

        # Precision, Recall, F1 (macro-averaged)
        precisions = []
        recalls = []
        f1s = []

        for i in range(self.num_classes):
            tp = confusion_matrix[i][i]
            fp = sum(confusion_matrix[j][i] for j in range(self.num_classes)) - tp
            fn = sum(confusion_matrix[i][j] for j in range(self.num_classes)) - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        return ClassificationMetrics(
            accuracy=accuracy,
            top_k_accuracy=top_k_accuracy,
            per_class_accuracy=per_class_accuracy,
            confusion_matrix=confusion_matrix,
            precision_macro=np.mean(precisions),
            recall_macro=np.mean(recalls),
            f1_macro=np.mean(f1s),
            num_samples=len(predictions),
            num_correct=num_correct,
        )

    def parse_classification_response(
        self,
        response: str,
        valid_classes: list[str] | None = None,
    ) -> tuple[int | None, str | None, float | None]:
        """Parse a classification response from an LLM.

        Returns:
            (predicted_label_idx, predicted_label_name, confidence)
        """
        valid_classes = valid_classes or self.class_names
        response_lower = response.lower().strip()

        # Try to find exact class name match
        for idx, class_name in enumerate(valid_classes):
            if class_name.lower() in response_lower:
                return idx, class_name, None

        # Try to find class index (e.g., "class 5", "label: 3")
        import re
        patterns = [
            r'class[:\s]+(\d+)',
            r'label[:\s]+(\d+)',
            r'answer[:\s]+(\d+)',
            r'^(\d+)$',
            r'\b(\d+)\b',
        ]

        for pattern in patterns:
            match = re.search(pattern, response_lower)
            if match:
                try:
                    idx = int(match.group(1))
                    if 0 <= idx < len(valid_classes):
                        return idx, valid_classes[idx], None
                except ValueError:
                    continue

        # Try to find the first mentioned class
        first_match_pos = len(response_lower) + 1
        first_match_idx = None
        first_match_name = None

        for idx, class_name in enumerate(valid_classes):
            pos = response_lower.find(class_name.lower())
            if pos != -1 and pos < first_match_pos:
                first_match_pos = pos
                first_match_idx = idx
                first_match_name = class_name

        if first_match_idx is not None:
            return first_match_idx, first_match_name, None

        return None, None, None

    def format_metrics_summary(self, metrics: ClassificationMetrics) -> str:
        """Format metrics as a human-readable summary."""
        lines = [
            f"Classification Results ({metrics.num_samples} samples)",
            f"=" * 50,
            f"Overall Accuracy: {metrics.accuracy:.2%} ({metrics.num_correct}/{metrics.num_samples})",
            "",
            "Top-K Accuracy:",
        ]

        for k, acc in sorted(metrics.top_k_accuracy.items()):
            lines.append(f"  Top-{k}: {acc:.2%}")

        lines.extend([
            "",
            f"Macro Precision: {metrics.precision_macro:.2%}",
            f"Macro Recall: {metrics.recall_macro:.2%}",
            f"Macro F1: {metrics.f1_macro:.2%}",
            "",
            "Per-Class Accuracy:",
        ])

        for class_name, acc in metrics.per_class_accuracy.items():
            lines.append(f"  {class_name}: {acc:.2%}")

        return "\n".join(lines)
