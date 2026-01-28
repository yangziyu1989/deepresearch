"""Result figure generation using matplotlib."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from deepresearch.core.config import StatisticalComparison
from deepresearch.core.state import ExperimentResult


@dataclass
class FigureConfig:
    """Configuration for figure generation."""

    width: float = 8.0
    height: float = 6.0
    dpi: int = 300
    font_size: int = 12
    style: str = "seaborn-v0_8-whitegrid"
    palette: str = "Set2"
    format: str = "pdf"  # pdf, png, svg


class ResultFigureGenerator:
    """Generates result figures for papers."""

    def __init__(
        self,
        output_dir: Path,
        config: FigureConfig | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or FigureConfig()

        # Set up matplotlib style
        try:
            plt.style.use(self.config.style)
        except Exception:
            plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({
            "font.size": self.config.font_size,
            "axes.titlesize": self.config.font_size + 2,
            "axes.labelsize": self.config.font_size,
            "xtick.labelsize": self.config.font_size - 1,
            "ytick.labelsize": self.config.font_size - 1,
            "legend.fontsize": self.config.font_size - 1,
        })

    def generate_comparison_bar_chart(
        self,
        results: dict[str, ExperimentResult],
        metric: str,
        title: str | None = None,
        filename: str = "comparison",
    ) -> Path:
        """Generate a bar chart comparing methods on a metric."""
        fig, ax = plt.subplots(
            figsize=(self.config.width, self.config.height),
            dpi=self.config.dpi,
        )

        # Extract data
        names = []
        values = []
        errors = []

        for exp_id, result in results.items():
            if result.status != "completed":
                continue
            names.append(exp_id)
            values.append(result.metrics.get(metric, 0.0))

            # Calculate std from raw outputs if available
            if result.raw_outputs:
                sample_values = [
                    r.get("metrics", {}).get(metric, 0.0)
                    for r in result.raw_outputs
                ]
                errors.append(np.std(sample_values) if sample_values else 0.0)
            else:
                errors.append(0.0)

        # Create bar chart
        colors = sns.color_palette(self.config.palette, len(names))
        bars = ax.bar(names, values, yerr=errors, capsize=5, color=colors)

        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title or f"Comparison on {metric}")
        ax.set_xticklabels(names, rotation=45, ha="right")

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.annotate(
                f"{val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=self.config.font_size - 2,
            )

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / f"{filename}.{self.config.format}"
        fig.savefig(output_path, format=self.config.format, bbox_inches="tight")
        plt.close(fig)

        return output_path

    def generate_ablation_chart(
        self,
        baseline_name: str,
        baseline_value: float,
        ablations: dict[str, float],
        metric: str,
        filename: str = "ablation",
    ) -> Path:
        """Generate an ablation study chart."""
        fig, ax = plt.subplots(
            figsize=(self.config.width, self.config.height),
            dpi=self.config.dpi,
        )

        # Prepare data
        names = [baseline_name] + list(ablations.keys())
        values = [baseline_value] + list(ablations.values())
        colors = ["#2ecc71"] + ["#e74c3c"] * len(ablations)  # Green for baseline, red for ablations

        # Create horizontal bar chart
        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, values, color=colors)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.set_title(f"Ablation Study: {metric}")

        # Add value labels
        for bar, val in zip(bars, values):
            ax.annotate(
                f"{val:.3f}",
                xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                ha="left",
                va="center",
                fontsize=self.config.font_size - 2,
            )

        # Add baseline reference line
        ax.axvline(x=baseline_value, color="#2ecc71", linestyle="--", alpha=0.7)

        plt.tight_layout()

        output_path = self.output_dir / f"{filename}.{self.config.format}"
        fig.savefig(output_path, format=self.config.format, bbox_inches="tight")
        plt.close(fig)

        return output_path

    def generate_line_plot(
        self,
        x_values: list[float],
        y_values: dict[str, list[float]],
        x_label: str,
        y_label: str,
        title: str | None = None,
        filename: str = "line_plot",
    ) -> Path:
        """Generate a line plot with multiple series."""
        fig, ax = plt.subplots(
            figsize=(self.config.width, self.config.height),
            dpi=self.config.dpi,
        )

        colors = sns.color_palette(self.config.palette, len(y_values))
        markers = ["o", "s", "^", "D", "v", "<", ">", "p"]

        for i, (name, values) in enumerate(y_values.items()):
            ax.plot(
                x_values,
                values,
                label=name,
                color=colors[i],
                marker=markers[i % len(markers)],
                markersize=6,
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title or f"{y_label} vs {x_label}")
        ax.legend()

        plt.tight_layout()

        output_path = self.output_dir / f"{filename}.{self.config.format}"
        fig.savefig(output_path, format=self.config.format, bbox_inches="tight")
        plt.close(fig)

        return output_path

    def generate_heatmap(
        self,
        data: list[list[float]],
        row_labels: list[str],
        col_labels: list[str],
        title: str | None = None,
        filename: str = "heatmap",
    ) -> Path:
        """Generate a heatmap."""
        fig, ax = plt.subplots(
            figsize=(self.config.width, self.config.height),
            dpi=self.config.dpi,
        )

        sns.heatmap(
            data,
            xticklabels=col_labels,
            yticklabels=row_labels,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            ax=ax,
        )

        ax.set_title(title or "Performance Heatmap")
        plt.tight_layout()

        output_path = self.output_dir / f"{filename}.{self.config.format}"
        fig.savefig(output_path, format=self.config.format, bbox_inches="tight")
        plt.close(fig)

        return output_path

    def generate_results_table(
        self,
        results: dict[str, ExperimentResult],
        metrics: list[str],
        filename: str = "results_table",
    ) -> Path:
        """Generate a results table as a figure."""
        fig, ax = plt.subplots(
            figsize=(self.config.width, self.config.height / 2),
            dpi=self.config.dpi,
        )
        ax.axis("off")

        # Prepare table data
        headers = ["Method"] + [m.replace("_", " ").title() for m in metrics]
        rows = []

        for exp_id, result in results.items():
            if result.status != "completed":
                continue
            row = [exp_id]
            for metric in metrics:
                val = result.metrics.get(metric, 0.0)
                row.append(f"{val:.3f}")
            rows.append(row)

        # Find best values for bolding
        best_idx = {}
        for i, metric in enumerate(metrics):
            values = [float(row[i + 1]) for row in rows]
            best_idx[i] = values.index(max(values))

        table = ax.table(
            cellText=rows,
            colLabels=headers,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(self.config.font_size)
        table.scale(1.2, 1.5)

        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor("#3498db")
            table[(0, i)].set_text_props(color="white", weight="bold")

        plt.tight_layout()

        output_path = self.output_dir / f"{filename}.{self.config.format}"
        fig.savefig(output_path, format=self.config.format, bbox_inches="tight")
        plt.close(fig)

        return output_path

    def generate_all_figures(
        self,
        results: dict[str, ExperimentResult],
        comparisons: list[StatisticalComparison],
        primary_metric: str,
    ) -> list[Path]:
        """Generate all standard result figures."""
        figures = []

        # Comparison bar chart
        try:
            path = self.generate_comparison_bar_chart(
                results, primary_metric, filename="main_comparison"
            )
            figures.append(path)
        except Exception:
            pass

        # Results table
        try:
            metrics = list(next(iter(results.values())).metrics.keys())
            path = self.generate_results_table(results, metrics[:5])
            figures.append(path)
        except Exception:
            pass

        return figures
