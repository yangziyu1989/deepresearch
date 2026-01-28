"""Paper section generation module."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepresearch.core.config import ResearchIdea, ValidationResult
from deepresearch.core.state import ExperimentResult, Paper
from deepresearch.modules.experiment.api_manager import APIManager
from deepresearch.providers.base import GenerationRequest, Message


SECTION_PROMPTS = {
    "abstract": """Write an abstract for a research paper with the following details:

Title: {title}
Research Idea: {description}
Methodology: {methodology}
Key Results: {results}
Main Findings: {findings}

Write a concise abstract (150-250 words) that:
1. States the problem/motivation
2. Describes the approach
3. Summarizes key results
4. Highlights the contribution

Output format: {format}
""",
    "introduction": """Write the Introduction section for a research paper.

Title: {title}
Research Idea: {description}
Hypothesis: {hypothesis}
Key Contributions: {contributions}
Related Work Summary: {related_work}

Write an introduction (500-800 words) that:
1. Motivates the problem
2. Describes the gap in existing work
3. Presents the research question/hypothesis
4. Summarizes contributions
5. Outlines the paper structure

Output format: {format}
""",
    "methodology": """Write the Methodology/Method section for a research paper.

Title: {title}
Methodology Description: {methodology}
Experiment Setup: {experiment_setup}
Datasets: {datasets}
Evaluation Metrics: {metrics}

Write a methodology section (600-1000 words) that:
1. Describes the proposed approach in detail
2. Explains the experimental setup
3. Lists datasets and evaluation metrics
4. Provides implementation details

Include mathematical notation where appropriate.
Output format: {format}
""",
    "results": """Write the Results section for a research paper.

Title: {title}
Hypothesis: {hypothesis}
Experiment Results: {results}
Statistical Comparisons: {comparisons}
Key Findings: {findings}

Write a results section (500-800 words) that:
1. Presents main experimental results
2. Compares with baselines
3. Reports statistical significance
4. Discusses ablation studies if applicable

Reference figures as Figure X and tables as Table X.
Output format: {format}
""",
    "conclusion": """Write the Conclusion section for a research paper.

Title: {title}
Key Findings: {findings}
Contributions: {contributions}
Limitations: {limitations}
Future Work: {future_work}

Write a conclusion (200-400 words) that:
1. Summarizes the main contributions
2. Discusses limitations
3. Suggests future research directions

Output format: {format}
""",
}


@dataclass
class PaperContent:
    """Complete paper content."""

    title: str
    abstract: str
    introduction: str
    methodology: str
    results: str
    conclusion: str
    references: list[str]
    figures: list[str]


class SectionWriter:
    """Generates paper sections using LLMs."""

    def __init__(
        self,
        api_manager: APIManager,
        output_dir: Path,
        output_format: str = "latex",
    ) -> None:
        self.api_manager = api_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_format = output_format

    async def write_section(
        self,
        section: str,
        context: dict[str, Any],
    ) -> str:
        """Write a specific paper section."""
        if section not in SECTION_PROMPTS:
            raise ValueError(f"Unknown section: {section}")

        # Add format to context
        context["format"] = self.output_format

        prompt = SECTION_PROMPTS[section].format(**context)

        request = GenerationRequest(
            messages=[
                Message(
                    role="system",
                    content=f"You are an expert academic writer. Write clear, precise, and well-structured {self.output_format} content for research papers.",
                ),
                Message(role="user", content=prompt),
            ],
            temperature=0.7,
            max_tokens=4096,
        )

        response = await self.api_manager.generate(request)
        return response.content

    async def write_full_paper(
        self,
        idea: ResearchIdea,
        results: dict[str, ExperimentResult],
        validation: ValidationResult,
        papers: list[Paper],
        figures: list[str],
    ) -> PaperContent:
        """Write all sections of a paper."""
        # Prepare context for each section
        results_summary = self._format_results_summary(results)
        comparisons_text = self._format_comparisons(validation.statistical_comparisons)
        related_work = self._format_related_work(papers)

        # Write each section
        abstract = await self.write_section("abstract", {
            "title": idea.title,
            "description": idea.description,
            "methodology": idea.methodology,
            "results": results_summary,
            "findings": ", ".join(validation.key_findings),
        })

        introduction = await self.write_section("introduction", {
            "title": idea.title,
            "description": idea.description,
            "hypothesis": idea.hypothesis,
            "contributions": ", ".join(idea.key_contributions),
            "related_work": related_work,
        })

        # Get experiment details for methodology
        experiment_setup = self._format_experiment_setup(results)
        datasets = self._format_datasets(results)
        metrics = self._format_metrics(results)

        methodology = await self.write_section("methodology", {
            "title": idea.title,
            "methodology": idea.methodology,
            "experiment_setup": experiment_setup,
            "datasets": datasets,
            "metrics": metrics,
        })

        results_section = await self.write_section("results", {
            "title": idea.title,
            "hypothesis": idea.hypothesis,
            "results": results_summary,
            "comparisons": comparisons_text,
            "findings": ", ".join(validation.key_findings),
        })

        conclusion = await self.write_section("conclusion", {
            "title": idea.title,
            "findings": ", ".join(validation.key_findings),
            "contributions": ", ".join(idea.key_contributions),
            "limitations": "Sample size limitations, single benchmark evaluation",
            "future_work": ", ".join(validation.suggested_followups),
        })

        # Format references
        references = self._format_references(papers)

        return PaperContent(
            title=idea.title,
            abstract=abstract,
            introduction=introduction,
            methodology=methodology,
            results=results_section,
            conclusion=conclusion,
            references=references,
            figures=figures,
        )

    def _format_results_summary(
        self,
        results: dict[str, ExperimentResult],
    ) -> str:
        """Format results for prompts."""
        lines = []
        for exp_id, result in results.items():
            if result.status != "completed":
                continue
            metrics = ", ".join(f"{k}={v:.3f}" for k, v in result.metrics.items())
            lines.append(f"- {exp_id}: {metrics}")
        return "\n".join(lines) or "No results available"

    def _format_comparisons(
        self,
        comparisons: list,
    ) -> str:
        """Format statistical comparisons."""
        if not comparisons:
            return "No statistical comparisons available"

        lines = []
        for c in comparisons:
            sig = "significant" if c.significant else "not significant"
            lines.append(
                f"- {c.method_a} vs {c.method_b} on {c.metric}: "
                f"p={c.p_value:.4f} ({sig})"
            )
        return "\n".join(lines)

    def _format_related_work(self, papers: list[Paper]) -> str:
        """Format related work summary."""
        if not papers:
            return "Limited prior work in this area"

        lines = []
        for paper in papers[:5]:
            lines.append(f"- {paper.title} ({paper.year})")
        return "\n".join(lines)

    def _format_experiment_setup(
        self,
        results: dict[str, ExperimentResult],
    ) -> str:
        """Format experiment setup description."""
        lines = ["Experiments were conducted using the following setup:"]
        for exp_id, result in results.items():
            lines.append(f"- {exp_id}: {len(result.raw_outputs)} samples")
        return "\n".join(lines)

    def _format_datasets(
        self,
        results: dict[str, ExperimentResult],
    ) -> str:
        """Format dataset information."""
        return "Standard benchmark datasets were used for evaluation."

    def _format_metrics(
        self,
        results: dict[str, ExperimentResult],
    ) -> str:
        """Format metrics information."""
        all_metrics = set()
        for result in results.values():
            all_metrics.update(result.metrics.keys())
        return ", ".join(all_metrics)

    def _format_references(self, papers: list[Paper]) -> list[str]:
        """Format paper references."""
        refs = []
        for paper in papers[:20]:
            authors = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                authors += " et al."
            refs.append(f"{authors}. {paper.title}. {paper.year}.")
        return refs

    def save_paper(
        self,
        paper: PaperContent,
        filename: str = "paper",
    ) -> Path:
        """Save paper to file."""
        if self.output_format == "latex":
            return self._save_latex(paper, filename)
        else:
            return self._save_markdown(paper, filename)

    def _save_latex(self, paper: PaperContent, filename: str) -> Path:
        """Save paper as LaTeX."""
        content = f"""\\documentclass{{article}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{hyperref}}
\\usepackage{{booktabs}}

\\title{{{paper.title}}}
\\author{{DeepResearch}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
{paper.abstract}
\\end{{abstract}}

\\section{{Introduction}}
{paper.introduction}

\\section{{Methodology}}
{paper.methodology}

\\section{{Results}}
{paper.results}

\\section{{Conclusion}}
{paper.conclusion}

\\bibliographystyle{{plain}}
\\begin{{thebibliography}}{{99}}
"""
        for i, ref in enumerate(paper.references):
            content += f"\\bibitem{{ref{i+1}}} {ref}\n"

        content += """\\end{thebibliography}
\\end{document}
"""
        output_path = self.output_dir / f"{filename}.tex"
        output_path.write_text(content)
        return output_path

    def _save_markdown(self, paper: PaperContent, filename: str) -> Path:
        """Save paper as Markdown."""
        content = f"""# {paper.title}

## Abstract

{paper.abstract}

## Introduction

{paper.introduction}

## Methodology

{paper.methodology}

## Results

{paper.results}

## Conclusion

{paper.conclusion}

## References

"""
        for i, ref in enumerate(paper.references):
            content += f"{i+1}. {ref}\n"

        output_path = self.output_dir / f"{filename}.md"
        output_path.write_text(content)
        return output_path
