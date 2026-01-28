"""Method figure generation using TikZ."""

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from deepresearch.core.config import ResearchIdea
from deepresearch.core.exceptions import FigureGenerationError
from deepresearch.modules.experiment.api_manager import APIManager
from deepresearch.providers.base import GenerationRequest, Message


TIKZ_GENERATION_PROMPT = """You are an expert at creating TikZ diagrams for academic papers. Generate a TikZ diagram for the following method/architecture.

Method Description:
{description}

Methodology:
{methodology}

Create a clear, professional TikZ diagram that illustrates the key components and flow of this method.

Requirements:
- Use tikzpicture environment
- Include all necessary \\usepackage commands at the top
- Use clear labels and arrows
- Make it suitable for a research paper
- Use a clean, professional style
- Keep it readable when scaled

Output the complete LaTeX code that can be compiled standalone, wrapped in ```latex ... ``` markers.
"""


@dataclass
class TikZConfig:
    """Configuration for TikZ figure generation."""

    latex_compiler: str = "pdflatex"
    output_format: str = "pdf"  # pdf, png, svg
    dpi: int = 300
    timeout: int = 30


class MethodFigureGenerator:
    """Generates method/architecture figures using TikZ."""

    def __init__(
        self,
        api_manager: APIManager,
        output_dir: Path,
        config: TikZConfig | None = None,
    ) -> None:
        self.api_manager = api_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or TikZConfig()

    async def generate_method_diagram(
        self,
        idea: ResearchIdea,
        filename: str = "method_diagram",
    ) -> Path:
        """Generate a method diagram for a research idea."""
        tikz_code = await self._generate_tikz_code(idea)

        # Compile TikZ to PDF
        pdf_path = self._compile_tikz(tikz_code, filename)

        # Convert to desired format if not PDF
        if self.config.output_format != "pdf":
            return self._convert_pdf(pdf_path, self.config.output_format)

        return pdf_path

    async def _generate_tikz_code(self, idea: ResearchIdea) -> str:
        """Generate TikZ code using LLM."""
        prompt = TIKZ_GENERATION_PROMPT.format(
            description=idea.description,
            methodology=idea.methodology,
        )

        request = GenerationRequest(
            messages=[
                Message(
                    role="system",
                    content="You are an expert LaTeX and TikZ diagram creator. Generate clean, compilable TikZ code.",
                ),
                Message(role="user", content=prompt),
            ],
            temperature=0.5,
            max_tokens=4096,
        )

        response = await self.api_manager.generate(request)

        # Extract TikZ code from response
        content = response.content
        if "```latex" in content:
            code = content.split("```latex")[1].split("```")[0].strip()
        elif "```" in content:
            code = content.split("```")[1].split("```")[0].strip()
        else:
            code = content.strip()

        return code

    def _compile_tikz(self, tikz_code: str, filename: str) -> Path:
        """Compile TikZ code to PDF."""
        # Ensure the code has a document wrapper
        if "\\documentclass" not in tikz_code:
            tikz_code = self._wrap_tikz_code(tikz_code)

        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = Path(tmpdir) / f"{filename}.tex"
            tex_path.write_text(tikz_code)

            try:
                # Run pdflatex
                result = subprocess.run(
                    [
                        self.config.latex_compiler,
                        "-interaction=nonstopmode",
                        "-output-directory",
                        tmpdir,
                        str(tex_path),
                    ],
                    capture_output=True,
                    timeout=self.config.timeout,
                    cwd=tmpdir,
                )

                pdf_path = Path(tmpdir) / f"{filename}.pdf"
                if not pdf_path.exists():
                    raise FigureGenerationError(
                        f"LaTeX compilation failed: {result.stderr.decode()[:500]}",
                        figure_type="tikz",
                    )

                # Copy to output directory
                output_path = self.output_dir / f"{filename}.pdf"
                output_path.write_bytes(pdf_path.read_bytes())
                return output_path

            except subprocess.TimeoutExpired:
                raise FigureGenerationError(
                    "LaTeX compilation timed out",
                    figure_type="tikz",
                )
            except FileNotFoundError:
                raise FigureGenerationError(
                    f"LaTeX compiler '{self.config.latex_compiler}' not found. "
                    "Please install texlive or similar.",
                    figure_type="tikz",
                )

    def _wrap_tikz_code(self, tikz_code: str) -> str:
        """Wrap TikZ code in a complete LaTeX document."""
        # Check if it already has necessary packages
        packages = []
        if "\\usepackage{tikz}" not in tikz_code:
            packages.append("\\usepackage{tikz}")
        if "\\usetikzlibrary" not in tikz_code:
            packages.append("\\usetikzlibrary{arrows.meta,positioning,shapes,fit,backgrounds}")

        package_str = "\n".join(packages)

        return f"""\\documentclass[tikz,border=10pt]{{standalone}}
{package_str}
\\begin{{document}}
{tikz_code}
\\end{{document}}
"""

    def _convert_pdf(self, pdf_path: Path, output_format: str) -> Path:
        """Convert PDF to another format."""
        output_path = pdf_path.with_suffix(f".{output_format}")

        try:
            if output_format == "png":
                # Use pdftoppm for PNG conversion
                subprocess.run(
                    [
                        "pdftoppm",
                        "-png",
                        "-r",
                        str(self.config.dpi),
                        "-singlefile",
                        str(pdf_path),
                        str(output_path.with_suffix("")),
                    ],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
            elif output_format == "svg":
                # Use pdf2svg for SVG conversion
                subprocess.run(
                    ["pdf2svg", str(pdf_path), str(output_path)],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
            else:
                return pdf_path

            return output_path

        except (subprocess.SubprocessError, FileNotFoundError) as e:
            # Return PDF if conversion fails
            return pdf_path

    async def generate_flowchart(
        self,
        steps: list[str],
        title: str,
        filename: str = "flowchart",
    ) -> Path:
        """Generate a simple flowchart from a list of steps."""
        # Generate TikZ code for flowchart
        nodes = []
        for i, step in enumerate(steps):
            node_name = f"step{i}"
            y_pos = -i * 1.5
            nodes.append(f"\\node[block] ({node_name}) at (0, {y_pos}) {{{step}}};")

        # Add arrows
        arrows = []
        for i in range(len(steps) - 1):
            arrows.append(f"\\draw[arrow] (step{i}) -- (step{i+1});")

        tikz_code = f"""\\begin{{tikzpicture}}[
    block/.style={{rectangle, draw, fill=blue!20, text width=8em, text centered, rounded corners, minimum height=2em}},
    arrow/.style={{->, >=stealth, thick}}
]
\\node[above] at (0, 0.5) {{\\textbf{{{title}}}}};
{chr(10).join(nodes)}
{chr(10).join(arrows)}
\\end{{tikzpicture}}"""

        return self._compile_tikz(tikz_code, filename)

    async def generate_architecture_diagram(
        self,
        components: list[dict],
        connections: list[tuple[str, str]],
        filename: str = "architecture",
    ) -> Path:
        """Generate an architecture diagram.

        Args:
            components: List of dicts with 'name', 'type' (input/output/process), 'position' (x, y)
            connections: List of (from_name, to_name) tuples
        """
        # Build nodes
        nodes = []
        style_map = {
            "input": "input",
            "output": "output",
            "process": "process",
        }

        for comp in components:
            style = style_map.get(comp.get("type", "process"), "process")
            pos = comp.get("position", (0, 0))
            nodes.append(
                f"\\node[{style}] ({comp['name'].replace(' ', '_')}) at ({pos[0]}, {pos[1]}) {{{comp['name']}}};"
            )

        # Build arrows
        arrows = []
        for from_name, to_name in connections:
            from_id = from_name.replace(" ", "_")
            to_id = to_name.replace(" ", "_")
            arrows.append(f"\\draw[arrow] ({from_id}) -- ({to_id});")

        tikz_code = f"""\\begin{{tikzpicture}}[
    input/.style={{rectangle, draw, fill=green!20, text width=6em, text centered, minimum height=2em}},
    output/.style={{rectangle, draw, fill=red!20, text width=6em, text centered, minimum height=2em}},
    process/.style={{rectangle, draw, fill=blue!20, text width=6em, text centered, rounded corners, minimum height=2em}},
    arrow/.style={{->, >=stealth, thick}}
]
{chr(10).join(nodes)}
{chr(10).join(arrows)}
\\end{{tikzpicture}}"""

        return self._compile_tikz(tikz_code, filename)
