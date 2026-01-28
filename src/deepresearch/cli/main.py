"""CLI entry point for DeepResearch."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from deepresearch.core.config import PipelineConfig, PipelineStage
from deepresearch.pipeline.research_pipeline import ResearchPipeline

app = typer.Typer(
    name="deepresearch",
    help="Automated AI Research Pipeline",
    add_completion=False,
)
console = Console()


def create_progress_callback(progress: Progress, task_id: int):
    """Create a progress callback for the pipeline."""

    def callback(stage: PipelineStage, message: str):
        progress.update(task_id, description=f"[cyan]{stage.value}[/cyan]: {message}")

    return callback


@app.command()
def run(
    topic: str = typer.Argument(..., help="Research topic to investigate"),
    output_dir: Path = typer.Option(
        Path("data/outputs"),
        "--output",
        "-o",
        help="Output directory for generated files",
    ),
    session_dir: Path = typer.Option(
        Path("data/sessions"),
        "--sessions",
        help="Directory for session state files",
    ),
    budget: float = typer.Option(
        50.0,
        "--budget",
        "-b",
        help="Maximum budget in USD",
    ),
    format: str = typer.Option(
        "latex",
        "--format",
        "-f",
        help="Output format (latex or markdown)",
    ),
    resume: Optional[str] = typer.Option(
        None,
        "--resume",
        "-r",
        help="Resume from a previous session ID",
    ),
):
    """Run the complete research pipeline on a topic."""
    console.print(Panel(f"[bold blue]DeepResearch[/bold blue]\n{topic}", expand=False))

    config = PipelineConfig(
        research_topic=topic,
        output_dir=output_dir,
        session_dir=session_dir,
        output_format=format,
    )
    config.api.total_budget_usd = budget

    pipeline = ResearchPipeline(config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Starting pipeline...", total=None)
        callback = create_progress_callback(progress, task)

        result = asyncio.run(
            pipeline.run(
                research_topic=topic if not resume else None,
                session_id=resume,
                progress_callback=callback,
            )
        )

    # Display results
    console.print()
    if result.success:
        console.print("[green]Pipeline completed successfully![/green]")
        console.print(f"Session ID: {result.session_id}")
        console.print(f"Total cost: ${result.total_cost_usd:.4f}")

        if result.paper_path:
            console.print(f"Paper: {result.paper_path}")

        if result.figure_paths:
            console.print(f"Figures: {len(result.figure_paths)} generated")
    else:
        console.print(f"[red]Pipeline failed: {result.error}[/red]")

    # Show stage summary
    if result.stage_results:
        table = Table(title="Stage Results")
        table.add_column("Stage")
        table.add_column("Status")
        table.add_column("Duration")
        table.add_column("Cost")

        for stage_name, stage_result in result.stage_results.items():
            status = "[green]OK[/green]" if stage_result.success else "[red]FAIL[/red]"
            table.add_row(
                stage_name,
                status,
                f"{stage_result.duration_seconds:.1f}s",
                f"${stage_result.cost_usd:.4f}",
            )

        console.print(table)


@app.command()
def list_sessions(
    session_dir: Path = typer.Option(
        Path("data/sessions"),
        "--sessions",
        help="Directory for session state files",
    ),
):
    """List all saved research sessions."""
    config = PipelineConfig(
        research_topic="",
        session_dir=session_dir,
    )
    pipeline = ResearchPipeline(config)
    sessions = pipeline.list_sessions()

    if not sessions:
        console.print("No sessions found.")
        return

    table = Table(title="Research Sessions")
    table.add_column("Session ID")
    table.add_column("Topic")
    table.add_column("Stage")
    table.add_column("Cost")

    for session_id in sessions:
        try:
            state = pipeline.load_session(session_id)
            table.add_row(
                session_id,
                state.research_topic[:50],
                state.current_stage.value,
                f"${state.total_cost_usd:.4f}",
            )
        except Exception:
            table.add_row(session_id, "Error loading", "-", "-")

    console.print(table)


@app.command()
def show(
    session_id: str = typer.Argument(..., help="Session ID to display"),
    session_dir: Path = typer.Option(
        Path("data/sessions"),
        "--sessions",
        help="Directory for session state files",
    ),
):
    """Show details of a research session."""
    config = PipelineConfig(
        research_topic="",
        session_dir=session_dir,
    )
    pipeline = ResearchPipeline(config)

    try:
        state = pipeline.load_session(session_id)
    except Exception as e:
        console.print(f"[red]Error loading session: {e}[/red]")
        raise typer.Exit(1)

    console.print(Panel(f"[bold]{state.research_topic}[/bold]", title="Research Topic"))

    # Show stages
    table = Table(title="Pipeline Progress")
    table.add_column("Stage")
    table.add_column("Status")

    for stage in PipelineStage:
        status = "[green]Complete[/green]" if state.is_stage_complete(stage) else "[yellow]Pending[/yellow]"
        if stage == state.current_stage and not state.is_stage_complete(stage):
            status = "[cyan]Current[/cyan]"
        table.add_row(stage.value, status)

    console.print(table)

    # Show research idea if available
    if state.research_idea:
        console.print(Panel(
            f"[bold]{state.research_idea.title}[/bold]\n\n"
            f"{state.research_idea.description}\n\n"
            f"Novelty Score: {state.novelty_score:.2f}",
            title="Research Idea",
        ))

    # Show experiment results if available
    if state.experiment_results:
        table = Table(title="Experiment Results")
        table.add_column("Experiment")
        table.add_column("Status")
        table.add_column("Metrics")

        for exp_id, result in state.experiment_results.items():
            metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in result.metrics.items())
            status_color = "green" if result.status == "completed" else "red"
            table.add_row(
                exp_id,
                f"[{status_color}]{result.status}[/{status_color}]",
                metrics_str or "-",
            )

        console.print(table)

    console.print(f"\nTotal Cost: ${state.total_cost_usd:.4f}")


@app.command()
def vision(
    topic: str = typer.Argument(..., help="Vision research topic"),
    datasets: str = typer.Option(
        "mnist,cifar10",
        "--datasets",
        "-d",
        help="Comma-separated datasets (mnist, cifar10, cifar100)",
    ),
    num_samples: int = typer.Option(
        100,
        "--samples",
        "-n",
        help="Number of samples per experiment",
    ),
    output_dir: Path = typer.Option(
        Path("data/outputs"),
        "--output",
        "-o",
        help="Output directory",
    ),
    budget: float = typer.Option(
        20.0,
        "--budget",
        "-b",
        help="Maximum budget in USD",
    ),
    resume: Optional[str] = typer.Option(
        None,
        "--resume",
        "-r",
        help="Resume from session ID",
    ),
):
    """Run vision research pipeline (MNIST, CIFAR-10)."""
    from deepresearch.pipeline.vision_pipeline import VisionResearchPipeline

    console.print(Panel(
        f"[bold blue]Vision Research[/bold blue]\n{topic}\nDatasets: {datasets}",
        expand=False
    ))

    dataset_list = [d.strip() for d in datasets.split(",")]

    config = PipelineConfig(
        research_topic=topic,
        output_dir=output_dir,
        session_dir=Path("data/sessions"),
        results_dir=Path("data/results"),
    )
    config.api.total_budget_usd = budget

    pipeline = VisionResearchPipeline(config, datasets=dataset_list)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Starting vision pipeline...", total=None)
        callback = create_progress_callback(progress, task)

        result = asyncio.run(
            pipeline.run(
                research_topic=topic if not resume else None,
                session_id=resume,
                progress_callback=callback,
            )
        )

    console.print()
    if result.success:
        console.print("[green]Vision pipeline completed![/green]")
        console.print(f"Session: {result.session_id}")
        console.print(f"Total cost: ${result.total_cost_usd:.4f}")

        if result.metrics_summary:
            table = Table(title="Experiment Results")
            table.add_column("Experiment")
            table.add_column("Accuracy")
            table.add_column("Status")

            for exp_id, metrics in result.metrics_summary.items():
                acc = metrics.get("accuracy", 0)
                status = metrics.get("status", "unknown")
                color = "green" if status == "completed" else "red"
                table.add_row(
                    exp_id,
                    f"{acc:.2%}",
                    f"[{color}]{status}[/{color}]",
                )
            console.print(table)

        if result.paper_path:
            console.print(f"Paper: {result.paper_path}")
    else:
        console.print(f"[red]Pipeline failed: {result.error}[/red]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    max_results: int = typer.Option(20, "--max", "-m", help="Maximum results"),
):
    """Search literature for a topic."""
    from deepresearch.modules.literature.searcher import LiteratureSearcher

    searcher = LiteratureSearcher()

    with console.status("Searching..."):
        papers = asyncio.run(searcher.search(query, max_results=max_results))

    if not papers:
        console.print("No papers found.")
        return

    table = Table(title=f"Literature Search: {query}")
    table.add_column("Title", max_width=60)
    table.add_column("Year")
    table.add_column("Citations")
    table.add_column("Source")

    for paper in papers:
        table.add_row(
            paper.title[:60],
            str(paper.year),
            str(paper.citations),
            paper.source,
        )

    console.print(table)
    console.print(f"\nFound {len(papers)} papers")


@app.command()
def cost(
    session_dir: Path = typer.Option(
        Path("data/sessions"),
        "--sessions",
        help="Directory for session state files",
    ),
):
    """Show total cost across all sessions."""
    config = PipelineConfig(
        research_topic="",
        session_dir=session_dir,
    )
    pipeline = ResearchPipeline(config)
    sessions = pipeline.list_sessions()

    total_cost = 0.0
    for session_id in sessions:
        try:
            state = pipeline.load_session(session_id)
            total_cost += state.total_cost_usd
        except Exception:
            pass

    console.print(f"Total cost across {len(sessions)} sessions: ${total_cost:.4f}")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
