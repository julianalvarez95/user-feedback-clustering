from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from feedback_clustering.config import SourceConfig, load_sources_config
from feedback_clustering.exceptions import FeedbackClusteringError
from feedback_clustering.clustering.kmeans import cluster_items
from feedback_clustering.embeddings.openai_embedder import embed_items
from feedback_clustering.ingestion.csv_loader import load_multiple
from feedback_clustering.labeling.openai_labeler import label_clusters
from feedback_clustering.output.markdown_export import generate_report

app = typer.Typer(help="Cluster user feedback from CSV exports.")

_console = Console()
_err_console = Console(stderr=True)


@app.command()
def run(
    input: Annotated[list[Path], typer.Option(help="Input CSV file(s)")] = [],
    text_col: Annotated[list[str] | None, typer.Option("--text-col")] = None,
    config: Annotated[Path | None, typer.Option("--config")] = None,
    clusters: Annotated[int | None, typer.Option("--clusters", "-k")] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    try:
        # 1. Resolve sources
        sources: list[SourceConfig]
        if config is not None:
            sources = load_sources_config(config)
        elif input:
            cols = list(text_col) if text_col else ["description"]
            sources = [
                SourceConfig(
                    name=f.stem,
                    file=f,
                    text_columns=cols,
                    id_column=None,
                    encoding="utf-8",
                )
                for f in input
            ]
        else:
            _err_console.print(
                "[red]Error:[/red] Provide at least one --input file or a --config YAML."
            )
            raise typer.Exit(1)

        # 2. Load CSV items
        _console.print(f"[bold]Loading[/bold] {len(sources)} source(s)...")
        items = load_multiple(sources)
        _console.print(f"Loaded [cyan]{len(items)}[/cyan] feedback items.")

        if len(items) == 0:
            _err_console.print("[red]Error:[/red] No feedback items found after loading.")
            raise typer.Exit(1)

        # 3. Embed
        _console.print("[bold]Embedding[/bold] feedback items...")
        items = embed_items(items, confirm=not yes)

        # 4. Cluster
        _console.print("[bold]Clustering[/bold]...")
        items, silhouette = cluster_items(items, n_clusters=clusters)
        _console.print(f"Silhouette score: [cyan]{silhouette:.3f}[/cyan]")

        # 5. Group items by cluster_id, sort by distance asc
        from collections import defaultdict
        groups: dict[int, list] = defaultdict(list)
        for item in items:
            groups[item.cluster_id].append(item)
        for cid in groups:
            groups[cid].sort(key=lambda x: x.distance_to_centroid or 0.0)

        clusters_raw = sorted(groups.items(), key=lambda x: x[0])

        # 6. Label clusters
        _console.print("[bold]Labeling[/bold] clusters with GPT-4o-mini...")
        labeled_clusters = label_clusters(list(clusters_raw))

        # Sort by size descending
        labeled_clusters.sort(key=lambda c: c.size, reverse=True)

        # 7. Generate report
        source_names = [s.name for s in sources]
        report = generate_report(
            clusters=labeled_clusters,
            sources=source_names,
            total_items=len(items),
            output_path=output,
            silhouette_score=silhouette,
        )

        if output:
            _console.print(f"[green]Report written to:[/green] {output}")
        else:
            _console.print(report)

    except FeedbackClusteringError as exc:
        _err_console.print(
            Panel(
                str(exc),
                title="[red]Error[/red]",
                border_style="red",
            )
        )
        raise typer.Exit(1)
