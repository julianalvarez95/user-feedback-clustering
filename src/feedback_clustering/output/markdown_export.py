from pathlib import Path

from feedback_clustering.models import Cluster


def generate_report(
    clusters: list[Cluster],
    sources: list[str],
    total_items: int,
    output_path: Path | None = None,
    silhouette_score: float | None = None,
) -> str:
    lines: list[str] = []

    lines.append("# User Feedback Clustering Report")
    lines.append("")

    lines.append("## Overview")
    lines.append("")
    lines.append(f"- **Total feedback items**: {total_items}")
    lines.append(f"- **Sources**: {', '.join(sources)}")
    lines.append(f"- **Clusters found**: {len(clusters)}")
    if silhouette_score is not None:
        quality = "good" if silhouette_score >= 0.35 else "low (below 0.35 threshold)"
        lines.append(f"- **Silhouette score**: {silhouette_score:.3f} ({quality})")
    lines.append("")

    lines.append("## Summary Table")
    lines.append("")
    lines.append("| # | Label | Size | Suggested Action |")
    lines.append("|---|-------|------|-----------------|")
    for i, cluster in enumerate(clusters, start=1):
        action_preview = cluster.suggested_action[:80] + "..." if len(cluster.suggested_action) > 80 else cluster.suggested_action
        lines.append(f"| {i} | {cluster.label} | {cluster.size} | {action_preview} |")
    lines.append("")

    lines.append("## Cluster Details")
    lines.append("")
    for i, cluster in enumerate(clusters, start=1):
        lines.append(f"### {i}. {cluster.label}")
        lines.append("")
        lines.append(f"**Size**: {cluster.size} items")
        lines.append("")
        lines.append(f"**Description**: {cluster.description}")
        lines.append("")
        lines.append(f"**Suggested Action**: {cluster.suggested_action}")
        lines.append("")
        lines.append("**Representative Examples**:")
        lines.append("")
        for example in cluster.representative_examples:
            lines.append(f"- {example}")
        lines.append("")

    report = "\n".join(lines)

    if output_path is not None:
        output_path.write_text(report, encoding="utf-8")

    return report
