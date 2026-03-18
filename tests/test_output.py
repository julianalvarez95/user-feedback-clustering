from pathlib import Path

import pytest

from feedback_clustering.models import Cluster
from feedback_clustering.output.markdown_export import generate_report


def make_clusters(sizes: list[int]) -> list[Cluster]:
    return [
        Cluster(
            id=i,
            label=f"Cluster Label {i}",
            description=f"Description for cluster {i}.",
            suggested_action=f"Action for cluster {i}.",
            size=size,
            representative_examples=[f"Example {j} for cluster {i}" for j in range(3)],
        )
        for i, size in enumerate(sizes)
    ]


def test_generate_report_structure() -> None:
    clusters = make_clusters([10, 5, 3])
    report = generate_report(
        clusters=clusters,
        sources=["crm.csv", "jira.csv"],
        total_items=18,
        silhouette_score=0.42,
    )

    assert "# User Feedback Clustering Report" in report
    assert "## Overview" in report
    assert "## Summary Table" in report
    assert "## Cluster Details" in report
    assert "crm.csv" in report
    assert "jira.csv" in report
    assert "0.420" in report


def test_generate_report_sorted_by_size() -> None:
    # Pass clusters pre-sorted by size desc (as CLI would do)
    clusters = make_clusters([15, 8, 2])
    report = generate_report(
        clusters=clusters,
        sources=["source.csv"],
        total_items=25,
    )

    # Cluster 0 (size=15) should appear before Cluster 1 (size=8) in the report
    pos_0 = report.index("Cluster Label 0")
    pos_1 = report.index("Cluster Label 1")
    pos_2 = report.index("Cluster Label 2")
    assert pos_0 < pos_1 < pos_2


def test_generate_report_writes_file(tmp_path: Path) -> None:
    clusters = make_clusters([5])
    output_path = tmp_path / "report.md"

    generate_report(
        clusters=clusters,
        sources=["source.csv"],
        total_items=5,
        output_path=output_path,
    )

    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "# User Feedback Clustering Report" in content


def test_generate_report_returns_string() -> None:
    clusters = make_clusters([3, 7])
    result = generate_report(
        clusters=clusters,
        sources=["test.csv"],
        total_items=10,
    )
    assert isinstance(result, str)
    assert len(result) > 0
