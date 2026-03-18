import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from feedback_clustering.cli import app
from feedback_clustering.exceptions import ClusteringError
from feedback_clustering.models import Cluster, FeedbackItem

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_CRM = FIXTURES / "sample_crm.csv"
SAMPLE_JIRA = FIXTURES / "sample_jira.csv"
SOURCES_YAML = FIXTURES / "sources.yaml"

runner = CliRunner()


def make_fake_embedding(text: str) -> list[float]:
    return [float(hash(text) % 100) / 100.0] * 8


def make_cluster(cluster_id: int, size: int) -> Cluster:
    return Cluster(
        id=cluster_id,
        label=f"Theme {cluster_id}",
        description=f"Description {cluster_id}",
        suggested_action=f"Action {cluster_id}",
        size=size,
        representative_examples=["Example 1", "Example 2"],
    )


def patched_embed(items: list[FeedbackItem], **kwargs: object) -> list[FeedbackItem]:
    import numpy as np
    rng = np.random.default_rng(42)
    for item in items:
        item.embedding = rng.random(8).tolist()
    return items


def patched_label(clusters_raw: list) -> list[Cluster]:
    return [make_cluster(cid, len(itms)) for cid, itms in clusters_raw]


def test_run_with_input_flag() -> None:
    with patch("feedback_clustering.cli.embed_items", side_effect=patched_embed), \
         patch("feedback_clustering.cli.label_clusters", side_effect=patched_label):
        result = runner.invoke(
            app,
            ["--input", str(SAMPLE_CRM), "--text-col", "description", "--yes"],
        )
    assert result.exit_code == 0, result.output


def test_run_with_config_flag() -> None:
    with patch("feedback_clustering.cli.embed_items", side_effect=patched_embed), \
         patch("feedback_clustering.cli.label_clusters", side_effect=patched_label):
        result = runner.invoke(
            app,
            ["--config", str(SOURCES_YAML), "--yes"],
        )
    assert result.exit_code == 0, result.output


def test_run_yes_flag_skips_confirmation() -> None:
    embed_called_with: dict = {}

    def capture_embed(items: list[FeedbackItem], **kwargs: object) -> list[FeedbackItem]:
        embed_called_with.update(kwargs)
        return patched_embed(items)

    with patch("feedback_clustering.cli.embed_items", side_effect=capture_embed), \
         patch("feedback_clustering.cli.label_clusters", side_effect=patched_label):
        result = runner.invoke(
            app,
            ["--input", str(SAMPLE_CRM), "--text-col", "description", "--yes"],
        )

    assert result.exit_code == 0
    assert embed_called_with.get("confirm") is False


def test_run_clustering_error_shows_panel() -> None:
    def raise_clustering(*args: object, **kwargs: object) -> None:
        raise ClusteringError("Simulated clustering failure")

    with patch("feedback_clustering.cli.embed_items", side_effect=patched_embed), \
         patch("feedback_clustering.cli.cluster_items", side_effect=raise_clustering):
        result = runner.invoke(
            app,
            ["--input", str(SAMPLE_CRM), "--text-col", "description", "--yes"],
        )

    assert result.exit_code == 1
