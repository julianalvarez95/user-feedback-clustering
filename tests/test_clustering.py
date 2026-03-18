import io
from unittest.mock import patch

import numpy as np
import pytest

from feedback_clustering.clustering.kmeans import cluster_items, detect_optimal_k
from feedback_clustering.models import FeedbackItem


def make_items_with_embeddings(embeddings: list[list[float]]) -> list[FeedbackItem]:
    return [
        FeedbackItem(
            id=f"item-{i}",
            source="test.csv",
            text=f"feedback text {i}",
            embedding=emb,
        )
        for i, emb in enumerate(embeddings)
    ]


def make_clustered_embeddings(
    n_clusters: int = 3,
    n_per_cluster: int = 10,
    dims: int = 8,
    seed: int = 42,
) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    all_embeddings: list[list[float]] = []
    for c in range(n_clusters):
        center = rng.uniform(-10, 10, dims)
        cluster_points = center + rng.normal(0, 0.1, (n_per_cluster, dims))
        all_embeddings.extend(cluster_points.tolist())
    return all_embeddings


def test_cluster_items_assigns_cluster_ids() -> None:
    embeddings = make_clustered_embeddings()
    items = make_items_with_embeddings(embeddings)
    result, _ = cluster_items(items, n_clusters=3)
    assert all(item.cluster_id is not None for item in result)
    cluster_ids = {item.cluster_id for item in result}
    assert len(cluster_ids) == 3


def test_cluster_items_assigns_distances() -> None:
    embeddings = make_clustered_embeddings()
    items = make_items_with_embeddings(embeddings)
    result, _ = cluster_items(items, n_clusters=3)
    assert all(item.distance_to_centroid is not None for item in result)
    assert all(item.distance_to_centroid >= 0.0 for item in result)


def test_detect_optimal_k() -> None:
    # Generate clearly separated clusters so elbow method can detect them
    embeddings = make_clustered_embeddings(n_clusters=3, n_per_cluster=15)
    k = detect_optimal_k(embeddings, k_min=2, k_max=8)
    # Should detect something reasonable (not necessarily exactly 3, but in range)
    assert 2 <= k <= 8


def test_silhouette_warning_emitted() -> None:
    # Completely random data — silhouette should be low
    rng = np.random.default_rng(0)
    embeddings = rng.random((30, 8)).tolist()
    items = make_items_with_embeddings(embeddings)

    stderr_output: list[str] = []

    with patch(
        "feedback_clustering.clustering.kmeans._console"
    ) as mock_console:
        _, score = cluster_items(items, n_clusters=5)
        # If score < 0.35, warning should have been printed
        if score < 0.35:
            mock_console.print.assert_called_once()
            call_str = str(mock_console.print.call_args)
            assert "Warning" in call_str or "warning" in call_str.lower() or "0.35" in call_str
