import numpy as np
from rich.console import Console
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from feedback_clustering.exceptions import ClusteringError
from feedback_clustering.models import FeedbackItem

_console = Console(stderr=True)

_SILHOUETTE_THRESHOLD = 0.35


def detect_optimal_k(
    embeddings: list[list[float]],
    k_min: int = 2,
    k_max: int = 15,
) -> int:
    n = len(embeddings)
    k_max = min(k_max, n // 2)
    k_max = max(k_max, k_min)

    X = np.array(embeddings)
    inertias: list[float] = []
    k_range = range(k_min, k_max + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(X)
        inertias.append(float(km.inertia_))

    if len(inertias) < 3:
        return k_min

    second_derivative = np.diff(inertias, n=2)
    best_index = int(np.argmax(second_derivative))
    # argmax of second derivative is relative to k_min+1
    optimal_k = k_min + best_index + 1
    return int(optimal_k)


def cluster_items(
    items: list[FeedbackItem],
    n_clusters: int | None = None,
    random_state: int = 42,
) -> tuple[list[FeedbackItem], float]:
    if not items:
        raise ClusteringError("Cannot cluster an empty list of items.")

    embeddings = [item.embedding for item in items]
    if any(len(e) == 0 for e in embeddings):
        raise ClusteringError(
            "Some items have empty embeddings. Run embed_items before clustering."
        )

    X = np.array(embeddings)

    if n_clusters is None:
        n_clusters = detect_optimal_k(embeddings)

    n_clusters = min(n_clusters, len(items))

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_

    for item, label in zip(items, labels):
        item.cluster_id = int(label)
        centroid = centroids[label]
        distance = float(np.linalg.norm(np.array(item.embedding) - centroid))
        item.distance_to_centroid = distance

    score: float
    if n_clusters > 1:
        score = float(silhouette_score(X, labels))
    else:
        score = 0.0

    if score < _SILHOUETTE_THRESHOLD:
        _console.print(
            f"[yellow]Warning:[/yellow] Silhouette score {score:.3f} is below the "
            f"target threshold of {_SILHOUETTE_THRESHOLD}. Cluster quality may be low."
        )

    return items, score
