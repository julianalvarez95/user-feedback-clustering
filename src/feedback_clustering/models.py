from dataclasses import dataclass, field


@dataclass
class FeedbackItem:
    id: str
    source: str
    text: str
    embedding: list[float] = field(default_factory=list)
    cluster_id: int | None = None
    distance_to_centroid: float | None = None


@dataclass
class Cluster:
    id: int
    label: str
    description: str
    suggested_action: str
    size: int
    representative_examples: list[str]
