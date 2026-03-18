import json
from unittest.mock import MagicMock, patch

import pytest

from feedback_clustering.labeling.openai_labeler import label_clusters
from feedback_clustering.models import FeedbackItem


def make_cluster_raw(
    cluster_id: int, texts: list[str]
) -> tuple[int, list[FeedbackItem]]:
    items = [
        FeedbackItem(
            id=f"item-{cluster_id}-{i}",
            source="test.csv",
            text=text,
            cluster_id=cluster_id,
            distance_to_centroid=float(i) * 0.1,
        )
        for i, text in enumerate(texts)
    ]
    return cluster_id, items


def make_valid_response(label: str = "Test Label") -> MagicMock:
    content = json.dumps(
        {
            "label": label,
            "description": "This is a test description.",
            "suggested_action": "Take this action.",
        }
    )
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def test_label_clusters_returns_correct_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    clusters_raw = [
        make_cluster_raw(0, ["Login broken", "Cannot login", "Auth fails"]),
        make_cluster_raw(1, ["Slow load", "Page timeout", "Dashboard slow"]),
    ]
    mock_response = make_valid_response()

    with patch("feedback_clustering.labeling.openai_labeler.openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_response

        result = label_clusters(clusters_raw)

    assert len(result) == 2


def test_label_clusters_uses_representative_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    texts = [f"Feedback item {i}" for i in range(6)]
    clusters_raw = [make_cluster_raw(0, texts)]
    mock_response = make_valid_response()

    with patch("feedback_clustering.labeling.openai_labeler.openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_response

        result = label_clusters(clusters_raw)

    # Representative examples = first 3 items (sorted by distance asc)
    assert len(result[0].representative_examples) == 3
    for example in result[0].representative_examples:
        assert example in texts


def test_label_clusters_fallback_on_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    clusters_raw = [make_cluster_raw(0, ["Some feedback"])]

    invalid_message = MagicMock()
    invalid_message.content = "NOT VALID JSON {{{"
    invalid_choice = MagicMock()
    invalid_choice.message = invalid_message
    invalid_response = MagicMock()
    invalid_response.choices = [invalid_choice]

    with patch("feedback_clustering.labeling.openai_labeler.openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = invalid_response

        with patch("feedback_clustering.labeling.openai_labeler._console"):
            result = label_clusters(clusters_raw)

    assert len(result) == 1
    assert "Cluster 0" in result[0].label or result[0].label != ""


def test_label_clusters_has_required_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    clusters_raw = [
        make_cluster_raw(0, ["Feedback A", "Feedback B", "Feedback C"]),
    ]
    mock_response = make_valid_response(label="Mobile Issues")

    with patch("feedback_clustering.labeling.openai_labeler.openai.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_response

        result = label_clusters(clusters_raw)

    cluster = result[0]
    assert cluster.label != ""
    assert cluster.description != ""
    assert cluster.suggested_action != ""
