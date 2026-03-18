import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from feedback_clustering.embeddings.cache import compute_hash, load_cache, save_cache
from feedback_clustering.embeddings.openai_embedder import embed_items, estimate_cost
from feedback_clustering.exceptions import ConfigurationError, UserCancelledError
from feedback_clustering.models import FeedbackItem


def make_items(texts: list[str]) -> list[FeedbackItem]:
    return [
        FeedbackItem(id=f"item-{i}", source="test.csv", text=text)
        for i, text in enumerate(texts)
    ]


def make_mock_embedding_response(texts: list[str]) -> MagicMock:
    response = MagicMock()
    response.data = []
    for text in texts:
        entry = MagicMock()
        entry.embedding = [0.1, 0.2, 0.3]
        response.data.append(entry)
    return response


def test_estimate_cost() -> None:
    texts = ["hello world", "this is a test sentence with more words"]
    total_tokens, cost = estimate_cost(texts)
    assert total_tokens > 0
    assert cost > 0.0
    # "hello world" = 2 words * 1.3 = 2 tokens; second = 8 words * 1.3 = 10 tokens
    assert total_tokens == int(2 * 1.3) + int(8 * 1.3)


def test_embed_items_uses_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    cache_path = tmp_path / "cache.json"
    items = make_items(["Login broken", "App crashes"])

    mock_response = make_mock_embedding_response(["Login broken", "App crashes"])

    with patch("feedback_clustering.embeddings.openai_embedder.openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = mock_response

        # First call
        embed_items(items, cache_path=cache_path, confirm=False)

        assert mock_client.embeddings.create.call_count == 1

        # Reset embeddings and call again — should use cache
        for item in items:
            item.embedding = []
        mock_client.embeddings.create.reset_mock()

        embed_items(items, cache_path=cache_path, confirm=False)
        mock_client.embeddings.create.assert_not_called()


def test_embed_items_skips_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    cache_path = tmp_path / "cache.json"
    items = make_items(["Cached text", "Uncached text"])

    # Pre-populate cache for the first item
    cached_hash = compute_hash("Cached text")
    cache = {cached_hash: [0.9, 0.8, 0.7]}
    save_cache(cache_path, cache)

    mock_response = make_mock_embedding_response(["Uncached text"])

    with patch("feedback_clustering.embeddings.openai_embedder.openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = mock_response

        embed_items(items, cache_path=cache_path, confirm=False)

        # API should only be called for the uncached item
        call_args = mock_client.embeddings.create.call_args
        assert call_args is not None
        # input kwarg
        input_texts = call_args.kwargs.get("input", call_args.args[1] if len(call_args.args) > 1 else [])
        assert "Uncached text" in input_texts
        assert "Cached text" not in input_texts

    # First item uses pre-cached embedding
    assert items[0].embedding == [0.9, 0.8, 0.7]


def test_embed_items_user_cancel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    cache_path = tmp_path / "cache.json"
    items = make_items(["Some feedback text"])

    with patch("feedback_clustering.embeddings.openai_embedder.openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        with patch("feedback_clustering.embeddings.openai_embedder._console") as mock_console:
            mock_console.input.return_value = "n"
            with pytest.raises(UserCancelledError):
                embed_items(items, cache_path=cache_path, confirm=True)


def test_missing_api_key_raises_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    items = make_items(["Some text"])
    # Patch os.environ.get inside the embedder to always return None for OPENAI_API_KEY
    with patch("feedback_clustering.embeddings.openai_embedder.os") as mock_os:
        mock_os.environ.get.return_value = None
        with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
            embed_items(items, cache_path=tmp_path / "cache.json", confirm=False)
