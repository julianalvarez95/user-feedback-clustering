import os
from pathlib import Path

import openai
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm

from feedback_clustering.embeddings.cache import compute_hash, load_cache, save_cache
from feedback_clustering.exceptions import ConfigurationError, UserCancelledError
from feedback_clustering.models import FeedbackItem

load_dotenv()

_console = Console()
_COST_PER_MILLION_TOKENS = 0.02


def estimate_cost(texts: list[str]) -> tuple[int, float]:
    total_tokens = sum(int(len(text.split()) * 1.3) for text in texts)
    cost = (total_tokens / 1_000_000) * _COST_PER_MILLION_TOKENS
    return total_tokens, cost


def embed_items(
    items: list[FeedbackItem],
    cache_path: Path = Path(".embeddings_cache.json"),
    batch_size: int = 100,
    confirm: bool = True,
) -> list[FeedbackItem]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ConfigurationError(
            "OPENAI_API_KEY environment variable is not set. "
            "Add it to your .env file or export it before running."
        )

    client = openai.OpenAI(api_key=api_key)
    cache = load_cache(cache_path)

    uncached_items: list[FeedbackItem] = []
    for item in items:
        h = compute_hash(item.text)
        if h in cache:
            item.embedding = cache[h]
        else:
            uncached_items.append(item)

    if uncached_items:
        uncached_texts = [it.text for it in uncached_items]
        total_tokens, cost = estimate_cost(uncached_texts)

        if confirm:
            _console.print(
                Panel(
                    f"[bold]Embedding cost preview[/bold]\n\n"
                    f"Items to embed: [cyan]{len(uncached_items)}[/cyan]\n"
                    f"Estimated tokens: [cyan]{total_tokens:,}[/cyan]\n"
                    f"Estimated cost: [cyan]${cost:.6f}[/cyan]",
                    title="OpenAI API",
                    border_style="yellow",
                )
            )
            answer = _console.input("Proceed? [y/N] ").strip().lower()
            if answer != "y":
                raise UserCancelledError("User cancelled embedding step.")

        batches = [
            uncached_items[i : i + batch_size]
            for i in range(0, len(uncached_items), batch_size)
        ]

        for batch in tqdm(batches, desc="Embedding batches", unit="batch"):
            texts = [it.text for it in batch]
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            for item, embedding_data in zip(batch, response.data):
                vector = embedding_data.embedding
                item.embedding = vector
                cache[compute_hash(item.text)] = vector

        save_cache(cache_path, cache)

    return items
