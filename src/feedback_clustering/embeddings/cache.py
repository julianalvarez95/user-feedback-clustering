import hashlib
import json
from pathlib import Path


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_cache(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def save_cache(path: Path, cache: dict[str, list[float]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(cache, fh)
