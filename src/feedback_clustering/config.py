from dataclasses import dataclass, field
from pathlib import Path

import yaml

from feedback_clustering.exceptions import ConfigurationError


@dataclass
class SourceConfig:
    name: str
    file: Path
    text_columns: list[str]
    id_column: str | None = None
    encoding: str = "utf-8"


def load_sources_config(path: Path) -> list[SourceConfig]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except FileNotFoundError:
        raise ConfigurationError(f"Config file not found: {path}")
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in config file: {exc}") from exc

    if not isinstance(raw, dict) or "sources" not in raw:
        raise ConfigurationError("Config file must contain a top-level 'sources' key")

    sources: list[SourceConfig] = []
    for i, entry in enumerate(raw["sources"]):
        if not isinstance(entry, dict):
            raise ConfigurationError(f"Source entry {i} is not a mapping")

        name = entry.get("name")
        file_raw = entry.get("file")
        text_columns = entry.get("text_columns")

        if not name:
            raise ConfigurationError(f"Source entry {i} is missing 'name'")
        if not file_raw:
            raise ConfigurationError(f"Source '{name}' is missing 'file'")
        if not text_columns or not isinstance(text_columns, list):
            raise ConfigurationError(
                f"Source '{name}' must have a non-empty 'text_columns' list"
            )

        sources.append(
            SourceConfig(
                name=str(name),
                file=Path(str(file_raw)),
                text_columns=[str(c) for c in text_columns],
                id_column=str(entry["id_column"]) if entry.get("id_column") else None,
                encoding=str(entry.get("encoding", "utf-8")),
            )
        )

    return sources
