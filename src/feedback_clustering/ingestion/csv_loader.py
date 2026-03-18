from pathlib import Path

import pandas as pd
from rich.console import Console

from feedback_clustering.config import SourceConfig
from feedback_clustering.exceptions import IngestionError
from feedback_clustering.models import FeedbackItem

_console = Console(stderr=True)


def load_csv(
    file: Path,
    text_columns: list[str],
    id_column: str | None,
    encoding: str,
    source_name: str,
) -> list[FeedbackItem]:
    try:
        df = pd.read_csv(file, encoding=encoding, dtype=str)
    except UnicodeDecodeError:
        _console.print(
            f"[yellow]Warning:[/yellow] Could not read '{file}' with encoding "
            f"'{encoding}', falling back to latin-1."
        )
        df = pd.read_csv(file, encoding="latin-1", dtype=str)

    # Normalize column names (strip surrounding whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Validate text columns
    missing = [col for col in text_columns if col not in df.columns]
    if missing:
        available = ", ".join(df.columns.tolist())
        raise IngestionError(
            f"Column(s) {missing} not found in '{file}'. "
            f"Available columns: {available}"
        )

    if id_column is not None and id_column not in df.columns:
        available = ", ".join(df.columns.tolist())
        raise IngestionError(
            f"ID column '{id_column}' not found in '{file}'. "
            f"Available columns: {available}"
        )

    items: list[FeedbackItem] = []
    for row_index, row in df.iterrows():
        parts = [
            str(row[col]).strip()
            for col in text_columns
            if pd.notna(row[col]) and str(row[col]).strip()
        ]
        text = " | ".join(parts)
        if not text:
            continue

        if id_column is not None:
            item_id = str(row[id_column]).strip() if pd.notna(row[id_column]) else f"{source_name}_{row_index}"
        else:
            item_id = f"{source_name}_{row_index}"

        items.append(
            FeedbackItem(
                id=item_id,
                source=file.name,
                text=text,
            )
        )

    return items


def load_multiple(sources: list[SourceConfig]) -> list[FeedbackItem]:
    all_items: list[FeedbackItem] = []
    for source in sources:
        items = load_csv(
            file=source.file,
            text_columns=source.text_columns,
            id_column=source.id_column,
            encoding=source.encoding,
            source_name=source.name,
        )
        all_items.extend(items)
    return all_items
