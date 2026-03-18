import io
from pathlib import Path

import pytest

from feedback_clustering.config import SourceConfig
from feedback_clustering.exceptions import IngestionError
from feedback_clustering.ingestion.csv_loader import load_csv, load_multiple

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_CRM = FIXTURES / "sample_crm.csv"
SAMPLE_JIRA = FIXTURES / "sample_jira.csv"


def test_load_csv_basic() -> None:
    items = load_csv(
        file=SAMPLE_CRM,
        text_columns=["description", "notes"],
        id_column="ticket_id",
        encoding="utf-8",
        source_name="crm",
    )
    assert len(items) == 10
    # Verify text concat for first row
    first = next(it for it in items if it.id == "CRM-001")
    assert "Login button not working on mobile" in first.text
    assert "Customer reported on iOS 16" in first.text
    assert " | " in first.text


def test_null_rows_are_filtered(tmp_path: Path) -> None:
    csv_content = "ticket_id,description\nT-001,Valid feedback\nT-002,\nT-003,\n"
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content, encoding="utf-8")

    items = load_csv(
        file=csv_file,
        text_columns=["description"],
        id_column="ticket_id",
        encoding="utf-8",
        source_name="test",
    )
    assert len(items) == 1
    assert items[0].id == "T-001"


def test_missing_column_raises_error() -> None:
    with pytest.raises(IngestionError, match="nonexistent_col"):
        load_csv(
            file=SAMPLE_CRM,
            text_columns=["nonexistent_col"],
            id_column=None,
            encoding="utf-8",
            source_name="crm",
        )


def test_id_fallback(tmp_path: Path) -> None:
    csv_content = "description\nFirst item\nSecond item\n"
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content, encoding="utf-8")

    items = load_csv(
        file=csv_file,
        text_columns=["description"],
        id_column=None,
        encoding="utf-8",
        source_name="mysource",
    )
    assert len(items) == 2
    assert items[0].id == "mysource_0"
    assert items[1].id == "mysource_1"


def test_load_multiple() -> None:
    sources = [
        SourceConfig(
            name="crm",
            file=SAMPLE_CRM,
            text_columns=["description", "notes"],
            id_column="ticket_id",
            encoding="utf-8",
        ),
        SourceConfig(
            name="jira",
            file=SAMPLE_JIRA,
            text_columns=["Summary", "Description"],
            id_column="Issue key",
            encoding="utf-8",
        ),
    ]
    items = load_multiple(sources)
    assert len(items) == 20


def test_encoding_fallback(tmp_path: Path) -> None:
    # Write a CSV with latin-1 encoding containing special characters
    content = "ticket_id,description\nT-001,Caf\xe9 au lait issue\n"
    csv_file = tmp_path / "latin1.csv"
    csv_file.write_bytes(content.encode("latin-1"))

    # Should not raise, should fall back to latin-1
    items = load_csv(
        file=csv_file,
        text_columns=["description"],
        id_column="ticket_id",
        encoding="utf-8",
        source_name="test",
    )
    assert len(items) == 1
    assert "Caf" in items[0].text
