# user-feedback-clustering

> CLI tool that ingests CSV exports, clusters feedback semantically, and produces a labeled Markdown report — reducing PM synthesis from hours to minutes.

## What it does

Product teams accumulate feedback across CRM exports, Jira tickets, and support queues. Reading and grouping hundreds of items by hand is slow and inconsistent. This tool ingests one or more CSV files, computes semantic embeddings via OpenAI, groups items into themes with KMeans, and labels each theme using GPT-4o-mini — producing a structured Markdown report with descriptions and suggested actions.

## How it works

```
CSV files
  -> ingestion    Normalize columns, concatenate text fields, filter nulls
  -> embeddings   text-embedding-3-small (1536-dim), SHA-256 cache
  -> clustering   KMeans via scikit-learn, auto-detect k via elbow method
  -> labeling     GPT-4o-mini generates label + description + suggested action
  -> report       Markdown with summary table and per-cluster detail
```

## Requirements

- Python 3.12+
- OpenAI API key (`text-embedding-3-small` + `gpt-4o-mini`)

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # then add your OPENAI_API_KEY
```

`.env` format:

```dotenv
OPENAI_API_KEY=sk-...

# Optional: override cache file location (default: .embeddings_cache.json)
# EMBEDDINGS_CACHE_PATH=.embeddings_cache.json
```

## Usage

### Single file

```bash
feedback-cluster run --input tickets.csv --text-col description
```

### Multiple sources with a config YAML

```bash
feedback-cluster run --config sources.yaml
```

### All options

| Option | Short | Type | Default | Description |
|---|---|---|---|---|
| `--input` | | `PATH` (repeatable) | — | Input CSV file(s). Can be specified multiple times. |
| `--text-col` | | `TEXT` (repeatable) | `description` | Column(s) to use as the feedback text. |
| `--config` | | `PATH` | — | YAML config file describing multiple sources. Overrides `--input`. |
| `--clusters` | `-k` | `INT` | auto | Number of clusters. If omitted, k is chosen via the elbow method. |
| `--output` | `-o` | `PATH` | stdout | Write report to a file instead of printing to stdout. |
| `--yes` | `-y` | flag | `false` | Skip the cost confirmation prompt before calling the OpenAI API. |

## Config file format

Use a YAML config to merge multiple CSV sources with different schemas:

```yaml
sources:
  - name: crm
    file: crm_export.csv
    text_columns: [description, notes]   # columns to concatenate as the feedback text
    id_column: ticket_id                 # column to use as item ID (optional)
    encoding: utf-8                      # utf-8 or latin-1

  - name: jira
    file: jira_export.csv
    text_columns: [Summary, Description]
    id_column: Issue key
    encoding: utf-8
```

| Field | Required | Description |
|---|---|---|
| `name` | yes | Label used in the report to identify this source |
| `file` | yes | Path to the CSV file (relative to the config file or absolute) |
| `text_columns` | yes | One or more columns whose values are joined to form the feedback text |
| `id_column` | no | Column to use as the item identifier; auto-generated if omitted |
| `encoding` | no | File encoding (`utf-8` or `latin-1`). Defaults to `utf-8`. |

## Sample output

```markdown
# User Feedback Clustering Report

## Overview

- **Total feedback items**: 10
- **Sources**: sample_crm
- **Clusters found**: 4
- **Silhouette score**: 0.028 (low — below 0.35 threshold)

## Summary Table

| #   | Label                                    | Size | Suggested Action                          |
| --- | ---------------------------------------- | ---- | ----------------------------------------- |
| 1   | Performance and Functionality Issues     | 5    | Conduct a comprehensive review of the ... |
| 2   | Authentication and Notification Problems | 3    | Investigate the email delivery system ... |
| 3   | Dark Mode Preference Issue               | 1    | Investigate the persistence mechanism ... |
| 4   | Irrelevant Search Results                | 1    | Review and enhance the search algorithm.. |

## Cluster Details

### 1. Performance and Functionality Issues

**Size**: 5 items

**Description**: Users are facing multiple performance and functionality problems,
including slow dashboard loading, mobile app crashes, and issues with billing and
data export features.

**Suggested Action**: Conduct a comprehensive review of the app's performance
metrics and functionality, prioritizing fixes for the mobile app and critical
features like billing and data export.

**Representative Examples**:

- Dashboard loads slowly
- Login button not working on mobile
- Mobile app crashes on startup
```

## Key features

- **Embedding cache** — items are hashed (SHA-256); already-embedded rows are skipped on re-runs, saving API cost.
- **Cost preview** — estimated token count and cost are shown before any OpenAI call; requires confirmation unless `--yes` is passed.
- **Auto k-detection** — if `-k` is not specified, the optimal number of clusters is selected via the elbow method.
- **Silhouette score gate** — the report flags scores below 0.35 so you know when clusters lack separation.
- **Multi-source ingestion** — merge CRM exports, Jira CSVs, or any tabular feedback into a single run.
- **Encoding fallback** — per-source `encoding` field accepts `utf-8` or `latin-1` to handle legacy exports.

## Development

```bash
# Run all tests
pytest -v

# Run a single test file
pytest tests/test_ingestion.py -v

# Run a single test
pytest tests/test_ingestion.py::test_null_rows_are_filtered -v
```

## License

MIT
