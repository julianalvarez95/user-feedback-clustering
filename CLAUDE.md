# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

CLI tool that ingests CSV exports (CRM + Jira), generates semantic embeddings, clusters feedback with KMeans, and produces a Markdown report with labeled themes and suggested actions. Target: reduce PM feedback synthesis from hours to minutes.

## Environment setup

```bash
source .venv/bin/activate   # Python 3.12 venv — all deps already installed
```

Required environment variables (load from `.env`, never hardcode):
- `OPENAI_API_KEY` — for `text-embedding-3-small` embeddings and `gpt-4o-mini` labeling

## Running / testing

```bash
# Run the CLI
feedback-cluster run --input tickets.csv --text-col description

# Multiple sources with config
feedback-cluster run --input crm.csv --input jira.csv --config sources.yaml

# Skip cost confirmation prompt
feedback-cluster run --input tickets.csv --yes

# Specify cluster count and output file
feedback-cluster run --input tickets.csv -k 8 -o report.md

# Run all tests
pytest

# Run a single test file
pytest tests/test_ingestion.py

# Run a single test
pytest tests/test_ingestion.py::test_null_rows_are_filtered
```

## Architecture

The pipeline is a linear sequence of stages, each in its own sub-package under `src/feedback_clustering/`:

```
CSV files
  → ingestion/     csv_loader.py       Normalize columns, concat text fields, filter nulls. Supports utf-8 and latin-1.
  → embeddings/    openai_embedder.py  OpenAI text-embedding-3-small (1536-dim). Cache in .embeddings_cache.json keyed by content hash.
  → clustering/    kmeans.py           KMeans via scikit-learn. Auto-detect k via elbow method (second derivative of inertia) if -k not specified.
  → labeling/      openai_labeler.py   OpenAI gpt-4o-mini generates label + description + suggested action per cluster.
  → output/        markdown_export.py  Writes report.md (or stdout). Includes summary table and per-cluster detail.
```

The pipeline is orchestrated in `src/feedback_clustering/cli.py`. All stage functions are pure (accept and return typed data); side effects (API calls, file I/O) are isolated within each stage module.

### Core data models (`src/feedback_clustering/models.py`)

Both use `@dataclass`. No Pydantic dependency.

**`FeedbackItem`**: `id`, `source` (CSV filename), `text` (concatenated columns), `embedding` (list[float]), `cluster_id` (int | None), `distance_to_centroid` (float | None)

**`Cluster`**: `id`, `label`, `description`, `suggested_action`, `size`, `representative_examples` (list[str] — top 3 item texts closest to centroid)

### Exception hierarchy (`src/feedback_clustering/exceptions.py`)

All exceptions inherit from `FeedbackClusteringError`. Specific subclasses: `IngestionError`, `EmbeddingError`, `ClusteringError`, `LabelingError`, `ConfigurationError`, `UserCancelledError`. The CLI catches `FeedbackClusteringError` and exits with code 1.

### Multi-source config (`sources.yaml`)

```yaml
sources:
  - name: crm
    file: crm_export.csv
    text_columns: [description, notes]
    id_column: ticket_id      # optional
    encoding: utf-8
  - name: jira
    file: jira_export.csv
    text_columns: [Summary, Description]
    id_column: Issue key
    encoding: utf-8
```

## Key constraints

- Embedding cost preview: estimated at `len(text.split()) * 1.3` tokens, `$0.02/M tokens`; shown before API call, skipped with `--yes`
- Embedding cache: `.embeddings_cache.json` keyed by SHA-256 hash of item text; loaded at start, saved after each batch
- Embedding batches: 100 items per API call (configurable via `batch_size` param in `embed_items`)
- Clusters ordered by size descending in the report
- Representative examples = items with smallest Euclidean distance to their cluster centroid
- Silhouette score target: > 0.35 — warning printed to stderr if below threshold, pipeline still completes
- Performance target: < 60s for 1,000 rows (excluding API latency)
