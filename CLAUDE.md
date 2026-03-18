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
# Run the CLI (once entrypoint is wired)
feedback-cluster run --input tickets.csv --text-col description

# Multiple sources with config
feedback-cluster run --input crm.csv --input jira.csv --config sources.yaml

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
  → clustering/    kmeans.py           KMeans via scikit-learn. Auto-detect k via elbow method if not specified.
  → labeling/      openai_labeler.py   OpenAI gpt-4o-mini generates label + description + suggested action per cluster.
  → output/        markdown_export.py  Writes report.md (or stdout). Includes summary table and per-cluster detail.
```

### Core data models (use `dataclasses` or `pydantic`)

**`FeedbackItem`**: `id`, `source` (CSV filename), `text` (concatenated columns), `embedding` (list[float]), `cluster_id` (int), `distance_to_centroid` (float)

**`Cluster`**: `id`, `label`, `description`, `suggested_action`, `size`, `representative_examples` (top 3 closest to centroid)

### Multi-source config (`sources.yaml`)

```yaml
sources:
  - name: crm
    file: crm_export.csv
    text_columns: [description, notes]
    id_column: ticket_id
    encoding: utf-8
  - name: jira
    file: jira_export.csv
    text_columns: [Summary, Description]
    id_column: Issue key
    encoding: utf-8
```

## Key constraints

- Embedding cost preview: show estimated token count and cost before calling OpenAI API, require confirmation
- Embedding cache: skip re-computation for rows already seen (hash-based)
- Clusters ordered by size descending in the report
- Representative examples = items with smallest Euclidean distance to their cluster centroid
- Performance target: < 60s for 1,000 rows (excluding API latency)
- Silhouette score target: > 0.35
