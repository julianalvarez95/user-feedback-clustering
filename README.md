# User Feedback Clustering

**Turn 200+ raw tickets into actionable product insights in under 5 minutes.**

Product Managers spend 3–5 hours every sprint reading, grouping, and summarizing user feedback. This tool eliminates that. Drop in your CRM and Jira exports, and get back a structured Markdown report with labeled themes, descriptions, and concrete actions your team can act on immediately.

---

## The problem it solves

Every sprint, feedback piles up across multiple tools: support CRM, Jira, user interviews. Someone has to read through hundreds of tickets, mentally group them, and write a summary that's consistent enough for the team to trust.

That process is slow, biased, and impossible to reproduce at scale.

**Before this tool:**
- 3–5 hours per sprint reading and categorizing 200+ tickets
- Different results depending on who does the synthesis
- No way to detect trends across periods
- Jira and CRM feedback living in silos, never crossed

**After:**
- Run one command, get a full thematic report in minutes
- Consistent, reproducible clusters every time
- Output ready to paste directly into Notion or a team doc

---

## How it works

```
CSV files (CRM + Jira exports)
  → ingestion      Normalize columns, concatenate text fields, filter nulls
  → embeddings     text-embedding-3-small (1536-dim) with SHA-256 cache
  → clustering     KMeans via scikit-learn, auto-detect k via elbow method
  → labeling       GPT-4o-mini generates label + description + suggested action
  → report         Markdown with summary table and per-cluster detail
```

The pipeline is intentionally linear and transparent — each stage is a pure function so you can inspect, test, or swap any piece independently.

---

## Demo output

Given a CSV of support tickets, the tool produces:

```markdown
# User Feedback Clustering Report
Generated: 2025-03-18 | Sources: crm_export.csv, jira_export.csv | Total items: 312

## Summary

| #  | Theme                              | Items | % of total |
|----|------------------------------------|-------|------------|
| 1  | Login & Authentication Issues      | 87    | 27.9%      |
| 2  | Performance / Slow Loading         | 64    | 20.5%      |
| 3  | Missing Export Functionality       | 41    | 13.1%      |
| 4  | Billing & Subscription Confusion   | 38    | 12.2%      |

---

## 1 · Login & Authentication Issues

**Description:** Users report being unable to log in after password resets,
session timeouts, and SSO configuration failures with company email aliases.

**Suggested action:** Prioritize an auth reliability audit. Focus on SSO edge
cases in staging and review session expiry logic against "remember me" state.

**Representative examples:**
- "Can't log in after password reset, keeps saying invalid credentials"
- "SSO with Google breaks when using company email alias"
- "Session expires after 10 minutes even with remember me checked"
```

---

## Quickstart

**Requirements:** Python 3.12+ · OpenAI API key

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # add your OPENAI_API_KEY

# Run on a single file
feedback-cluster run --input tickets.csv --text-col description

# Merge multiple sources (CRM + Jira)
feedback-cluster run --config sources.yaml

# Save to file instead of stdout
feedback-cluster run --input tickets.csv -o report.md
```

The tool shows a **cost estimate before calling any API** — you confirm, then it runs.

---

## Web UI

Prefer a browser? There's a full Streamlit interface:

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

**What you get:**

- Upload one or more CSV files, configure text columns and encoding per file
- Live preview of each file before running
- Cluster count: auto-detect (elbow method) or fixed via slider
- Step indicator: Upload → Cost Estimate → Running → Results
- Quality metrics: silhouette score with guidance (poor / moderate / good)
- Cluster size bar chart + 2D PCA scatter plot colored by theme
- Styled cards per cluster: description, suggested action, representative examples
- One-click `report.md` download
- Actionable error messages — no raw stack traces

---

## CLI reference

| Option | Short | Default | Description |
|---|---|---|---|
| `--input PATH` | | — | Input CSV file(s). Repeatable. |
| `--text-col TEXT` | | `description` | Column(s) to use as feedback text. Repeatable. |
| `--config PATH` | | — | YAML config for multiple sources. Overrides `--input`. |
| `--clusters INT` | `-k` | auto | Fixed cluster count. Default: elbow method. |
| `--output PATH` | `-o` | stdout | Write report to file. |
| `--yes` | `-y` | false | Skip cost confirmation prompt. |

---

## Multi-source config

Use a YAML config to merge CSVs with different schemas in one run:

```yaml
# sources.yaml
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

| Field | Required | Description |
|---|---|---|
| `name` | yes | Source label used in the report |
| `file` | yes | Path to CSV (relative to config or absolute) |
| `text_columns` | yes | Columns concatenated as the feedback text |
| `id_column` | no | Item identifier; auto-generated if omitted |
| `encoding` | no | `utf-8` or `latin-1`. Defaults to `utf-8`. |

---

## Key engineering decisions

| Decision | Why |
|---|---|
| **Embedding cache (SHA-256)** | Re-running on the same data skips already-embedded rows — keeps API costs near zero on iterative runs |
| **Cost preview before API call** | You see estimated tokens and cost before committing. No surprises. |
| **Auto k-detection via elbow method** | Removes a parameter most users shouldn't have to tune |
| **Silhouette score gate** | Report flags scores below 0.35 so you know when cluster separation is weak |
| **Pure-function pipeline** | Each stage accepts typed data and returns typed data — no hidden state, fully testable |
| **Multi-source ingestion** | Merge any number of CSV exports with different schemas in a single run |

---

## Run tests

```bash
pytest -v                                              # all tests
pytest tests/test_ingestion.py -v                      # single module
pytest tests/test_ingestion.py::test_null_rows_are_filtered -v  # single test
```

---

## Project context

This tool was built from a real pain point: manually synthesizing sprint feedback across a CRM and Jira takes too long and introduces inconsistency. The goal was a zero-friction workflow — one command, a config file, and output you can share directly with your team.

The full PRD is included in this repo (`PRD.md`) to show the product thinking behind the technical decisions.

---

## License

MIT
