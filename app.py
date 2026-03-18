"""Streamlit frontend for the user-feedback-clustering pipeline."""

from __future__ import annotations

import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA

from feedback_clustering.clustering.kmeans import cluster_items
from feedback_clustering.config import SourceConfig
from feedback_clustering.embeddings.openai_embedder import embed_items, estimate_cost
from feedback_clustering.exceptions import FeedbackClusteringError
from feedback_clustering.ingestion.csv_loader import load_multiple
from feedback_clustering.labeling.openai_labeler import label_clusters
from feedback_clustering.models import Cluster, FeedbackItem
from feedback_clustering.output.markdown_export import generate_report

st.set_page_config(
    page_title="Feedback Clustering",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# State machine
# Stage values:
#   None / absent  — initial (no run started)
#   "cost_preview" — CSV loaded, waiting for user to confirm embedding cost
#   "running"      — embedding + clustering + labeling in progress
#   "done"         — results ready
# ---------------------------------------------------------------------------

def _reset_pipeline() -> None:
    for key in ("stage", "items_loaded", "cost_info", "file_configs_state",
                "items", "clusters", "silhouette", "report"):
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Configuration")

api_key: str = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    value=os.getenv("OPENAI_API_KEY", ""),
)

cluster_mode: str = st.sidebar.radio(
    "Number of clusters",
    options=["Auto-detect", "Manual"],
    index=0,
)
manual_k: int | None = None
if cluster_mode == "Manual":
    manual_k = st.sidebar.slider("Clusters", min_value=2, max_value=15, value=5)

use_cache: bool = st.sidebar.checkbox("Use embedding cache", value=True)

# ---------------------------------------------------------------------------
# Main area — Step 1: Upload
# ---------------------------------------------------------------------------
st.title("User Feedback Clustering")
st.markdown(
    "Upload one or more CSV exports, configure text columns, and run the pipeline "
    "to automatically group feedback into labeled themes."
)

st.header("1. Upload CSV files")

uploaded_files = st.file_uploader(
    "Choose CSV file(s)",
    type=["csv"],
    accept_multiple_files=True,
)

file_configs: list[dict] = []

if uploaded_files:
    for uf in uploaded_files:
        st.subheader(f"File: {uf.name}")

        try:
            sample_df = pd.read_csv(uf, nrows=0, dtype=str)
            uf.seek(0)
        except Exception as exc:
            st.error(f"Could not read columns from {uf.name}: {exc}")
            continue

        columns = sample_df.columns.str.strip().tolist()

        col1, col2 = st.columns(2)
        with col1:
            text_cols = st.multiselect(
                "Text columns (will be concatenated)",
                options=columns,
                default=columns[:1],
                key=f"text_{uf.name}",
            )
        with col2:
            id_col = st.selectbox(
                "ID column (optional)",
                options=["(none)"] + columns,
                index=0,
                key=f"id_{uf.name}",
            )

        file_configs.append(
            {
                "name": uf.name,
                "content": uf.read(),
                "text_cols": text_cols,
                "id_col": None if id_col == "(none)" else id_col,
            }
        )
        uf.seek(0)

# ---------------------------------------------------------------------------
# Run button — triggers Stage 1: load CSV + show cost preview
# ---------------------------------------------------------------------------
can_run = (
    bool(uploaded_files)
    and bool(api_key)
    and all(cfg["text_cols"] for cfg in file_configs)
)

if uploaded_files and not api_key:
    st.warning("Set your OpenAI API Key in the sidebar to enable the pipeline.")
if uploaded_files and any(not cfg["text_cols"] for cfg in file_configs):
    st.warning("Select at least one text column for each file.")

if st.button("Run Pipeline", disabled=not can_run, type="primary"):
    _reset_pipeline()
    os.environ["OPENAI_API_KEY"] = api_key

    try:
        with st.spinner("Loading CSV files…"):
            sources: list[SourceConfig] = []
            tmp_paths: list[Path] = []

            for cfg in file_configs:
                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                    tmp.write(cfg["content"])
                    tmp_path = Path(tmp.name)
                    tmp_paths.append(tmp_path)

                sources.append(
                    SourceConfig(
                        name=cfg["name"],
                        file=tmp_path,
                        text_columns=cfg["text_cols"],
                        id_column=cfg["id_col"],
                        encoding="utf-8",
                    )
                )

            items_loaded = load_multiple(sources)

            # Clean up temp files immediately — data is in memory
            for p in tmp_paths:
                p.unlink(missing_ok=True)

        if not items_loaded:
            st.error("No items loaded — check that your text columns contain data.")
        else:
            tokens, cost = estimate_cost([it.text for it in items_loaded])
            st.session_state["stage"] = "cost_preview"
            st.session_state["items_loaded"] = items_loaded
            st.session_state["cost_info"] = (tokens, cost)
            st.session_state["file_configs_state"] = {
                "api_key": api_key,
                "manual_k": manual_k,
                "use_cache": use_cache,
                "source_names": [cfg["name"] for cfg in file_configs],
            }
            st.rerun()

    except FeedbackClusteringError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")

# ---------------------------------------------------------------------------
# Stage: cost_preview — show estimate and wait for confirmation
# ---------------------------------------------------------------------------
if st.session_state.get("stage") == "cost_preview":
    tokens, cost = st.session_state["cost_info"]
    n_items = len(st.session_state["items_loaded"])

    st.info(
        f"**Embedding cost preview**\n\n"
        f"- Items to embed: **{n_items}**\n"
        f"- Estimated tokens: **{tokens:,}**\n"
        f"- Estimated cost: **${cost:.6f}**"
    )

    col_proceed, col_cancel = st.columns([1, 5])
    with col_proceed:
        if st.button("Proceed with embedding", type="primary"):
            st.session_state["stage"] = "running"
            st.rerun()
    with col_cancel:
        if st.button("Cancel"):
            _reset_pipeline()
            st.rerun()

# ---------------------------------------------------------------------------
# Stage: running — embed → cluster → label
# ---------------------------------------------------------------------------
if st.session_state.get("stage") == "running":
    cfg_state = st.session_state["file_configs_state"]
    os.environ["OPENAI_API_KEY"] = cfg_state["api_key"]

    try:
        items: list[FeedbackItem] = st.session_state["items_loaded"]

        with st.status("Generating embeddings…", expanded=True) as status:
            cache_path = (
                Path(".embeddings_cache.json")
                if cfg_state["use_cache"]
                else Path(tempfile.mktemp(suffix=".json"))
            )
            items = embed_items(items, cache_path=cache_path, confirm=False)
            status.update(
                label=f"Embeddings ready for {len(items)} items.",
                state="complete",
            )

        with st.status("Clustering…", expanded=True) as status:
            items, silhouette = cluster_items(
                items, n_clusters=cfg_state["manual_k"]
            )
            n_clusters_found = len({it.cluster_id for it in items})
            status.update(
                label=f"Clustered into {n_clusters_found} groups "
                f"(silhouette: {silhouette:.3f}).",
                state="complete",
            )

        with st.status("Labeling clusters with GPT-4o-mini…", expanded=True) as status:
            grouped: dict[int, list[FeedbackItem]] = defaultdict(list)
            for item in items:
                if item.cluster_id is not None:
                    grouped[item.cluster_id].append(item)

            clusters_raw: list[tuple[int, list[FeedbackItem]]] = []
            for cluster_id, cluster_items_list in sorted(grouped.items()):
                cluster_items_list.sort(key=lambda x: x.distance_to_centroid or 0.0)
                clusters_raw.append((cluster_id, cluster_items_list))

            clusters = label_clusters(clusters_raw)
            clusters.sort(key=lambda c: c.size, reverse=True)

            status.update(
                label=f"Labeled {len(clusters)} clusters.",
                state="complete",
            )

        report = generate_report(
            clusters=clusters,
            sources=cfg_state["source_names"],
            total_items=len(items),
            silhouette_score=silhouette,
        )

        st.session_state["items"] = items
        st.session_state["clusters"] = clusters
        st.session_state["silhouette"] = silhouette
        st.session_state["report"] = report
        st.session_state["stage"] = "done"
        st.rerun()

    except FeedbackClusteringError as exc:
        st.error(str(exc))
        st.session_state["stage"] = None
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        st.session_state["stage"] = None

# ---------------------------------------------------------------------------
# Stage: done — show results
# ---------------------------------------------------------------------------
if st.session_state.get("stage") == "done":
    items = st.session_state["items"]
    clusters = st.session_state["clusters"]
    silhouette = st.session_state["silhouette"]
    report = st.session_state["report"]

    st.header("2. Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total items", len(items))
    col2.metric("Clusters found", len(clusters))
    col3.metric(
        "Silhouette score",
        f"{silhouette:.3f}",
        delta="good" if silhouette >= 0.35 else "below threshold",
        delta_color="normal" if silhouette >= 0.35 else "inverse",
    )

    st.subheader("Cluster map")
    embeddings_matrix = np.array([it.embedding for it in items])
    coords = PCA(n_components=2).fit_transform(embeddings_matrix)

    scatter_df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "cluster": [str(it.cluster_id) for it in items],
            "text": [it.text[:120] for it in items],
        }
    )
    fig = px.scatter(
        scatter_df,
        x="x",
        y="y",
        color="cluster",
        hover_data={"text": True, "x": False, "y": False},
        labels={"cluster": "Cluster"},
        title="2D PCA projection of embeddings",
    )
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster details")
    for cluster in clusters:
        with st.expander(f"**{cluster.label}** — {cluster.size} items"):
            st.markdown(f"**Description:** {cluster.description}")
            st.markdown(f"**Suggested action:** {cluster.suggested_action}")
            st.markdown("**Representative examples:**")
            for example in cluster.representative_examples:
                st.markdown(f"- {example}")

    st.subheader("Download report")
    st.download_button(
        label="Download report.md",
        data=report,
        file_name="report.md",
        mime="text/markdown",
    )

    if st.button("Start over"):
        _reset_pipeline()
        st.rerun()
