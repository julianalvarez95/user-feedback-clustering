"""Streamlit frontend for the user-feedback-clustering pipeline."""

from __future__ import annotations

import io
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
from feedback_clustering.embeddings.cache import compute_hash, load_cache
from feedback_clustering.embeddings.openai_embedder import embed_items, estimate_cost
from feedback_clustering.exceptions import (
    ConfigurationError,
    EmbeddingError,
    FeedbackClusteringError,
    IngestionError,
)
from feedback_clustering.ingestion.csv_loader import load_multiple
from feedback_clustering.labeling.openai_labeler import label_clusters
from feedback_clustering.models import Cluster, FeedbackItem
from feedback_clustering.output.markdown_export import generate_report

st.set_page_config(
    page_title="Feedback Clustering",
    page_icon="🔍",
    layout="wide",
)

_SILHOUETTE_THRESHOLD = 0.35

# ---------------------------------------------------------------------------
# State machine
# Stage values:
#   None / absent  — initial (no run started)
#   "cost_preview" — CSV loaded, waiting for user to confirm embedding cost
#   "running"      — embedding + clustering + labeling in progress
#   "done"         — results ready
# ---------------------------------------------------------------------------


def _reset_pipeline() -> None:
    for key in (
        "stage",
        "items_loaded",
        "cost_info",
        "file_configs_state",
        "items",
        "clusters",
        "silhouette",
        "report",
    ):
        st.session_state.pop(key, None)


def _show_error(exc: Exception) -> None:
    """Show contextual error message based on exception type."""
    if isinstance(exc, IngestionError):
        st.error(
            f"**Ingestion error:** {exc}\n\n"
            "Tip: check that the selected text columns match the CSV headers, "
            "or try switching to latin-1 encoding."
        )
    elif isinstance(exc, ConfigurationError):
        st.error(
            f"**Configuration error:** {exc}\n\n"
            "Tip: verify that your OpenAI API Key is correct and active."
        )
    elif isinstance(exc, EmbeddingError):
        st.error(
            f"**Embedding error:** {exc}\n\n"
            "Tip: check your OpenAI API quota, or wait a moment and retry."
        )
    elif isinstance(exc, FeedbackClusteringError):
        st.error(f"**{type(exc).__name__}:** {exc}")
    else:
        st.error(f"**Unexpected error ({type(exc).__name__}):** {exc}")


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

if st.session_state.get("stage") is not None:
    st.sidebar.divider()
    if st.sidebar.button("Start Over", type="secondary"):
        _reset_pipeline()
        st.rerun()

# ---------------------------------------------------------------------------
# Step indicator stepper
# ---------------------------------------------------------------------------
_stage = st.session_state.get("stage")
_step_map: dict[str | None, int] = {None: 0, "cost_preview": 1, "running": 2, "done": 3}
_active = _step_map.get(_stage, 0)
_steps = ["Upload & Configure", "Cost Estimate", "Running", "Results"]

_step_parts: list[str] = []
for _i, _label in enumerate(_steps):
    if _i < _active:
        _num_color = "#4F46E5"
        _num_text = "#4F46E5"
        _circle_bg = "#EEF2FF"
        _circle_border = "2px solid #4F46E5"
        _label_color = "#4F46E5"
        _weight = "600"
    elif _i == _active:
        _num_color = "#4F46E5"
        _num_text = "#FFFFFF"
        _circle_bg = "#4F46E5"
        _circle_border = "2px solid #4F46E5"
        _label_color = "#4F46E5"
        _weight = "700"
    else:
        _num_color = "#CBD5E1"
        _num_text = "#94A3B8"
        _circle_bg = "#F1F5F9"
        _circle_border = "2px solid #CBD5E1"
        _label_color = "#94A3B8"
        _weight = "400"

    _step_parts.append(
        f'<div style="display:flex;flex-direction:column;align-items:center;gap:4px;min-width:120px;">'
        f'<div style="width:32px;height:32px;border-radius:50%;border:{_circle_border};'
        f'background:{_circle_bg};display:flex;align-items:center;justify-content:center;'
        f'font-weight:700;color:{_num_text};font-size:14px;">{_i + 1}</div>'
        f'<span style="font-size:12px;font-weight:{_weight};color:{_label_color};'
        f'white-space:nowrap;">{_label}</span>'
        f"</div>"
    )
    if _i < len(_steps) - 1:
        _connector = "#4F46E5" if _i < _active else "#CBD5E1"
        _step_parts.append(
            f'<div style="flex:1;height:2px;background:{_connector};margin-top:-16px;"></div>'
        )

st.markdown(
    '<div style="display:flex;align-items:flex-start;justify-content:center;'
    'gap:0;padding:16px 0 24px 0;">'
    + "".join(_step_parts)
    + "</div>",
    unsafe_allow_html=True,
)

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

        col1, col2, col3 = st.columns([3, 2, 1])
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
        with col3:
            encoding = st.selectbox(
                "Encoding",
                options=["utf-8", "latin-1"],
                index=0,
                key=f"enc_{uf.name}",
            )

        content = uf.read()
        file_configs.append(
            {
                "name": uf.name,
                "content": content,
                "text_cols": text_cols,
                "id_col": None if id_col == "(none)" else id_col,
                "encoding": encoding,
            }
        )
        uf.seek(0)

        if text_cols:
            with st.expander("Preview first 5 rows →", expanded=False):
                try:
                    preview_df = pd.read_csv(
                        io.BytesIO(content),
                        nrows=5,
                        dtype=str,
                        encoding=encoding,
                        on_bad_lines="skip",
                    )
                    preview_df.columns = preview_df.columns.str.strip()
                    available_cols = [c for c in text_cols if c in preview_df.columns]
                    if available_cols:
                        preview_df["text_preview"] = (
                            preview_df[available_cols].fillna("").agg(" ".join, axis=1)
                        )
                        id_display = (
                            id_col
                            if (id_col and id_col != "(none)" and id_col in preview_df.columns)
                            else None
                        )
                        if id_display:
                            preview_df = preview_df.rename(columns={id_display: "id"})
                        cols_to_show = (["id"] if id_display else []) + ["text_preview"]
                        st.dataframe(preview_df[cols_to_show], use_container_width=True)
                    else:
                        st.info("Selected text columns not found in preview.")
                except Exception as exc:
                    st.warning(f"Could not generate preview: {exc}")

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
                        encoding=cfg["encoding"],
                    )
                )

            items_loaded = load_multiple(sources)

            # Clean up temp files immediately — data is in memory
            for p in tmp_paths:
                p.unlink(missing_ok=True)

        if not items_loaded:
            st.error("No items loaded — check that your text columns contain data.")
        else:
            # Count cached vs uncached to give accurate cost preview
            existing_cache: dict[str, list[float]] = {}
            if use_cache:
                existing_cache = load_cache(Path(".embeddings_cache.json"))

            all_texts = [it.text for it in items_loaded]
            cached_count = sum(1 for t in all_texts if compute_hash(t) in existing_cache)
            uncached_count = len(items_loaded) - cached_count
            uncached_texts = [t for t in all_texts if compute_hash(t) not in existing_cache]

            if uncached_texts:
                tokens, cost = estimate_cost(uncached_texts)
            else:
                tokens, cost = 0, 0.0

            st.session_state["stage"] = "cost_preview"
            st.session_state["items_loaded"] = items_loaded
            st.session_state["cost_info"] = {
                "tokens": tokens,
                "cost": cost,
                "cached_count": cached_count,
                "uncached_count": uncached_count,
            }
            st.session_state["file_configs_state"] = {
                "api_key": api_key,
                "manual_k": manual_k,
                "use_cache": use_cache,
                "source_names": [cfg["name"] for cfg in file_configs],
            }
            st.rerun()

    except FeedbackClusteringError as exc:
        _show_error(exc)
    except Exception as exc:
        _show_error(exc)

# ---------------------------------------------------------------------------
# Stage: cost_preview — show estimate and wait for confirmation
# ---------------------------------------------------------------------------
if st.session_state.get("stage") == "cost_preview":
    cost_info = st.session_state["cost_info"]
    n_items = len(st.session_state["items_loaded"])
    tokens = cost_info["tokens"]
    cost = cost_info["cost"]
    cached_count = cost_info["cached_count"]
    uncached_count = cost_info["uncached_count"]

    st.info(
        f"**Embedding cost preview**\n\n"
        f"- Items to embed: **{n_items}**\n"
        f"- Cached (free): **{cached_count}**\n"
        f"- Uncached (will call API): **{uncached_count}**\n"
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
            items, silhouette = cluster_items(items, n_clusters=cfg_state["manual_k"])
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
        _show_error(exc)
        st.session_state["stage"] = None
    except Exception as exc:
        _show_error(exc)
        st.session_state["stage"] = None

# ---------------------------------------------------------------------------
# Stage: done — show results
# ---------------------------------------------------------------------------
if st.session_state.get("stage") == "done":
    items = st.session_state["items"]
    clusters: list[Cluster] = st.session_state["clusters"]
    silhouette = st.session_state["silhouette"]
    report = st.session_state["report"]

    st.header("2. Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total items", len(items))
    col2.metric("Clusters found", len(clusters))
    col3.metric(
        "Silhouette score",
        f"{silhouette:.3f}",
        delta="good" if silhouette >= _SILHOUETTE_THRESHOLD else "below threshold",
        delta_color="normal" if silhouette >= _SILHOUETTE_THRESHOLD else "inverse",
    )

    # Silhouette score guidance
    if silhouette < 0.20:
        st.error(
            f"Cluster quality is poor (silhouette {silhouette:.3f} < 0.20). "
            "Consider reviewing your column selection, filtering noise, or gathering more data."
        )
    elif silhouette < _SILHOUETTE_THRESHOLD:
        st.warning(
            f"Cluster quality is moderate (silhouette {silhouette:.3f}, target ≥ {_SILHOUETTE_THRESHOLD}). "
            "Try adjusting the number of clusters or adding more text columns."
        )
    else:
        st.success("Cluster quality is good.")

    # Build cluster label lookup for scatter
    cluster_id_to_label: dict[int, str] = {c.id: c.label for c in clusters}
    multi_source = len({it.source for it in items}) > 1

    st.subheader("Cluster map")
    embeddings_matrix = np.array([it.embedding for it in items])
    coords = PCA(n_components=2).fit_transform(embeddings_matrix)

    scatter_data: dict[str, list] = {
        "x": coords[:, 0].tolist(),
        "y": coords[:, 1].tolist(),
        "cluster": [
            cluster_id_to_label.get(it.cluster_id, str(it.cluster_id)) for it in items
        ],
        "text": [it.text[:120] for it in items],
    }
    if multi_source:
        scatter_data["source"] = [it.source for it in items]

    scatter_df = pd.DataFrame(scatter_data)
    hover_cols: dict = {"text": True, "x": False, "y": False}
    if multi_source:
        hover_cols["source"] = True

    bar_df = pd.DataFrame(
        {"label": [c.label for c in clusters], "size": [c.size for c in clusters]}
    ).sort_values("size", ascending=True)

    chart_col, scatter_col = st.columns([1, 2])
    with chart_col:
        fig_bar = px.bar(
            bar_df,
            x="size",
            y="label",
            orientation="h",
            color="size",
            color_continuous_scale="Blues",
            labels={"size": "Items", "label": "Cluster"},
            title="Cluster sizes",
        )
        fig_bar.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_bar, use_container_width=True)

    with scatter_col:
        fig = px.scatter(
            scatter_df,
            x="x",
            y="y",
            color="cluster",
            hover_data=hover_cols,
            labels={"cluster": "Cluster"},
            title="2D PCA projection of embeddings",
        )
        fig.update_traces(marker=dict(size=6, opacity=0.7))
        st.plotly_chart(fig, use_container_width=True)

    # Styled cluster cards
    st.subheader("Cluster details")
    st.markdown(
        """
        <style>
        .cluster-card {
            border-left: 4px solid #4F46E5;
            background: #F8F9FB;
            border-radius: 4px;
            padding: 16px 20px;
            margin-bottom: 12px;
        }
        .cluster-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .cluster-title {
            font-size: 1.05em;
            font-weight: 700;
            color: #1A1A2E;
        }
        .cluster-badge {
            background: #4F46E5;
            color: white;
            padding: 2px 12px;
            border-radius: 12px;
            font-size: 0.82em;
            font-weight: 600;
            white-space: nowrap;
        }
        .cluster-card p { margin: 4px 0 8px 0; color: #1A1A2E; }
        .cluster-card ul { margin: 4px 0 0 0; padding-left: 18px; color: #374151; }
        .cluster-card li { margin-bottom: 3px; font-size: 0.93em; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    for cluster in clusters:
        examples_html = "".join(
            f"<li>{ex[:200]}</li>" for ex in cluster.representative_examples
        )
        st.markdown(
            f'<div class="cluster-card">'
            f'<div class="cluster-card-header">'
            f'<span class="cluster-title">{cluster.label}</span>'
            f'<span class="cluster-badge">{cluster.size} items</span>'
            f"</div>"
            f"<p><strong>Description:</strong> {cluster.description}</p>"
            f"<p><strong>Suggested action:</strong> {cluster.suggested_action}</p>"
            f"<p><strong>Representative examples:</strong></p>"
            f"<ul>{examples_html}</ul>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.subheader("Download report")
    st.download_button(
        label="Download report.md",
        data=report,
        file_name="report.md",
        mime="text/markdown",
    )
