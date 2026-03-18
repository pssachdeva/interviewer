"""Interactive browser for opacity rationale UMAP plots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "opacity_rationale_clustering"
MECHANISMS = [
    "voice_opacity",
    "vulnerability_opacity",
    "provenance_opacity",
    "attention_opacity",
    "investment_opacity",
]
COLOR_OPTIONS = {
    "Cluster": "cluster_label",
    "Opacity level": "level",
    "Opacity form": "form",
    "Transcript type": "transcript_type",
}
LEVEL_ORDER = ["none", "potential", "clear"]
FORM_ORDER = ["none", "production", "avoidance", "mixed"]
TYPE_ORDER = ["work", "science", "creativity"]
LEVEL_COLORS = {
    "none": "#cbd5e1",
    "potential": "#f59e0b",
    "clear": "#dc2626",
}
FORM_COLORS = {
    "none": "#cbd5e1",
    "production": "#2563eb",
    "avoidance": "#059669",
    "mixed": "#7c3aed",
}
TYPE_COLORS = {
    "work": "#2563eb",
    "science": "#059669",
    "creativity": "#d97706",
}


st.set_page_config(
    page_title="Opacity UMAP Browser",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        .stApp {
            background: #111315;
            color: #ece7df;
        }

        .block-container {
            max-width: 1600px;
            padding-top: 1.2rem;
            padding-bottom: 1.2rem;
        }

        div[data-baseweb="segmented-control"] {
            background: transparent;
            gap: 0.35rem;
        }

        div[data-baseweb="segmented-control"] [role="radiogroup"] {
            gap: 0.35rem;
        }

        div[data-baseweb="segment"] {
            background: #1d2125;
            border-radius: 999px;
            border: 1px solid #353b42;
        }

        div[data-baseweb="segment"] label {
            color: #cfc7bc !important;
        }

        div[data-baseweb="segment"]:has(input:checked) {
            background: #efe5d2;
            border-color: #c4a97e;
        }

        div[data-baseweb="segment"]:has(input:checked) label {
            color: #1e1811 !important;
        }

        .detail-card {
            background: #1a1e22;
            border: 1px solid #2d3339;
            border-radius: 16px;
            padding: 1rem 1.1rem;
        }

        .detail-card.centered {
            text-align: center;
        }

        .metric-chip {
            display: inline-block;
            padding: 0.25rem 0.55rem;
            border-radius: 999px;
            background: #262b31;
            border: 1px solid #384049;
            margin-right: 0.4rem;
            margin-bottom: 0.35rem;
            font-size: 0.9rem;
        }

        [data-testid="stTextInput"] input {
            background: #1a1e22;
            color: #ece7df;
            border-color: #353b42;
        }

        [data-testid="stDataFrame"] {
            background: #1a1e22;
            border-radius: 16px;
        }

        .detail-section-title {
            font-size: 0.92rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #b7c0ca;
            margin-bottom: 0.6rem;
        }

        .side-panel {
            position: sticky;
            top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def mechanism_title(mechanism: str) -> str:
    return mechanism.replace("_opacity", "").replace("_", " ").title()


def transcript_type(transcript_id: str) -> str:
    prefix = str(transcript_id).split("_", 1)[0]
    if prefix in {"work", "science", "creativity"}:
        return prefix
    return "unknown"


@st.cache_data(show_spinner=False)
def load_mechanism_points(artifacts_dir: str, mechanism: str) -> pd.DataFrame:
    path = Path(artifacts_dir) / mechanism / "clustered_rationales.csv"
    df = pd.read_csv(path)
    df["transcript_type"] = df["transcript_id"].map(transcript_type)
    df["cluster_label"] = df["cluster_label"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def list_available_mechanisms(artifacts_dir: str) -> list[str]:
    root = Path(artifacts_dir)
    return [mechanism for mechanism in MECHANISMS if (root / mechanism / "clustered_rationales.csv").exists()]


def cluster_palette(labels: list[str]) -> dict[str, str]:
    palette = px.colors.qualitative.Bold + px.colors.qualitative.Set3 + px.colors.qualitative.Safe
    return {label: palette[index % len(palette)] for index, label in enumerate(labels)}


def color_settings(df: pd.DataFrame, color_label: str) -> tuple[str, dict[str, str], list[str] | None]:
    column = COLOR_OPTIONS[color_label]
    if column == "cluster_label":
        ordered = sorted(df[column].unique(), key=lambda value: int(str(value).replace("C", "")))
        return column, cluster_palette(ordered), ordered
    if column == "level":
        return column, LEVEL_COLORS, LEVEL_ORDER
    if column == "form":
        return column, FORM_COLORS, FORM_ORDER
    if column == "transcript_type":
        present = [value for value in TYPE_ORDER if value in set(df[column])]
        return column, TYPE_COLORS, present
    raise ValueError(f"Unsupported color label: {color_label}")


def build_figure(df: pd.DataFrame, color_label: str):
    color_column, color_map, category_orders = color_settings(df, color_label)
    fig = px.scatter(
        df,
        x="umap_x",
        y="umap_y",
        color=color_column,
        color_discrete_map=color_map,
        category_orders={color_column: category_orders} if category_orders is not None else None,
        hover_name="transcript_id",
        custom_data=["row_id"],
        hover_data={
            "cluster_label": True,
            "level": True,
            "form": True,
            "transcript_type": True,
            "umap_x": ":.3f",
            "umap_y": ":.3f",
            "rationale": False,
        },
        labels={
            "umap_x": "UMAP 1",
            "umap_y": "UMAP 2",
            "cluster_label": "Cluster",
            "level": "Opacity level",
            "form": "Opacity form",
            "transcript_type": "Transcript type",
        },
        template="plotly_white",
        height=780,
    )
    fig.update_traces(
        marker={"size": 8, "opacity": 0.82, "line": {"width": 0.35, "color": "#f7f7f4"}},
        selected={"marker": {"size": 12, "opacity": 1.0}},
    )
    fig.update_layout(
        clickmode="event+select",
        dragmode="pan",
        legend_title_text=color_label,
        paper_bgcolor="#fffdf8",
        plot_bgcolor="#fffdf8",
        margin={"l": 10, "r": 10, "t": 28, "b": 10},
    )
    return fig


def selected_row(selection: dict, df: pd.DataFrame) -> pd.Series | None:
    state = selection or {}
    if "selection" in state:
        points = state.get("selection", {}).get("points", [])
    else:
        points = state.get("points", [])
    if not points:
        return None
    row_id = points[0].get("customdata", [None])[0]
    if row_id is None:
        return None
    match = df.loc[df["row_id"] == row_id]
    if match.empty:
        return None
    return match.iloc[0]


st.title("Opacity Rationale UMAP Browser")
st.caption(
    "Pan, zoom, hover, and click a point to inspect the full rationale. "
    "Each mechanism is clustered independently."
)

artifacts_dir = st.text_input("Artifacts directory", value=str(DEFAULT_ARTIFACTS_DIR))
available_mechanisms = list_available_mechanisms(artifacts_dir)
if not available_mechanisms:
    st.error(f"No clustered rationale files found under {artifacts_dir}.")
    st.stop()

control_left, control_right = st.columns([1.45, 1], vertical_alignment="center")
with control_left:
    mechanism = st.segmented_control(
        "Opacity mechanism",
        available_mechanisms,
        default=available_mechanisms[0],
        format_func=mechanism_title,
        key="mechanism_selector",
        width="stretch",
    )
with control_right:
    color_label = st.segmented_control(
        "Color points by",
        list(COLOR_OPTIONS),
        default="Cluster",
        key="color_selector",
        width="stretch",
    )

df = load_mechanism_points(artifacts_dir, mechanism)
fig = build_figure(df, color_label=color_label)

main_col, side_col = st.columns([1.65, 0.95], gap="large", vertical_alignment="top")
with main_col:
    selection = st.plotly_chart(
        fig,
        width="stretch",
        theme=None,
        on_select="rerun",
        selection_mode=("points", "box", "lasso"),
        key=f"plot_{mechanism}_{COLOR_OPTIONS[color_label]}",
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "modeBarButtonsToRemove": ["autoScale2d"],
        },
    )

with side_col:
    row = selected_row(selection, df)
    if row is None:
        st.markdown("**Selected point**")
        st.write("Click a point in the scatter to inspect its rationale.")
    else:
        st.markdown(f"### `{row['transcript_id']}`")
        st.markdown(
            f"<span class='metric-chip'>Cluster {row['cluster_label']}</span>"
            f"<span class='metric-chip'>Level: {row['level']}</span>"
            f"<span class='metric-chip'>Form: {row['form']}</span>"
            f"<span class='metric-chip'>Type: {row['transcript_type']}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='detail-section-title'>Rationale</div>", unsafe_allow_html=True)
    if row is None:
        st.write("No point selected.")
    else:
        st.write(row["rationale"])

    st.markdown("<div class='detail-section-title'>Evidence</div>", unsafe_allow_html=True)
    if row is None:
        st.write("No point selected.")
    elif isinstance(row.get("evidence"), str) and row["evidence"].strip():
        st.write(row["evidence"])
    else:
        st.write("No evidence text for this point.")

    st.markdown("<div class='detail-section-title'>Summary</div>", unsafe_allow_html=True)
    if row is None:
        st.write("No point selected.")
    elif isinstance(row.get("summary"), str) and row["summary"].strip():
        st.write(row["summary"])
    else:
        st.write("No summary text for this point.")
