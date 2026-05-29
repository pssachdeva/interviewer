import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_lego.labels import bold_text

from interviewer.opacity_columns import MECHANISM_KEYS
from interviewer.plotting import (
    build_cluster_combo_prop_table,
    configure_plot_style,
    load_task_activity_clusters,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/task_activity_clustering"
OUTPUT_PATH = PROJECT_ROOT / "artifacts/tasks_opacity_mechanism_grid.pdf"
OUTPUT_PNG_PATH = PROJECT_ROOT / "artifacts/tasks_opacity_mechanism_grid.png"

FIGSIZE = (16, 6.5)
DPI = 300
NROWS = 2
TOP_N = 5
WRAP_WIDTH = 28
TARGET_COMBOS = ["clear production", "clear avoidance"]
COMBO_COLORS = {
    "clear production": "#009E73",
    "clear avoidance": "#0072B2",
}
MECHANISM_TITLES = {
    "voice_opacity": "Voice",
    "vulnerability_opacity": "Vulnerability",
    "provenance_opacity": "Provenance",
    "attention_opacity": "Attention",
    "investment_opacity": "Investment",
}
ROW_LABELS = {
    "clear production": "Clear\nProduction",
    "clear avoidance": "Clear\nAvoidance",
}
PANEL_EDGE_COLOR = "#dddddd"
PANEL_EDGE_WIDTH = 0.8
PANEL_LABEL_X = 0.05
PANEL_LABEL_Y0 = 0.93
PANEL_LABEL_STEP = 0.20
BAR_X = 0.64
BAR_WIDTH = 0.25
BAR_HEIGHT = 0.055
BAR_BACKGROUND_COLOR = "#eeeeee"
BAR_EDGE_COLOR = "#cccccc"
BAR_EDGE_WIDTH = 0.35
PERCENT_X_OFFSET = 0.018
BAR_Y_OFFSET = 0.075
TEXT_SIZE = 9
TITLE_SIZE = 15
ROW_LABEL_SIZE = 13
TITLE_PAD = 8
ROW_LABEL_X = -0.04
ROW_LABEL_Y = {
    "clear production": 0.73,
    "clear avoidance": 0.27,
}
PERCENT_COLOR = "#444444"
SAVE_PAD_INCHES = 0.02


def draw_ranked_list(
    ax: plt.Axes,
    table,
    *,
    combo: str,
    color: str,
) -> None:
    """Draw one ranked cluster list panel."""
    top = table[combo].sort_values(ascending=False)
    top = top[top > 0].head(TOP_N)
    ax.set_axis_off()
    ax.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            facecolor="none",
            edgecolor=PANEL_EDGE_COLOR,
            linewidth=PANEL_EDGE_WIDTH,
            clip_on=False,
        )
    )

    percent_x = BAR_X + BAR_WIDTH + PERCENT_X_OFFSET
    for item_idx, (label, value) in enumerate(top.items()):
        y = PANEL_LABEL_Y0 - item_idx * PANEL_LABEL_STEP
        wrapped = textwrap.fill(str(label), width=WRAP_WIDTH)
        ax.text(
            PANEL_LABEL_X,
            y,
            wrapped,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=TEXT_SIZE,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2},
            zorder=3,
        )

        bar_y = y - BAR_Y_OFFSET
        ax.add_patch(
            Rectangle(
                (BAR_X, bar_y),
                BAR_WIDTH,
                BAR_HEIGHT,
                transform=ax.transAxes,
                facecolor=BAR_BACKGROUND_COLOR,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                zorder=1,
            )
        )
        ax.add_patch(
            Rectangle(
                (BAR_X, bar_y),
                BAR_WIDTH * value,
                BAR_HEIGHT,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="none",
                zorder=2,
            )
        )
        ax.text(
            percent_x,
            bar_y + BAR_HEIGHT / 2,
            f"{value * 100:.0f}\\%",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=TEXT_SIZE,
            color=PERCENT_COLOR,
        )


def main() -> None:
    """Build and save the task activity mechanism grid."""
    configure_plot_style()

    clustered = load_task_activity_clusters(ARTIFACTS_DIR, MECHANISM_KEYS)
    prop_tables = {
        mechanism: build_cluster_combo_prop_table(frame, TARGET_COMBOS)
        for mechanism, frame in clustered.items()
    }

    fig, axes = plt.subplots(
        nrows=NROWS,
        ncols=len(MECHANISM_KEYS),
        figsize=FIGSIZE,
        dpi=DPI,
        constrained_layout=True,
    )

    for col, mechanism in enumerate(MECHANISM_KEYS):
        axes[0, col].set_title(
            bold_text(MECHANISM_TITLES[mechanism]),
            fontsize=TITLE_SIZE,
            pad=TITLE_PAD,
        )

    for row, combo in enumerate(TARGET_COMBOS):
        for col, mechanism in enumerate(MECHANISM_KEYS):
            draw_ranked_list(
                axes[row, col],
                prop_tables[mechanism],
                combo=combo,
                color=COMBO_COLORS[combo],
            )

    for combo, label in ROW_LABELS.items():
        fig.text(
            ROW_LABEL_X,
            ROW_LABEL_Y[combo],
            bold_text(label),
            ha="center",
            va="center",
            fontsize=ROW_LABEL_SIZE,
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    fig.savefig(OUTPUT_PNG_PATH, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)


if __name__ == "__main__":
    main()
