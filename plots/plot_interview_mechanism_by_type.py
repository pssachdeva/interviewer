from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_lego.colors import get_default_ccycle
from mpl_lego.labels import bold_text

from interviewer.plotting import (
    available_mechanisms,
    build_level_fraction_table,
    configure_plot_style,
    mechanism_label,
    subset_masks,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "outputs/runs/exp0.3/results.csv"
OUTPUT_PATH = PROJECT_ROOT / "artifacts/interview_mechanism_by_type.pdf"
OUTPUT_PNG_PATH = PROJECT_ROOT / "artifacts/interview_mechanism_by_type.png"

FIGSIZE = (20, 5)
DPI = 300
NROWS = 1
NCOLS = 3
LEVEL_ORDER = ["none", "potential", "clear"]
RANDOM_SEED = 7
N_BOOTSTRAP = 1000
CI_LOWER = 2.5
CI_UPPER = 97.5
BAR_WIDTH = 0.22
ERROR_CAPSIZE = 3
ERROR_COLOR = "black"
TITLE_SIZE = 18
XLABEL = "Mechanism"
YLABEL = "Fraction of Interviews"
AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 14
YLABEL_PAD = 10
Y_LIMITS = (0, 1.05)
AXIS_FACE_COLOR = "whitesmoke"
GRID_ALPHA = 0.7
GRID_LINESTYLE = "--"
LEGEND_COLUMNS = 1
LEGEND_LOCATION = "center left"
LEGEND_ANCHOR = (0.915, 0.5)
LEGEND_FRAME = True
LEGEND_LABEL_SIZE = 14
LEGEND_TITLE_SIZE = 16
TIGHT_LAYOUT_RECT = (0.04, 0, 0.92, 1)
SAVE_PAD_INCHES = 0.02


def main() -> None:
    """Build and save the opacity level plot faceted by interview type."""
    configure_plot_style()

    results = pd.read_csv(RESULTS_PATH)
    mechanisms = available_mechanisms(results)
    palette = get_default_ccycle()

    fig, axes = plt.subplots(
        NROWS,
        NCOLS,
        figsize=FIGSIZE,
        dpi=DPI,
        sharey=True,
    )
    legend_handles = []
    legend_labels = []

    for panel_idx, (panel_title, mask) in enumerate(subset_masks(results).items()):
        subset = results.loc[mask].copy()
        level_fractions = build_level_fraction_table(
            subset,
            mechanisms,
            LEVEL_ORDER,
            rng=np.random.default_rng(RANDOM_SEED + panel_idx),
            n_bootstrap=N_BOOTSTRAP,
            ci_lower=CI_LOWER,
            ci_upper=CI_UPPER,
        )

        ax = axes[panel_idx]
        x_positions = np.arange(len(mechanisms))
        for idx, level in enumerate(LEVEL_ORDER):
            chart_data = (
                level_fractions[level_fractions["level"] == level]
                .set_index("mechanism")
                .reindex(mechanisms)
                .fillna({"fraction": 0.0, "ci_lower": 0.0, "ci_upper": 0.0})
            )
            offsets = x_positions + (idx - 1) * BAR_WIDTH
            heights = chart_data["fraction"].to_numpy()
            yerr = np.vstack(
                [
                    heights - chart_data["ci_lower"].to_numpy(),
                    chart_data["ci_upper"].to_numpy() - heights,
                ]
            )
            bars = ax.bar(
                offsets,
                heights,
                width=BAR_WIDTH,
                label=level,
                color=palette[idx % len(palette)],
                yerr=yerr,
                capsize=ERROR_CAPSIZE,
                ecolor=ERROR_COLOR,
                linewidth=0,
            )
            if panel_idx == 0:
                legend_handles.append(bars)
                legend_labels.append(level)

        ax.set_xticks(x_positions)
        ax.set_xticklabels([mechanism_label(mechanism) for mechanism in mechanisms])
        ax.set_xlabel(bold_text(XLABEL), fontsize=AXIS_LABEL_SIZE)
        if panel_idx == 0:
            ax.set_ylabel(
                bold_text(YLABEL),
                fontsize=AXIS_LABEL_SIZE,
                labelpad=YLABEL_PAD,
            )
        ax.set_title(bold_text(panel_title), fontsize=TITLE_SIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
        ax.set_ylim(*Y_LIMITS)
        ax.set_facecolor(AXIS_FACE_COLOR)
        ax.grid(axis="y", linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)
        ax.set_axisbelow(True)

    fig.legend(
        legend_handles,
        legend_labels,
        loc=LEGEND_LOCATION,
        ncol=LEGEND_COLUMNS,
        frameon=LEGEND_FRAME,
        bbox_to_anchor=LEGEND_ANCHOR,
        title=bold_text("Level"),
        prop={"size": LEGEND_LABEL_SIZE},
        title_fontsize=LEGEND_TITLE_SIZE,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=TIGHT_LAYOUT_RECT)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    fig.savefig(OUTPUT_PNG_PATH, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)


if __name__ == "__main__":
    main()
