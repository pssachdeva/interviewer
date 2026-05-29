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
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "outputs/runs/exp0.3/results.csv"
OUTPUT_PATH = PROJECT_ROOT / "artifacts/all_interview_opacity.pdf"
OUTPUT_PNG_PATH = PROJECT_ROOT / "artifacts/all_interview_opacity.png"

FIGSIZE = (8, 4)
DPI = 300
LEVEL_ORDER = ["none", "potential", "clear"]
RANDOM_SEED = 7
N_BOOTSTRAP = 1000
CI_LOWER = 2.5
CI_UPPER = 97.5
BAR_WIDTH = 0.22
ERROR_CAPSIZE = 3
ERROR_COLOR = "black"
XLABEL = "Mechanism"
YLABEL = "Fraction of samples"
AXIS_LABEL_SIZE = 18
TICK_LABEL_SIZE = 16
Y_LIMITS = (0, 1)
AXIS_FACE_COLOR = "whitesmoke"
GRID_ALPHA = 0.7
GRID_LINESTYLE = "--"
SAVE_PAD_INCHES = 0.02


def main() -> None:
    """Build and save the all-interviews opacity level plot."""
    configure_plot_style()

    results = pd.read_csv(RESULTS_PATH)
    mechanisms = available_mechanisms(results)
    level_fractions = build_level_fraction_table(
        results,
        mechanisms,
        LEVEL_ORDER,
        rng=np.random.default_rng(RANDOM_SEED),
        n_bootstrap=N_BOOTSTRAP,
        ci_lower=CI_LOWER,
        ci_upper=CI_UPPER,
    )

    palette = get_default_ccycle()
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    x_positions = np.arange(len(mechanisms))

    for idx, level in enumerate(LEVEL_ORDER):
        subset = (
            level_fractions[level_fractions["level"] == level]
            .set_index("mechanism")
            .reindex(mechanisms)
            .fillna({"fraction": 0.0, "ci_lower": 0.0, "ci_upper": 0.0})
        )
        offsets = x_positions + (idx - 1) * BAR_WIDTH
        heights = subset["fraction"].to_numpy()
        yerr = np.vstack(
            [
                heights - subset["ci_lower"].to_numpy(),
                subset["ci_upper"].to_numpy() - heights,
            ]
        )
        ax.bar(
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

    ax.set_xticks(x_positions)
    ax.set_xticklabels([mechanism_label(mechanism) for mechanism in mechanisms])
    ax.set_xlabel(bold_text(XLABEL), fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(bold_text(YLABEL), fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.set_ylim(*Y_LIMITS)
    ax.legend(title=bold_text("Level"))
    ax.set_facecolor(AXIS_FACE_COLOR)
    ax.grid(axis="y", linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)
    ax.set_axisbelow(True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    fig.savefig(OUTPUT_PNG_PATH, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)


if __name__ == "__main__":
    main()
