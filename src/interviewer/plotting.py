"""Shared helpers for project figure scripts."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from mpl_lego.style import use_latex_style

from interviewer.opacity_columns import MECHANISM_KEYS, form_column, level_column

OKABE_ITO_COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
]


def configure_plot_style() -> None:
    """Apply the shared matplotlib style for static project figures."""
    use_latex_style()
    plt.rcParams["axes.prop_cycle"] = cycler(color=OKABE_ITO_COLORS)


def mechanism_label(mechanism: str) -> str:
    """Convert a mechanism key into a human-readable label."""
    return mechanism.replace("_opacity", "").replace("_", " ").title()


def available_mechanisms(results: pd.DataFrame) -> list[str]:
    """Return opacity mechanisms represented in a results dataframe."""
    return [
        mechanism
        for mechanism in MECHANISM_KEYS
        if level_column(mechanism) in results
    ]


def bootstrap_category_fractions(
    values: np.ndarray,
    categories: list[str],
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    ci_lower: float,
    ci_upper: float,
) -> pd.DataFrame:
    """Estimate category fractions and bootstrap confidence intervals."""
    n_values = len(values)
    bootstrap_fractions = {category: [] for category in categories}

    if n_values:
        for _ in range(n_bootstrap):
            sample = values[rng.integers(0, n_values, n_values)]
            for category in categories:
                bootstrap_fractions[category].append(np.mean(sample == category))

    observed = (
        pd.Series(values).value_counts(normalize=True)
        if n_values
        else pd.Series(dtype=float)
    )

    rows = []
    for category in categories:
        fraction = float(observed.get(category, 0.0))
        if bootstrap_fractions[category]:
            lower = float(np.percentile(bootstrap_fractions[category], ci_lower))
            upper = float(np.percentile(bootstrap_fractions[category], ci_upper))
        else:
            lower = 0.0
            upper = 0.0
        rows.append(
            {
                "category": category,
                "fraction": fraction,
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )

    return pd.DataFrame(rows)


def build_level_fraction_table(
    results: pd.DataFrame,
    mechanisms: list[str],
    level_order: list[str],
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    ci_lower: float,
    ci_upper: float,
) -> pd.DataFrame:
    """Build level fractions by opacity mechanism."""
    frames = []
    for mechanism in mechanisms:
        values = results[level_column(mechanism)].dropna().to_numpy()
        fractions = bootstrap_category_fractions(
            values,
            level_order,
            rng=rng,
            n_bootstrap=n_bootstrap,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )
        fractions["mechanism"] = mechanism
        frames.append(fractions.rename(columns={"category": "level"}))

    return pd.concat(frames, ignore_index=True)


def build_form_fraction_table(
    results: pd.DataFrame,
    mechanisms: list[str],
    form_order: list[str],
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    ci_lower: float,
    ci_upper: float,
) -> pd.DataFrame:
    """Build form fractions by mechanism among cases with mechanism evidence."""
    frames = []
    for mechanism in mechanisms:
        level_col = level_column(mechanism)
        mechanism_form_col = form_column(mechanism)
        conditioned = (
            results.loc[results[level_col].ne("none"), mechanism_form_col]
            .dropna()
            .to_numpy()
        )
        conditioned = conditioned[conditioned != "none"]
        fractions = bootstrap_category_fractions(
            conditioned,
            form_order,
            rng=rng,
            n_bootstrap=n_bootstrap,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )
        fractions["mechanism"] = mechanism
        frames.append(fractions.rename(columns={"category": "form"}))

    return pd.concat(frames, ignore_index=True)


def subset_masks(results: pd.DataFrame) -> dict[str, pd.Series]:
    """Return the interview-type masks used by the notebook figures."""
    transcript_ids = results["transcript_id"].astype(str)
    return {
        "Work Interviews": transcript_ids.str.startswith("work_"),
        "Creativity Interviews": transcript_ids.str.startswith("creativity_"),
        "Science Interviews": transcript_ids.str.startswith("science_"),
    }


def load_task_activity_clusters(
    artifacts_dir: Path,
    mechanisms: list[str],
) -> dict[str, pd.DataFrame]:
    """Load clustered task activity rows joined to cluster labels."""
    activity_frames = []
    for mechanism in mechanisms:
        activity_frames.append(
            pd.read_csv(artifacts_dir / mechanism / "clustered_activities.csv")
        )

    activities = pd.concat(activity_frames, ignore_index=True)
    labels = pd.read_csv(artifacts_dir / "cluster_labels.csv")
    clustered = activities.merge(labels, on=["mechanism", "cluster_id"], how="left")
    return {
        mechanism: clustered[clustered["mechanism"] == mechanism].copy()
        for mechanism in mechanisms
    }


def build_cluster_combo_prop_table(
    frame: pd.DataFrame,
    target_combos: list[str],
) -> pd.DataFrame:
    """Build within-cluster proportions for selected level/form combinations."""
    return (
        frame.assign(combo=lambda df: df["level"] + " " + df["form"])
        .groupby(["cluster_label_phrase", "combo"])
        .size()
        .unstack(fill_value=0)
        .pipe(lambda df: df.div(df.sum(axis=1), axis=0))
        .reindex(columns=target_combos, fill_value=0)
    )
