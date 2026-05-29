from pathlib import Path

import numpy as np
import pandas as pd
import plot_all_interview_opacity as all_interview_opacity
import plot_interview_mechanism_by_type as interview_mechanism_by_type
import plot_opacity_form as opacity_form

from interviewer.opacity_columns import form_column, level_column
from interviewer.plotting import (
    available_mechanisms,
    build_form_fraction_table,
    build_level_fraction_table,
    mechanism_label,
    subset_masks,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "outputs/runs/exp0.3/results.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ALL_INTERVIEW_OUTPUT_PATH = ARTIFACTS_DIR / "all_interview_opacity_values.md"
INTERVIEW_TYPE_OUTPUT_PATH = ARTIFACTS_DIR / "interview_mechanism_by_type_values.md"
OPACITY_FORM_OUTPUT_PATH = ARTIFACTS_DIR / "opacity_form_values.md"

FRACTION_DECIMALS = 6
PERCENT_DECIMALS = 1


def _format_fraction(value: float) -> str:
    return f"{value:.{FRACTION_DECIMALS}f}"


def _format_percent(value: float) -> str:
    return f"{value * 100:.{PERCENT_DECIMALS}f}%"


def _escape_markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def _to_markdown(frame: pd.DataFrame) -> str:
    headers = [_escape_markdown_cell(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False):
        cells = [_escape_markdown_cell(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_markdown_table(path: Path, title: str, frame: pd.DataFrame) -> None:
    path.write_text(f"# {title}\n\n{_to_markdown(frame)}\n", encoding="utf-8")


def _level_counts(
    results: pd.DataFrame,
    mechanisms: list[str],
    levels: list[str],
) -> dict[tuple[str, str], tuple[int, int]]:
    counts = {}
    for mechanism in mechanisms:
        values = results[level_column(mechanism)].dropna()
        denominator = len(values)
        value_counts = values.value_counts()
        for level in levels:
            counts[(mechanism, level)] = (int(value_counts.get(level, 0)), denominator)
    return counts


def _form_counts(
    results: pd.DataFrame,
    mechanisms: list[str],
    forms: list[str],
) -> dict[tuple[str, str], tuple[int, int]]:
    counts = {}
    for mechanism in mechanisms:
        level_mask = results[level_column(mechanism)].ne("none")
        values = (
            results.loc[level_mask, form_column(mechanism)]
            .dropna()
            .loc[lambda series: series.ne("none")]
        )
        denominator = len(values)
        value_counts = values.value_counts()
        for form in forms:
            counts[(mechanism, form)] = (int(value_counts.get(form, 0)), denominator)
    return counts


def _level_export_rows(
    fractions: pd.DataFrame,
    counts: dict[tuple[str, str], tuple[int, int]],
    mechanisms: list[str],
    levels: list[str],
    *,
    panel: str | None = None,
) -> list[dict[str, object]]:
    rows = []
    indexed = fractions.set_index(["mechanism", "level"])
    for mechanism in mechanisms:
        for level in levels:
            row = indexed.loc[(mechanism, level)]
            count, denominator = counts[(mechanism, level)]
            export_row = {
                "Mechanism": mechanism_label(mechanism),
                "Level": level,
                "Fraction": _format_fraction(row["fraction"]),
                "Percent": _format_percent(row["fraction"]),
                "CI lower": _format_fraction(row["ci_lower"]),
                "CI upper": _format_fraction(row["ci_upper"]),
                "Count": count,
                "Denominator": denominator,
            }
            if panel is not None:
                export_row = {"Interview type": panel, **export_row}
            rows.append(export_row)
    return rows


def _form_export_rows(
    fractions: pd.DataFrame,
    counts: dict[tuple[str, str], tuple[int, int]],
    mechanisms: list[str],
    forms: list[str],
    *,
    panel: str,
) -> list[dict[str, object]]:
    rows = []
    indexed = fractions.set_index(["mechanism", "form"])
    for mechanism in mechanisms:
        for form in forms:
            row = indexed.loc[(mechanism, form)]
            count, denominator = counts[(mechanism, form)]
            rows.append(
                {
                    "Interview type": panel,
                    "Mechanism": mechanism_label(mechanism),
                    "Form": form,
                    "Fraction": _format_fraction(row["fraction"]),
                    "Percent": _format_percent(row["fraction"]),
                    "CI lower": _format_fraction(row["ci_lower"]),
                    "CI upper": _format_fraction(row["ci_upper"]),
                    "Count": count,
                    "Denominator": denominator,
                }
            )
    return rows


def build_all_interview_opacity_values(results: pd.DataFrame) -> pd.DataFrame:
    """Build the table of bar values for the all-interviews opacity figure."""
    mechanisms = available_mechanisms(results)
    levels = all_interview_opacity.LEVEL_ORDER
    fractions = build_level_fraction_table(
        results,
        mechanisms,
        levels,
        rng=np.random.default_rng(all_interview_opacity.RANDOM_SEED),
        n_bootstrap=all_interview_opacity.N_BOOTSTRAP,
        ci_lower=all_interview_opacity.CI_LOWER,
        ci_upper=all_interview_opacity.CI_UPPER,
    )
    counts = _level_counts(results, mechanisms, levels)
    rows = _level_export_rows(fractions, counts, mechanisms, levels)
    return pd.DataFrame(rows)


def build_interview_type_opacity_values(results: pd.DataFrame) -> pd.DataFrame:
    """Build the table of bar values for the interview-type opacity figure."""
    mechanisms = available_mechanisms(results)
    levels = interview_mechanism_by_type.LEVEL_ORDER
    rows = []
    for panel_idx, (panel, mask) in enumerate(subset_masks(results).items()):
        subset = results.loc[mask].copy()
        fractions = build_level_fraction_table(
            subset,
            mechanisms,
            levels,
            rng=np.random.default_rng(
                interview_mechanism_by_type.RANDOM_SEED + panel_idx
            ),
            n_bootstrap=interview_mechanism_by_type.N_BOOTSTRAP,
            ci_lower=interview_mechanism_by_type.CI_LOWER,
            ci_upper=interview_mechanism_by_type.CI_UPPER,
        )
        rows.extend(
            _level_export_rows(
                fractions,
                _level_counts(subset, mechanisms, levels),
                mechanisms,
                levels,
                panel=panel,
            )
        )
    return pd.DataFrame(rows)


def build_opacity_form_values(results: pd.DataFrame) -> pd.DataFrame:
    """Build the table of bar values for the opacity-form figure."""
    mechanisms = available_mechanisms(results)
    forms = opacity_form.FORM_ORDER
    rows = []
    for panel_idx, (panel, mask) in enumerate(subset_masks(results).items()):
        subset = results.loc[mask].copy()
        fractions = build_form_fraction_table(
            subset,
            mechanisms,
            forms,
            rng=np.random.default_rng(opacity_form.RANDOM_SEED + panel_idx),
            n_bootstrap=opacity_form.N_BOOTSTRAP,
            ci_lower=opacity_form.CI_LOWER,
            ci_upper=opacity_form.CI_UPPER,
        )
        rows.extend(
            _form_export_rows(
                fractions,
                _form_counts(subset, mechanisms, forms),
                mechanisms,
                forms,
                panel=panel,
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Save Markdown tables for each non-grid bar figure."""
    results = pd.read_csv(RESULTS_PATH)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    _write_markdown_table(
        ALL_INTERVIEW_OUTPUT_PATH,
        "All interview opacity bar values",
        build_all_interview_opacity_values(results),
    )
    _write_markdown_table(
        INTERVIEW_TYPE_OUTPUT_PATH,
        "Interview mechanism by type bar values",
        build_interview_type_opacity_values(results),
    )
    _write_markdown_table(
        OPACITY_FORM_OUTPUT_PATH,
        "Opacity form bar values",
        build_opacity_form_values(results),
    )


if __name__ == "__main__":
    main()
