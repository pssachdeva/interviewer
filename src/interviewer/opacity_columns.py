"""Shared constants for opacity mechanisms and exported CSV columns."""

from __future__ import annotations


MECHANISM_KEYS = [
    "voice_opacity",
    "vulnerability_opacity",
    "provenance_opacity",
    "attention_opacity",
    "investment_opacity",
]

LEVELS = {"none", "potential", "clear"}
FORMS = {"production", "avoidance", "mixed", "none"}

def level_column(mechanism: str) -> str:
    return f"{mechanism}_level"


def form_column(mechanism: str) -> str:
    return f"{mechanism}_form"


def rationale_column(mechanism: str) -> str:
    return f"{mechanism}_rationale"


def evidence_column(mechanism: str) -> str:
    return f"{mechanism}_evidence"


def present_column(mechanism: str) -> str:
    return f"{mechanism}_present"


def mechanism_columns(mechanism: str) -> list[str]:
    return [
        level_column(mechanism),
        form_column(mechanism),
        rationale_column(mechanism),
        evidence_column(mechanism),
    ]


OPACITY_RESULTS_COLUMNS = [
    column
    for mechanism in MECHANISM_KEYS
    for column in mechanism_columns(mechanism)
]
