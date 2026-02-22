"""Data loading utilities for the Anthropic Interviewer dataset."""

from datasets import load_dataset
import pandas as pd


DATASET_NAME = "Anthropic/AnthropicInterviewer"


def load_interviews(split: str | None = None) -> pd.DataFrame:
    """Load the Anthropic Interviewer dataset.

    Args:
        split: Optional split to load ('workforce', 'creatives', 'scientists').
               If None, loads all splits combined.

    Returns:
        DataFrame with columns: transcript_id, text, split
    """
    if split:
        ds = load_dataset(DATASET_NAME, split=split)
        return ds.to_pandas()

    # Load all splits and combine
    dfs = []
    for s in ["workforce", "creatives", "scientists"]:
        ds = load_dataset(DATASET_NAME, split=s)
        df = ds.to_pandas()
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def get_split_counts() -> dict[str, int]:
    """Get the number of interviews in each split."""
    return {
        "workforce": 1000,
        "creatives": 125,
        "scientists": 125,
    }
