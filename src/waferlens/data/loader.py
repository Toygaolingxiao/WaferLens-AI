"""Load and normalise WM-811K (LSWMD.pkl).

The raw release stores ``failureType`` as a one-element array (or an empty
array for unlabelled rows). Everything downstream expects a plain string
label or ``None``, so normalisation happens here, once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RAW_PATH = Path("data/raw/LSWMD.pkl")

DEFECT_PATTERNS = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Random",
    "Scratch",
    "Near-Full",
]
ALL_LABELS = ["none", *DEFECT_PATTERNS]


def _normalise_label(value: object) -> str | None:
    """Collapse the raw failureType cell to a plain label or None."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return None
        value = value[0]
    label = str(value).strip()
    return label if label in ALL_LABELS else None


def load_lswmd(path: str | Path = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """Read LSWMD.pkl and add a clean ``label`` column (None when unlabelled)."""
    df = pd.read_pickle(path)
    df = df.rename(columns={"failureType": "failure_type_raw"})
    df["label"] = df["failure_type_raw"].map(_normalise_label)
    return df


def labelled_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that carry a defect-pattern label — the supervised dataset."""
    return df[df["label"].notna()].reset_index(drop=True)


def class_counts(df: pd.DataFrame) -> pd.Series:
    """Label distribution of the labelled subset, in canonical label order."""
    counts = labelled_subset(df)["label"].value_counts()
    return counts.reindex(ALL_LABELS).fillna(0).astype(int)


def stage1_label(label: str) -> str:
    """Binary stage-1 target: defective vs none."""
    return "none" if label == "none" else "defective"
