"""Lot-grouped train/test splitting.

Wafers from the same lot share process conditions, so a row-wise random split
leaks lot identity into both sides and inflates scores. All splits in this
project are grouped by ``lotName`` — this module is the only way to make one.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def grouped_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    group_col: str = "lotName",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into (train, test) with no lot appearing in both."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    overlap = set(train[group_col]) & set(test[group_col])
    assert not overlap, f"lot leakage across split: {sorted(overlap)[:5]}"
    return train, test
