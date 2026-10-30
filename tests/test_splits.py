"""Split hygiene: no lot may appear on both sides of a grouped split."""

import numpy as np
import pandas as pd

from waferlens.data.loader import _normalise_label, stage1_label
from waferlens.data.splits import grouped_split


def _toy_frame() -> pd.DataFrame:
    rows = []
    for lot in range(20):
        for wafer in range(5):
            rows.append({"lotName": f"LOT{lot}", "waferIndex": wafer, "label": "Center"})
    return pd.DataFrame(rows)


def test_grouped_split_has_no_lot_overlap():
    train, test = grouped_split(_toy_frame(), test_size=0.25, seed=0)
    assert not set(train["lotName"]) & set(test["lotName"])
    assert len(train) + len(test) == 100


def test_label_normalisation():
    assert _normalise_label(np.array(["Edge-Ring"])) == "Edge-Ring"
    assert _normalise_label(np.array([])) is None
    assert _normalise_label("none") == "none"
    assert _normalise_label("bogus") is None


def test_stage1_label():
    assert stage1_label("none") == "none"
    assert stage1_label("Scratch") == "defective"
