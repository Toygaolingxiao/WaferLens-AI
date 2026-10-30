"""L4 SPC computations: p-chart limits and EWMA series.

Pure computations here; plotting lives in the analysis notebooks/scripts so
the numbers stay unit-testable.
"""

from __future__ import annotations

import numpy as np


def p_chart_limits(defect_counts: np.ndarray, sample_sizes: np.ndarray) -> dict:
    """Centre line and 3-sigma control limits for a p-chart (varying n)."""
    counts = np.asarray(defect_counts, dtype=float)
    sizes = np.asarray(sample_sizes, dtype=float)
    p_bar = counts.sum() / sizes.sum()
    sigma = np.sqrt(p_bar * (1.0 - p_bar) / sizes)
    return {
        "center": float(p_bar),
        "ucl": p_bar + 3.0 * sigma,
        "lcl": np.clip(p_bar - 3.0 * sigma, 0.0, None),
        "proportions": counts / sizes,
    }


def ewma(values: np.ndarray, lam: float = 0.2) -> np.ndarray:
    """Exponentially weighted moving average series."""
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = lam * values[i] + (1.0 - lam) * out[i - 1]
    return out


# TODO(owner): Cp/Cpk demonstration on a continuous process variable, and the
# side-by-side "when p-chart, when EWMA" note for the design log.
