"""Hand-crafted wafer map features for route A.

Wafer maps vary in size, so spatial features are computed on a fixed grid:
the map is block-pooled to ``GRID x GRID`` cells and each cell contributes its
defect density. On top of the grid, a few scalar descriptors summarise the
radial profile and edge behaviour — the quantities an engineer reads first.

Die states: 0 = no die, 1 = good die, 2 = defective die.
"""

from __future__ import annotations

import numpy as np

GRID = 8


def _defect_mask(wafer_map: np.ndarray) -> np.ndarray:
    return (wafer_map == 2).astype(float)


def _die_mask(wafer_map: np.ndarray) -> np.ndarray:
    return (wafer_map > 0).astype(float)


def block_pool(values: np.ndarray, grid: int = GRID) -> np.ndarray:
    """Average-pool a 2D array to (grid, grid), padding edges as needed."""
    h, w = values.shape
    pad_h = (-h) % grid
    pad_w = (-w) % grid
    padded = np.pad(values, ((0, pad_h), (0, pad_w)))
    ph, pw = padded.shape[0] // grid, padded.shape[1] // grid
    return padded.reshape(grid, ph, grid, pw).mean(axis=(1, 3))


def density_grid_features(wafer_map: np.ndarray) -> np.ndarray:
    """Defect density per grid cell (defective dies / dies present)."""
    defects = block_pool(_defect_mask(wafer_map))
    dies = block_pool(_die_mask(wafer_map))
    return np.divide(defects, dies, out=np.zeros_like(defects), where=dies > 0).ravel()


def scalar_features(wafer_map: np.ndarray) -> dict[str, float]:
    """Radial / edge summary statistics of the defect distribution."""
    defects = _defect_mask(wafer_map)
    dies = _die_mask(wafer_map)
    total_defects = defects.sum()
    total_dies = dies.sum()
    h, w = wafer_map.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.nonzero(defects)
    if total_defects == 0:
        return {
            "defect_rate": 0.0,
            "mean_radius_norm": 0.0,
            "edge_share": 0.0,
            "centre_share": 0.0,
        }
    radii = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_radius = np.sqrt(cy**2 + cx**2) or 1.0
    radii_norm = radii / max_radius
    return {
        "defect_rate": float(total_defects / total_dies) if total_dies else 0.0,
        "mean_radius_norm": float(radii_norm.mean()),
        "edge_share": float((radii_norm > 0.85).mean()),
        "centre_share": float((radii_norm < 0.3).mean()),
    }


def extract_features(wafer_map: np.ndarray) -> np.ndarray:
    """Full route-A feature vector: density grid + scalar summaries."""
    scalars = scalar_features(wafer_map)
    return np.concatenate([density_grid_features(wafer_map), list(scalars.values())])


# TODO(owner): defect-cluster features — e.g. connected-component count and
# largest-cluster share of defects. Expected to separate Scratch / Loc from
# Random; add once route A's first numbers are in, and record the delta.
