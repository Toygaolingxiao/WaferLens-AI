"""Route A baseline: feature vectors + RandomForest.

Deliberately plain — this is the number every fancier route must beat, and
the imbalance strategy (class_weight vs resampling) is compared on top of it
rather than assumed.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    class_weight: str | None = "balanced",
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return model
