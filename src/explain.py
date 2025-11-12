from typing import Dict, List

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.base import ClassifierMixin


def compute_permutation_importance(
    model: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
    random_state: int = 42,
) -> List[Dict[str, float]]:
    """
    Compute permutation feature importance on the provided data.

    Returns a list of dicts with:
      - feature: feature name
      - importance_mean: mean decrease in score
      - importance_std: std of the decrease
    sorted by importance_mean in descending order.
    """
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=None,
    )

    importances: List[Dict[str, float]] = []
    for i, name in enumerate(feature_names):
        importances.append(
            {
                "feature": name,
                "importance_mean": float(result.importances_mean[i]),
                "importance_std": float(result.importances_std[i]),
            }
        )

    # Sort by absolute importance (descending)
    importances.sort(key=lambda d: abs(d["importance_mean"]), reverse=True)
    return importances
