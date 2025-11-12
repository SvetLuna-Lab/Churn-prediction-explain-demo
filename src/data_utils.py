import csv
import os
from typing import List, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "churn_samples.csv")


def load_churn_data(path: str = DATA_PATH) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load churn samples from CSV into feature matrix X, target vector y,
    and a list of feature names.

    Features:
        tenure_months, monthly_charges, total_charges,
        contract_type, num_support_calls, has_promo

    Target:
        churn (0 or 1)
    """
    X: List[List[float]] = []
    y: List[int] = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        feature_names = [
            "tenure_months",
            "monthly_charges",
            "total_charges",
            "contract_type",
            "num_support_calls",
            "has_promo",
        ]
        for row in reader:
            features = [
                float(row["tenure_months"]),
                float(row["monthly_charges"]),
                float(row["total_charges"]),
                float(row["contract_type"]),
                float(row["num_support_calls"]),
                float(row["has_promo"]),
            ]
            churn = int(row["churn"])

            X.append(features)
            y.append(churn)

    return np.array(X, dtype=float), np.array(y, dtype=int), feature_names
