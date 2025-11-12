import json
import os
from typing import Any, Dict, List

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data_utils import PROJECT_ROOT, load_churn_data
from explain import compute_permutation_importance

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "churn_model.joblib")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "churn_experiment_results.json")


def main() -> None:
    # Load data
    X, y, feature_names = load_churn_data()

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    # Train a simple logistic regression model
    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model accuracy on test set
    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))

    # Baseline: predict the majority class from the training set
    majority_class = int(y_train.mean() >= 0.5)
    y_baseline = [majority_class] * len(y_test)
    baseline_accuracy = float(accuracy_score(y_test, y_baseline))

    # Compute permutation feature importance on the test set
    feature_importances = compute_permutation_importance(
        model,
        X_test,
        y_test,
        feature_names=feature_names,
    )

    # Persist model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # Build a result payload
    results: Dict[str, Any] = {
        "metrics": {
            "accuracy": accuracy,
            "baseline_accuracy": baseline_accuracy,
        },
        "majority_class": majority_class,
        "top_features": feature_importances,
        "feature_names": feature_names,
        "model_path": MODEL_PATH,
    }

    # Print short summary
    print("=== Churn prediction experiment summary ===")
    print(f"Accuracy          : {accuracy:.3f}")
    print(f"Baseline accuracy : {baseline_accuracy:.3f}")
    print("\nTop features (by permutation importance):")
    for item in feature_importances:
        mean_imp = item["importance_mean"]
        print(f"  {item['feature']}: {mean_imp:.4f}")

    # Save detailed JSON
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDetailed results saved to: {RESULTS_PATH}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
