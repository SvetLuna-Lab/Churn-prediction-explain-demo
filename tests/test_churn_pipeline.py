import os
import unittest

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data_utils import PROJECT_ROOT, load_churn_data
from explain import compute_permutation_importance


class TestChurnPipeline(unittest.TestCase):
    def test_load_churn_data_shapes(self):
        X, y, feature_names = load_churn_data()
        # At least one sample and one feature
        self.assertGreater(X.shape[0], 0)
        self.assertGreater(X.shape[1], 0)
        self.assertEqual(X.shape[1], len(feature_names))
        self.assertEqual(X.shape[0], y.shape[0])

    def test_model_beats_baseline_or_equal(self):
        X, y, _ = load_churn_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
            stratify=y,
        )

        model = LogisticRegression(solver="liblinear", random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        majority_class = int(y_train.mean() >= 0.5)
        y_baseline = [majority_class] * len(y_test)
        baseline_acc = accuracy_score(y_test, y_baseline)

        # For this small synthetic dataset, we expect the model to be
        # at least not worse than the majority baseline.
        self.assertGreaterEqual(acc, baseline_acc)

    def test_permutation_importance_length(self):
        X, y, feature_names = load_churn_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
            stratify=y,
        )

        model = LogisticRegression(solver="liblinear", random_state=42)
        model.fit(X_train, y_train)

        importances = compute_permutation_importance(
            model,
            X_test,
            y_test,
            feature_names=feature_names,
            n_repeats=5,
            random_state=42,
        )

        # We should get one importance entry per feature.
        self.assertEqual(len(importances), len(feature_names))


if __name__ == "__main__":
    unittest.main()
