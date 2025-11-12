# Churn-prediction-explain-demo

Small demo project for **churn prediction and feature importance**.

The goal of this repository is to show how to:

- train a simple churn prediction model on a tiny tabular dataset,
- evaluate model accuracy against a majority-class baseline,
- compute **permutation feature importance** to understand which features drive churn,
- save both the model and an experiment report as artifacts.

This is a lightweight example of classic ML + basic explainability.

---

## Repository structure

```text
churn-prediction-explain-demo/
├─ data/
│  └─ churn_samples.csv          # small synthetic churn dataset
├─ models/
│  └─ churn_model.joblib         # created after running run_churn_experiment.py
├─ src/
│  ├─ __init__.py
│  ├─ data_utils.py              # load features/labels from CSV
│  ├─ explain.py                 # permutation feature importance
│  └─ run_churn_experiment.py    # train model, compute metrics & importance, save report
├─ tests/
│  ├─ __init__.py
│  └─ test_churn_pipeline.py     # sanity tests for data/model/explanation
├─ README.md
├─ requirements.txt
└─ .gitignore



Requirements

Python 3.10+

Dependencies from requirements.txt:

scikit-learn
joblib
numpy


Install them (ideally in a virtual environment):

pip install -r requirements.txt



Data

The data/churn_samples.csv file contains a tiny synthetic dataset with the following columns:

tenure_months – number of months the customer has been with the company

monthly_charges – monthly payment amount

total_charges – total amount paid over the relationship

contract_type – 0 = month-to-month, 1 = long-term contract

num_support_calls – number of calls to customer support

has_promo – 0/1 flag indicating if a promotion is active

churn – target label: 0 = stayed, 1 = churned

This dataset is only for demonstration and is not based on real customers.


Experiment script

The main script is src/run_churn_experiment.py. It:

Loads the churn data from data/churn_samples.csv.

Splits the data into train and test sets.

Trains a LogisticRegression classifier (scikit-learn).

Evaluates:

model accuracy on the test set,

a majority-class baseline accuracy (always predicting the most frequent class).

Computes permutation feature importance on the test set.

Saves:

the trained model to models/churn_model.joblib,

a detailed JSON report to churn_experiment_results.json.


Run from the project root:

python src/run_churn_experiment.py


Example console output:

=== Churn prediction experiment summary ===
Accuracy          : 0.800
Baseline accuracy : 0.600

Top features (by permutation importance):
  contract_type: 0.1500
  num_support_calls: 0.1200
  tenure_months: 0.0900
  ...


Output artifacts

After running src/run_churn_experiment.py, you should see:

models/
└─ churn_model.joblib

churn_experiment_results.json


The JSON report contains:

test accuracy and baseline accuracy,

majority-class label used for the baseline,

feature names,

permutation feature importance for each feature.


Example structure:

{
  "metrics": {
    "accuracy": 0.8,
    "baseline_accuracy": 0.6
  },
  "majority_class": 0,
  "top_features": [
    {
      "feature": "contract_type",
      "importance_mean": 0.15,
      "importance_std": 0.02
    },
    ...
  ],
  "feature_names": [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "contract_type",
    "num_support_calls",
    "has_promo"
  ],
  "model_path": "models/churn_model.joblib"
}



Tests

The project includes a small test suite:

tests/test_churn_pipeline.py:

checks that data loading returns consistent shapes,

verifies that the trained model is not worse than the majority-class baseline on this synthetic dataset,

verifies that permutation importance returns one entry per feature.


Run tests from the project root:

python -m unittest discover -s tests



Extending the demo

Possible next steps:

Replace the tiny synthetic dataset with a larger, more realistic churn dataset.

Try different models (e.g., random forest, gradient boosting) and compare metrics.

Add more evaluation metrics (precision, recall, F1-score, ROC-AUC).

Separate train/validation/test splits and introduce simple model selection.

Visualize feature importance with bar plots or integrate with an explainability library.

