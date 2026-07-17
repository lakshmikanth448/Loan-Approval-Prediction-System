"""Synthetic dataset generator and training script for loan approval prediction."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "loan_applications.csv"
MODEL_PATH = ROOT / "loan_approval_model.joblib"
METRICS_PATH = ROOT / "model_metrics.json"


def build_dataset(rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Create a fictional dataset; target follows a realistic but non-production rule."""
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 66, rows)
    income = np.round(rng.lognormal(np.log(55000), 0.55, rows), 0)
    employment_years = np.minimum(rng.integers(0, 31, rows), age - 18)
    credit_score = np.clip(rng.normal(690, 75, rows), 300, 850).round().astype(int)
    loan_amount = np.round(rng.lognormal(np.log(18000), 0.65, rows), 0)
    loan_term = rng.choice([12, 24, 36, 48, 60], rows, p=[.08, .18, .35, .20, .19])
    debt_to_income = np.clip(rng.beta(2.2, 4.2, rows) * 0.8, 0.03, 0.75).round(3)
    dependents = rng.choice([0, 1, 2, 3, 4], rows, p=[.36, .28, .22, .10, .04])
    education = rng.choice(["High School", "Bachelor", "Master", "Doctorate"], rows, p=[.34, .42, .20, .04])
    self_employed = rng.choice(["No", "Yes"], rows, p=[.78, .22])
    property_area = rng.choice(["Urban", "Semiurban", "Rural"], rows, p=[.42, .36, .22])
    existing_loans = rng.integers(0, 4, rows)

    # A synthetic approval score with controlled noise avoids encoding any real policy.
    score = (
        (credit_score - 625) / 55 + (income - loan_amount * 1.7) / 50000
        - debt_to_income * 2.4 + employment_years / 16 - existing_loans * .3
        + (property_area == "Semiurban") * .15 + (education == "Master") * .10
        + rng.normal(0, .55, rows)
    )
    approved = np.where(score > 0, "Approved", "Rejected")
    return pd.DataFrame({
        "age": age, "annual_income": income, "employment_years": employment_years,
        "credit_score": credit_score, "loan_amount": loan_amount, "loan_term_months": loan_term,
        "debt_to_income": debt_to_income, "dependents": dependents, "education": education,
        "self_employed": self_employed, "property_area": property_area,
        "existing_loans": existing_loans, "loan_status": approved,
    })


def train() -> None:
    data = build_dataset()
    data.to_csv(DATA_PATH, index=False)
    features = data.drop(columns="loan_status")
    target = data["loan_status"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=.2, random_state=42, stratify=target
    )
    categorical = features.select_dtypes(include="object").columns.tolist()
    numeric = [c for c in features.columns if c not in categorical]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=250, min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=-1
        )),
    ])
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {"accuracy": round(float(accuracy_score(y_test, predictions)), 3),
               "report": classification_report(y_test, predictions, output_dict=True)}
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Dataset saved: {DATA_PATH.name} ({len(data):,} fictional applications)")
    print(f"Model saved: {MODEL_PATH.name}")
    print(f"Test accuracy: {metrics['accuracy']:.1%}")


if __name__ == "__main__":
    train()

