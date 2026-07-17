from pathlib import Path
import json

import joblib
import pandas as pd
from flask import Flask, render_template, request

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "loan_approval_model.joblib"
METRICS_PATH = ROOT / "model_metrics.json"
app = Flask(__name__)
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

FIELDS = ["age", "annual_income", "employment_years", "credit_score", "loan_amount", "loan_term_months", "debt_to_income", "dependents", "education", "self_employed", "property_area", "existing_loans"]
NUMERIC = {"age": int, "annual_income": float, "employment_years": int, "credit_score": int, "loan_amount": float, "loan_term_months": int, "debt_to_income": float, "dependents": int, "existing_loans": int}

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    values = {}
    if request.method == "POST" and model:
        values = request.form.to_dict()
        row = {field: NUMERIC[field](values[field]) if field in NUMERIC else values[field] for field in FIELDS}
        frame = pd.DataFrame([row])
        label = model.predict(frame)[0]
        probability = model.predict_proba(frame)[0]
        approved_index = list(model.classes_).index("Approved")
        result = {"label": label, "probability": round(float(probability[approved_index]) * 100, 1)}
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None
    return render_template("index.html", result=result, values=values, metrics=metrics, model_ready=model is not None)

if __name__ == "__main__":
    # Disable Flask's reloader so only one predictable local server is started.
    app.run(host="127.0.0.1", port=5000, debug=False)
