# Loan Approval Prediction System

An educational, synthetic-data Flask demo. It is not suitable for real lending decisions.

## Run

```powershell
py -m pip install -r requirements.txt
py ml.py
py app.py
```

Then visit `http://127.0.0.1:5000`.

`ml.py` creates 5,000 fictional applications and trains a preprocessing + Random Forest pipeline. `app.py` loads that pipeline and presents the form UI.
