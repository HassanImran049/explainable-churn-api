# Subscription Churn & Explainable Retention Engine

An end-to-end, containerized machine learning microservice that identifies at-risk subscribers before cancellation, assigns actionable risk tiers, and returns per-customer **SHAP** drivers via a FastAPI REST endpoint and zero-dependency interactive dashboard.

---

## Model Performance & Inference Latency

Evaluated via **5-Fold Stratified Cross-Validation** and a **20% hold-out test split**, optimizing for **ROC-AUC** and **Recall** under class imbalance (~26.5% positive churn rate):

| Model | Implementation | ROC-AUC | Recall | Avg Single-Request Latency |
| :--- | :--- | :---: | :---: | :---: |
| **Logistic Regression** | Custom NumPy (Weighted Log-Loss & Gradient Descent) | 0.8415 | 0.7844 | ~0.4 ms |
| **Linear SVM** | Custom NumPy (Weighted Hinge Loss & Sub-Gradient Descent) | 0.8330 | 0.8240 | ~0.4 ms |
| **LightGBM (Tuned)** | `LGBMClassifier` (`RandomizedSearchCV` + `class_weight="balanced"`) | **0.8487** | **0.8048** | **11.39 ms** |

---

## The Business Problem

In recurring-revenue businesses, acquiring a new subscriber costs **5x–7x more** than retaining an existing one—yet most retention teams rely on lagging metrics (like a cancellation request) when it is already too late to intervene. 

Furthermore, a raw probability score alone (`0.83`) does not tell a retention specialist *what action to take*. This system solves both problems:
1. **Early Risk Detection:** Scores active subscribers on billing, tenure, contract commitment, and service-adoption signals, prioritizing high recall (`80.5%`) so at-risk accounts are not missed.
2. **Actionable Reason Codes:** Computes local **SHAP (SHapley Additive exPlanations)** values on every prediction, surfacing the top 3 concrete drivers pushing a specific customer toward churn (e.g., month-to-month billing friction or missing security/support add-ons).

---

## Key Engineering Highlights

* **Zero Training/Serving Skew:** Encapsulates silent type-coercion edge cases (e.g., whitespace strings in billing totals), median/mode imputation, scaling, and one-hot encoding inside a single scikit-learn `Pipeline` (`DataCleaner` + `ColumnTransformer`).
* **First-Principles Benchmarking:** Built vectorized **Logistic Regression**, **Linear SVM**, and **Stratified K-Fold** splitters from scratch in NumPy to establish leak-free linear baselines before scaling to regularized gradient-boosted trees (`LightGBM`).
* **Strict API Contracts & Graceful Missingness:** Enforces runtime schema validation with **Pydantic v2** (`422 Unprocessable Entity` on invalid payloads) while allowing minor background attributes to default to `None` for pipeline imputation.
* **Offline-Ready Interactive UI:** Ships with a lightweight, zero-CDN dashboard (`GET /`) for real-time what-if simulation alongside OpenAPI docs (`GET /docs`).
* **Production Containerization:** Packaged into a slim `python:3.11-slim` Docker image with automated **Pytest** API verification.

---

## API Usage (`POST /predict`)

### Request Payload
Send full subscriber records or only core behavioral/billing drivers (optional background fields are imputed automatically):

```json
{
  "customerID": "7590-VVHEG",
  "SeniorCitizen": 0,
  "tenure": 1,
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "TechSupport": "No",
  "Contract": "Month-to-month",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.0
}
```

### Actionable Response (`200 OK`)
```json
{
  "customer_id": "7590-VVHEG",
  "churn_probability": 0.9124,
  "risk_tier": "high",
  "top_risk_factors": [
    { "feature": "Contract_Month-to-month", "impact": "+0.74" },
    { "feature": "tenure", "impact": "+0.69" },
    { "feature": "OnlineSecurity_No", "impact": "+0.24" }
  ]
}
```

---

## Project Structure

```text
├── data/               # Raw subscription dataset (excluded from version control)
├── models/             # Serialized model_bundle.joblib (Pipeline + SHAP TreeExplainer)
├── src/
│   ├── __init__.py
│   ├── preprocess.py   # Custom DataCleaner transformer & ColumnTransformer pipeline
│   ├── train.py        # Tuned LightGBM training, evaluation, & SHAP artifact export
│   ├── predict.py      # FastAPI service, Pydantic v2 schemas, & local SHAP extraction
│   └── index.html      # Interactive what-if simulator & global SHAP feature chart
├── tests/
│   └── test_api.py     # Pytest suite (200 OK, 422 validation, probability bounds)
├── Dockerfile          # Slim Python 3.11 container with libgomp1 runtime
└── requirements.txt
```

---

## Quickstart

### 1. Run via Docker (Recommended)
```bash
docker build -t churn-retention-api .
docker run -p 8000:8000 churn-retention-api
```
* **Interactive Dashboard:** `http://localhost:8000/`
* **Swagger API Docs:** `http://localhost:8000/docs`

### 2. Run Locally (`venv`)
```bash
python -m venv venv
venv\Scripts\activate        # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt

# Train model bundle & run API test suite
python -m src.train
python -m pytest -v

# Start server
uvicorn src.predict:app --reload
```
