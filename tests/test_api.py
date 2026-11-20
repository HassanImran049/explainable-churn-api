import pytest
from fastapi.testclient import TestClient

from src.predict import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


VALID_CUSTOMER = {
    "customerID": "7590-VVHEG",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": "29.85",
}

def test_predict_returns_200_ok(client):
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200

    data = response.json()
    assert data["customer_id"] == "7590-VVHEG"
    assert data["risk_tier"] in {"low", "medium", "high"}
    assert len(data["top_risk_factors"]) == 3
    assert "feature" in data["top_risk_factors"][0]
    assert "impact" in data["top_risk_factors"][0]

def test_predict_invalid_payload_returns_422(client):
    bad_payload = VALID_CUSTOMER.copy()
    bad_payload["tenure"] = -5  # Violates Field(..., ge=0)
    bad_payload["Contract"] = "Weekly"  # Violates Literal options

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422

def test_churn_probability_within_bounds(client):
    # Also tests the TotalCharges blank string edge case (" ")
    edge_payload = VALID_CUSTOMER.copy()
    edge_payload["tenure"] = 0
    edge_payload["TotalCharges"] = " "

    response = client.post("/predict", json=edge_payload)
    assert response.status_code == 200

    prob = response.json()["churn_probability"]
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0