from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = Path("models/model_bundle.joblib")
ml_artifacts = {}

# ==========================================
# 1. PYDANTIC INPUT & OUTPUT SCHEMAS
# ==========================================
from typing import Literal, Optional, Union

class CustomerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Core 8 Drivers (Required)
    customerID: str = Field(default="DEMO-USER", min_length=1)
    SeniorCitizen: int = Field(..., ge=0, le=1)
    tenure: int = Field(..., ge=0)
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., ge=0.0)

    # Minor Background Fields (Optional -> Imputed by Pipeline if None)
    gender: Optional[Literal["Male", "Female"]] = None
    Partner: Optional[Literal["Yes", "No"]] = None
    Dependents: Optional[Literal["Yes", "No"]] = None
    PhoneService: Optional[Literal["Yes", "No"]] = None
    MultipleLines: Optional[Literal["Yes", "No", "No phone service"]] = None
    OnlineBackup: Optional[Literal["Yes", "No", "No internet service"]] = None
    DeviceProtection: Optional[Literal["Yes", "No", "No internet service"]] = None
    StreamingTV: Optional[Literal["Yes", "No", "No internet service"]] = None
    StreamingMovies: Optional[Literal["Yes", "No", "No internet service"]] = None
    PaperlessBilling: Optional[Literal["Yes", "No"]] = None
    TotalCharges: Optional[Union[float, int, str]] = None


class RiskFactor(BaseModel):
    feature: str
    impact: str

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    risk_tier: Literal['low', 'medium', 'high']
    top_risk_factors: list[RiskFactor]

# ==========================================
# 2. APP LIFESPAN (LOAD MODEL ONCE AT STARTUP)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model bundle not found at {MODEL_PATH}. Run py -m src.train first")
    bundle = joblib.load(MODEL_PATH)
    ml_artifacts["pipeline"] = bundle["pipeline"]
    ml_artifacts["explainer"] = bundle["explainer"]
    ml_artifacts["feature_names"] = bundle["feature_names"]
    yield
    ml_artifacts.clear()

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def serve_ui():
    return FileResponse("src/index.html")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# ==========================================
# 3. ACTIONABLE PREDICTION ENDPOINT
# ==========================================
@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerInput):
    if "pipeline" not in ml_artifacts:
        raise HTTPException(status_code=503, detail="Model artifacts not found.")

    pipeline = ml_artifacts['pipeline']
    explainer = ml_artifacts['explainer']
    feature_names = ml_artifacts['feature_names']

    input_df = pd.DataFrame([customer.model_dump()])

    prob = float(pipeline.predict_proba(input_df)[0, 1])

    if prob >= 0.70:
        risk_tier = "high"
    elif prob >= 0.40:
        risk_tier = "medium"
    else:
        risk_tier = "low"

    transformed_row = pipeline.named_steps["preprocessor"].transform(input_df)
    shap_vals = explainer.shap_values(transformed_row)

    # Safer handling for list, 3D array, or 2D array
    if isinstance(shap_vals, list):
        row_shap = shap_vals[1][0]
    elif getattr(shap_vals, "ndim", 0) == 3:
        row_shap = shap_vals[0, :, 1]
    else:
        row_shap = shap_vals[0]


    paired = list(zip(feature_names, row_shap))
    top_3 = sorted(paired, key=lambda x: abs(x[-1]), reverse=True)[:3]

    top_risk_factors = [
        RiskFactor(feature=feat, impact=f"{float(val):+.2f}")
        for feat, val in top_3
    ]

    return PredictionResponse(
        customer_id=customer.customerID,
        churn_probability=round(prob, 4),
        risk_tier=risk_tier,
        top_risk_factors=top_risk_factors
    )