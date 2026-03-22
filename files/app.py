import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model

app = FastAPI(title="Lending Club Loan Prediction API", version="1.0.0")

# Load model and scaler at startup
MODEL_PATH = os.getenv("MODEL_PATH", "lending_club_model.keras")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")

model = None
scaler = None
feature_columns = None


@app.on_event("startup")
def load_artifacts():
    global model, scaler, feature_columns
    try:
        model = load_model(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise

    try:
        with open(SCALER_PATH, "rb") as f:
            artifacts = pickle.load(f)
            scaler = artifacts["scaler"]
            feature_columns = artifacts["feature_columns"]
        print(f"✅ Scaler loaded from {SCALER_PATH}")
        print(f"✅ Feature columns: {feature_columns}")
    except Exception as e:
        print(f"❌ Failed to load scaler: {e}")
        raise


class LoanApplication(BaseModel):
    loan_amnt: float
    int_rate: float
    installment: float
    annual_inc: float
    dti: float
    open_acc: float
    pub_rec: float
    revol_bal: float
    revol_util: float
    total_acc: float
    mort_acc: float
    earliest_cr_line_year: int
    term: int  # 36 or 60
    grade: str  # A, B, C, D, E, F, G
    home_ownership: str  # RENT, OWN, MORTGAGE, OTHER
    verification_status: str  # Not Verified, Source Verified, Verified
    purpose: str  # debt_consolidation, credit_card, home_improvement, etc.
    address: str  # 5-digit zip code
    application_type: str  # Individual or Joint App

    class Config:
        json_schema_extra = {
            "example": {
                "loan_amnt": 10000,
                "int_rate": 11.44,
                "installment": 329.48,
                "annual_inc": 117000,
                "dti": 26.24,
                "open_acc": 16,
                "pub_rec": 0,
                "revol_bal": 36369,
                "revol_util": 41.8,
                "total_acc": 25,
                "mort_acc": 0,
                "earliest_cr_line_year": 1990,
                "term": 36,
                "grade": "B",
                "home_ownership": "MORTGAGE",
                "verification_status": "Not Verified",
                "purpose": "debt_consolidation",
                "address": "22690",
                "application_type": "Individual"
            }
        }


def preprocess(data: LoanApplication) -> np.ndarray:
    """Replicate the notebook preprocessing pipeline."""
    row = {
        "loan_amnt": data.loan_amnt,
        "int_rate": data.int_rate,
        "installment": data.installment,
        "annual_inc": data.annual_inc,
        "dti": data.dti,
        "open_acc": data.open_acc,
        "pub_rec": data.pub_rec,
        "revol_bal": data.revol_bal,
        "revol_util": data.revol_util,
        "total_acc": data.total_acc,
        "mort_acc": data.mort_acc,
        "earliest_cr_line_year": data.earliest_cr_line_year,
        "term": data.term,
        # categorical
        "grade": data.grade,
        "home_ownership": data.home_ownership.upper().replace("NONE", "OTHER").replace("ANY", "OTHER"),
        "verification_status": data.verification_status,
        "purpose": data.purpose,
        "address": str(data.address)[-5:],
        "application_type": data.application_type,
    }

    df = pd.DataFrame([row])

    # One-hot encode categoricals (drop_first=True, same as training)
    cat_cols = ["grade", "home_ownership", "verification_status", "purpose", "address", "application_type"]
    dummies = pd.get_dummies(df[cat_cols], drop_first=True)
    df = df.drop(columns=cat_cols)
    df = pd.concat([df, dummies], axis=1)

    # Align with training feature columns
    df = df.reindex(columns=feature_columns, fill_value=0)

    return scaler.transform(df.values)


@app.get("/")
def root():
    return {"message": "Lending Club Loan Prediction API", "status": "running"}


@app.get("/health")
def health():
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


@app.post("/predict")
def predict(application: LoanApplication):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        X = preprocess(application)
        prob = float(model.predict(X)[0][0])
        prediction = int(prob > 0.5)
        return {
            "prediction": prediction,
            "label": "Fully Paid" if prediction == 1 else "Charged Off",
            "probability": round(prob, 4),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
