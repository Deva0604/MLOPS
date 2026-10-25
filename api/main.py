
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "road_risk_model.joblib"

app = FastAPI(
    title="SafeRoute MLOps API",
    description=(
        "Predicts the existing dynamic road-risk category "
        "using a trained machine-learning pipeline."
    ),
    version="1.0.0",
)

model_bundle = None


class RoadRiskRequest(BaseModel):
    road_length_m: float = Field(ge=0)
    current_speed_kmh: float = Field(ge=0)
    free_flow_speed_kmh: float = Field(gt=0)
    congestion_ratio: float = Field(ge=0)
    temperature_c: float
    humidity_percent: float = Field(ge=0, le=100)
    precipitation_mm: float = Field(ge=0)
    wind_speed_kmh: float = Field(ge=0)
    road_closure: bool
    highway: str


@app.on_event("startup")
def load_model():
    global model_bundle

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run python src/train.py first."
        )

    model_bundle = joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {
        "project": "SafeRoute MLOps",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model_bundle is not None,
    }


@app.get("/model-info")
def model_info():
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    return {
        "model_type": type(model_bundle["pipeline"].steps[-1][1]).__name__,
        "target": model_bundle["target"],
        "classes": list(model_bundle["classes"]),
        "features": model_bundle["feature_columns"],
    }


@app.post("/predict")
def predict(request: RoadRiskRequest):
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    try:
        input_data = pd.DataFrame([request.model_dump()])
        pipeline = model_bundle["pipeline"]

        prediction = pipeline.predict(input_data)[0]
        result = {"predicted_risk_category": str(prediction)}

        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(input_data)[0]
            result["class_probabilities"] = {
                str(label): round(float(probability), 4)
                for label, probability in zip(
                    pipeline.classes_, probabilities
                )
            }

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc
