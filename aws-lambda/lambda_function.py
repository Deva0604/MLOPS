import json
import os

# Lambda inference uses serial processing.
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import joblib
import numpy as np
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "road_risk_model.joblib",
)

FEATURE_COLUMNS = [
    "road_length_m",
    "current_speed_kmh",
    "free_flow_speed_kmh",
    "congestion_ratio",
    "temperature_c",
    "humidity_percent",
    "precipitation_mm",
    "wind_speed_kmh",
    "road_closure",
    "highway",
]

_model_bundle = None


def load_model():
    global _model_bundle

    if _model_bundle is None:
        _model_bundle = joblib.load(MODEL_PATH)

    return _model_bundle


def lambda_handler(event, context):
    try:
        body = event

        if isinstance(event, dict) and "body" in event:
            body = event["body"]

        if isinstance(body, str):
            body = json.loads(body)

        if not isinstance(body, dict):
            raise ValueError("Input must be a JSON object.")

        # Accept either {"features": {...}} or a flat feature object.
        features = body.get("features", body)

        if not isinstance(features, dict):
            raise ValueError("'features' must be a JSON object.")

        missing = [
            name for name in FEATURE_COLUMNS
            if name not in features
        ]

        if missing:
            return response(400, {
                "error": "Missing required features",
                "missing_features": missing,
            })

        row = {name: features[name] for name in FEATURE_COLUMNS}

        numeric_features = [
            "road_length_m",
            "current_speed_kmh",
            "free_flow_speed_kmh",
            "congestion_ratio",
            "temperature_c",
            "humidity_percent",
            "precipitation_mm",
            "wind_speed_kmh",
        ]

        for name in numeric_features:
            value = row[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return response(400, {
                    "error": f"Feature '{name}' must be numeric."
                })
            if not np.isfinite(value):
                return response(400, {
                    "error": f"Feature '{name}' must be finite."
                })

        for name in ["road_closure", "highway"]:
            if not isinstance(row[name], str):
                return response(400, {
                    "error": f"Feature '{name}' must be a string."
                })

        bundle = load_model()
        pipeline = bundle["pipeline"]

        # Avoid Joblib multiprocessing warnings in AWS Lambda.
        if hasattr(pipeline, "named_steps"):
            model_step = pipeline.named_steps.get("model")
            if model_step is not None and hasattr(model_step, "n_jobs"):
                model_step.n_jobs = 1

        # Preserve the exact feature order used during training.
        feature_columns = bundle.get("feature_columns", FEATURE_COLUMNS)
        ordered_row = {name: row[name] for name in feature_columns}

        import pandas as pd

        input_frame = pd.DataFrame([ordered_row])
        prediction = pipeline.predict(input_frame)[0]

        result = {
            "prediction": str(prediction),
            "target": bundle.get("target", "dynamic_risk_category"),
            "model_classes": [
                str(value)
                for value in bundle.get("classes", [])
            ],
        }

        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(input_frame)[0]
            classes = pipeline.classes_

            result["probabilities"] = {
                str(label): round(float(probability), 6)
                for label, probability in zip(classes, probabilities)
            }

        return response(200, result)

    except (json.JSONDecodeError, ValueError) as exc:
        return response(400, {"error": str(exc)})

    except Exception:
        print("Prediction failed.")
        import traceback
        traceback.print_exc()
        return response(500, {"error": "Internal prediction error"})


def response(status_code, payload):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
        },
        "body": json.dumps(payload),
    }
