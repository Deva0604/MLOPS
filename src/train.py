from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "dynamic_road_risk.csv"
PARAMS_PATH = ROOT / "params.yaml"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TARGET = "dynamic_risk_category"

NUMERIC = [
    "road_length_m",
    "current_speed_kmh",
    "free_flow_speed_kmh",
    "congestion_ratio",
    "temperature_c",
    "humidity_percent",
    "precipitation_mm",
    "wind_speed_kmh",
]

CATEGORICAL = ["road_closure", "highway"]


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    params = yaml.safe_load(PARAMS_PATH.read_text())
    df = pd.read_csv(DATA_PATH)

    required = NUMERIC + CATEGORICAL + [TARGET]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(str).str.strip().str.upper()

    allowed_classes = {"LOW", "MEDIUM", "HIGH", "CLOSED"}
    df = df[df[TARGET].isin(allowed_classes)].copy()

    X = df[NUMERIC + CATEGORICAL].copy()
    y = df[TARGET].copy()

    # Convert numeric inputs and handle infinite values.
    for column in NUMERIC:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    X[NUMERIC] = X[NUMERIC].replace(
        [float("inf"), float("-inf")], float("nan")
    )

    # Standardize categorical inputs.
    for column in CATEGORICAL:
        X[column] = X[column].astype("string").fillna("UNKNOWN").astype(str)

    if y.nunique() < 2 or y.value_counts().min() < 2:
        raise ValueError("Each target class needs at least two records.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(params["test_size"]),
        random_state=int(params["random_state"]),
        stratify=y,
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("categorical", categorical_pipeline, CATEGORICAL),
    ])

    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("model", RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            class_weight="balanced",
            random_state=int(params["random_state"]),
            n_jobs=-1,
        )),
    ])

    # Track experiments locally with MLflow.
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    tracking_db = ROOT / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")
    mlflow.set_experiment("SafeRoute-Road-Risk")

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_test, predictions)
            ),
            "macro_f1": float(
                f1_score(y_test, predictions, average="macro", zero_division=0)
            ),
            "weighted_f1": float(
                f1_score(y_test, predictions, average="weighted", zero_division=0)
            ),
            "training_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        }

        mlflow.log_params(params)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("target", TARGET)
        mlflow.log_metrics(metrics)

        model_path = MODEL_DIR / "road_risk_model.joblib"

        joblib.dump({
            "pipeline": pipeline,
            "feature_columns": NUMERIC + CATEGORICAL,
            "target": TARGET,
            "classes": ["LOW", "MEDIUM", "HIGH", "CLOSED"],
        }, model_path)

        report = {
            **metrics,
            "classification_report": classification_report(
                y_test,
                predictions,
                labels=["LOW", "MEDIUM", "HIGH", "CLOSED"],
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix_labels": ["LOW", "MEDIUM", "HIGH", "CLOSED"],
            "confusion_matrix": confusion_matrix(
                y_test,
                predictions,
                labels=["LOW", "MEDIUM", "HIGH", "CLOSED"],
            ).tolist(),
        }

        report_path = REPORT_DIR / "evaluation_metrics.json"
        report_path.write_text(json.dumps(report, indent=2))

        mlflow.log_artifact(str(report_path))
        mlflow.sklearn.log_model(
                pipeline,
                name="road_risk_model",
                skops_trusted_types=[
            "numpy.dtype",
            "sklearn.tree._tree.Tree",
    ],
)

        print("\n=== SAFEROUTE MODEL EVALUATION ===")
        print(json.dumps(metrics, indent=2))
        print("\nClassification report:")
        print(classification_report(y_test, predictions, zero_division=0))
        print("\nConfusion matrix:")
        print(report["confusion_matrix"])
        print(f"\nModel saved: {model_path}")
        print(f"Evaluation saved: {report_path}")


if __name__ == "__main__":
    main()
