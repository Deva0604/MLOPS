from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dynamic_road_risk.csv"
REPORTS = ROOT / "reports"

TARGET = "dynamic_risk_category"

REQUIRED = [
    TARGET,
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

def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    df = pd.read_csv(DATA)

    missing = [col for col in REQUIRED if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    df[TARGET] = df[TARGET].astype(str).str.strip().str.upper()

    expected = {"LOW", "MEDIUM", "HIGH", "CLOSED"}
    unexpected = sorted(set(df[TARGET].dropna()) - expected)

    if unexpected:
        raise ValueError(f"Unexpected target labels: {unexpected}")

    report = {
        "dataset": str(DATA.relative_to(ROOT)),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_values": int(df.isna().sum().sum()),
        "target_distribution": {
            str(k): int(v)
            for k, v in df[TARGET].value_counts().items()
        },
        "required_columns": REQUIRED,
        "validation": "PASSED",
    }

    output = REPORTS / "data_validation.json"
    output.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"\nValidation report saved: {output}")

if __name__ == "__main__":
    main()
