# SafeRoute MLOps

An MLOps project for estimating road-risk categories from road,
traffic, and weather features using a reproducible machine-learning
pipeline.

## Prediction target
dynamic_risk_category: LOW, MEDIUM, HIGH, or CLOSED.

## Pipeline
1. Validate input data.
2. Preprocess numerical and categorical features.
3. Train and evaluate a Random Forest classifier.
4. Track experiments with MLflow.
5. Serve predictions through FastAPI.
6. Run automated tests.
7. Package the application with Docker.
8. Automate checks with GitHub Actions.

## Limitations
The target represents an existing calculated risk category. The model
estimates that category and is not independently validated as a
predictor of future accidents. Performance metrics are measured on
held-out records from the available dataset.
# MLOPS
