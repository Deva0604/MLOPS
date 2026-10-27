import joblib

path = "model/road_risk_model.joblib"
model = joblib.load(path)

print("Model type:", type(model))
print("Saved object type:", type(model).__name__)

if isinstance(model, dict):
    print("Saved keys:", list(model.keys()))
    pipeline = model.get("pipeline")
    if pipeline is not None:
        print("Pipeline type:", type(pipeline))
        print("Pipeline steps:", list(pipeline.named_steps.keys()))
    print("Feature columns:", model.get("feature_columns"))
    print("Target:", model.get("target"))
    print("Classes:", model.get("classes"))
else:
    print("Pipeline steps:", list(model.named_steps.keys()))
