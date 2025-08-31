
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from .schemas import Features
from .utils import ARTIFACTS_DIR

app = FastAPI(title="Student Performance Predictor", version="0.1.0")

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"

@app.on_event("startup")
def _load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        model = None

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(model)}

@app.post("/predict")
def predict(features: Features):
    if model is None:
        return {"error": "Model not loaded. Train first."}
    df = pd.DataFrame([features.dict()])
    pred = float(model.predict(df)[0])
    return {"prediction": pred}
