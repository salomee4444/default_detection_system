# serving/app.py
from pathlib import Path
from typing import Any, Dict, List
import os
import urllib.request

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

# --- Paths & config ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "model_pipeline.joblib"

API_TOKEN = os.getenv("API_TOKEN", "change-me")
THRESHOLD = float(os.getenv("PRED_THRESHOLD", "0.30"))
MODEL_URL = os.getenv("MODEL_URL", "")

# --- Ensure artifact exists (download if necessary) ---
if not ARTIFACT_PATH.exists():
    if MODEL_URL:
        ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            print(f"Downloading model from {MODEL_URL} -> {ARTIFACT_PATH}")
            urllib.request.urlretrieve(MODEL_URL, ARTIFACT_PATH)
        except Exception as e:
            raise RuntimeError(f"Model artifact not found and download failed: {e}")
    else:
        raise RuntimeError(
            f"Model artifact not found at {ARTIFACT_PATH} and MODEL_URL is not set."
        )

# --- Load pipeline once on startup ---
pipe = joblib.load(ARTIFACT_PATH)

# --- Schemas ---
class PredictIn(BaseModel):
    row_id: str = Field(..., description="PK of the row in your DB")
    features: Dict[str, Any]

class PredictOut(BaseModel):
    row_id: str
    risk_score: float
    target: int

class PredictBatchIn(BaseModel):
    rows: List[PredictIn]

class PredictBatchOut(BaseModel):
    results: List[PredictOut]

# --- App ---
app = FastAPI(title="Default Detection API", version="1.0.0")

def _auth_or_401(auth_header: str):
    if auth_header != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.get("/health")
def health():
    return {"ok": True, "artifact": str(ARTIFACT_PATH.name)}

def _score_row(row_id: str, features: Dict[str, Any]) -> PredictOut:
    X = pd.DataFrame([features])
    if hasattr(pipe, "predict_proba"):
        proba = float(pipe.predict_proba(X)[0, 1])
    elif hasattr(pipe, "decision_function"):
        s = float(pipe.decision_function(X)[0])
        proba = 1.0 / (1.0 + np.exp(-s))
    else:
        proba = float(int(pipe.predict(X)[0]))
    target = int(proba >= THRESHOLD)
    return PredictOut(row_id=row_id, risk_score=round(proba, 6), target=target)

@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn, authorization: str = Header(default="")):
    _auth_or_401(authorization)
    return _score_row(payload.row_id, payload.features)

@app.post("/predict_batch", response_model=PredictBatchOut)
def predict_batch(payload: PredictBatchIn, authorization: str = Header(default="")):
    _auth_or_401(authorization)
    results = [_score_row(r.row_id, r.features) for r in payload.rows]
    return PredictBatchOut(results=results)