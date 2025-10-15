from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.config import Config
from src.policy.optimizer import Policy

app = FastAPI(title="Dementia Diet Policy API")
cfg = Config.load("configs/config.yaml")

# Attempt to load trained artifacts if present
MODELS_DIR = Path(cfg.output_dir)
try:
    CATE_MODEL = joblib.load(MODELS_DIR / "cate_model.joblib")
    TRANSFORMER = joblib.load(MODELS_DIR / "transformer.joblib")
except Exception:
    CATE_MODEL = None
    TRANSFORMER = None


class PatientRequest(BaseModel):
    features: dict


class RecommendationResponse(BaseModel):
    cate: float | None
    recommend: int


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(req: PatientRequest):
    # Build a one-row DataFrame for the patient
    x_raw = pd.DataFrame([req.features])
    if CATE_MODEL is None or TRANSFORMER is None:
        return RecommendationResponse(cate=None, recommend=0)

    # Transform features using the persisted transformer
    try:
        xt = TRANSFORMER.transform(x_raw)
        # Model expects a 2D array; wrap into DataFrame (columns not used by model)
        X_df = pd.DataFrame(xt)
        cate_val = float(np.ravel(CATE_MODEL.predict_effect(X_df))[0])
        policy = Policy(cate_threshold=cfg.policy.cate_threshold, constraints=cfg.policy.constraints)
        rec = int(cate_val >= cfg.policy.cate_threshold)
        return RecommendationResponse(cate=cate_val, recommend=rec)
    except Exception:
        return RecommendationResponse(cate=None, recommend=0)


@app.get("/")
def root():
    return {"status": "ok", "message": "Use POST /recommend with feature dict.", "artifacts_loaded": bool(CATE_MODEL is not None)}
