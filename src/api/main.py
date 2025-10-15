from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd

from src.config import Config
from src.data.preprocess import preprocess
from src.models.heterogeneous_te import CATEModel
from src.policy.optimizer import Policy

app = FastAPI(title="Dementia Diet Policy API")
cfg = Config.load("configs/config.yaml")


class PatientRequest(BaseModel):
    features: dict


class RecommendationResponse(BaseModel):
    cate: float
    recommend: int


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(req: PatientRequest):
    # Build a one-row DataFrame for the patient
    x = pd.DataFrame([req.features])
    # Minimal preprocessing: assume training transformer is not persisted in this MVP
    # For a production system, persist and load the fitted transformer and model artifacts
    # Here, we fit on the fly (not ideal) or expect the app to be called after training pipeline runs
    # Return a placeholder if not trained
    # For now: use simple defaults
    cate_model = CATEModel(method=cfg.hte_model.method, base=cfg.hte_model.base_learner, random_state=cfg.random_state)
    # Without training, we cannot produce a real estimate; return neutral response
    try:
        cate = float(np.nan)
        rec = 0
    except Exception:
        cate = float("nan")
        rec = 0
    return RecommendationResponse(cate=cate, recommend=rec)


@app.get("/")
def root():
    return {"status": "ok", "message": "Use POST /recommend with feature dict."}
