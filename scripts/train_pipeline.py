from __future__ import annotations
from pathlib import Path
import os
import json
import joblib
import numpy as np
import pandas as pd

from src.config import Config
from src.data.preprocess import load_dataset, preprocess
from src.causal.target_trial import estimate_propensity, dr_estimators
from src.models.heterogeneous_te import CATEModel
from src.policy.optimizer import Policy
from src.sim.simulation import rollout_effect

# Optional MLflow
try:
    import mlflow
    MLFLOW = True
except Exception:
    MLFLOW = False


def main():
    cfg = Config.load("configs/config.yaml")
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = load_dataset(cfg.dataset.path, cfg.dataset.file_pattern)

    print("Preprocessing...")
    art = preprocess(
        df=df,
        id_col=cfg.columns.id,
        t_col=cfg.columns.treatment,
        y_col=cfg.columns.outcome,
        numeric=cfg.columns.features_numeric,
        categorical=cfg.columns.features_categorical,
        imputation=cfg.preprocessing.imputation_strategy,
        scale_numeric=cfg.preprocessing.scale_numeric,
    )

    print("Estimating propensity and DR pseudo-outcomes...")
    pr = estimate_propensity(art.X, art.t, model=cfg.causal.propensity_model, random_state=cfg.random_state)
    dr = dr_estimators(art.X, art.t, art.y, pr.pscore, random_state=cfg.random_state)

    print("Fitting CATE model...")
    cate_model = CATEModel(method=cfg.hte_model.method, base=cfg.hte_model.base_learner, random_state=cfg.random_state)
    cate_model.fit(art.X, art.t, art.y)
    cate = cate_model.predict_effect(art.X)

    print("Optimizing policy and simulating rollout...")
    policy = Policy(cate_threshold=cfg.policy.cate_threshold, constraints=cfg.policy.constraints)
    rec = policy.recommend(art.X, cate)
    sim = rollout_effect(cate, rec.to_numpy())

    # Save artifacts
    print("Saving artifacts...")
    joblib.dump(cate_model, outdir / "cate_model.joblib")
    joblib.dump(art.transformer, outdir / "transformer.joblib")
    np.save(outdir / "cate.npy", cate)
    rec.to_csv(outdir / "recommendations.csv", index=True)
    with open(outdir / "propensity.json", "w", encoding="utf-8") as f:
        json.dump({"auc": pr.auc, "model": pr.model_name}, f)
    with open(outdir / "simulation.json", "w", encoding="utf-8") as f:
        json.dump(sim, f)

    print("Summary:")
    print({"propensity_auc": pr.auc, **sim})

    if MLFLOW:
        print("Logging to MLflow...")
        mlflow.set_experiment(cfg.project_name)
        with mlflow.start_run(run_name="initial_run"):
            mlflow.log_params({
                "propensity_model": cfg.causal.propensity_model,
                "estimator": cfg.causal.estimator,
                "hte_method": cfg.hte_model.method,
                "hte_base": cfg.hte_model.base_learner,
                "cate_threshold": cfg.policy.cate_threshold,
            })
            mlflow.log_metrics({
                "propensity_auc": pr.auc,
                "prevented_cases": sim["prevented_cases_est"],
                "n_treated": sim["n_treated"],
            })
            mlflow.log_artifact(str(outdir / "recommendations.csv"))
            mlflow.log_artifact(str(outdir / "simulation.json"))
            # Model artifacts
            mlflow.log_artifact(str(outdir / "cate_model.joblib"))
            mlflow.log_artifact(str(outdir / "transformer.joblib"))


if __name__ == "__main__":
    main()
