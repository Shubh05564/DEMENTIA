from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from src.config import Config
from src.data.preprocess import load_dataset, preprocess
from src.causal.target_trial import estimate_propensity, dr_estimators
from src.models.heterogeneous_te import CATEModel
from src.policy.optimizer import Policy
from src.sim.simulation import rollout_effect

st.set_page_config(page_title="Dementia Diet Policy", layout="wide")

cfg = Config.load("configs/config.yaml")
st.title("Personalized Diet Recommendations to Reduce Dementia Risk")
st.caption("Target trial emulation, CATE modeling, and policy optimization")

# Sidebar: data source selection
st.sidebar.header("Data source")
data_source = st.sidebar.radio(
    "Choose data source",
    ["Configured path", "Upload CSVs"],
    index=0,
)
st.sidebar.write(f"Configured dataset path: {cfg.dataset.path}")

run_btn = st.sidebar.button("Run demo")

if run_btn:
    with st.spinner("Loading and preprocessing data..."):
        df = None
        if data_source == "Configured path":
            try:
                df = load_dataset(cfg.dataset.path, cfg.dataset.file_pattern)
            except Exception as e:
                st.warning(f"Could not load configured data: {e}")
        if df is None and data_source == "Upload CSVs":
            uploads = st.sidebar.file_uploader("Upload one or more CSV files", type=["csv"], accept_multiple_files=True)
            if not uploads:
                st.stop()
            import pandas as pd
            df = pd.concat([pd.read_csv(u) for u in uploads], ignore_index=True)
        if df is None:
            st.stop()
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

    st.success("Preprocessing complete.")

    with st.spinner("Estimating propensity and DR pseudo-outcomes..."):
        pr = estimate_propensity(art.X, art.t, model=cfg.causal.propensity_model, random_state=cfg.random_state)
        dr = dr_estimators(art.X, art.t, art.y, pr.pscore, random_state=cfg.random_state)
    st.write({"propensity_auc": pr.auc, "propensity_model": pr.model_name})

    with st.spinner("Fitting CATE model and computing effects..."):
        cate_model = CATEModel(method=cfg.hte_model.method, base=cfg.hte_model.base_learner, random_state=cfg.random_state)
        cate_model.fit(art.X, art.t, art.y)
        cate = cate_model.predict_effect(art.X)

    with st.spinner("Optimizing policy and simulating rollout..."):
        policy = Policy(cate_threshold=cfg.policy.cate_threshold, constraints=cfg.policy.constraints)
        rec = policy.recommend(art.X, cate)
        sim = rollout_effect(cate, rec.to_numpy())

    st.subheader("Results")
    st.metric("Estimated prevented cases (sum of effects)", f"{sim['prevented_cases_est']:.2f}")
    st.metric("Individuals recommended diet", str(int(rec.sum())))

    st.subheader("Distribution of predicted individual risk reduction (CATE)")
    st.bar_chart(pd.DataFrame({"CATE": cate}))

    st.subheader("Sample recommendations")
    st.dataframe(pd.DataFrame({
        "cate": cate,
        "recommend": rec.to_numpy(),
    }).head(20))

st.sidebar.markdown("---")
st.sidebar.write("Edit configs/config.yaml to match your data columns.")
