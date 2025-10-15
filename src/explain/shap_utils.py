from __future__ import annotations
import numpy as np
import pandas as pd
import shap


def shap_values(model, X: pd.DataFrame):
    # Try tree explainer; fallback to kernel
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
    except Exception:
        explainer = shap.KernelExplainer(model.predict, X.sample(min(200, len(X)), random_state=0))
        sv = explainer.shap_values(X, nsamples=200)
    return sv
