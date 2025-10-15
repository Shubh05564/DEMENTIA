from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Literal
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None


@dataclass
class PropensityResult:
    pscore: np.ndarray
    auc: float
    model_name: str


def estimate_propensity(X: pd.DataFrame, t: pd.Series, model: Literal["logistic", "xgboost"] = "logistic", random_state: int = 42) -> PropensityResult:
    X_train, X_val, t_train, t_val = train_test_split(X, t, test_size=0.2, random_state=random_state, stratify=t)

    if model == "xgboost" and xgb is not None:
        clf = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=random_state)
    else:
        clf = LogisticRegression(max_iter=200, solver="lbfgs")
        model = "logistic"

    clf.fit(X_train, t_train)
    p_val = clf.predict_proba(X_val)[:, 1]
    auc = float(roc_auc_score(t_val, p_val))
    p_all = clf.predict_proba(X)[:, 1]

    # Bound away from 0/1 for stability
    eps = 1e-3
    p_all = np.clip(p_all, eps, 1 - eps)

    return PropensityResult(pscore=p_all, auc=auc, model_name=model)


def dr_estimators(X: pd.DataFrame, t: pd.Series, y: pd.Series, pscore: np.ndarray, random_state: int = 42) -> pd.DataFrame:
    """Compute doubly robust pseudo-outcomes for CATE learning (binary outcome baseline)."""
    from sklearn.ensemble import RandomForestClassifier

    # Outcome models m0, m1
    m0 = RandomForestClassifier(n_estimators=300, random_state=random_state)
    m1 = RandomForestClassifier(n_estimators=300, random_state=random_state)
    m0.fit(X[t == 0], y[t == 0])
    m1.fit(X[t == 1], y[t == 1])
    mu0 = m0.predict_proba(X)[:, 1]
    mu1 = m1.predict_proba(X)[:, 1]

    w = t.to_numpy()
    y_np = y.to_numpy()

    # DR pseudo-outcome for uplift
    # tau_hat = ( (w - p) / (p*(1-p)) ) * (y - mu_w) + (mu1 - mu0)
    p = pscore
    mu_w = w * mu1 + (1 - w) * mu0
    tau_dr = ((w - p) / (p * (1 - p))) * (y_np - mu_w) + (mu1 - mu0)

    return pd.DataFrame({
        "mu0": mu0,
        "mu1": mu1,
        "tau_dr": tau_dr,
        "pscore": p,
    }, index=X.index)
