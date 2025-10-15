from __future__ import annotations
import numpy as np
import pandas as pd


def rollout_effect(cate: np.ndarray, recommend: np.ndarray, base_incidence: float = 0.1) -> dict:
    """Estimate prevented cases under policy rollout.
    cate: predicted individual risk reduction (absolute)
    recommend: 0/1 policy decisions
    base_incidence: baseline dementia incidence used for scaling (toy)
    """
    cate = np.asarray(cate)
    recommend = np.asarray(recommend)
    treated_effect = (cate * recommend).sum()
    prevented = max(0.0, treated_effect)
    n_treated = int(recommend.sum())
    return {
        "n_treated": n_treated,
        "prevented_cases_est": float(prevented),
        "avg_effect_treated": float(treated_effect / max(1, n_treated)),
    }
