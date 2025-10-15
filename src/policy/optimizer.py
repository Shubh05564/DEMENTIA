from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

try:
    import cvxpy as cp
except Exception:  # pragma: no cover
    cp = None


@dataclass
class Policy:
    cate_threshold: float = 0.02
    constraints: dict | None = None

    def recommend(self, X: pd.DataFrame, cate: np.ndarray) -> pd.Series:
        # Simple thresholding policy, placeholder for constrained optimization
        cate = np.asarray(cate).ravel()
        rec = (cate >= self.cate_threshold).astype(int)
        # Optionally apply feasibility constraints (placeholder)
        # Real constraints would require mapping features to diet nutrient totals
        return pd.Series(rec, index=X.index, name="diet_recommendation")

    def optimize_population(self, X: pd.DataFrame, cate: np.ndarray, budget: int | None = None) -> pd.Series:
        if budget is None or cp is None:
            return self.recommend(X, cate)
        # Example: select top-K individuals to treat under a budget using cvxpy
        n = len(X)
        x = cp.Variable(n, boolean=True)
        c = np.asarray(cate).ravel()
        obj = cp.Maximize(c @ x)
        constraints = []
        if budget is not None:
            constraints.append(cp.sum(x) <= budget)
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.ECOS_BB, warm_start=True)
        sol = np.array(x.value).round().astype(int).reshape(-1)
        return pd.Series(sol, index=X.index, name="diet_recommendation")
