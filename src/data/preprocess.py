from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import glob
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline


@dataclass
class PreprocessArtifacts:
    X: pd.DataFrame
    y: pd.Series
    t: pd.Series
    transformer: ColumnTransformer


def load_dataset(directory: Path, pattern: str) -> pd.DataFrame:
    csv_paths = sorted(glob.glob(str(directory / pattern)))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {directory} with pattern '{pattern}'.")
    dfs = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    return df


def build_transformer(numeric: list[str], categorical: list[str], imputation: str = "iterative", scale_numeric: bool = True) -> ColumnTransformer:
    if imputation == "iterative":
        num_imputer = IterativeImputer(random_state=0, sample_posterior=True)
    else:
        strategy = "mean" if imputation not in {"mean", "median"} else imputation
        num_imputer = SimpleImputer(strategy=strategy)

    num_steps = [("imputer", num_imputer)]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformer = ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, categorical),
    ], remainder="drop")
    return transformer


def preprocess(df: pd.DataFrame, id_col: str, t_col: str, y_col: str, numeric: list[str], categorical: list[str], imputation: str, scale_numeric: bool) -> PreprocessArtifacts:
    # Drop rows with missing T/Y minimally; transformers handle X
    df = df.dropna(subset=[t_col, y_col])
    X = df[numeric + categorical].copy()
    y = df[y_col].astype(float)
    t = df[t_col]
    # If treatment is categorical, binarize to 0/1 for baseline; extend later for multi-arm
    if t.dtype == object:
        t = (t.astype(str) != "control").astype(int)
    else:
        t = t.astype(int)

    transformer = build_transformer(numeric, categorical, imputation, scale_numeric)
    X_tr = transformer.fit_transform(X)
    # Recreate DataFrame with transformed columns for interpretability
    # Build feature names
    num_features = numeric
    cat_features = list(transformer.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical))
    feat_names = num_features + cat_features
    X_df = pd.DataFrame(X_tr, columns=feat_names, index=df.index)

    return PreprocessArtifacts(X=X_df, y=y, t=t, transformer=transformer)
