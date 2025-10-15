from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class Columns:
    id: str
    treatment: str
    outcome: str
    time: str | None
    features_numeric: list[str]
    features_categorical: list[str]


@dataclass
class DatasetCfg:
    path: Path
    file_pattern: str


@dataclass
class PreprocessingCfg:
    imputation_strategy: str
    scale_numeric: bool


@dataclass
class CausalCfg:
    propensity_model: str
    estimator: str


@dataclass
class HTECfg:
    method: str
    base_learner: str


@dataclass
class PolicyCfg:
    type: str
    cate_threshold: float
    constraints: dict


@dataclass
class AppCfg:
    port_streamlit: int
    port_fastapi: int


@dataclass
class Config:
    project_name: str
    random_state: int
    output_dir: Path
    dataset: DatasetCfg
    columns: Columns
    preprocessing: PreprocessingCfg
    causal: CausalCfg
    hte_model: HTECfg
    policy: PolicyCfg
    app: AppCfg

    @staticmethod
    def load(path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        # Normalize paths
        dataset_path = Path(cfg["dataset"]["path"]).expanduser()
        output_dir = Path(cfg["output_dir"]).expanduser()
        return Config(
            project_name=cfg["project_name"],
            random_state=int(cfg["random_state"]),
            output_dir=output_dir,
            dataset=DatasetCfg(path=dataset_path, file_pattern=cfg["dataset"]["file_pattern"]),
            columns=Columns(
                id=cfg["columns"]["id"],
                treatment=cfg["columns"]["treatment"],
                outcome=cfg["columns"]["outcome"],
                time=cfg["columns"].get("time"),
                features_numeric=cfg["columns"].get("features_numeric", []) or [],
                features_categorical=cfg["columns"].get("features_categorical", []) or [],
            ),
            preprocessing=PreprocessingCfg(
                imputation_strategy=cfg["preprocessing"]["imputation_strategy"],
                scale_numeric=bool(cfg["preprocessing"]["scale_numeric"]),
            ),
            causal=CausalCfg(
                propensity_model=cfg["causal"]["propensity_model"],
                estimator=cfg["causal"]["estimator"],
            ),
            hte_model=HTECfg(
                method=cfg["h te_model"]["method"],
                base_learner=cfg["h te_model"]["base_learner"],
            ),
            policy=PolicyCfg(
                type=cfg["policy"]["type"],
                cate_threshold=float(cfg["policy"]["cate_threshold"]),
                constraints=cfg["policy"].get("constraints", {}),
            ),
            app=AppCfg(
                port_streamlit=int(cfg["app"]["port_streamlit"]),
                port_fastapi=int(cfg["app"]["port_fastapi"]),
            ),
        )
