"""Leitura da configuração e resolução de caminhos (sem caminhos absolutos fixos)."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    # Sobrescritas por ambiente (úteis no Kaggle, sem editar o YAML):
    #   PE_DATA_ROOT     raiz da competição
    #   PE_FEATURES_DIR  pasta com exams.csv e slices/ (ex.: dataset de saída da extração)
    #   PE_RESULTS_DIR   pasta de resultados
    if os.environ.get("PE_DATA_ROOT"):
        cfg["data"]["root"] = os.environ["PE_DATA_ROOT"]
    if os.environ.get("PE_FEATURES_DIR"):
        cfg["features"]["out_dir"] = os.environ["PE_FEATURES_DIR"]
    if os.environ.get("PE_RESULTS_DIR"):
        cfg["results_dir"] = os.environ["PE_RESULTS_DIR"]
    return cfg


def repo_path(rel: str | os.PathLike) -> Path:
    """Caminho relativo à raiz do repositório (ou absoluto, se já for)."""
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def data_path(cfg: dict, *parts: str) -> Path:
    return Path(cfg["data"]["root"]).joinpath(*parts)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
