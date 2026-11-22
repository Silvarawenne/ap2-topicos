"""Agregação corte -> exame.

Duas estratégias comparadas no trabalho:
1. Pooling de características: para cada descritor, estatísticas sobre os cortes do exame
   (média, desvio, máximo, percentil 90). O máximo/p90 preserva sinais focais (um trombo em
   poucos cortes), que a média diluiria.
2. Pooling de probabilidades: um classificador de corte é treinado e suas probabilidades
   são agregadas por exame (máximo, média dos top-k). Coerente com a regra hierárquica
   do desafio: exame positivo <=> ao menos um corte positivo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import STUDY

FEATURE_PREFIXES = ("int_", "tex_", "grad_", "wav_", "noroi_int_", "noroi_tex_")
POOL_STATS = ("mean", "std", "max", "p90")


def feature_columns(df: pd.DataFrame, prefixes=FEATURE_PREFIXES) -> list[str]:
    return [c for c in df.columns if c.startswith(tuple(prefixes))]


def family_columns(cols: list[str], family: str) -> list[str]:
    """Seleciona colunas de uma família ('intensity', 'texture', ..., 'all', 'noroi')."""
    pref = {"intensity": "int_", "texture": "tex_", "gradient": "grad_", "wavelet": "wav_"}
    if family == "all":
        return [c for c in cols if not c.startswith("noroi_")]
    if family == "noroi":
        return [c for c in cols if c.startswith("noroi_")]
    if family == "roi_matched":  # mesmas famílias do noroi, mas com ROI (ablação justa)
        return [c for c in cols if c.startswith(("int_", "tex_"))
                and not c.startswith(("int_roi_", "int_lung_", "int_vessel_", "int_defect_"))]
    p = pref[family]
    return [c for c in cols if c.startswith(p)]


def pool_features(slices: pd.DataFrame, stats=POOL_STATS) -> pd.DataFrame:
    ok = slices[slices["status"] == "ok"]
    cols = feature_columns(ok)
    g = ok.groupby(STUDY)[cols]
    parts = []
    if "mean" in stats:
        parts.append(g.mean().add_suffix("__mean"))
    if "std" in stats:
        parts.append(g.std(ddof=0).add_suffix("__std"))
    if "max" in stats:
        parts.append(g.max().add_suffix("__max"))
    if "p90" in stats:
        parts.append(g.quantile(0.9).add_suffix("__p90"))
    pooled = pd.concat(parts, axis=1)
    pooled["n_slices_used"] = ok.groupby(STUDY).size()
    return pooled.reset_index()


def pool_probabilities(study_ids: np.ndarray, probs: np.ndarray, k: int = 5) -> pd.DataFrame:
    df = pd.DataFrame({STUDY: study_ids, "p": probs})

    def topk(x):
        return np.sort(x.values)[-k:].mean()

    g = df.groupby(STUDY)["p"]
    return pd.DataFrame({"max": g.max(), f"top{k}": g.apply(topk), "mean": g.mean()}).reset_index()
