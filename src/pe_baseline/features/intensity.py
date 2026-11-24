"""Descritores de intensidade em HU dentro da ROI torácica.

Justificativa: o TEP aparece como falha de enchimento (≈20–100 HU) dentro de uma artéria
opacificada (≈250–500 HU). Como HU é uma unidade física calibrada, histogramas e frações
de pixels em faixas de HU têm significado clínico direto, ao contrário de níveis de cinza.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

HIST_EDGES = np.array([-1024, -900, -700, -500, -300, -100, 0, 50, 100, 150,
                       200, 300, 400, 600, 1000, 3072], dtype=np.float32)
PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def _stats(values: np.ndarray, prefix: str) -> dict:
    out = {}
    if values.size < 10:
        values = np.zeros(10, dtype=np.float32)
    out[f"{prefix}mean"] = float(values.mean())
    out[f"{prefix}std"] = float(values.std())
    out[f"{prefix}skew"] = float(np.nan_to_num(stats.skew(values)))
    out[f"{prefix}kurt"] = float(np.nan_to_num(stats.kurtosis(values)))
    for p, v in zip(PCTS, np.percentile(values, PCTS)):
        out[f"{prefix}p{p}"] = float(v)
    return out


def extract(hu: np.ndarray, masks, cfg: dict, use_roi: bool = True, prefix: str = "int_") -> dict:
    region = masks.roi if use_roi else np.ones_like(masks.roi, dtype=bool)
    vals = hu[region]
    out = _stats(vals, prefix)
    hist, _ = np.histogram(vals, bins=HIST_EDGES)
    hist = hist / max(hist.sum(), 1)
    for i, h in enumerate(hist):
        out[f"{prefix}hist{i:02d}"] = float(h)
    if use_roi:
        n_roi = max(int(masks.roi.sum()), 1)
        n_vessel = int(masks.vessel.sum())
        n_defect = int(masks.defect.sum())
        out[f"{prefix}roi_area"] = n_roi / masks.roi.size
        out[f"{prefix}lung_frac"] = float(masks.lung.sum()) / n_roi
        out[f"{prefix}vessel_frac"] = n_vessel / n_roi
        out[f"{prefix}defect_frac"] = n_defect / n_roi
        out[f"{prefix}defect_per_vessel"] = n_defect / max(n_vessel, 1)
        vessel_vals = hu[masks.vessel]
        out[f"{prefix}vessel_mean"] = float(vessel_vals.mean()) if vessel_vals.size else 0.0
        out[f"{prefix}vessel_p90"] = float(np.percentile(vessel_vals, 90)) if vessel_vals.size else 0.0
        out[f"{prefix}roi_fallback"] = float(masks.roi_fallback)
    return out
