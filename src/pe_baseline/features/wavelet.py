"""Decomposição wavelet 2D (Mallat, 1989) — textura multiescala.

Energia, média absoluta e desvio dos coeficientes de cada sub-banda (aproximação e
detalhes H/V/D em 2 níveis, Daubechies-2) sobre o recorte da ROI na janela CTPA.
Sub-bandas de detalhe fino respondem a estruturas pequenas (vasos segmentares),
as grossas à distribuição global de contraste.
"""
from __future__ import annotations

import numpy as np
import pywt

from ..dicom_io import window
from ..preprocess import roi_bbox


def extract(hu: np.ndarray, masks, cfg: dict, prefix: str = "wav_") -> dict:
    pre, fcfg = cfg["preprocess"], cfg["features"]
    level, width = pre["windows"]["ctpa"]
    img = window(hu, level, width)
    ys, xs = roi_bbox(masks.roi)
    crop = img[ys, xs] * masks.roi[ys, xs]
    coeffs = pywt.wavedec2(crop, fcfg["wavelet"], level=fcfg["wavelet_levels"])
    bands = {"A": coeffs[0]}
    for lvl, (ch, cv, cd) in enumerate(coeffs[1:], start=1):
        # coeffs[1] é o nível mais grosso
        k = fcfg["wavelet_levels"] - lvl + 1
        bands.update({f"H{k}": ch, f"V{k}": cv, f"D{k}": cd})
    out = {}
    for name, c in bands.items():
        a = np.abs(c)
        out[f"{prefix}{name}_energy"] = float(np.mean(c ** 2))
        out[f"{prefix}{name}_absmean"] = float(a.mean())
        out[f"{prefix}{name}_std"] = float(c.std())
    return out
