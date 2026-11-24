"""Textura: GLCM/Haralick e LBP uniforme rotação-invariante.

* GLCM (Haralick et al., 1973) sobre a imagem na janela CTPA, quantizada em 32 níveis,
  restrita ao recorte da ROI; propriedades médias sobre 4 ângulos (invariância
  aproximada à rotação) para cada distância.
* LBP (Ojala et al., 2002), variante uniforme e rotação-invariante: invariante a mudanças
  monotônicas de intensidade, o que reduz o efeito de diferenças de contraste entre
  aquisições. Histograma calculado apenas sobre pixels da ROI.
"""
from __future__ import annotations

import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from ..dicom_io import window
from ..preprocess import roi_bbox

GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


def glcm_features(img01: np.ndarray, mask: np.ndarray, levels: int, distances, prefix: str) -> dict:
    ys, xs = roi_bbox(mask)
    crop = img01[ys, xs]
    m = mask[ys, xs]
    q = np.minimum((crop * levels).astype(np.int32), levels - 1)
    # Pixels fora da ROI recebem um nível extra que é descartado da matriz.
    q = np.where(m, q, levels).astype(np.uint8)
    glcm = graycomatrix(q, distances=distances, angles=ANGLES, levels=levels + 1,
                        symmetric=True, normed=False)
    glcm = glcm[:levels, :levels].astype(np.float64)
    sums = glcm.sum(axis=(0, 1), keepdims=True)
    glcm = glcm / np.where(sums == 0, 1, sums)
    out = {}
    for prop in GLCM_PROPS:
        vals = graycoprops(glcm, prop)          # (n_dist, n_angles)
        for di, d in enumerate(distances):
            out[f"{prefix}glcm_{prop}_d{d}"] = float(np.nan_to_num(vals[di].mean()))
    return out


def lbp_features(img01: np.ndarray, mask: np.ndarray, configs, prefix: str) -> dict:
    out = {}
    img8 = (img01 * 255).astype(np.uint8)
    for P, R in configs:
        codes = local_binary_pattern(img8, P, R, method="uniform")
        n_bins = P + 2
        hist, _ = np.histogram(codes[mask], bins=n_bins, range=(0, n_bins))
        hist = hist / max(hist.sum(), 1)
        for i, h in enumerate(hist):
            out[f"{prefix}lbp_P{P}R{R}_{i:02d}"] = float(h)
    return out


def extract(hu: np.ndarray, masks, cfg: dict, use_roi: bool = True, prefix: str = "tex_") -> dict:
    pre, fcfg = cfg["preprocess"], cfg["features"]
    level, width = pre["windows"]["ctpa"]
    img = window(hu, level, width)
    mask = masks.roi if use_roi else np.ones_like(masks.roi, dtype=bool)
    out = glcm_features(img, mask, fcfg["glcm_levels"], fcfg["glcm_distances"], prefix)
    out.update(lbp_features(img, mask, fcfg["lbp"], prefix))
    return out
