"""Gradiente e bordas: HOG (Dalal & Triggs, 2005) + histograma de magnitude de Sobel.

O HOG captura a geometria local das bordas (contorno de vasos, interface trombo/contraste)
de forma robusta a pequenas variações de intensidade. É calculado sobre o recorte da ROI
redimensionado para 128x128 na janela mediastinal, com células grandes (32 px) para manter
a dimensionalidade baixa (324 dimensões).
"""
from __future__ import annotations

import numpy as np
from skimage.feature import hog
from skimage.filters import sobel
from skimage.transform import resize

from ..dicom_io import window
from ..preprocess import roi_bbox

SOBEL_EDGES = np.linspace(0, 0.5, 11)


def extract(hu: np.ndarray, masks, cfg: dict, prefix: str = "grad_") -> dict:
    pre, fcfg = cfg["preprocess"], cfg["features"]
    level, width = pre["windows"]["mediastinum"]
    img = window(hu, level, width)
    ys, xs = roi_bbox(masks.roi)
    crop = img[ys, xs] * masks.roi[ys, xs]
    size = fcfg["hog_size"]
    crop = resize(crop, (size, size), order=1, anti_aliasing=True)
    h = hog(crop, orientations=9, pixels_per_cell=(fcfg["hog_cell"],) * 2,
            cells_per_block=(2, 2), block_norm="L2-Hys", feature_vector=True)
    out = {f"{prefix}hog{i:03d}": float(v) for i, v in enumerate(h)}
    mag = sobel(img)[masks.roi]
    hist, _ = np.histogram(np.clip(mag, 0, 0.5 - 1e-9), bins=SOBEL_EDGES)
    hist = hist / max(hist.sum(), 1)
    for i, v in enumerate(hist):
        out[f"{prefix}sobel{i:02d}"] = float(v)
    out[f"{prefix}sobel_mean"] = float(mag.mean()) if mag.size else 0.0
    return out
