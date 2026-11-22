"""Pré-processamento por corte: redimensionamento e máscaras anatômicas.

ROI torácica: a embolia pulmonar ocorre nas artérias pulmonares (tronco, principais,
lobares, segmentares), que estão no mediastino e nos hilos/parênquima. Por isso a ROI
é o *casco convexo dos pulmões*: ele contém pulmões + mediastino e exclui mesa,
parede torácica, braços e ar externo, que só adicionam variância de aquisição.

Máscaras derivadas (usadas pelos descritores de intensidade):
* vaso: ROI ∩ HU ∈ [150, 600]  -> lúmen arterial opacificado por contraste.
* falha: ROI ∩ HU ∈ [0, 120] cercada por contraste (buraco no lúmen ou vizinhança
  majoritariamente vaso) -> densidade de trombo cercada de contraste,
  que é o sinal radiológico direto de TEP (falha de enchimento).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image, disk
from skimage.segmentation import clear_border
from skimage.transform import resize


@dataclass
class Masks:
    roi: np.ndarray
    lung: np.ndarray
    vessel: np.ndarray
    defect: np.ndarray
    roi_fallback: bool


def resize_hu(hu: np.ndarray, out_size: int) -> np.ndarray:
    if hu.shape == (out_size, out_size):
        return hu.astype(np.float32)
    # anti_aliasing evita serrilhado; order=1 (bilinear) preserva HU médios.
    return resize(hu, (out_size, out_size), order=1, preserve_range=True,
                  anti_aliasing=True).astype(np.float32)


def body_mask(hu: np.ndarray) -> np.ndarray:
    body = hu > -500
    body = ndi.binary_opening(body, structure=disk(2))
    lab = label(body)
    if lab.max() == 0:
        return np.ones_like(body, dtype=bool)
    largest = max(regionprops(lab), key=lambda r: r.area).label
    return ndi.binary_fill_holes(lab == largest)


def lung_mask(hu: np.ndarray, thr: float = -400) -> np.ndarray:
    air = hu < thr
    air = clear_border(label(air)) > 0        # remove ar externo (conectado à borda)
    lab = label(air)
    min_area = 0.002 * hu.size
    regions = [r for r in regionprops(lab) if r.area >= min_area]
    regions = sorted(regions, key=lambda r: r.area, reverse=True)[:2]
    mask = np.zeros_like(air, dtype=bool)
    for r in regions:
        mask[lab == r.label] = True
    return mask


def compute_masks(hu: np.ndarray, cfg_pre: dict) -> Masks:
    lung = lung_mask(hu, cfg_pre["lung_threshold_hu"])
    fallback = lung.sum() < 0.01 * hu.size
    if fallback:
        roi = body_mask(hu)
    else:
        roi = ndi.binary_dilation(convex_hull_image(lung), structure=disk(3))
    v_lo, v_hi = cfg_pre["vessel_hu"]
    d_lo, d_hi = cfg_pre["defect_hu"]
    vessel = roi & (hu >= v_lo) & (hu <= v_hi)
    # Falha de enchimento = densidade de partes moles CERCADA por contraste:
    #  (a) dentro de um "buraco" do lúmen opacificado (sinal do anel de contraste), ou
    #  (b) com fração de vaso > 0.55 na vizinhança 7x7 (falha parcialmente cercada).
    # Isso descarta a borda do vaso, onde o volume parcial (contraste <-> partes moles)
    # produz HU na mesma faixa do trombo. Limitação: trombo mural colado à parede, sem
    # contraste ao redor, não é capturado.
    soft = roi & ~vessel & (hu >= d_lo) & (hu <= d_hi)
    holes = ndi.binary_fill_holes(vessel) & ~vessel
    vessel_frac = ndi.uniform_filter(vessel.astype(np.float32), size=7)
    defect = soft & (holes | (vessel_frac > 0.55))
    return Masks(roi=roi, lung=lung, vessel=vessel, defect=defect, roi_fallback=bool(fallback))


def roi_bbox(mask: np.ndarray, pad: int = 2) -> tuple[slice, slice]:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return slice(None), slice(None)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, mask.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, mask.shape[1])
    return slice(y0, y1), slice(x0, x1)
