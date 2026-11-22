"""Leitura DICOM, conversão para HU e janelamento.

Decisões:
* HU = pixel * RescaleSlope + RescaleIntercept (obrigatório em TC; valores brutos
  variam entre fabricantes e não são comparáveis).
* Valores de preenchimento fora do FOV (ex.: -2000/-3024) são recortados em -1024 HU (ar).
* Ordenação dos cortes por ImagePositionPatient[2] (eixo z), não por InstanceNumber,
  que não é confiável entre fabricantes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom

META_TAGS = [
    "PatientID", "Modality", "Manufacturer", "KVP", "SliceThickness",
    "PixelSpacing", "RescaleSlope", "RescaleIntercept", "WindowCenter",
    "WindowWidth", "Rows", "Columns", "ImagePositionPatient", "InstanceNumber",
]


@dataclass
class Slice:
    hu: np.ndarray          # float32, HU
    meta: dict


def _first(v):
    """Tags como WindowCenter podem ser multivalor; usa o primeiro valor."""
    if isinstance(v, (pydicom.multival.MultiValue, list, tuple)):
        return v[0] if len(v) else None
    return v


def read_meta(ds: pydicom.Dataset) -> dict:
    meta = {}
    for tag in META_TAGS:
        v = getattr(ds, tag, None)
        if tag == "PixelSpacing" and v is not None:
            meta["pixel_spacing_row"] = float(v[0])
            meta["pixel_spacing_col"] = float(v[1])
            continue
        if tag == "ImagePositionPatient" and v is not None:
            meta["z"] = float(v[2])
            continue
        v = _first(v)
        if isinstance(v, (pydicom.valuerep.DSfloat, pydicom.valuerep.IS)):
            v = float(v)
        meta[tag] = v if v is None or isinstance(v, (int, float)) else str(v)
    return meta


def read_slice(path: str | Path, hu_clip=(-1024, 3071)) -> Slice:
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    hu = arr * slope + intercept
    hu = np.clip(hu, hu_clip[0], hu_clip[1])
    return Slice(hu=hu, meta=read_meta(ds))


def read_z(path: str | Path) -> float:
    """Lê apenas o cabeçalho para obter z (rápido; usado para ordenar cortes)."""
    ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None:
        return float(ipp[2])
    return float(getattr(ds, "InstanceNumber", 0))


def window(hu: np.ndarray, level: float, width: float) -> np.ndarray:
    """Janelamento linear para [0, 1]."""
    lo, hi = level - width / 2.0, level + width / 2.0
    return np.clip((hu - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
