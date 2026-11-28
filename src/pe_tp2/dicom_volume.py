"""Volume uint8 de um exame lido direto dos DICOMs, IGUAL ao gravado na extração completa
(pe_baseline.extract.process_exam com save_volumes): cortes ordenados por z (ImagePositionPatient,
ou InstanceNumber sem ela), HU recortado, redimensionado para 256 px e janelado (CTPA) em uint8.

Usado na submissão tardia: a competição é de código e, ao submeter, o Kaggle roda o notebook de
novo sobre o teste OCULTO (outros exames). Então a inferência tem de partir dos DICOMs, dentro do
notebook, e não de predições pré-calculadas. Cada arquivo é lido UMA vez (cabeçalho e pixels);
a extração lia o cabeçalho antes para ordenar, com o mesmo resultado.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom

from pe_baseline.dicom_io import window
from pe_baseline.preprocess import resize_hu


def _z(ds) -> float:
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None:
        return float(ipp[2])
    return float(getattr(ds, "InstanceNumber", 0))


def read_exam(exam_dir: str | Path, sops: list[str], size: int, hu_clip, level: float,
              width: float):
    """-> (sops na ordem de z, volume uint8 (n, size, size), ok (n,) bool).
    Corte ilegível: fatia zerada e ok=False (sem predição), como na extração."""
    exam_dir = Path(exam_dir)
    zs, imgs, ok = [], [], []
    for sop in sops:
        try:
            ds = pydicom.dcmread(str(exam_dir / f"{sop}.dcm"))
            z = _z(ds)
        except Exception:                                   # nem o cabeçalho abre
            zs.append(np.inf)
            imgs.append(None)
            ok.append(False)
            continue
        zs.append(z)
        try:
            arr = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
            intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
            hu = np.clip(arr * slope + intercept, hu_clip[0], hu_clip[1])
            small = resize_hu(hu, size)
            imgs.append(np.round(window(small, level, width) * 255.0).astype(np.uint8))
            ok.append(True)
        except Exception:
            imgs.append(None)
            ok.append(False)
    order = np.argsort(np.asarray(zs, float), kind="stable")
    vol = np.zeros((len(sops), size, size), np.uint8)
    for j, i in enumerate(order):
        if imgs[i] is not None:
            vol[j] = imgs[i]
    return [sops[i] for i in order], vol, np.asarray(ok, bool)[order]
