"""Extração de características por corte, paralela e retomável.

Estratégia volumétrica (decisão central do trabalho): características 2D POR CORTE,
depois agregadas por exame (pooling). Motivo: o rótulo de corte (pe_present_on_image)
existe para todos os cortes, o que permite (i) um classificador de corte supervisionado e
(ii) agregação corte -> exame coerente com a hierarquia de rótulos do desafio
(exame positivo <=> existe ao menos um corte positivo). Descritores 3D diretos sobre um
volume de ~250 cortes diluiriam um trombo que ocupa poucos cortes.

Os cortes de cada exame são ordenados por z e subamostrados com passo fixo
(sample.slice_stride), o que preserva a cobertura do volume inteiro.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import data_path
from .aggregate import FEATURE_PREFIXES
from .data import SOP, SERIES, STUDY
from .dicom_io import read_slice, read_z
from .features import gradient, intensity, texture, wavelet
from .preprocess import compute_masks, peak_hu, resize_hu

META_KEEP = ["PatientID", "Manufacturer", "KVP", "SliceThickness", "pixel_spacing_row",
             "RescaleSlope", "RescaleIntercept", "WindowCenter", "WindowWidth", "Rows", "Columns"]


def slice_features(hu: np.ndarray, cfg: dict) -> dict:
    pre = cfg["preprocess"]
    peak = peak_hu(hu, pre["out_size"])
    hu = resize_hu(hu, pre["out_size"])
    masks = compute_masks(hu, pre, peak)
    feats = {}
    feats.update(intensity.extract(hu, masks, cfg))
    feats.update(texture.extract(hu, masks, cfg))
    feats.update(gradient.extract(hu, masks, cfg))
    feats.update(wavelet.extract(hu, masks, cfg))
    if cfg["features"].get("compute_noroi", False):
        feats.update(intensity.extract(hu, masks, cfg, use_roi=False, prefix="noroi_int_"))
        feats.update(texture.extract(hu, masks, cfg, use_roi=False, prefix="noroi_tex_"))
    return feats


def select_slices(exam_rows: pd.DataFrame, exam_dir: Path, stride: int) -> pd.DataFrame:
    rows = exam_rows.copy()
    rows["path"] = [exam_dir / f"{s}.dcm" for s in rows[SOP]]
    rows["z"] = [read_z(p) for p in rows["path"]]
    # Ordem crânio-caudal pode variar; usamos z crescente e a posição relativa.
    rows = rows.sort_values("z").reset_index(drop=True)
    rows["z_rank"] = np.arange(len(rows)) / max(len(rows) - 1, 1)
    return rows.iloc[::stride]


def process_exam(study: str, exam_rows: pd.DataFrame, cfg: dict) -> list[dict]:
    series = exam_rows[SERIES].iloc[0]
    exam_dir = data_path(cfg, cfg["data"]["train_dir"], study, series)
    chosen = select_slices(exam_rows, exam_dir, cfg["sample"]["slice_stride"])
    out = []
    for _, r in chosen.iterrows():
        try:
            sl = read_slice(r["path"], tuple(cfg["preprocess"]["hu_clip"]))
            feats = slice_features(sl.hu, cfg)
            meta = {f"meta_{k}": sl.meta.get(k) for k in META_KEEP}
            out.append({STUDY: study, SOP: r[SOP], "z_rank": r["z_rank"],
                        "pe_present_on_image": int(r["pe_present_on_image"]),
                        "status": "ok", **meta, **feats})
        except Exception as exc:  # arquivo corrompido/ilegível: registrado, não silenciado
            out.append({STUDY: study, SOP: r[SOP], "status": f"error: {exc!r}"[:300]})
    return out


def run_extraction(cfg: dict, train: pd.DataFrame, studies: list[str], out_dir: Path,
                   shard: int = 0, n_shards: int = 1, chunk_size: int = 40,
                   n_jobs: int = -1, verbose: bool = True) -> list[Path]:
    """Processa os exames do shard em blocos; blocos já gravados são pulados (retomável)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    studies = sorted(studies)[shard::n_shards]
    by_study = {k: v for k, v in train[train[STUDY].isin(studies)].groupby(STUDY)}
    chunks = [studies[i:i + chunk_size] for i in range(0, len(studies), chunk_size)]
    written = []
    for ci, chunk in enumerate(chunks):
        path = out_dir / f"slices_s{shard:02d}of{n_shards:02d}_c{ci:04d}.parquet"
        if path.exists():
            written.append(path)
            continue
        t0 = time.time()
        try:
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_exam)(s, by_study[s], cfg) for s in chunk)
        except Exception:
            traceback.print_exc()
            raise
        rows = [row for exam in results for row in exam]
        df = pd.DataFrame(rows)
        for c in df.columns:
            if c.startswith("meta_"):
                df[c] = df[c].astype(str)
            elif c.startswith(FEATURE_PREFIXES):
                df[c] = df[c].astype("float32")   # metade do disco; precisão suficiente
        df.to_parquet(path, index=False)
        written.append(path)
        if verbose:
            print(f"[shard {shard}] bloco {ci + 1}/{len(chunks)}: {len(chunk)} exames, "
                  f"{len(df)} cortes, {time.time() - t0:.0f}s -> {path.name}", flush=True)
    return written


def load_slice_features(feat_dir: Path) -> pd.DataFrame:
    files = sorted(Path(feat_dir).glob("slices_*.parquet"))
    if not files:
        raise FileNotFoundError(f"Nenhum parquet de características em {feat_dir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
