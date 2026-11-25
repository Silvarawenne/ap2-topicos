"""Extração de características por corte, paralela e retomável.

Estratégia volumétrica (decisão central do trabalho): características 2D POR CORTE,
depois agregadas por exame (pooling). Motivo: o rótulo de corte (pe_present_on_image)
existe para todos os cortes, o que permite (i) um classificador de corte supervisionado e
(ii) agregação corte -> exame coerente com a hierarquia de rótulos do desafio
(exame positivo <=> existe ao menos um corte positivo). Descritores 3D diretos sobre um
volume de ~250 cortes diluiriam um trombo que ocupa poucos cortes.

Os cortes de cada exame são ordenados por z e subamostrados com passo fixo
(sample.slice_stride, TP1) ou usados todos (full.slice_stride = 1, TP2).

TP2 (save_volumes): na MESMA leitura do DICOM, cada corte também é gravado como imagem
uint8 de 256x256 na janela CTPA, em um volume por exame ordenado por z. Os volumes de um
bloco ficam juntos em volumes_<bloco>.npz (um membro <StudyInstanceUID>.npy por exame,
lido sob demanda com np.load(...)[study]); a coluna vol_idx do parquet diz qual fatia do
volume corresponde a cada SOPInstanceUID. Cortes ilegíveis ficam como fatia zerada e
status de erro (lista de excluídos).
"""
from __future__ import annotations

import faulthandler
import multiprocessing as mp
import queue
import signal
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import data_path
from .aggregate import FEATURE_PREFIXES
from .data import SOP, SERIES, STUDY
from .dicom_io import read_slice, read_z, window
from .features import gradient, intensity, texture, wavelet
from .preprocess import compute_masks, peak_hu, resize_hu

META_KEEP = ["PatientID", "Manufacturer", "KVP", "SliceThickness", "pixel_spacing_row",
             "RescaleSlope", "RescaleIntercept", "WindowCenter", "WindowWidth", "Rows", "Columns"]

HAS_ALARM = hasattr(signal, "SIGALRM")          # não existe no Windows (testes locais)


class SliceTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise SliceTimeout("tempo limite por corte excedido")


def to_uint8(hu_small: np.ndarray, level: float, width: float) -> np.ndarray:
    """Corte já redimensionado (HU) -> uint8 na janela dada (entrada das redes no TP2)."""
    return np.round(window(hu_small, level, width) * 255.0).astype(np.uint8)


def slice_features(hu: np.ndarray, cfg: dict, hu_small: np.ndarray | None = None) -> dict:
    pre = cfg["preprocess"]
    peak = peak_hu(hu, pre["out_size"])
    hu = resize_hu(hu, pre["out_size"]) if hu_small is None else hu_small
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


def process_exam(study: str, exam_rows: pd.DataFrame, cfg: dict, stride: int | None = None,
                 src_dir: str | None = None, vol_dir: Path | None = None) -> list[dict]:
    series = exam_rows[SERIES].iloc[0]
    exam_dir = data_path(cfg, src_dir or cfg["data"]["train_dir"], study, series)
    stride = stride or cfg["sample"]["slice_stride"]
    chosen = select_slices(exam_rows, exam_dir, stride)
    size = int(cfg["preprocess"]["out_size"])
    vol = np.zeros((len(chosen), size, size), np.uint8) if vol_dir is not None else None
    lvl, wid = cfg["preprocess"]["windows"][cfg.get("full", {}).get("volume_window", "ctpa")]
    out = []
    limit = int(cfg["preprocess"].get("slice_timeout_s", 90))
    # Proteção contra corte que trava (observado no Kaggle): vira erro registrado, não bloco parado.
    # faulthandler imprime a pilha se mesmo assim o processo ficar parado (diagnóstico no log).
    faulthandler.dump_traceback_later(limit * 4, repeat=True, file=sys.stderr)
    if HAS_ALARM:
        signal.signal(signal.SIGALRM, _on_alarm)
    for i, (_, r) in enumerate(chosen.iterrows()):
        base = {STUDY: study, SOP: r[SOP], "z_rank": r["z_rank"], "vol_idx": i,
                "pe_present_on_image": int(r.get("pe_present_on_image", -1))}
        try:
            if HAS_ALARM:
                signal.alarm(limit)
            sl = read_slice(r["path"], tuple(cfg["preprocess"]["hu_clip"]))
            small = resize_hu(sl.hu, size)
            feats = slice_features(sl.hu, cfg, small)
            if vol is not None:
                vol[i] = to_uint8(small, lvl, wid)
            meta = {f"meta_{k}": sl.meta.get(k) for k in META_KEEP}
            out.append({**base, "status": "ok", **meta, **feats})
        except Exception as exc:  # arquivo corrompido/ilegível: registrado, não silenciado
            out.append({**base, "status": f"error: {exc!r}"[:300]})
        finally:
            if HAS_ALARM:
                signal.alarm(0)
    faulthandler.cancel_dump_traceback_later()
    if vol is not None:
        tmp = Path(vol_dir) / f"{study}.npy.part"
        with open(tmp, "wb") as fh:
            np.save(fh, vol)
        tmp.replace(Path(vol_dir) / f"{study}.npy")
    return out


def _child(q, study, exam_rows, cfg, kw):
    try:
        q.put(("ok", process_exam(study, exam_rows, cfg, **kw)))
    except BaseException as exc:  # noqa: BLE001
        q.put(("err", repr(exc)[:300]))


def process_exam_guarded(study: str, exam_rows: pd.DataFrame, cfg: dict, **kw) -> list[dict]:
    """Roda o exame em processo separado com limite de tempo duro. Travamento em código nativo
    (que o alarme não interrompe) mata só esse exame, que fica registrado como excluído."""
    if "fork" not in mp.get_all_start_methods():   # Windows (testes locais): sem proteção
        return process_exam(study, exam_rows, cfg, **kw)
    limit = float(cfg["preprocess"].get("exam_timeout_s", 900))
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(q, study, exam_rows, cfg, kw))
    p.start()
    try:
        kind, payload = q.get(timeout=limit)
    except queue.Empty:
        kind, payload = "err", f"timeout do exame ({limit:.0f}s)"
    finally:
        if p.is_alive():
            p.kill()
        p.join()
    if kind == "ok":
        return payload
    sops = exam_rows[SOP].tolist()
    return [{STUDY: study, SOP: sop, "vol_idx": -1,
             "status": f"error: exame excluído: {payload}"[:300]} for sop in sops]


def bundle_volumes(staging: Path, studies: list[str], dest: Path) -> list[str]:
    """Junta os .npy dos exames do bloco num único .npz (zip de .npy, legível por np.load).
    Poucos arquivos grandes em vez de milhares: mais simples de subir como Dataset no Kaggle."""
    tmp = dest.with_name(dest.name + ".part")
    present = []
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for s in studies:
            f = staging / f"{s}.npy"
            if f.exists():                      # exame excluído por timeout não tem volume
                zf.write(f, arcname=f"{s}.npy")
                present.append(s)
    tmp.replace(dest)
    for s in present:
        (staging / f"{s}.npy").unlink()
    return present


def run_extraction(cfg: dict, train: pd.DataFrame, studies: list[str], out_dir: Path,
                   shard: int = 0, n_shards: int = 1, chunk_size: int = 40,
                   n_jobs: int = -1, verbose: bool = True, stride: int | None = None,
                   save_volumes: bool = False) -> list[Path]:
    """Processa os exames do shard em blocos; blocos já gravados são pulados (retomável).

    `train` pode ter a coluna `source` ("train"/"test"): a pasta de cada exame é
    cfg.data.train_dir ou cfg.full.test_dir. Com save_volumes, grava também
    volumes_<bloco>.npz (ver docstring do módulo) ANTES do parquet, que marca o bloco pronto.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    studies = sorted(studies)[shard::n_shards]
    rows = train[train[STUDY].isin(studies)]
    by_study = {k: v for k, v in rows.groupby(STUDY)}
    src_dirs = {"train": cfg["data"]["train_dir"], "test": cfg.get("full", {}).get("test_dir", "test")}
    source = rows.groupby(STUDY)["source"].first() if "source" in rows else None
    staging = out_dir / "_staging"
    if save_volumes:
        staging.mkdir(exist_ok=True)

    def kw(s):
        src = source[s] if source is not None else "train"
        return dict(stride=stride, src_dir=src_dirs[src], vol_dir=staging if save_volumes else None)

    chunks = [studies[i:i + chunk_size] for i in range(0, len(studies), chunk_size)]
    written = []
    for ci, chunk in enumerate(chunks):
        tag = f"s{shard:02d}of{n_shards:02d}_c{ci:04d}"
        path = out_dir / f"slices_{tag}.parquet"
        if path.exists():
            written.append(path)
            continue
        t0 = time.time()
        try:
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_exam_guarded)(s, by_study[s], cfg, **kw(s)) for s in chunk)
        except Exception:
            traceback.print_exc()
            raise
        df = pd.DataFrame([row for exam in results for row in exam])
        if source is not None:
            df["source"] = df[STUDY].map(source)
        if save_volumes:
            vol_name = f"volumes_{tag}.npz"
            have = set(bundle_volumes(staging, chunk, out_dir / vol_name))
            df["vol_file"] = np.where(df[STUDY].isin(have), vol_name, "")
        for c in df.columns:
            if c.startswith("meta_"):
                df[c] = df[c].astype(str)
            elif c.startswith(FEATURE_PREFIXES):
                df[c] = df[c].astype("float32")   # metade do disco; precisão suficiente
        df.to_parquet(path, index=False)
        written.append(path)
        if verbose:
            n_err = int((df["status"] != "ok").sum())
            print(f"[shard {shard}] bloco {ci + 1}/{len(chunks)}: {len(chunk)} exames, "
                  f"{len(df)} cortes ({n_err} com erro), {time.time() - t0:.0f}s -> {path.name}",
                  flush=True)
    if save_volumes and staging.exists() and not any(staging.iterdir()):
        staging.rmdir()
    return written


def load_volume(npz_path: Path | str, study: str) -> np.ndarray:
    """Volume uint8 (n_cortes, 256, 256) de um exame, na ordem de z (vol_idx)."""
    with np.load(npz_path) as z:
        return z[study]


def load_slice_features(feat_dir: Path) -> pd.DataFrame:
    files = sorted(Path(feat_dir).glob("slices_*.parquet"))
    if not files:
        raise FileNotFoundError(f"Nenhum parquet de características em {feat_dir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
