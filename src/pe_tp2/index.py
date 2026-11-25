"""Índice de cortes do TP2: junta os shards da extração completa com a partição fixa.

Cada linha = um corte (SOPInstanceUID) com o exame, a posição no volume (vol_idx), o rótulo
de corte, o arquivo .npz que guarda o volume e a partição:
  train / val / test  -> partição fixa por exame do treino oficial (splits/split_full_train.csv)
  official_test       -> teste oficial do Kaggle (sem rótulos; só para a submissão tardia)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

STUDY, SOP = "StudyInstanceUID", "SOPInstanceUID"
INDEX_COLS = [STUDY, SOP, "vol_idx", "z_rank", "pe_present_on_image", "status", "source", "vol_file"]


def find_slice_dirs(roots, max_depth: int = 6) -> list[Path]:
    """Pastas que contêm slices_*.parquet com volumes (saídas dos shards da extração).
    Não desce nas pastas de imagens da competição (train/ e test/ ao lado de train.csv)."""
    found = []
    for root in roots:
        root = Path(root)
        base_depth = len(root.parts)
        for cur, dirs, files in os.walk(root):
            if any(f.startswith("volumes_") and f.endswith(".npz") for f in files):
                found.append(Path(cur))
            if "train.csv" in files or len(Path(cur).parts) - base_depth >= max_depth:
                dirs[:] = []                          # pasta da competição ou fundo demais
    return sorted(set(found))


def load_index(slice_dirs, split_csv: str | Path, exclude_indeterminate: bool = True) -> pd.DataFrame:
    parts = []
    for d in slice_dirs:
        for f in sorted(Path(d).glob("slices_*.parquet")):
            df = pd.read_parquet(f, columns=INDEX_COLS)
            df["vol_path"] = np.where(df["vol_file"] != "", [str(Path(d) / v) for v in df["vol_file"]], "")
            parts.append(df.drop(columns="vol_file"))
    if not parts:
        raise FileNotFoundError(f"nenhum slices_*.parquet com volumes em {list(slice_dirs)}")
    idx = pd.concat(parts, ignore_index=True)
    dup = idx.duplicated(SOP)
    if dup.any():                                      # o mesmo shard anexado duas vezes
        idx = idx[~dup]
    split = pd.read_csv(split_csv)[[STUDY, "split", "label3", "pe"]]
    idx = idx.merge(split, on=STUDY, how="left")
    off = idx["source"] == "test"
    idx.loc[off, "split"] = "official_test"
    idx.loc[off, "pe"] = -1
    idx.loc[off, "label3"] = "unknown"
    if idx["split"].isna().any():
        raise ValueError("exames do treino sem partição: a partição não corresponde à extração")
    if exclude_indeterminate:                          # sem verdade binária (mesma regra do TP1)
        idx = idx[idx["label3"] != "indeterminate"]
    idx["pe"] = idx["pe"].astype(int)
    idx["ok"] = (idx["status"] == "ok") & (idx["vol_path"] != "")
    return idx.reset_index(drop=True)


def counts(idx: pd.DataFrame) -> pd.DataFrame:
    """Contagens por partição (exames, cortes, cortes positivos, excluídos) para o artigo."""
    g = idx.groupby("split")
    return pd.DataFrame({
        "exams": g[STUDY].nunique(),
        "exams_pe": idx[idx["pe"] == 1].groupby("split")[STUDY].nunique(),
        "slices": g.size(),
        "slices_pos": idx[idx["pe_present_on_image"] == 1].groupby("split").size(),
        "slices_excluded": idx[~idx["ok"]].groupby("split").size(),
    }).fillna(0).astype(int)
