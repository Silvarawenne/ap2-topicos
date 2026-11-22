"""Tabela de exames, partição fixa por exame e amostra estratificada do TP1.

Unidade de amostra: o exame (StudyInstanceUID). No RSNA-STR PE, cada exame corresponde a
um paciente e a uma única série de angio-TC; logo, partir por exame = partir por paciente.
(Verificado na EDA comparando PatientID dos cabeçalhos DICOM.)

Rótulo de exame do TP1 (binário): pe = 1 se negative_exam_for_pe == 0 e indeterminate == 0;
pe = 0 se negative_exam_for_pe == 1. Exames 'indeterminate' (qualidade insuficiente para
laudo) não têm verdade binária e são excluídos da amostra do TP1, mas recebem partição
para uso futuro.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

EXAM_LABELS = [
    "negative_exam_for_pe", "indeterminate", "qa_motion", "qa_contrast", "flow_artifact",
    "rv_lv_ratio_gte_1", "rv_lv_ratio_lt_1", "leftsided_pe", "rightsided_pe", "central_pe",
    "chronic_pe", "acute_and_chronic_pe", "true_filling_defect_not_pe",
]
STUDY, SERIES, SOP = "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"


def load_train(csv_path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def exam_table(train: pd.DataFrame) -> pd.DataFrame:
    labels = [c for c in EXAM_LABELS if c in train.columns]
    g = train.groupby(STUDY)
    exams = g[labels].first()
    exams[SERIES] = g[SERIES].first()
    exams["n_slices"] = g.size()
    exams["n_pos_slices"] = g["pe_present_on_image"].sum()
    exams["n_series"] = g[SERIES].nunique()
    exams["label3"] = np.select(
        [exams["negative_exam_for_pe"] == 1, exams["indeterminate"] == 1],
        ["negative", "indeterminate"], default="pe")
    exams["pe"] = (exams["label3"] == "pe").astype(int)
    return exams.reset_index()


def make_full_split(exams: pd.DataFrame, test_frac: float, val_frac: float, seed: int) -> pd.DataFrame:
    """Partição train/val/test do treino oficial completo, estratificada por label3."""
    ids = exams[STUDY].to_numpy(dtype=object)
    strat = exams["label3"].to_numpy(dtype=object)
    rest, test = train_test_split(ids, test_size=test_frac, stratify=strat, random_state=seed)
    rest_strat = exams.set_index(STUDY).loc[rest, "label3"].to_numpy(dtype=object)
    train, val = train_test_split(rest, test_size=val_frac / (1 - test_frac),
                                  stratify=rest_strat, random_state=seed)
    split = pd.Series("train", index=ids)
    split[val] = "val"
    split[test] = "test"
    out = exams[[STUDY, "label3", "pe", "n_slices"]].copy()
    out["split"] = out[STUDY].map(split)
    return out


def _stratified_take(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df
    take, _ = train_test_split(df, train_size=n, stratify=df["pe"], random_state=seed)
    return take


def make_sample(split: pd.DataFrame, n_dev: int, n_test: int, seed: int) -> pd.DataFrame:
    binary = split[split["label3"] != "indeterminate"]
    dev = _stratified_take(binary[binary["split"].isin(["train", "val"])], n_dev, seed)
    test = _stratified_take(binary[binary["split"] == "test"], n_test, seed)
    dev = dev.assign(subset="dev")
    test = test.assign(subset="test")
    return pd.concat([dev, test]).sort_values(["subset", STUDY]).reset_index(drop=True)
