"""Testes do pipeline com dados sintéticos (rodam sem os dados reais):  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from pe_baseline.config import load_config  # noqa: E402
from pe_baseline.data import exam_table, make_full_split, make_sample  # noqa: E402
from pe_baseline.dicom_io import read_slice, window  # noqa: E402
from pe_baseline.evaluate import exam_folds, splits_from_folds  # noqa: E402
from pe_baseline.extract import run_extraction  # noqa: E402
from pe_baseline.preprocess import compute_masks, resize_hu  # noqa: E402
from synthetic import generate, make_image  # noqa: E402


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    generate(root, n_exams=30, n_slices=6, size=96, seed=1)
    return root


def test_hu_conversion(synth):
    f = next((synth / "train").rglob("*.dcm"))
    sl = read_slice(f)
    assert sl.hu.min() >= -1024 and sl.hu.max() < 1000   # intercept -1024 aplicado
    assert "z" in sl.meta and sl.meta["Modality"] == "CT"


def test_window_range():
    w = window(np.array([-1000.0, 100.0, 2000.0]), 100, 700)
    assert w[0] == 0 and w[-1] == 1 and 0 < w[1] < 1


def test_defect_mask_specific():
    rng = np.random.default_rng(0)
    cfg = load_config()
    pos = resize_hu(make_image(rng, 128, True), 256)
    neg = resize_hu(make_image(rng, 128, False), 256)
    mp, mn = compute_masks(pos, cfg["preprocess"]), compute_masks(neg, cfg["preprocess"])
    assert mp.defect.sum() > 10 * max(mn.defect.sum(), 1)
    assert not mp.roi_fallback and mp.vessel.sum() > 0


def test_split_is_by_exam_and_fixed(synth):
    exams = exam_table(pd.read_csv(synth / "train.csv"))
    s1 = make_full_split(exams, 0.2, 0.1, 42)
    s2 = make_full_split(exams, 0.2, 0.1, 42)
    assert s1.equals(s2)                                   # determinístico
    assert s1["StudyInstanceUID"].is_unique                # um exame -> uma partição
    sample = make_sample(s1, 15, 5, 42)
    assert set(sample.loc[sample.subset == "test", "StudyInstanceUID"]) <= \
        set(s1.loc[s1.split == "test", "StudyInstanceUID"])
    assert "indeterminate" not in set(sample["label3"])


def test_folds_have_no_group_leak():
    ids = np.array([f"e{i}" for i in range(40)])
    y = np.array([0, 1] * 20)
    folds = exam_folds(ids, y, 5, 0)
    groups = np.repeat(ids, 3)                             # 3 cortes por exame
    for tr, te in splits_from_folds(groups, folds):
        assert not set(groups[tr]) & set(groups[te])


def test_extraction_end_to_end(synth, tmp_path):
    cfg = load_config()
    cfg["data"]["root"] = str(synth)
    cfg["sample"]["slice_stride"] = 2
    train = pd.read_csv(synth / "train.csv")
    studies = sorted(train["StudyInstanceUID"].unique())[:4]
    files = run_extraction(cfg, train, studies, tmp_path, chunk_size=2, n_jobs=1, verbose=False)
    df = pd.concat([pd.read_parquet(f) for f in files])
    assert len(df) == 4 * 3 and (df["status"] == "ok").all()
    assert {"int_defect_per_vessel", "tex_glcm_contrast_d1", "grad_hog000", "wav_A_energy"} <= set(df.columns)
    # retomável: rodar de novo não reprocessa
    again = run_extraction(cfg, train, studies, tmp_path, chunk_size=2, n_jobs=1, verbose=False)
    assert again == files


def test_full_extraction_with_volumes(synth, tmp_path):
    """TP2: todos os cortes, treino + 'teste oficial' (sem rótulo), volumes uint8 por exame."""
    from pe_baseline.extract import load_volume, to_uint8
    cfg = load_config()
    cfg["data"]["root"] = str(synth)
    cfg["full"]["test_dir"] = "train"            # o sintético só tem train/; reaproveita a pasta
    train = pd.read_csv(synth / "train.csv")
    studies = sorted(train["StudyInstanceUID"].unique())[:4]
    test_ids = studies[2:]
    table = train.assign(source=np.where(train["StudyInstanceUID"].isin(test_ids), "test", "train"))
    table.loc[table["source"] == "test", "pe_present_on_image"] = -1
    files = run_extraction(cfg, table, studies, tmp_path, chunk_size=2, n_jobs=1, verbose=False,
                           stride=1, save_volumes=True)
    df = pd.concat([pd.read_parquet(f) for f in files])
    assert len(df) == 4 * 6 and (df["status"] == "ok").all()          # todos os cortes
    assert set(df.loc[df["source"] == "test", "pe_present_on_image"]) == {-1}
    assert not (tmp_path / "_staging").exists()
    for study, g in df.groupby("StudyInstanceUID"):
        npz = tmp_path / g["vol_file"].iloc[0]
        vol = load_volume(npz, study)
        assert vol.dtype == np.uint8 and vol.shape == (6, 256, 256)
        assert sorted(g["vol_idx"]) == list(range(6))
        # a fatia vol_idx corresponde ao SOP certo (ordem por z preservada)
        r = g.iloc[3]
        series = train.loc[train["SOPInstanceUID"] == r["SOPInstanceUID"], "SeriesInstanceUID"].iloc[0]
        sl = read_slice(synth / "train" / study / series / f"{r['SOPInstanceUID']}.dcm")
        lvl, wid = cfg["preprocess"]["windows"]["ctpa"]
        assert np.array_equal(vol[int(r["vol_idx"])], to_uint8(resize_hu(sl.hu, 256), lvl, wid))
        assert g.sort_values("vol_idx")["z_rank"].is_monotonic_increasing
