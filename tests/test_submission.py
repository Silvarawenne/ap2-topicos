"""Submissão tardia: regras de consistência, formato, calibração e métrica (dados sintéticos)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pe_tp2 import submission as S  # noqa: E402

STUDY, SOP = S.STUDY, S.SOP


def fake_train(n_exams=60, n_slices=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_exams):
        kind = ["neg", "pe", "pe", "neg", "indet"][e % 5]
        lab = dict.fromkeys(["pe_present_on_image", *S.EXAM_LABELS], 0)
        lab["negative_exam_for_pe"] = int(kind == "neg")
        lab["indeterminate"] = int(kind == "indet")
        if kind == "pe":
            lab.update(rightsided_pe=1, central_pe=int(e % 3 == 0), rv_lv_ratio_lt_1=int(e % 2 == 0),
                       rv_lv_ratio_gte_1=int(e % 2 == 1), chronic_pe=int(e % 7 == 0))
        pos = set(rng.choice(n_slices, 4, replace=False)) if kind == "pe" else set()
        for i in range(n_slices):
            rows.append({STUDY: f"E{e:03d}", SOP: f"E{e:03d}_{i:03d}", **lab,
                         "pe_present_on_image": int(i in pos)})
    train = pd.DataFrame(rows)
    ex = train.groupby(STUDY).first().reset_index()
    split = pd.DataFrame({STUDY: ex[STUDY], "split": ["train", "train", "val", "test"] * (len(ex) // 4)})
    return train, split


def test_prior_correction_inverts_subsampling():
    # modelo calibrado no treino subamostrado: chance = verdadeira / q  ->  corrigida = verdadeira
    q, p_true = 0.1, np.array([0.01, 0.2, 0.5])
    odds_model = p_true / (1 - p_true) / q
    p_model = odds_model / (1 + odds_model)
    assert np.allclose(S.prior_correct(p_model, q), p_true, atol=1e-6)


@pytest.mark.parametrize("seed", range(5))
def test_any_input_becomes_consistent(seed):
    rng = np.random.default_rng(seed)
    studies = pd.Index([f"S{i}" for i in range(80)])
    exams = pd.DataFrame(rng.random((80, 9)), index=studies, columns=S.EXAM_LABELS)
    images = pd.DataFrame({STUDY: np.repeat(studies, 5), SOP: [f"x{i}" for i in range(400)],
                           "p": rng.random(400)})
    positive = pd.Series(rng.random(80) > 0.5, index=studies)
    before = S.check_consistency(exams, images)
    assert sum(before.values()) > 0                      # entrada aleatória viola as regras
    e, im = S.make_consistent(exams, images, positive)
    assert S.check_consistency(e, im) == dict.fromkeys(before, 0)
    assert ((im.groupby(STUDY)["p"].max().reindex(positive.index) > 0.5) == positive).all()
    assert np.abs(e.to_numpy() - exams.to_numpy()).max() <= 1   # só probabilidades válidas
    assert e.to_numpy().min() >= 0 and e.to_numpy().max() <= 1


def test_build_format_and_metric():
    train, split = fake_train()
    pri = S.train_priors(train, split, neg_ratio=1)
    assert 0 < pri["q"] <= 1 and pri["given_pe"]["rightsided_pe"] == 1.0
    test_ids = split.loc[split["split"] == "test", STUDY]
    truth = train[train[STUDY].isin(test_ids) & (train["indeterminate"] == 0)]
    # predição "boa": corte = rótulo suavizado; exame = máximo
    sl = truth[[STUDY, SOP]].assign(p=np.where(truth["pe_present_on_image"] == 1, 0.9, 0.05))
    score = sl.groupby(STUDY)["p"].max()
    cal = S.fit_platt((score > 0.5).astype(int), score)
    e, im = S.build(sl, score, pri, cal)
    assert S.check_consistency(e, im) == dict.fromkeys(S.check_consistency(e, im), 0)
    long = S.to_long(e, im)
    assert len(long) == len(im) + 9 * len(e) and long["id"].is_unique
    assert long["id"].str.endswith("_negative_exam_for_pe").sum() == len(e)
    good = S.weighted_log_loss(e, im, truth)["weighted_log_loss"]
    e0, im0 = S.build(sl, score, pri, cal, constant=True)
    base = S.weighted_log_loss(e0, im0, truth)["weighted_log_loss"]
    assert good < base


def test_script_end_to_end(tmp_path):
    """Usa o mundo sintético do TP2 (extração + uma execução de ajuste fino curta)."""
    import yaml
    pytest.importorskip("torch")
    sys.path.insert(0, str(ROOT / "tests"))
    from pe_baseline.config import load_config
    from pe_baseline.data import exam_table
    from pe_baseline.extract import run_extraction
    from synthetic import generate
    data = generate(tmp_path / "data", n_exams=24, n_slices=6, size=96, seed=3, indet=2)
    train = pd.read_csv(data / "train.csv").assign(source="train")
    cfg = load_config()
    cfg["data"]["root"] = str(data)
    studies = sorted(train[STUDY].unique())
    off = studies[-3:]
    table = train.copy()
    table.loc[table[STUDY].isin(off), "source"] = "test"
    cfg["full"]["test_dir"] = "train"
    run_extraction(cfg, table, studies, tmp_path / "inputs" / "s0" / "slices", chunk_size=8, n_jobs=1,
                   verbose=False, stride=1, save_volumes=True)
    exams = exam_table(train[~train[STUDY].isin(off)])
    sp = exams[[STUDY, "label3", "pe", "n_slices"]].copy()
    sp["split"] = sp.groupby("label3").cumcount().map(lambda i: ["train", "val", "train", "test"][i % 4])
    sp.to_csv(tmp_path / "split.csv", index=False)
    cfg["split"]["file"] = str(tmp_path / "split.csv")
    cfg["tp2"].update(inputs=str(tmp_path / "inputs"), out_dir=str(tmp_path / "runs"), epochs=1,
                      batch_size=4, eval_batch_size=8, num_workers=0, neg_ratio=1, n_bootstrap=20)
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "10_tp2.py"), "finetune", "--model", "resnet50",
                        "--seeds", "0", "--sequential", "--max-steps", "1", "--config", str(tmp_path / "cfg.yaml"),
                        "--no-pretrained"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    # o "test.csv oficial" e o sample_submission do mundo sintético
    test = train[train[STUDY].isin(off)][[STUDY, "SeriesInstanceUID", SOP]]
    test.to_csv(data / "test.csv", index=False)
    ids = list(test[SOP]) + [f"{s}_{c}" for s in off for c in S.EXAM_LABELS]
    pd.DataFrame({"id": ids, "label": 0.5}).to_csv(data / "sample_submission.csv", index=False)
    run = tmp_path / "runs" / "ft_resnet50_cosine_main_s0"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "11_submission.py"), "--runs", str(run), str(run),
                        "--out", str(tmp_path / "sub"), "--config", str(tmp_path / "cfg.yaml"),
                        "--train-csv", str(data / "train.csv"), "--test-csv", str(data / "test.csv")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    sub = pd.read_csv(tmp_path / "sub" / "submission.csv")
    assert list(sub["id"]) == ids and sub["label"].between(0, 1).all()
    rep = __import__("json").loads((tmp_path / "sub" / "report.json").read_text(encoding="utf-8"))
    assert sum(rep["official"]["consistency"].values()) == 0
    # inferência direto dos DICOMs (notebook de submissão): mesma saída que a extração + treino
    pri_json = tmp_path / "priors.json"          # como o notebook 03 grava (sem train.csv na submissão)
    pri_json.write_text(__import__("json").dumps(rep["priors"]), encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "13_kaggle_infer.py"), "--runs", str(run),
                        "--out", str(tmp_path / "sub13"), "--config", str(tmp_path / "cfg.yaml"),
                        "--data-root", str(data), "--workers", "0", "--batch", "4", "--priors", str(pri_json)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    sub13 = pd.read_csv(tmp_path / "sub13" / "submission.csv")
    assert list(sub13["id"]) == ids
    rep13 = __import__("json").loads((tmp_path / "sub13" / "submission_report.json").read_text(encoding="utf-8"))
    par = rep13["parity_vs_extraction"][run.name]
    assert par["n_exams"] == len(off) and par["max_abs_diff"] < 1e-4, par
    assert sum(rep13["consistency"].values()) == 0 and rep13["exams_without_prediction"] == 0
    assert np.allclose(sub13["label"], sub["label"], atol=1e-4)
