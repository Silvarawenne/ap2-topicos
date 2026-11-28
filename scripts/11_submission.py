"""Submissão tardia (10 rótulos) a partir de uma ou mais execuções do TP2 (sementes = média).

  python scripts/11_submission.py --runs runs/ft_resnet50_cosine_main_s0 runs/ft_resnet50_cosine_main_s1 \
      --out results/submission_resnet50

Entradas de cada execução: exam_{val,test,official_test}.csv e slices_{test,official_test}.parquet.
Do desafio: train.csv (prevalências e verdade do nosso teste) e test.csv (todas as imagens a prever).
Saídas em --out: submission.csv (id,label), report.json (calibração, priors, verificação das regras,
log loss ponderada no NOSSO teste com e sem modelo). Não usa GPU nem internet: roda num notebook
do Kaggle com a internet desligada (exigência da competição para submeter).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from pe_baseline.config import data_path, load_config, repo_path  # noqa: E402
from pe_tp2 import submission as S  # noqa: E402

STUDY, SOP = S.STUDY, S.SOP


def load_runs(runs, k):
    """Média entre execuções: escore de exame (top-k) e probabilidade por corte."""
    col = f"top{k}"
    ex, sl = {}, {}
    ex["val"] = S.val_exam_scores(runs, k)
    for split in ("test", "official_test"):
        parts = [pd.read_csv(Path(r) / f"exam_{split}.csv")[[STUDY, "pe", col]] for r in runs]
        ex[split] = pd.concat(parts).groupby(STUDY).agg(pe=("pe", "first"), score=(col, "mean"))
    for split in ("test", "official_test"):
        parts = [pd.read_parquet(Path(r) / f"slices_{split}.parquet") for r in runs]
        sl[split] = pd.concat(parts).groupby([STUDY, SOP], as_index=False)["p"].mean()
    return ex, sl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--train-csv", default=None)
    ap.add_argument("--test-csv", default=None)
    ap.add_argument("--sample", default=None, help="sample_submission.csv para conferir os ids")
    ap.add_argument("--neg-ratio", type=float, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    t = cfg["tp2"]
    neg_ratio = args.neg_ratio if args.neg_ratio is not None else t["neg_ratio"]
    train_csv = Path(args.train_csv) if args.train_csv else data_path(cfg, cfg["data"]["train_csv"])
    test_csv = Path(args.test_csv) if args.test_csv else data_path(cfg, cfg["full"]["test_csv"])
    sample = Path(args.sample) if args.sample else test_csv.with_name("sample_submission.csv")
    out = repo_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_csv)
    split = pd.read_csv(repo_path(cfg["split"]["file"]))
    priors = S.train_priors(train, split, neg_ratio)
    ex, sl = load_runs(args.runs, t["topk"])
    cal = S.fit_platt(ex["val"]["pe"], ex["val"]["score"])
    rep = {"runs": [str(r) for r in args.runs], "neg_ratio": neg_ratio, "priors": priors, "platt_val": cal}

    # nosso teste (exames com verdade binária): log loss ponderada com e sem modelo
    ids = ex["test"].index
    truth = train[train[STUDY].isin(ids)]
    for name, const in (("model", False), ("prevalence_only", True)):
        e, im = S.build(sl["test"], ex["test"]["score"], priors, cal, truth[[STUDY, SOP]], constant=const)
        rep[f"our_test_{name}"] = {**S.weighted_log_loss(e, im, truth), "consistency": S.check_consistency(e, im)}
    print("log loss ponderada no nosso teste: modelo = {:.4f} | só prevalências = {:.4f}".format(
        rep["our_test_model"]["weighted_log_loss"], rep["our_test_prevalence_only"]["weighted_log_loss"]))

    # teste oficial
    test = pd.read_csv(test_csv)
    e, im = S.build(sl["official_test"], ex["official_test"]["score"], priors, cal, test[[STUDY, SOP]])
    sub, cons = S.finalize(e, im, pd.read_csv(sample) if sample.exists() else None)
    rep["official"] = {"rows": len(sub), "exams": len(e), "images": len(im), "consistency": cons,
                       "exams_positive": int((e["negative_exam_for_pe"] < 0.5).sum()),
                       "images_with_prediction": int(test[SOP].isin(sl["official_test"][SOP]).sum()),
                       "checked_against_sample": bool(sample.exists())}
    sub.to_csv(out / "submission.csv", index=False)
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    print(json.dumps(rep["official"], indent=2))
    print("gravado:", out / "submission.csv")


if __name__ == "__main__":
    main()
