"""Orquestração dos experimentos do TP1 (todas as tabelas saem daqui).

Fluxo:
  1. Grade descritor x modelo no nível de EXAME (pooling de características), CV aninhada.
  2. Classificador de CORTE + agregação de probabilidades corte->exame, mesmas dobras.
  3. Ablações sobre a melhor configuração (desbalanceamento, pooling, ROI).
  4. Avaliação final ÚNICA no conjunto de teste (configuração escolhida só pela CV).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .aggregate import POOL_STATS, family_columns, feature_columns, pool_features, pool_probabilities
from .config import repo_path
from .data import STUDY
from .evaluate import (bootstrap_ci, exam_folds, fit_tuned, metrics, nested_cv, scores_of,
                       splits_from_folds, summarize)
from .extract import load_slice_features

SUBGROUPS_POS = ["central_pe", "leftsided_pe", "rightsided_pe", "chronic_pe",
                 "acute_and_chronic_pe", "rv_lv_ratio_gte_1", "rv_lv_ratio_lt_1"]
SUBGROUPS_ALL = ["qa_motion", "qa_contrast", "flow_artifact", "true_filling_defect_not_pe"]


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def load_tables(cfg: dict):
    feat_dir = repo_path(cfg["features"]["out_dir"])
    slices = load_slice_features(feat_dir / "slices")
    sample = pd.read_csv(repo_path(cfg["sample"]["file"]))
    exam_info = pd.read_csv(feat_dir / "exams.csv")
    slices = slices.merge(sample[[STUDY, "subset"]], on=STUDY, how="inner")
    n_err = int((slices["status"] != "ok").sum())
    errors = slices.loc[slices["status"] != "ok", [STUDY, "SOPInstanceUID", "status"]]
    slices = slices[slices["status"] == "ok"].reset_index(drop=True)
    pooled = pool_features(slices)
    exams = sample.merge(pooled, on=STUDY, how="inner")
    exams = exams.merge(exam_info.drop(columns=["label3", "pe", "n_slices"], errors="ignore"),
                        on=STUDY, how="left")
    log(f"cortes ok={len(slices)} erros={n_err} | exames={len(exams)} "
        f"(dev={int((exams.subset == 'dev').sum())}, test={int((exams.subset == 'test').sum())})")
    return slices, exams, errors


def _pool_cols(cols, stats):
    return [c for c in cols if c.rsplit("__", 1)[-1] in stats]


def exam_grid(dev: pd.DataFrame, cfg: dict, folds: pd.Series, out: Path):
    seed, cv = cfg["seed"], cfg["cv"]
    all_cols = feature_columns(dev)
    rows, fold_tables, oofs = [], [], []
    splits = splits_from_folds(dev[STUDY], folds)
    y, g = dev["pe"].values, dev[STUDY].values
    for fam in cfg["experiments"]["families"]:
        cols = family_columns(all_cols, fam)
        X = dev[cols].values
        for model in cfg["experiments"]["exam_models"]:
            t0 = time.time()
            res = nested_cv(X, y, g, model, seed, cv, outer_splits=splits, search_jobs=cv["n_jobs"])
            summ = summarize(res["folds"])
            rows.append({"family": fam, "model": model, "n_features": len(cols), **summ,
                         "seconds": time.time() - t0})
            fold_tables.append(res["folds"].assign(family=fam, model=model))
            oofs.append(pd.DataFrame({STUDY: g, "y": y, "score": res["oof"], "family": fam,
                                      "model": model, "fold": dev[STUDY].map(folds).values}))
            log(f"exame | {fam:9s} x {model:7s} AUC={summ['auc_roc_mean']:.3f}±{summ['auc_roc_std']:.3f} "
                f"({time.time() - t0:.0f}s)")
    table = pd.DataFrame(rows)
    table.to_csv(out / "table_exam_cv.csv", index=False)
    pd.concat(fold_tables).to_csv(out / "table_exam_cv_folds.csv", index=False)
    pd.concat(oofs).to_parquet(out / "oof_exam.parquet", index=False)
    return table


def slice_level(slices_dev: pd.DataFrame, dev: pd.DataFrame, cfg: dict, folds: pd.Series, out: Path):
    seed, cv = cfg["seed"], cfg["cv"]
    cols = family_columns(feature_columns(slices_dev), "all") + ["z_rank"]
    X, y, g = slices_dev[cols].values, slices_dev["pe_present_on_image"].values, slices_dev[STUDY].values
    splits = splits_from_folds(g, folds)
    exam_y = dev.set_index(STUDY)["pe"]
    rows, agg_rows, oofs = [], [], []
    for model in cfg["experiments"]["slice_models"]:
        t0 = time.time()
        res = nested_cv(X, y, g, model, seed, cv, outer_splits=splits, model_jobs=cv["n_jobs"],
                        search_jobs=1, n_iter=cv["slice_n_iter"])
        summ = summarize(res["folds"])
        rows.append({"model": model, "n_features": len(cols), **summ, "seconds": time.time() - t0})
        log(f"corte | {model:7s} AUC={summ['auc_roc_mean']:.3f}±{summ['auc_roc_std']:.3f} ({time.time() - t0:.0f}s)")
        for k, (_, te) in enumerate(splits):
            agg = pool_probabilities(g[te], res["oof"][te], k=cv["topk"])
            yy = agg[STUDY].map(exam_y).values
            for how in ["max", f"top{cv['topk']}", "mean"]:
                agg_rows.append({"model": model, "pool": how, "fold": k,
                                 "auc_roc": roc_auc_score(yy, agg[how]),
                                 "auc_pr": average_precision_score(yy, agg[how])})
        oofs.append(pd.DataFrame({STUDY: g, "y": y, "score": res["oof"], "model": model}))
    table = pd.DataFrame(rows)
    table.to_csv(out / "table_slice_cv.csv", index=False)
    agg = pd.DataFrame(agg_rows)
    agg_sum = agg.groupby(["model", "pool"]).agg(
        auc_roc_mean=("auc_roc", "mean"), auc_roc_std=("auc_roc", "std"),
        auc_pr_mean=("auc_pr", "mean"), auc_pr_std=("auc_pr", "std")).reset_index()
    agg_sum.to_csv(out / "table_slice2exam_cv.csv", index=False)
    pd.concat(oofs).to_parquet(out / "oof_slice.parquet", index=False)
    return table, agg_sum


def ablations(dev: pd.DataFrame, cfg: dict, folds: pd.Series, best_family: str, best_model: str, out: Path):
    seed, cv = cfg["seed"], cfg["cv"]
    all_cols = feature_columns(dev)
    splits = splits_from_folds(dev[STUDY], folds)
    y, g = dev["pe"].values, dev[STUDY].values
    rows = []

    def run(name, variant, cols, model=best_model, imbalance="class_weight"):
        res = nested_cv(dev[cols].values, y, g, model, seed, cv, imbalance=imbalance,
                        outer_splits=splits, search_jobs=cv["n_jobs"])
        s = summarize(res["folds"])
        rows.append({"ablation": name, "variant": variant, "model": model, "n_features": len(cols), **s})
        log(f"ablação | {name}: {variant} AUC={s['auc_roc_mean']:.3f}±{s['auc_roc_std']:.3f}")

    base_cols = family_columns(all_cols, best_family)
    for imb in ["none", "class_weight", "smote"]:
        run("desbalanceamento", imb, base_cols, imbalance=imb)
    for stats in [("mean",), ("mean", "max"), POOL_STATS]:
        run("pooling", "+".join(stats), _pool_cols(base_cols, stats))
    run("roi", "com ROI (int+tex)", family_columns(all_cols, "roi_matched"))
    run("roi", "sem ROI (int+tex)", family_columns(all_cols, "noroi"))
    table = pd.DataFrame(rows)
    table.to_csv(out / "table_ablations.csv", index=False)
    return table


def subgroup_analysis(test: pd.DataFrame, score_col: str, thr: float) -> pd.DataFrame:
    rows = []
    pred = (test[score_col] >= thr).astype(int)
    pos = test["pe"] == 1
    for c in SUBGROUPS_POS:
        if c in test:
            m = pos & (test[c] == 1)
            rows.append({"subgroup": c, "n": int(m.sum()), "metric": "sensibilidade",
                         "value": pred[m].mean() if m.any() else np.nan})
    for c in SUBGROUPS_ALL:
        if c in test:
            for v in [0, 1]:
                m = test[c] == v
                yy = test.loc[m, "pe"]
                rows.append({"subgroup": f"{c}={v}", "n": int(m.sum()), "metric": "auc_roc",
                             "value": roc_auc_score(yy, test.loc[m, score_col]) if yy.nunique() > 1 else np.nan})
    if "meta_Manufacturer" in test:
        for v, d in test.groupby("meta_Manufacturer"):
            rows.append({"subgroup": f"fabricante={v}", "n": len(d), "metric": "auc_roc",
                         "value": roc_auc_score(d["pe"], d[score_col]) if d["pe"].nunique() > 1 else np.nan})
    return pd.DataFrame(rows)


def grouped_permutation_importance(est, X, y, groups_of_cols: dict, cols: list, seed: int, n_repeats=10):
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, scores_of(est, X))
    rows = []
    for name, gcols in groups_of_cols.items():
        idx = [cols.index(c) for c in gcols if c in cols]
        if not idx:
            continue
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[:, idx] = X[perm][:, idx]   # permuta o bloco inteiro junto
            drops.append(base - roc_auc_score(y, scores_of(est, Xp)))
        rows.append({"group": name, "n_features": len(idx), "auc_drop_mean": np.mean(drops),
                     "auc_drop_std": np.std(drops, ddof=1)})
    return pd.DataFrame(rows).sort_values("auc_drop_mean", ascending=False)


def final_evaluation(slices: pd.DataFrame, exams: pd.DataFrame, cfg: dict, best_family: str,
                     best_model: str, best_slice_model: str, out: Path):
    seed, cv = cfg["seed"], cfg["cv"]
    dev, test = exams[exams.subset == "dev"], exams[exams.subset == "test"]
    cols = family_columns(feature_columns(exams), best_family)
    rows, preds = [], test[[STUDY, "pe"]].copy()
    for name, model in [("baseline trivial", "dummy"), (f"{best_family}+{best_model}", best_model)]:
        est, thr, params = fit_tuned(model, dev[cols].values.astype(np.float32), dev["pe"].values,
                                     dev[STUDY].values, seed, cv, "class_weight", search_jobs=cv["n_jobs"])
        s = scores_of(est, test[cols].values.astype(np.float32))
        m = metrics(test["pe"].values, s, thr)
        auc, lo, hi = bootstrap_ci(test["pe"].values, s, roc_auc_score, cv["n_bootstrap"], seed) \
            if model != "dummy" else (0.5, 0.5, 0.5)
        ap, aplo, aphi = bootstrap_ci(test["pe"].values, s, average_precision_score, cv["n_bootstrap"], seed) \
            if model != "dummy" else (m["prevalence"],) * 3
        rows.append({"config": name, **m, "threshold": thr, "auc_roc_ci_low": lo, "auc_roc_ci_high": hi,
                     "auc_pr_ci_low": aplo, "auc_pr_ci_high": aphi, "params": str(params)})
        preds[f"score_{model}"] = s
        if model == best_model:
            final_est, final_thr = est, thr

    # Classificador de corte (melhor da CV) -> probabilidades no teste -> agregação por exame
    scols = family_columns(feature_columns(slices), "all") + ["z_rank"]
    sdev, stest = slices[slices.subset == "dev"], slices[slices.subset == "test"]
    sest, sthr, _ = fit_tuned(best_slice_model, sdev[scols].values.astype(np.float32),
                              sdev["pe_present_on_image"].values, sdev[STUDY].values, seed,
                              {**cv, "n_iter": cv["slice_n_iter"]}, "class_weight",
                              model_jobs=cv["n_jobs"], search_jobs=1)
    ss = scores_of(sest, stest[scols].values.astype(np.float32))
    slice_m = metrics(stest["pe_present_on_image"].values, ss, sthr)
    stest_pred = stest[[STUDY, "SOPInstanceUID", "z_rank", "pe_present_on_image"]].assign(score=ss)
    stest_pred.to_parquet(out / "test_slice_predictions.parquet", index=False)
    agg = pool_probabilities(stest[STUDY].values, ss, k=cv["topk"])
    preds = preds.merge(agg.rename(columns=lambda c: c if c == STUDY else f"slice_{c}"), on=STUDY)
    for how in ["max", f"top{cv['topk']}"]:
        auc, lo, hi = bootstrap_ci(preds["pe"], preds[f"slice_{how}"], roc_auc_score, cv["n_bootstrap"], seed)
        rows.append({"config": f"corte({best_slice_model})->exame [{how}]", "auc_roc": auc,
                     "auc_roc_ci_low": lo, "auc_roc_ci_high": hi,
                     "auc_pr": average_precision_score(preds["pe"], preds[f"slice_{how}"])})
    table = pd.DataFrame(rows)
    table.to_csv(out / "table_test_final.csv", index=False)
    pd.DataFrame([{"level": "corte", "model": best_slice_model, **slice_m, "threshold": sthr}]).to_csv(
        out / "table_test_slice.csv", index=False)

    test_full = test.merge(preds.drop(columns=["pe"]), on=STUDY)
    test_full[[STUDY, "pe"] + [c for c in test_full if c.startswith(("score_", "slice_"))]
              + [c for c in SUBGROUPS_POS + SUBGROUPS_ALL if c in test_full]].to_csv(
        out / "test_predictions.csv", index=False)
    subgroup_analysis(test_full, f"score_{best_model}", final_thr).to_csv(out / "table_subgroups.csv", index=False)

    fam_groups = {f: family_columns(cols, f) for f in ["intensity", "texture", "gradient", "wavelet"]}
    fam_groups.update({f"pool:{s}": [c for c in cols if c.endswith(f"__{s}")] for s in POOL_STATS})
    imp = grouped_permutation_importance(final_est, test[cols].values.astype(np.float32),
                                         test["pe"].values, fam_groups, cols, seed)
    imp.to_csv(out / "table_importance_groups.csv", index=False)
    json.dump({"best_family": best_family, "best_model": best_model,
               "best_slice_model": best_slice_model, "threshold": final_thr},
              open(out / "final_config.json", "w"), indent=2)
    log("teste final:\n" + table[["config", "auc_roc", "auc_roc_ci_low", "auc_roc_ci_high", "auc_pr"]]
        .to_string(index=False))
    return table
