"""Protocolo experimental: validação cruzada aninhada por exame, métricas e bootstrap.

* Dobras externas: StratifiedGroupKFold (grupo = exame/paciente) -> estimativa de desempenho.
* Dobras internas: StratifiedGroupKFold só com o treino da dobra externa -> ajuste de
  hiperparâmetros (RandomizedSearchCV, AUC-ROC) e escolha do limiar (Youden) sobre
  predições fora-da-dobra internas. A dobra externa de teste nunca participa de escolha alguma.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold

from .models import build


def scores_of(est, X) -> np.ndarray:
    if hasattr(est, "predict_proba"):
        try:
            return est.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return est.decision_function(X)


def youden_threshold(y, s) -> float:
    if len(np.unique(y)) < 2 or np.allclose(s, s[0]):
        return np.inf  # constante -> prediz sempre negativo (classe majoritária)
    fpr, tpr, thr = roc_curve(y, s)
    return float(thr[np.argmax(tpr - fpr)])


def metrics(y, s, thr) -> dict:
    y = np.asarray(y)
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "auc_roc": roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan,
        "auc_pr": average_precision_score(y, s) if len(np.unique(y)) > 1 else np.nan,
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "f1": f1_score(y, pred, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "accuracy": (tp + tn) / len(y),
        "prevalence": y.mean(),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    return out


def fit_tuned(model: str, X, y, groups, seed: int, cfg_cv: dict, imbalance: str,
              model_jobs: int = 1, search_jobs: int = -1):
    """Ajusta hiperparâmetros + limiar usando SOMENTE (X, y). Retorna (estimador, limiar, params)."""
    est, space = build(model, seed, imbalance, model_jobs)
    inner = StratifiedGroupKFold(cfg_cv["inner_folds"], shuffle=True, random_state=seed)
    splits = list(inner.split(X, y, groups))
    if space:
        search = RandomizedSearchCV(est, space, n_iter=cfg_cv["n_iter"], scoring="roc_auc",
                                    cv=splits, random_state=seed, n_jobs=search_jobs, refit=False)
        search.fit(X, y)
        params = search.best_params_
        est.set_params(**params)
    else:
        params = {}
    # Limiar: Youden sobre predições fora-da-dobra internas (sem tocar o teste externo).
    oof = np.zeros(len(y), dtype=float)
    for tr, va in splits:
        m = clone(est).fit(X[tr], y[tr])
        oof[va] = scores_of(m, X[va])
    thr = youden_threshold(y, oof)
    est.fit(X, y)
    return est, thr, params


def exam_folds(exam_ids, exam_y, n_folds: int, seed: int) -> pd.Series:
    """Atribui cada exame a uma dobra externa (fixa, reutilizada em todos os experimentos)."""
    exam_ids, exam_y = np.asarray(exam_ids), np.asarray(exam_y)
    sgkf = StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)
    fold = np.empty(len(exam_ids), dtype=int)
    for k, (_, te) in enumerate(sgkf.split(exam_ids, exam_y, exam_ids)):
        fold[te] = k
    return pd.Series(fold, index=exam_ids, name="fold")


def splits_from_folds(groups, fold_of_group: pd.Series):
    f = pd.Series(np.asarray(groups)).map(fold_of_group).values
    return [(np.where(f != k)[0], np.where(f == k)[0]) for k in sorted(np.unique(f))]


def nested_cv(X, y, groups, model: str, seed: int, cfg_cv: dict, imbalance="class_weight",
              model_jobs: int = 1, search_jobs: int = -1, outer_splits=None,
              n_iter: int | None = None) -> dict:
    X, y, groups = np.asarray(X, dtype=np.float32), np.asarray(y), np.asarray(groups)
    if n_iter is not None:
        cfg_cv = {**cfg_cv, "n_iter": n_iter}
    if outer_splits is None:
        outer = StratifiedGroupKFold(cfg_cv["outer_folds"], shuffle=True, random_state=seed)
        outer_splits = list(outer.split(X, y, groups))
    fold_rows, oof = [], np.full(len(y), np.nan)
    for k, (tr, te) in enumerate(outer_splits):
        assert not set(groups[tr]) & set(groups[te]), "vazamento: grupo em treino e teste"
        t0 = time.time()
        est, thr, params = fit_tuned(model, X[tr], y[tr], groups[tr], seed + k, cfg_cv,
                                     imbalance, model_jobs, search_jobs)
        s = scores_of(est, X[te])
        oof[te] = s
        row = {"fold": k, **metrics(y[te], s, thr), "threshold": thr,
               "fit_seconds": time.time() - t0, "params": str(params)}
        fold_rows.append(row)
    folds = pd.DataFrame(fold_rows)
    return {"folds": folds, "oof": oof, "splits": outer_splits}


def summarize(folds: pd.DataFrame, cols=("auc_roc", "auc_pr", "sensitivity", "specificity",
                                         "f1", "balanced_acc")) -> dict:
    out = {}
    for c in cols:
        out[f"{c}_mean"] = folds[c].mean()
        out[f"{c}_std"] = folds[c].std(ddof=1)
    return out


def bootstrap_ci(y, s, metric_fn=roc_auc_score, n_boot=2000, seed=0, alpha=0.05):
    """IC por bootstrap com reamostragem por exame (cada linha = um exame)."""
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(metric_fn(y[idx], s[idx]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric_fn(y, s)), float(lo), float(hi)
