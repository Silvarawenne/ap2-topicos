"""Agregação por exame, métricas, IC por bootstrap (nível de exame = paciente) e testes pareados.

* Exame = paciente nesta base (DICOMs anonimizados; suposição declarada), então reamostrar
  exames é o bootstrap em nível de paciente exigido pelo enunciado.
* DeLong (DeLong et al., 1988), implementação rápida de Sun e Xu (2014): compara duas AUCs
  correlacionadas (mesmos exames). Bootstrap pareado: mesma reamostragem para os dois modelos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .index import STUDY


def exam_scores(study: np.ndarray, p: np.ndarray, k: int = 5) -> pd.DataFrame:
    """Probabilidade do exame = média das k maiores probabilidades de corte (como no TP1):
    coerente com a hierarquia (exame positivo <=> algum corte positivo) e menos sensível a um
    único corte ruidoso que o máximo."""
    df = pd.DataFrame({STUDY: study, "p": p}).sort_values([STUDY, "p"], ascending=[True, False])
    df["r"] = df.groupby(STUDY).cumcount()
    top = df[df["r"] < k].groupby(STUDY)["p"].mean().rename(f"top{k}")
    g = df.groupby(STUDY)["p"]
    return pd.concat([top, g.max().rename("max"), g.mean().rename("mean")], axis=1).reset_index()


def youden(y, s) -> float:
    fpr, tpr, thr = roc_curve(y, s)
    return float(thr[np.argmax(tpr - fpr)])


def metrics_at(y, s, thr) -> dict:
    y, s = np.asarray(y), np.asarray(s)
    pred = s >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    return {"auc_roc": roc_auc_score(y, s), "auc_pr": average_precision_score(y, s),
            "sensitivity": sens, "specificity": spec, "balanced_acc": (sens + spec) / 2,
            "f1": 2 * tp / max(2 * tp + fp + fn, 1), "threshold": thr,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(y)}


def bootstrap_ci(y, s, n_boot=2000, seed=0, alpha=0.05, fn=roc_auc_score):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(fn(y[i], s[i]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(fn(y, s)), float(lo), float(hi)


def paired_bootstrap(y, s1, s2, n_boot=2000, seed=0, fn=roc_auc_score):
    """Diferença fn(s1) - fn(s2) com IC 95% e p bilateral (proporção de reamostras com sinal
    oposto ao observado, x2)."""
    rng = np.random.default_rng(seed)
    y, s1, s2 = map(np.asarray, (y, s1, s2))
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            d.append(fn(y[i], s1[i]) - fn(y[i], s2[i]))
    d = np.asarray(d)
    obs = fn(y, s1) - fn(y, s2)
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": float(obs), "ci_lo": float(lo), "ci_hi": float(hi), "p": float(p)}


def _midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong(y, s1, s2) -> dict:
    """Teste de DeLong para duas AUCs correlacionadas (Sun e Xu, 2014)."""
    y = np.asarray(y)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    preds = np.vstack([s1, s2]).astype(float)
    aucs, v01, v10 = [], [], []
    for s in preds:
        tx, ty, tz = _midrank(s[pos]), _midrank(s[neg]), _midrank(np.concatenate([s[pos], s[neg]]))
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        v01.append((tz[:m] - tx) / n)
        v10.append(1.0 - (tz[m:] - ty) / m)
    v01, v10 = np.array(v01), np.array(v10)
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = aucs[0] - aucs[1]
    z = diff / np.sqrt(var) if var > 0 else 0.0
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "diff": float(diff),
            "z": float(z), "p": float(2 * st.norm.sf(abs(z)))}
