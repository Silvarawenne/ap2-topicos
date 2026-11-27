"""Agregação das execuções do TP2: tabelas por execução e por configuração, conjunto de sementes
e comparações pareadas no NOSSO teste (1.424 exames; exame = paciente).

* Seleção do "melhor" de cada família SEMPRE pela AUC média de validação, nunca pelo teste.
* Conjunto de sementes: média do escore de exame (top-k) das 3 sementes.
* Comparações pareadas (mesmos exames): DeLong (Sun e Xu, 2014) e bootstrap pareado por exame;
  p-valores corrigidos por Holm dentro da família de comparações planejadas.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .stats import bootstrap_ci, delong, paired_bootstrap

STUDY = "StudyInstanceUID"
NAMES = {"resnet50": "ResNet-50 (ImageNet)", "radimagenet_resnet50": "ResNet-50 (RadImageNet)",
         "deit_s": "DeiT-S (ImageNet)", "dinov2_s": "DINOv2 ViT-S/14"}


def config_of(m: dict) -> str:
    if m.get("regime") == "frozen":
        return f"frozen_{m['model']}"
    return f"ft_{m['model']}_{m['strategy']}_{m['aug']}"


def label_of(cfg: str) -> str:
    p = cfg.split("_")
    if p[0] == "frozen":
        return f"{NAMES.get('_'.join(p[1:]), cfg)} · congelado"
    aug, strat = p[-1], p[-2]
    return f"{NAMES.get('_'.join(p[1:-2]), cfg)} · ajuste fino · {strat} · aum. {aug}"


def collect(roots) -> pd.DataFrame:
    rows, seen = [], set()
    for root in roots:
        for f in sorted(Path(root).rglob("metrics.json")):
            m = json.loads(f.read_text(encoding="utf-8"))
            if "test" not in m or "val" not in m:
                continue
            cfg, seed = config_of(m), int(m["seed"])
            if (cfg, seed) in seen:                     # mesma execução baixada duas vezes
                continue
            seen.add((cfg, seed))
            t, v = m["test"], m["val"]
            secs = m.get("total_seconds") or (m.get("embed_seconds", 0) + m.get("probe_seconds", 0))
            rows.append({"config": cfg, "seed": seed, "regime": m["regime"], "model": m["model"],
                         "strategy": m.get("strategy", ""), "aug": m.get("aug", ""),
                         "val_auc": v["auc_roc"], "test_auc": t["auc_roc"], "test_ci_lo": t["auc_ci_lo"],
                         "test_ci_hi": t["auc_ci_hi"], "test_auc_pr": t["auc_pr"],
                         "test_sens": t["sensitivity"], "test_spec": t["specificity"],
                         "test_bal_acc": t["balanced_acc"], "threshold": t["threshold"],
                         "best_epoch": m.get("best_epoch"), "minutes": secs / 60,
                         "trainable_params": m.get("trainable_params"), "dir": str(f.parent)})
    if not rows:
        raise FileNotFoundError(f"nenhum metrics.json em {roots}")
    return pd.DataFrame(rows).sort_values(["config", "seed"]).reset_index(drop=True)


def summary(runs: pd.DataFrame) -> pd.DataFrame:
    g = runs.groupby("config")
    out = pd.DataFrame({
        "n_seeds": g.size(),
        "val_auc_mean": g["val_auc"].mean(), "val_auc_sd": g["val_auc"].std(ddof=1),
        "test_auc_mean": g["test_auc"].mean(), "test_auc_sd": g["test_auc"].std(ddof=1),
        "test_auc_pr_mean": g["test_auc_pr"].mean(),
        "test_sens_mean": g["test_sens"].mean(), "test_spec_mean": g["test_spec"].mean(),
        "minutes_mean": g["minutes"].mean(), "trainable_params": g["trainable_params"].first(),
    })
    out.insert(0, "label", [label_of(c) for c in out.index])
    return out.sort_values("val_auc_mean", ascending=False)


def exam_scores(runs: pd.DataFrame, split: str, k: int = 5) -> dict[str, pd.DataFrame]:
    """config -> DataFrame (índice = exame): pe e escore médio das sementes (e de cada uma)."""
    out = {}
    for cfg, g in runs.groupby("config"):
        parts = []
        for _, r in g.iterrows():
            ex = pd.read_csv(Path(r["dir"]) / f"exam_{split}.csv", usecols=[STUDY, "pe", f"top{k}"])
            parts.append(ex.set_index(STUDY).rename(columns={f"top{k}": f"s{r['seed']}"}))
        df = pd.concat(parts, axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        df["score"] = df.filter(regex=r"^s\d+$").mean(axis=1)
        out[cfg] = df
    return out


def ensemble_table(scores: dict, n_boot: int, seed: int = 0) -> pd.DataFrame:
    rows = []
    for cfg, df in scores.items():
        auc, lo, hi = bootstrap_ci(df["pe"], df["score"], n_boot, seed)
        rows.append({"config": cfg, "label": label_of(cfg), "ens_test_auc": auc, "ens_ci_lo": lo, "ens_ci_hi": hi,
                     "n_exams": len(df)})
    return pd.DataFrame(rows).set_index("config")


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(1.0, run)
    return adj


def best_by_val(summ: pd.DataFrame, pattern: str) -> str | None:
    cand = summ[summ.index.str.match(pattern)]
    return None if cand.empty else cand["val_auc_mean"].idxmax()


def planned_comparisons(summ: pd.DataFrame) -> list[tuple[str, str, str]]:
    """(nome, A, B). Escolhas pela validação; pares inexistentes são omitidos."""
    cnn = best_by_val(summ, r"^ft_resnet50_")
    vit = best_by_val(summ, r"^ft_(deit_s|dinov2_s)_")
    med = best_by_val(summ, r"^ft_radimagenet_resnet50_")
    top = best_by_val(summ, r"^ft_(resnet50|deit_s|dinov2_s)_")
    comps = [("melhor CNN x melhor ViT (ajuste fino)", cnn, vit),
             ("melhor ImageNet/auto-sup. x RadImageNet (ajuste fino)", top, med)]
    for m in ("resnet50", "radimagenet_resnet50", "deit_s", "dinov2_s"):
        comps.append((f"ajuste fino x congelado: {NAMES[m]}", best_by_val(summ, rf"^ft_{m}_"), f"frozen_{m}"))
    comps += [("aumento: principal x nenhum (ResNet-50)", "ft_resnet50_cosine_main", "ft_resnet50_cosine_none"),
              ("aumento: forte x principal (ResNet-50)", "ft_resnet50_cosine_strong", "ft_resnet50_cosine_main"),
              ("ajuste fino: LLRD x cosseno (DINOv2)", "ft_dinov2_s_llrd_main", "ft_dinov2_s_cosine_main"),
              ("ajuste fino: cosseno x constante (DINOv2)", "ft_dinov2_s_cosine_main", "ft_dinov2_s_const_main")]
    out, seen = [], set()
    for name, a, b in comps:
        if a and b and a != b and a in summ.index and b in summ.index and (a, b) not in seen:
            seen.add((a, b))
            out.append((name, a, b))
    return out


def compare(scores: dict, comps, n_boot: int, seed: int = 0) -> pd.DataFrame:
    rows = []
    for name, a, b in comps:
        da, db = scores[a], scores[b]
        common = da.index.intersection(db.index)
        y = da.loc[common, "pe"].to_numpy()
        sa, sb = da.loc[common, "score"].to_numpy(), db.loc[common, "score"].to_numpy()
        d = delong(y, sa, sb)
        pb = paired_bootstrap(y, sa, sb, n_boot, seed)
        rows.append({"comparison": name, "A": a, "B": b, "auc_A": d["auc1"], "auc_B": d["auc2"],
                     "diff": d["diff"], "diff_ci_lo": pb["ci_lo"], "diff_ci_hi": pb["ci_hi"],
                     "p_delong": d["p"], "p_bootstrap": pb["p"], "n_exams": len(common)})
    df = pd.DataFrame(rows)
    if len(df):
        df["p_delong_holm"] = holm(df["p_delong"])
    return df


def fmt_pm(m, s, nd=3):
    return f"{m:.{nd}f} ± {s:.{nd}f}" if pd.notna(s) else f"{m:.{nd}f}"


def markdown_tables(summ, ens, comp) -> str:
    t = summ.join(ens[["ens_test_auc", "ens_ci_lo", "ens_ci_hi"]])
    lines = ["## Configurações (3 sementes; ordenadas pela AUC de validação)", "",
             "| Configuração | n | AUC val | AUC teste | AUC teste (conjunto, IC 95%) | AUC-PR teste | Sens. | Espec. | min/exec. |",
             "|---|---|---|---|---|---|---|---|---|"]
    for cfg, r in t.iterrows():
        lines.append(f"| {r['label']} | {r['n_seeds']} | {fmt_pm(r['val_auc_mean'], r['val_auc_sd'])} | "
                     f"{fmt_pm(r['test_auc_mean'], r['test_auc_sd'])} | {r['ens_test_auc']:.3f} "
                     f"[{r['ens_ci_lo']:.3f}, {r['ens_ci_hi']:.3f}] | {r['test_auc_pr_mean']:.3f} | "
                     f"{r['test_sens_mean']:.3f} | {r['test_spec_mean']:.3f} | {r['minutes_mean']:.0f} |")
    lines += ["", "## Comparações pareadas no teste (conjunto de sementes)", "",
              "| Comparação | A | B | AUC A | AUC B | Δ [IC 95% bootstrap] | p DeLong | p Holm |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in comp.iterrows():
        lines.append(f"| {r['comparison']} | {r['A']} | {r['B']} | {r['auc_A']:.3f} | {r['auc_B']:.3f} | "
                     f"{r['diff']:+.3f} [{r['diff_ci_lo']:+.3f}, {r['diff_ci_hi']:+.3f}] | "
                     f"{r['p_delong']:.2g} | {r['p_delong_holm']:.2g} |")
    return "\n".join(lines) + "\n"
