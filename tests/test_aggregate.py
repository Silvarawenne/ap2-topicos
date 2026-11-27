"""Agregação do TP2 com execuções falsas (metrics.json + exam_test.csv)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pe_tp2 import aggregate as A  # noqa: E402


def fake_run(root, regime, model, seed, strength, strategy="cosine", aug="main"):
    rng = np.random.default_rng(seed)
    n = 200
    y = (np.arange(n) % 3 == 0).astype(int)
    s = 1 / (1 + np.exp(-(strength * y + rng.normal(0, 1, n))))
    cfg = f"frozen_{model}" if regime == "frozen" else f"ft_{model}_{strategy}_{aug}"
    d = Path(root) / f"{cfg}_s{seed}"
    d.mkdir(parents=True)
    pd.DataFrame({A.STUDY: [f"e{i}" for i in range(n)], "top5": s, "pe": y}).to_csv(d / "exam_test.csv", index=False)
    met = {"auc_roc": 0.5 + strength / 10, "auc_pr": 0.5, "sensitivity": 0.5, "specificity": 0.5,
           "balanced_acc": 0.5, "threshold": 0.5, "auc_ci_lo": 0.4, "auc_ci_hi": 0.6}
    m = {"regime": regime, "model": model, "seed": seed, "val": met, "test": met, "total_seconds": 60}
    if regime != "frozen":
        m.update(strategy=strategy, aug=aug)
    (d / "metrics.json").write_text(json.dumps(m))


def test_aggregate_end_to_end(tmp_path):
    for seed in range(3):
        fake_run(tmp_path, "ft", "resnet50", seed, 2.0)
        fake_run(tmp_path, "ft", "dinov2_s", seed, 3.0, strategy="llrd")
        fake_run(tmp_path, "frozen", "resnet50", seed, 0.5)
    fake_run(tmp_path / "copia", "ft", "resnet50", 0, 2.0)          # duplicada: ignorada
    runs = A.collect([tmp_path])
    assert len(runs) == 9
    summ = A.summary(runs)
    assert summ.index[0] == "ft_dinov2_s_llrd_main" and (summ["n_seeds"] == 3).all()
    comps = A.planned_comparisons(summ)
    names = [c[0] for c in comps]
    assert "melhor CNN x melhor ViT (ajuste fino)" in names
    assert any(a == "ft_resnet50_cosine_main" and b == "frozen_resnet50" for _, a, b in comps)
    scores = A.exam_scores(runs, "test")
    comp = A.compare(scores, comps, n_boot=100)
    row = comp.set_index("comparison").loc["melhor CNN x melhor ViT (ajuste fino)"]
    assert row["diff"] < 0 and row["p_delong_holm"] >= row["p_delong"]
    md = A.markdown_tables(summ, A.ensemble_table(scores, 50), comp)
    assert "Comparações pareadas" in md


def test_holm():
    assert np.allclose(A.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
