"""Tabelas e estatísticas do TP2 a partir das saídas baixadas das sessões (sem GPU).

  python scripts/12_aggregate.py --roots kaggle_outputs --out results/tp2

Saídas em --out: runs.csv (uma linha por execução), summary.csv (média ± dp por configuração),
ensemble.csv (AUC do conjunto de sementes com IC 95% por bootstrap de exames),
comparisons.csv (DeLong + bootstrap pareado, Holm) e tables.md (as mesmas tabelas, legíveis).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pe_baseline.config import load_config, repo_path  # noqa: E402
from pe_tp2 import aggregate as A  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["kaggle_outputs"])
    ap.add_argument("--out", default="results/tp2")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    t = load_config(args.config)["tp2"]
    out = repo_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = A.collect([repo_path(r) for r in args.roots])
    summ = A.summary(runs)
    scores = A.exam_scores(runs, "test", t["topk"])
    ens = A.ensemble_table(scores, t["n_bootstrap"])
    comp = A.compare(scores, A.planned_comparisons(summ), t["n_bootstrap"])
    runs.drop(columns="dir").to_csv(out / "runs.csv", index=False)
    summ.to_csv(out / "summary.csv")
    ens.to_csv(out / "ensemble.csv")
    comp.to_csv(out / "comparisons.csv", index=False)
    md = A.markdown_tables(summ, ens, comp)
    (out / "tables.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
