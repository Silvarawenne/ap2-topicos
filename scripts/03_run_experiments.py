"""Passo 3 — todos os experimentos e tabelas do TP1.

  python scripts/03_run_experiments.py              # grade + corte + ablações + teste final
  python scripts/03_run_experiments.py --skip-slice # pula o nível de corte (mais caro)

Saídas em results/: table_*.csv, oof_*.parquet, test_*.csv/parquet, final_config.json,
environment.txt (versões das bibliotecas usadas).
"""
import argparse
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from pe_baseline.config import load_config, repo_path, seed_everything  # noqa: E402
from pe_baseline.data import STUDY  # noqa: E402
from pe_baseline.evaluate import exam_folds  # noqa: E402
from pe_baseline import experiments as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-slice", action="store_true")
    ap.add_argument("--skip-ablations", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    out = repo_path(cfg["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (out / "environment.txt").write_text(f"python {platform.python_version()}\n{freeze}")

    slices, exams, errors = E.load_tables(cfg)
    errors.to_csv(out / "excluded_slices.csv", index=False)
    dev = exams[exams.subset == "dev"].reset_index(drop=True)
    folds = exam_folds(dev[STUDY], dev["pe"], cfg["cv"]["outer_folds"], cfg["seed"])
    folds.rename_axis(STUDY).reset_index().to_csv(out / "cv_folds.csv", index=False)

    grid = E.exam_grid(dev, cfg, folds, out)
    cand = grid[grid.model != "dummy"].sort_values("auc_roc_mean", ascending=False).iloc[0]
    best_family, best_model = cand["family"], cand["model"]
    E.log(f"melhor (CV): {best_family} x {best_model} AUC={cand['auc_roc_mean']:.3f}")

    best_slice_model = "lgbm"
    if not args.skip_slice:
        sdev = slices[slices.subset == "dev"].reset_index(drop=True)
        stable, _ = E.slice_level(sdev, dev, cfg, folds, out)
        best_slice_model = (stable[stable.model != "dummy"]
                            .sort_values("auc_roc_mean", ascending=False).iloc[0]["model"])
    if not args.skip_ablations:
        E.ablations(dev, cfg, folds, best_family, best_model, out)
    E.final_evaluation(slices, exams, cfg, best_family, best_model, best_slice_model, out)
    E.log(f"Tabelas em {out}")


if __name__ == "__main__":
    main()
