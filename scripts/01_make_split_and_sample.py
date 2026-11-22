"""Passo 1 — partição fixa do treino completo + amostra estratificada do TP1.

Saídas (versionadas no repositório, pois definem o protocolo):
  splits/split_full_train.csv   StudyInstanceUID, label3, pe, n_slices, split (train/val/test)
  splits/sample_tp1.csv         exames da amostra do TP1 (subset = dev/test)
Saídas não versionadas:
  features/exams.csv            rótulos de exame (derivados de train.csv) para análises
  results/sample_counts.csv     contagens por classe (para o README)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from pe_baseline.config import data_path, load_config, repo_path, seed_everything  # noqa: E402
from pe_baseline.data import exam_table, load_train, make_full_split, make_sample  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--force-split", action="store_true",
                    help="regera a partição mesmo se já existir (NÃO faça depois de congelada)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    train = load_train(data_path(cfg, cfg["data"]["train_csv"]))
    exams = exam_table(train)
    feat_dir = repo_path(cfg["features"]["out_dir"])
    feat_dir.mkdir(parents=True, exist_ok=True)
    exams.to_csv(feat_dir / "exams.csv", index=False)

    split_file = repo_path(cfg["split"]["file"])
    split_file.parent.mkdir(parents=True, exist_ok=True)
    if split_file.exists() and not args.force_split:
        split = pd.read_csv(split_file)
        print(f"Partição congelada encontrada: {split_file.name} (não regerada)")
        assert set(split["StudyInstanceUID"]) == set(exams["StudyInstanceUID"]), \
            "a partição salva não corresponde ao train.csv atual"
    else:
        split = make_full_split(exams, cfg["split"]["test_frac"], cfg["split"]["val_frac"], cfg["seed"])
        split.to_csv(split_file, index=False)
        print(f"Partição criada: {split_file}")

    s = cfg["sample"]
    sample = make_sample(split, s["n_dev_exams"], s["n_test_exams"], cfg["seed"])
    sample_file = repo_path(s["file"])
    sample.to_csv(sample_file, index=False)

    counts = (sample.groupby(["subset", "label3"]).agg(exams=("StudyInstanceUID", "size"),
                                                       slices_total=("n_slices", "sum"))
              .reset_index())
    counts["slices_used_approx"] = (counts["slices_total"] / s["slice_stride"]).round().astype(int)
    full = split.groupby(["split", "label3"]).size().unstack(fill_value=0)
    out = repo_path(cfg["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out / "sample_counts.csv", index=False)
    full.to_csv(out / "split_counts.csv")
    print("\nPartição do treino completo (exames):\n", full.to_string())
    print("\nAmostra TP1:\n", counts.to_string(index=False))
    print(f"\nTotal: {len(sample)} exames de {len(exams)} "
          f"({100 * len(sample) / len(exams):.1f}%), ~{counts['slices_used_approx'].sum()} cortes")


if __name__ == "__main__":
    main()
