"""Passo 2 — extração de características por corte (paralela, retomável, fatiável).

Exemplos:
  python scripts/02_extract_features.py                       # tudo de uma vez
  python scripts/02_extract_features.py --shard 0 --n-shards 3 # 1/3 dos exames (outra sessão faz 1 e 2)
  python scripts/02_extract_features.py --limit 5              # teste rápido com 5 exames

Saída: features/slices/slices_sXXofYY_cZZZZ.parquet (um arquivo por bloco de exames).
Se a sessão cair, rode o mesmo comando de novo: blocos prontos são pulados.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from pe_baseline.config import data_path, load_config, repo_path, seed_everything  # noqa: E402
from pe_baseline.data import STUDY, load_train  # noqa: E402
from pe_baseline.extract import run_extraction  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--chunk-size", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None, help="só os N primeiros exames (teste)")
    ap.add_argument("--out-dir", default=None, help="padrão: <features.out_dir>/slices")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    sample = pd.read_csv(repo_path(cfg["sample"]["file"]))
    studies = sample[STUDY].tolist()
    if args.limit:
        studies = sorted(studies)[: args.limit]
    train = load_train(data_path(cfg, cfg["data"]["train_csv"]))
    out_dir = Path(args.out_dir) if args.out_dir else repo_path(cfg["features"]["out_dir"]) / "slices"
    t0 = time.time()
    files = run_extraction(cfg, train, studies, out_dir, args.shard, args.n_shards,
                           args.chunk_size, args.n_jobs)
    print(f"Concluído: {len(files)} blocos em {out_dir} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
