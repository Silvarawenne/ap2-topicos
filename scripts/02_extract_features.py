"""Passo 2 — extração de características por corte (paralela, retomável, fatiável).

Exemplos:
  python scripts/02_extract_features.py                       # TP1: amostra, 1 corte a cada 2
  python scripts/02_extract_features.py --shard 0 --n-shards 3 # 1/3 dos exames (outra sessão faz 1 e 2)
  python scripts/02_extract_features.py --limit 5              # teste rápido com 5 exames
  python scripts/02_extract_features.py --full --shard 0 --n-shards 10
      # TP2: treino oficial completo + teste oficial, TODOS os cortes, e grava também os
      # volumes uint8 256x256 (janela CTPA) usados pelas redes

Saída: <out-dir>/slices_sXXofYY_cZZZZ.parquet (um arquivo por bloco de exames) e, com --full,
volumes_sXXofYY_cZZZZ.npz ao lado. Se a sessão cair, rode o mesmo comando de novo: blocos
prontos são pulados.
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


def full_table(cfg: dict) -> pd.DataFrame:
    """Treino oficial completo + teste oficial (sem rótulos), com a coluna `source`."""
    train = load_train(data_path(cfg, cfg["data"]["train_csv"])).assign(source="train")
    test_csv = data_path(cfg, cfg["full"]["test_csv"])
    if not test_csv.exists():
        print(f"AVISO: {test_csv} não encontrado; só o treino será processado", flush=True)
        return train
    test = pd.read_csv(test_csv).assign(source="test", pe_present_on_image=-1)
    return pd.concat([train, test], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--full", action="store_true",
                    help="TP2: todos os exames (treino + teste oficial), todos os cortes, com volumes")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--chunk-size", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None, help="só os N primeiros exames (teste)")
    ap.add_argument("--out-dir", default=None,
                    help="padrão: <features.out_dir>/slices (TP1) ou <full.out_dir>/slices (--full)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])

    if args.full:
        table = full_table(cfg)
        studies = sorted(table[STUDY].unique())
        cfg["preprocess"]["exam_timeout_s"] = cfg["full"]["exam_timeout_s"]
        stride, default_out = cfg["full"]["slice_stride"], cfg["full"]["out_dir"]
    else:
        table = load_train(data_path(cfg, cfg["data"]["train_csv"]))
        studies = pd.read_csv(repo_path(cfg["sample"]["file"]))[STUDY].tolist()
        stride, default_out = None, cfg["features"]["out_dir"]
    if args.limit:
        studies = sorted(studies)[: args.limit]
    out_dir = Path(args.out_dir) if args.out_dir else repo_path(default_out) / "slices"
    t0 = time.time()
    files = run_extraction(cfg, table, studies, out_dir, args.shard, args.n_shards,
                           args.chunk_size, args.n_jobs, stride=stride, save_volumes=args.full)
    print(f"Concluído: {len(files)} blocos em {out_dir} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
