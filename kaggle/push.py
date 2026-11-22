"""Publica o código como Dataset privado e envia um kernel (estágio) para rodar no Kaggle.

Pré-requisito: CLI do Kaggle autenticada (token em ~/.kaggle ou KAGGLE_API_TOKEN).

  python kaggle/push.py code                          # cria/atualiza o Dataset <user>/ap2-topicos-code
  python kaggle/push.py kernel smoke --limit 20
  python kaggle/push.py kernel extract --shard 0 --n-shards 1
  python kaggle/push.py kernel experiments --features-kernel <user>/tp1-extract-s0
  kaggle kernels status <user>/<slug>  |  kaggle kernels output <user>/<slug> -p out/
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMP = "rsna-str-pulmonary-embolism-detection"
CODE_ITEMS = ["src", "scripts", "configs", "splits", "notebooks", "requirements-kaggle.txt", "README.md"]


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def push_code(user: str, slug: str):
    with tempfile.TemporaryDirectory() as d:
        stage = Path(d) / "pkg"
        root = stage / "ap2-topicos"
        root.mkdir(parents=True)
        for item in CODE_ITEMS:
            src = ROOT / item
            (shutil.copytree if src.is_dir() else shutil.copy)(src, root / item)
        for p in root.rglob("__pycache__"):
            shutil.rmtree(p)
        meta = {"title": slug, "id": f"{user}/{slug}", "licenses": [{"name": "other"}]}
        (stage / "dataset-metadata.json").write_text(json.dumps(meta))
        exists = subprocess.run(["kaggle", "datasets", "status", f"{user}/{slug}"],
                                capture_output=True, text=True).returncode == 0
        if exists:
            run(["kaggle", "datasets", "version", "-p", str(stage), "-m", "update", "-r", "zip", "-q"])
        else:
            run(["kaggle", "datasets", "create", "-p", str(stage), "-r", "zip", "-q"])


def push_kernel(user, stage, code_slug, shard, n_shards, limit, features_kernels, gpu):
    slug = f"tp1-{stage}" + (f"-s{shard}" if stage == "extract" else "")
    src = (ROOT / "kaggle" / "kernel_runner.py").read_text()
    src = (src.replace("__STAGE__", stage).replace("__SHARD__", str(shard))
              .replace("__N_SHARDS__", str(n_shards)).replace("__LIMIT__", str(limit or "")))
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "run.py").write_text(src)
        meta = {
            "id": f"{user}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_internet": True,
            "competition_sources": [COMP], "dataset_sources": [f"{user}/{code_slug}"],
            "kernel_sources": features_kernels or [],
        }
        (Path(d) / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        run(["kaggle", "kernels", "push", "-p", d])
    print(f"acompanhe: kaggle kernels status {user}/{slug}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["code", "kernel"])
    ap.add_argument("stage", nargs="?", choices=["smoke", "eda", "extract", "experiments"])
    ap.add_argument("--user", default=os.environ.get("KAGGLE_OWNER", "rawennedasilvaleite"))
    ap.add_argument("--code-slug", default="ap2-topicos-code")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--features-kernel", action="append", default=None)
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    if a.what == "code":
        push_code(a.user, a.code_slug)
    else:
        push_kernel(a.user, a.stage, a.code_slug, a.shard, a.n_shards, a.limit, a.features_kernel, a.gpu)


if __name__ == "__main__":
    main()
