"""TP2 — redes pré-treinadas sobre os volumes da extração completa.

  python scripts/10_tp2.py counts                         # contagens por partição (artigo)
  python scripts/10_tp2.py bench --models resnet50 deit_s dinov2_s
  python scripts/10_tp2.py frozen --model resnet50        # embeddings + sonda linear, 3 sementes
  python scripts/10_tp2.py finetune --model resnet50 --strategy cosine --aug main --seeds 0 1 2

Com 2 GPUs (Kaggle T4 x2), `finetune` roda duas sementes em paralelo, uma por GPU.
Cada execução grava em <out_dir>/<nome>/: history.csv, run.json, metrics.json,
exam_{val,test,official_test}.csv, slices_{test,official_test}.parquet e os pesos.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pe_baseline.config import load_config, repo_path  # noqa: E402


def tp2_cfg(args):
    cfg = load_config(args.config)
    t = cfg["tp2"]
    t["inputs"] = os.environ.get("PE_TP2_INPUTS", t["inputs"])
    t["out_dir"] = os.environ.get("PE_TP2_OUT", t["out_dir"])
    for k in ("epochs", "batch_size", "num_workers", "max_steps", "neg_ratio"):
        v = getattr(args, k, None)
        if v is not None:
            t[k] = v
    return cfg, t


def get_index(cfg, t, log=print):
    from pe_tp2.index import find_slice_dirs, load_index
    dirs = find_slice_dirs(str(t["inputs"]).split(os.pathsep))
    log(f"{len(dirs)} pastas de extração encontradas")
    idx = load_index(dirs, repo_path(cfg["split"]["file"]), t["exclude_indeterminate"])
    return idx


def env_info() -> dict:
    import torch
    import timm
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {"gpu": gpu, "n_gpus": torch.cuda.device_count(), "torch": torch.__version__,
            "timm": timm.__version__, "python": platform.python_version()}


def out_dir(t, name) -> Path:
    d = repo_path(t["out_dir"]) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_eval(d: Path, idx, t, preds: dict, extra: dict):
    """preds[split] = (pos, p). Grava cortes/exames, limiar de Youden na validação e métricas."""
    from pe_tp2.engine import exam_frame, slice_frame
    from pe_tp2.stats import bootstrap_ci, metrics_at, youden
    k = f"top{t['topk']}"
    ex = {}
    for s, (pos, p) in preds.items():
        sl = slice_frame(idx, pos, p)
        if s in ("test", "official_test"):
            sl[["StudyInstanceUID", "SOPInstanceUID", "p"]].to_parquet(d / f"slices_{s}.parquet", index=False)
        ex[s] = exam_frame(sl, t["topk"])
        ex[s].to_csv(d / f"exam_{s}.csv", index=False)
    thr = youden(ex["val"]["pe"], ex["val"][k])
    m = {"threshold_from": "val (Youden)", **extra}
    for s in ("val", "test"):
        if s in ex:
            y, sc = ex[s]["pe"].to_numpy(), ex[s][k].to_numpy()
            m[s] = metrics_at(y, sc, thr)
            auc, lo, hi = bootstrap_ci(y, sc, t["n_bootstrap"])
            m[s].update(auc_ci_lo=lo, auc_ci_hi=hi)
    (d / "metrics.json").write_text(json.dumps(m, indent=2, default=float))
    return m


# ------------------------------------------------------------------------- comandos

def cmd_counts(args):
    from pe_tp2.index import counts
    cfg, t = tp2_cfg(args)
    idx = get_index(cfg, t)
    c = counts(idx)
    d = out_dir(t, "counts")
    c.to_csv(d / "counts.csv")
    print(c.to_string())


def cmd_bench(args):
    """Vazão por modelo: treino e inferência só na GPU (entrada sintética) e leitura real."""
    import torch
    from pe_tp2 import augment
    from pe_tp2.data import TrainSlices, loader
    from pe_tp2.engine import _autocast, device
    from pe_tp2.models import Net
    cfg, t = tp2_cfg(args)
    dev = device()
    rows = []
    for name in args.models:
        model = Net(name, pretrained=not args.no_pretrained).to(dev)
        if model.family == "cnn":
            model = model.to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
        x = torch.rand(t["batch_size"], 3, 256, 256, device=dev).half()
        y = torch.randint(0, 2, (t["batch_size"],), device=dev).float()

        def step():
            with _autocast(dev):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x).float(), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        for _ in range(3):
            step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.steps):
            step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        train_ips = args.steps * t["batch_size"] / (time.time() - t0)
        model.eval()
        xe = torch.rand(t["eval_batch_size"], 3, 256, 256, device=dev).half()
        with torch.inference_mode(), _autocast(dev):
            model(xe)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(args.steps):
                model(xe)
            if dev.type == "cuda":
                torch.cuda.synchronize()
        infer_ips = args.steps * t["eval_batch_size"] / (time.time() - t0)
        mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0
        rows.append({"model": name, "train_img_s": train_ips, "infer_img_s": infer_ips, "gpu_mem_gb": mem})
        print(rows[-1], flush=True)
        del model, opt
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    # leitura real (CPU): quantos cortes 2.5D aumentados por segundo os workers entregam
    try:
        idx = get_index(cfg, t)
        ds = TrainSlices(idx[idx["split"] == "train"], t["neg_ratio"], 0, augment.build("main"), t["offset"])
        t0, n = time.time(), 0
        for xb, _ in loader(ds, t["batch_size"], t["num_workers"]):
            n += len(xb)
            if n >= args.loader_slices:
                break
        load_ips = n / (time.time() - t0)
        print({"loader_img_s": load_ips, "slices_per_epoch": len(ds)}, flush=True)
        for r in rows:
            r.update(loader_img_s=load_ips, slices_per_epoch=len(ds))
    except FileNotFoundError as e:
        print("sem volumes para medir a leitura:", e)
    df = pd.DataFrame(rows)
    if "slices_per_epoch" in df:
        eff = np.minimum(df["train_img_s"], df["loader_img_s"])
        df["min_per_epoch"] = df["slices_per_epoch"] / eff / 60
    d = out_dir(t, "bench")
    df.assign(**env_info()).to_csv(d / "bench.csv", index=False)
    print(df.to_string())


def cmd_frozen(args):
    import torch
    from pe_tp2.engine import embed_splits, exam_frame, probe_predict, slice_frame, train_probe, device
    from pe_tp2.models import Net
    from sklearn.metrics import roc_auc_score
    cfg, t = tp2_cfg(args)
    idx = get_index(cfg, t)
    dev = device()
    model = Net(args.model, pretrained=not args.no_pretrained).to(dev)
    t0 = time.time()
    splits = [s for s in ("train", "val", "test", "official_test") if (idx["split"] == s).any()]
    emb = embed_splits(model, idx, t, splits)
    t_embed = time.time() - t0
    # weight decay da sonda: escolhido na validação, semente 0
    k = f"top{t['topk']}"
    sel = {}
    for wd in t["probe"]["weight_decays"]:
        w, b = train_probe(emb, idx, t, 0, wd)
        pos, p = probe_predict(emb["val"], w, b)
        ex = exam_frame(slice_frame(idx, pos, p), t["topk"])
        sel[wd] = roc_auc_score(ex["pe"], ex[k])
        print(f"sonda wd={wd}: AUC val (exame) = {sel[wd]:.4f}", flush=True)
    wd = max(sel, key=sel.get)
    for seed in args.seeds or t["seeds"]:
        t1 = time.time()
        w, b = train_probe(emb, idx, t, seed, wd)
        name = f"frozen_{args.model}_s{seed}"
        d = out_dir(t, name)
        preds = {s: probe_predict(emb[s], w, b) for s in splits if s != "train"}
        torch.save({"model": args.model, "head_w": w.half(), "head_b": b.half()}, d / "head.pt")
        extra = {"regime": "frozen", "model": args.model, "seed": seed, "probe_wd": wd,
                 "probe_val_auc_by_wd": sel, "embed_seconds": t_embed, "probe_seconds": time.time() - t1,
                 "trainable_params": int(w.numel() + 1), **env_info()}
        m = save_eval(d, idx, t, preds, extra)
        (d / "run.json").write_text(json.dumps(extra, indent=2, default=float))
        print(name, "AUC teste =", round(m["test"]["auc_roc"], 4), flush=True)


def run_finetune_one(args, seed: int):
    import torch
    from pe_tp2.engine import device, fit, predict
    from pe_tp2.models import Net
    cfg, t = tp2_cfg(args)
    idx = get_index(cfg, t)
    dev = device()
    name = f"ft_{args.model}_{args.strategy}_{args.aug}_s{seed}"
    d = out_dir(t, name)
    log_f = open(d / "log.txt", "a", encoding="utf-8")

    def log(msg):
        print(f"[{name}] {msg}", flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    model = Net(args.model, pretrained=not args.no_pretrained,
                drop_path=t["drop_path"] if "dinov2" in args.model or "deit" in args.model else 0.0).to(dev)
    if model.family == "cnn":
        model = model.to(memory_format=torch.channels_last)
    t0 = time.time()
    hist, info = fit(model, idx, t, args.strategy, args.aug, seed, log)
    t_fit = time.time() - t0
    hist.to_csv(d / "history.csv", index=False)
    preds = {}
    for s in ("val", "test", "official_test"):
        sub = idx[idx["split"] == s]
        if len(sub):
            preds[s] = predict(model, sub, t["eval_batch_size"], t["num_workers"], t["offset"])
    sd = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
    torch.save({"model": args.model, "state_dict": sd}, d / "model.pt")
    extra = {"regime": "finetune", "model": args.model, "strategy": args.strategy, "aug": args.aug,
             "seed": seed, "fit_seconds": t_fit, "total_seconds": time.time() - t0, **info,
             "hparams": {k: t[k] for k in ("epochs", "batch_size", "lr", "weight_decay", "warmup_frac",
                                           "llrd_decay", "neg_ratio", "offset", "drop_path")},
             "load_info": model.load_info, **env_info()}
    m = save_eval(d, idx, t, preds, extra)
    (d / "run.json").write_text(json.dumps(extra, indent=2, default=float))
    log(f"AUC teste = {m['test']['auc_roc']:.4f} (época {info['best_epoch']})")


def cmd_finetune(args):
    import torch
    seeds = args.seeds or load_config(args.config)["tp2"]["seeds"]
    n_gpu = torch.cuda.device_count()
    if len(seeds) == 1 or n_gpu < 2 or args.sequential:
        for s in seeds:
            run_finetune_one(args, s)
        return
    # uma semente por GPU, em paralelo; os workers de CPU são divididos entre os processos
    workers = max(1, (os.cpu_count() or 4) // n_gpu)
    pending, running = list(seeds), []
    base = [sys.executable, "-u", __file__, "finetune", "--model", args.model, "--strategy", args.strategy,
            "--aug", args.aug, "--num-workers", str(workers), "--sequential"]
    for opt in ("epochs", "max_steps", "neg_ratio", "batch_size"):
        v = getattr(args, opt)
        if v is not None:
            base += [f"--{opt.replace('_', '-')}", str(v)]
    if args.no_pretrained:
        base.append("--no-pretrained")
    if args.config:
        base += ["--config", args.config]
    free = list(range(n_gpu))
    while pending or running:
        while pending and free:
            g, s = free.pop(0), pending.pop(0)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}
            running.append((subprocess.Popen(base + ["--seeds", str(s)], env=env), g))
        time.sleep(10)
        for p, g in list(running):
            if p.poll() is not None:
                if p.returncode != 0:
                    print(f"semente na GPU {g} falhou (código {p.returncode})", flush=True)
                running.remove((p, g))
                free.append(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["counts", "bench", "frozen", "finetune"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--models", nargs="+", default=["resnet50", "deit_s", "dinov2_s"])
    ap.add_argument("--strategy", default="cosine")
    ap.add_argument("--aug", default="main")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--neg-ratio", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="limite de passos por época (teste)")
    ap.add_argument("--steps", type=int, default=20, help="bench: passos medidos")
    ap.add_argument("--loader-slices", type=int, default=5000, help="bench: cortes lidos")
    ap.add_argument("--sequential", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true", help="só para testes sem internet")
    args = ap.parse_args()
    {"counts": cmd_counts, "bench": cmd_bench, "frozen": cmd_frozen, "finetune": cmd_finetune}[args.cmd](args)


if __name__ == "__main__":
    main()
