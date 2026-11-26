"""TP2 — redes pré-treinadas sobre os volumes da extração completa.

  python scripts/10_tp2.py counts                         # contagens por partição (artigo)
  python scripts/10_tp2.py bench --models resnet50 deit_s dinov2_s
  python scripts/10_tp2.py frozen --model resnet50        # embeddings + sonda linear, 3 sementes
  python scripts/10_tp2.py finetune --model resnet50 --strategy cosine --aug main --seeds 0 1 2

  python scripts/10_tp2.py queue --jobs frozen:resnet50 ft:resnet50:cosine:main:0 ...

`queue` (e `finetune` com várias sementes) roda uma tarefa por GPU ao mesmo tempo (Kaggle T4 x2:
duas em paralelo; a próxima entra assim que uma GPU libera). Log de cada tarefa em <out>/logs/.
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


def _sysinfo(t) -> dict:
    import shutil
    info = {"cpus": os.cpu_count()}
    try:
        with open("/proc/meminfo") as f:
            info["ram_gb"] = round(int(f.readline().split()[1]) / 2**20, 1)
    except OSError:
        pass
    for p in ("/kaggle/working", "/tmp", str(repo_path(t["out_dir"]))):
        if os.path.exists(p):
            info[f"free_gb[{p}]"] = round(shutil.disk_usage(p).free / 2**30, 1)
    return info


def _io_probe(idx, n_exams=8) -> dict:
    """Leitura dos volumes: sequencial crua (arquivo ainda não lido) e np.load por exame,
    a frio e a quente (cache do SO). Separa gargalo de disco/rede de gargalo de CPU."""
    from pe_tp2.data import VolumeCache
    files = sorted(idx.loc[idx["ok"], "vol_path"].unique())
    out = {"npz_files": len(files)}
    if len(files) < 2:
        return out
    a, b = files[len(files) // 2], files[-1]
    t0, n = time.time(), 0
    with open(a, "rb") as f:
        while n < 1 << 30:
            chunk = f.read(16 << 20)
            if not chunk:
                break
            n += len(chunk)
    out["raw_read_MBps"] = n / 1e6 / max(time.time() - t0, 1e-9)
    studies = idx.loc[idx["vol_path"] == b, "StudyInstanceUID"].unique()[:n_exams]
    for tag in ("cold", "warm"):
        t0 = time.time()
        for s in studies:
            VolumeCache(1).get(b, s)
        out[f"npz_exam_s_{tag}"] = (time.time() - t0) / max(len(studies), 1)
    return out


def _loader_rate(ds, t, n_slices, skip) -> tuple[float, float]:
    """(cortes/s em regime, s até o 1o lote): descarta os `skip` primeiros (enchimento)."""
    from pe_tp2.data import loader
    t_start = time.time()
    t_first = t0 = None
    n = 0
    for xb, _ in loader(ds, t["batch_size"], t["num_workers"]):
        if t_first is None:
            t_first = time.time() - t_start
        n += len(xb)
        if t0 is None and n >= skip:
            t0, n0 = time.time(), n
        if t0 is not None and n - n0 >= n_slices:
            break
    if t0 is None:
        return float("nan"), t_first or float("nan")
    return (n - n0) / max(time.time() - t0, 1e-9), t_first


def cmd_bench(args):
    """Vazão: (1) GPU, por modelo, com entrada sintética uint8 + aumento 'main' na GPU;
    (2) custo do aumento na GPU; (3) leitura real: disco, workers no treino e na avaliação."""
    import torch
    from pe_tp2 import augment
    from pe_tp2.data import EvalSlices, TrainSlices
    from pe_tp2.engine import _autocast, device
    from pe_tp2.models import Net
    cfg, t = tp2_cfg(args)
    dev = device()
    sysinfo = _sysinfo(t)
    print(sysinfo, flush=True)
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    xb = torch.randint(0, 256, (t["batch_size"], 3, 256, 256), dtype=torch.uint8, device=dev)
    aug_rates = {}
    for pol in ("main", "strong"):
        tf = augment.build(pol)
        tf(augment.to_float(xb))
        sync()
        t0 = time.time()
        for _ in range(args.steps):
            tf(augment.to_float(xb))
        sync()
        aug_rates[f"aug_{pol}_img_s"] = args.steps * len(xb) / (time.time() - t0)
    print(aug_rates, flush=True)
    rows = []
    tf = augment.build("main")
    for name in args.models:
        model = Net(name, pretrained=not args.no_pretrained).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
        y = torch.randint(0, 2, (t["batch_size"],), device=dev).float()

        def step():
            x = tf(augment.to_float(xb))
            with _autocast(dev):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x).float(), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        for _ in range(3):
            step()
        sync()
        t0 = time.time()
        for _ in range(args.steps):
            step()
        sync()
        train_ips = args.steps * t["batch_size"] / (time.time() - t0)
        model.eval()
        xe = torch.randint(0, 256, (t["eval_batch_size"], 3, 256, 256), dtype=torch.uint8, device=dev)
        with torch.inference_mode(), _autocast(dev):
            model.features(augment.to_float(xe))
            sync()
            t0 = time.time()
            for _ in range(args.steps):
                model.features(augment.to_float(xe))
            sync()
        infer_ips = args.steps * t["eval_batch_size"] / (time.time() - t0)
        mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0
        rows.append({"model": name, "train_img_s": train_ips, "infer_img_s": infer_ips, "gpu_mem_gb": mem})
        print(rows[-1], flush=True)
        del model, opt
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    df = pd.DataFrame(rows)
    read = {}
    try:
        idx = get_index(cfg, t)
        read.update(_io_probe(idx))
        print(read, flush=True)
        tr = TrainSlices(idx[idx["split"] == "train"], t["neg_ratio"], 0, t["offset"])
        skip = min(4 * t["batch_size"] * max(t["num_workers"], 1), args.loader_slices)
        read["loader_train_img_s"], read["loader_train_first_batch_s"] = _loader_rate(
            tr, t, args.loader_slices, skip + tr.buffer * max(t["num_workers"], 1))
        ev = EvalSlices(idx[idx["split"] == "val"])
        read["loader_eval_img_s"], _ = _loader_rate(ev, t, args.loader_slices, skip)
        read["slices_per_epoch_this_input"] = len(tr)
        print(read, flush=True)
        # época estimada para o treino inteiro (este input pode ser só parte dos shards)
        n_train_ex = idx.loc[idx["split"] == "train", "StudyInstanceUID"].nunique()
        sp = pd.read_csv(repo_path(cfg["split"]["file"]))
        n_all = int(((sp["split"] == "train") & (sp["label3"] != "indeterminate")).sum())
        frac = n_train_ex / n_all
        full_epoch = len(tr) / max(frac, 1e-9)
        eff = np.minimum(df["train_img_s"], read["loader_train_img_s"])
        df["min_per_epoch_full"] = full_epoch / eff / 60
        df["est_slices_per_epoch_full"] = int(full_epoch)
    except FileNotFoundError as e:
        print("sem volumes para medir a leitura:", e)
    d = out_dir(t, "bench")
    df.assign(**env_info(), **sysinfo, **aug_rates, **read).to_csv(d / "bench.csv", index=False)
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
                 "trainable_params": int(w.numel() + 1), "load_info": model.load_info, **env_info()}
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


def _job_cmd(args, job: str, workers: int) -> list[str]:
    """'frozen:<modelo>' ou 'ft:<modelo>:<estratégia>:<aumento>:<semente>' -> linha de comando."""
    p = job.split(":")
    if p[0] == "frozen" and len(p) == 2:
        cmd = ["frozen", "--model", p[1]]
    elif p[0] == "ft" and len(p) == 5:
        cmd = ["finetune", "--model", p[1], "--strategy", p[2], "--aug", p[3], "--seeds", p[4], "--sequential"]
    else:
        raise ValueError(f"tarefa inválida: {job!r} (use frozen:MODELO ou ft:MODELO:ESTRAT:AUG:SEMENTE)")
    cmd = [sys.executable, "-u", __file__, *cmd, "--num-workers", str(workers)]
    for opt in ("epochs", "max_steps", "neg_ratio", "batch_size"):
        v = getattr(args, opt, None)
        if v is not None:
            cmd += [f"--{opt.replace('_', '-')}", str(v)]
    if args.no_pretrained:
        cmd.append("--no-pretrained")
    if args.config:
        cmd += ["--config", args.config]
    return cmd


def run_queue(args, jobs: list[str]):
    """Fila de tarefas: uma por GPU ao mesmo tempo (a próxima entra quando uma GPU libera).
    Os workers de CPU são divididos entre as GPUs. Log de cada tarefa em <out>/logs/."""
    import torch
    _, t = tp2_cfg(args)
    n_gpu = max(1, torch.cuda.device_count())
    workers = args.num_workers if args.num_workers is not None else max(1, (os.cpu_count() or 4) // n_gpu)
    cmds = [(j, _job_cmd(args, j, workers)) for j in jobs]          # valida tudo antes de começar
    logs = out_dir(t, "logs")
    pending, running, failed, t0 = list(cmds), [], [], time.time()
    free = list(range(n_gpu))
    print(f"{len(jobs)} tarefas, {n_gpu} GPU(s), {workers} workers cada", flush=True)
    while pending or running:
        while pending and free:
            g = free.pop(0)
            job, cmd = pending.pop(0)
            env = dict(os.environ)
            if torch.cuda.is_available():
                env["CUDA_VISIBLE_DEVICES"] = str(g)
            f = open(logs / (job.replace(":", "_") + ".log"), "w")
            running.append((subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT), g, job, f, time.time()))
            print(f"[{(time.time() - t0) / 60:6.1f} min] início  {job}  (GPU {g})", flush=True)
        time.sleep(1 if args.no_pretrained else 15)
        for item in list(running):
            p, g, job, f, ts = item
            if p.poll() is None:
                continue
            f.close()
            running.remove(item)
            free.append(g)
            ok = p.returncode == 0
            print(f"[{(time.time() - t0) / 60:6.1f} min] {'fim   ' if ok else 'FALHOU'}  {job}  "
                  f"({(time.time() - ts) / 60:.1f} min)", flush=True)
            if not ok:
                failed.append(job)
                print("".join(open(f.name, encoding="utf-8", errors="replace").readlines()[-30:]), flush=True)
    print(f"fila concluída em {(time.time() - t0) / 60:.1f} min; falhas: {failed or 'nenhuma'}", flush=True)
    if failed:
        sys.exit(1)


def cmd_queue(args):
    run_queue(args, args.jobs)


def cmd_finetune(args):
    seeds = args.seeds or load_config(args.config)["tp2"]["seeds"]
    if args.sequential:
        for s in seeds:
            run_finetune_one(args, s)
        return
    run_queue(args, [f"ft:{args.model}:{args.strategy}:{args.aug}:{s}" for s in seeds])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["counts", "bench", "frozen", "finetune", "queue"])
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
    ap.add_argument("--loader-slices", type=int, default=8000, help="bench: cortes lidos (em regime)")
    ap.add_argument("--sequential", action="store_true")
    ap.add_argument("--jobs", nargs="+", default=[], help="queue: frozen:MODELO | ft:MODELO:ESTRAT:AUG:SEMENTE")
    ap.add_argument("--no-pretrained", action="store_true", help="só para testes sem internet")
    args = ap.parse_args()
    {"counts": cmd_counts, "bench": cmd_bench, "frozen": cmd_frozen, "finetune": cmd_finetune,
     "queue": cmd_queue}[args.cmd](args)


if __name__ == "__main__":
    main()
