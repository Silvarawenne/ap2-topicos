"""Treino (ajuste fino), inferência, embeddings congelados e sondagem linear.

Seleção (época, limiar, weight decay da sonda) usa SÓ a validação; o teste nunca entra em
escolha alguma. Estratégias de ajuste fino (ablação da Seção 4):
  const   AdamW, taxa constante
  cosine  AdamW, aquecimento linear + decaimento cosseno (Loshchilov e Hutter, 2017)  [principal]
  llrd    cosine + decaimento da taxa por camada (taxas discriminativas)
  lpft    cosine, com a 1a época só na cabeça (sondagem linear) e depois tudo (Kumar et al., 2022)
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import augment
from .data import EvalSlices, TrainSlices, loader, steps_per_epoch
from .index import SOP, STUDY
from .models import Net, count_params, param_groups, set_backbone_trainable
from .stats import exam_scores

STRATEGIES = ("const", "cosine", "llrd", "lpft")


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast(dev):
    return torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda")


@torch.inference_mode()
def predict(model: Net, idx: pd.DataFrame, batch_size: int, workers: int, offset: int = 1,
            features: bool = False):
    """Probabilidade (ou embedding) de TODOS os cortes válidos de idx, na ordem de idx.index."""
    dev = next(model.parameters()).device
    model.eval()
    ds = EvalSlices(idx, offset)
    pos_all, out = [], []
    for x, pos in loader(ds, batch_size, workers):
        with _autocast(dev):
            x = x.to(dev, non_blocking=True)
            y = model.features(x) if features else torch.sigmoid(model(x).float())
        out.append(y.float().cpu().numpy().astype(np.float16 if features else np.float32))
        pos_all.append(pos.numpy())
    pos = np.concatenate(pos_all)
    vals = np.concatenate(out)
    order = np.argsort(pos)
    return pos[order], vals[order]


def slice_frame(idx: pd.DataFrame, pos: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    sub = idx.loc[pos, [STUDY, SOP, "split", "pe", "pe_present_on_image"]].copy()
    sub["p"] = p
    return sub


def exam_frame(slices: pd.DataFrame, k: int) -> pd.DataFrame:
    ex = exam_scores(slices[STUDY].to_numpy(), slices["p"].to_numpy(), k)
    lab = slices.groupby(STUDY)[["pe", "split"]].first()
    return ex.merge(lab, left_on=STUDY, right_index=True)


def val_auc(model, idx_val, cfg) -> tuple[float, pd.DataFrame]:
    from sklearn.metrics import roc_auc_score
    pos, p = predict(model, idx_val, cfg["eval_batch_size"], cfg["num_workers"], cfg["offset"])
    ex = exam_frame(slice_frame(idx_val, pos, p), cfg["topk"])
    return float(roc_auc_score(ex["pe"], ex[f"top{cfg['topk']}"])), ex


def fit(model: Net, idx: pd.DataFrame, cfg: dict, strategy: str, aug: str, seed: int,
        log=print) -> tuple[pd.DataFrame, dict]:
    assert strategy in STRATEGIES, strategy
    dev = next(model.parameters()).device
    torch.manual_seed(seed)
    tr_idx, va_idx = idx[idx["split"] == "train"], idx[idx["split"] == "val"]
    ds = TrainSlices(tr_idx, cfg["neg_ratio"], seed, augment.build(aug), cfg["offset"])
    steps = steps_per_epoch(ds, cfg["batch_size"])
    lr = cfg["lr"][model.family]
    groups = param_groups(model, lr, cfg["weight_decay"],
                          cfg["llrd_decay"] if strategy == "llrd" else None)
    opt = torch.optim.AdamW(groups, lr=lr)
    total = cfg["epochs"] * steps
    warm = 0 if strategy == "const" else int(cfg["warmup_frac"] * total)

    def mult(step):
        if strategy == "const":
            return 1.0
        if step < warm:
            return (step + 1) / max(warm, 1)
        t = min(1.0, (step - warm) / max(total - warm, 1))
        return 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, mult)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    hist, best, best_state, step = [], -1.0, None, 0
    for epoch in range(cfg["epochs"]):
        lp = strategy == "lpft" and epoch == 0
        set_backbone_trainable(model, not lp)
        ds.set_epoch(epoch)
        model.train()
        t0, n_seen, loss_sum = time.time(), 0, 0.0
        for x, y in loader(ds, cfg["batch_size"], cfg["num_workers"]):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with _autocast(dev):
                loss = F.binary_cross_entropy_with_logits(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if step < total:
                sched.step()
            step += 1
            n_seen += len(y)
            loss_sum += loss.item() * len(y)
            if cfg.get("max_steps") and step >= cfg["max_steps"] * (epoch + 1):
                break
        t_train = time.time() - t0
        t1 = time.time()
        auc, _ = val_auc(model, va_idx, cfg)
        row = {"epoch": epoch, "loss": loss_sum / max(n_seen, 1), "val_auc": auc,
               "slices": n_seen, "train_s": t_train, "val_s": time.time() - t1,
               "img_per_s": n_seen / max(t_train, 1e-9), "lr_end": opt.param_groups[0]["lr"],
               "head_only": lp}
        hist.append(row)
        log(json.dumps(row))
        if auc > best:
            best, best_state = auc, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
    set_backbone_trainable(model, True)
    model.load_state_dict(best_state)
    hist = pd.DataFrame(hist)
    info = {"best_epoch": int(hist["val_auc"].idxmax()), "best_val_auc": best, "steps_per_epoch": steps,
            "slices_per_epoch_expected": len(ds), "n_pos_train": ds.n_pos,
            "trainable_params": count_params(model), "total_params": count_params(model, False)}
    return hist, info


# ----------------------------------------------------------------- regime congelado

def embed_splits(model: Net, idx: pd.DataFrame, cfg: dict, splits, log=print) -> dict:
    out = {}
    for s in splits:
        t0 = time.time()
        sub = idx[idx["split"] == s]
        pos, f = predict(model, sub, cfg["eval_batch_size"], cfg["num_workers"], cfg["offset"],
                         features=True)
        out[s] = (pos, f)
        log(f"embeddings {s}: {f.shape} em {time.time() - t0:.0f}s ({len(pos) / max(time.time() - t0, 1e-9):.0f} cortes/s)")
    return out


def train_probe(emb: dict, idx: pd.DataFrame, cfg: dict, seed: int, weight_decay: float):
    """Regressão logística (cabeça linear) sobre embeddings padronizados, minibatch AdamW,
    peso de classe balanceado. Devolve (w, b) já na escala original dos embeddings."""
    dev = device()
    pos, X = emb["train"]
    y = torch.tensor(idx.loc[pos, "pe_present_on_image"].to_numpy(), dtype=torch.float32)
    X = torch.from_numpy(X)
    mu, sd = X.float().mean(0), X.float().std(0).clamp_min(1e-6)
    g = torch.Generator().manual_seed(seed)
    lin = torch.nn.Linear(X.shape[1], 1).to(dev)
    torch.manual_seed(seed)
    torch.nn.init.zeros_(lin.weight)
    torch.nn.init.zeros_(lin.bias)
    opt = torch.optim.AdamW(lin.parameters(), lr=cfg["probe"]["lr"], weight_decay=weight_decay)
    pw = torch.tensor((y == 0).sum() / max((y == 1).sum(), 1), device=dev)
    Xd, yd, mud, sdd = X.to(dev), y.to(dev), mu.to(dev), sd.to(dev)
    bs = cfg["probe"]["batch_size"]
    for _ in range(cfg["probe"]["epochs"]):
        perm = torch.randperm(len(yd), generator=g).to(dev)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            logit = lin((Xd[b].float() - mud) / sdd).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logit, yd[b], pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
    w = (lin.weight.detach().squeeze(0) / sdd).cpu()
    b = (lin.bias.detach() - (lin.weight.detach().squeeze(0) * mud / sdd).sum()).cpu()
    return w, b


def probe_predict(emb_split, w, b) -> tuple[np.ndarray, np.ndarray]:
    pos, X = emb_split
    logit = torch.from_numpy(X).float() @ w + b
    return pos, torch.sigmoid(logit).numpy()
