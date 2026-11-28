"""Submissão tardia com inferência a partir dos DICOMs (roda DENTRO do notebook de submissão).

A competição RSNA-STR PE é de código: ao submeter, o Kaggle roda o notebook de novo, sem
internet, sobre o teste OCULTO (650 exames públicos + ~1.517 privados). Predições pré-calculadas
dos 650 exames públicos não servem; este script refaz tudo a partir dos DICOMs de test.csv:

  DICOM -> volume uint8 256x256 na janela CTPA (o mesmo da extração; pe_tp2.dicom_volume)
        -> pilhas 2.5D [i-1, i, i+1] -> cada modelo (sementes) -> p por corte e top-k por exame
        -> média das sementes -> calibração (prior do corte + Platt da validação) -> 10 rótulos
        -> regras de consistência -> submission.csv (na ordem do sample_submission)

  python scripts/13_kaggle_infer.py --runs /kaggle/input/<pesos>/ft_dinov2_s_llrd_main_s* \
      --out /kaggle/working

Na submissão o notebook não pode usar o treino da competição: as prevalências do treino vêm de
priors.json (gravado pelo notebook 03 ao lado do código, ou --priors); sem ele, são calculadas
de train.csv (uso local/testes).
Cada pasta de --runs precisa de model.pt e exam_val.csv (calibração). Se tiver também
exam_official_test.csv (predições do treino para os 650 exames públicos), o relatório mostra a
concordância entre este caminho (DICOM -> rede) e o da extração, como verificação.
Proteção de tempo (--time-limit-h): passado o limite, os exames restantes recebem as prevalências
do treino (uma submissão pior, mas válida, em vez de estourar o tempo do Kaggle).
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pe_baseline.config import data_path, load_config, repo_path  # noqa: E402
from pe_tp2 import submission as S  # noqa: E402
from pe_tp2.dicom_volume import read_exam  # noqa: E402

STUDY, SOP, SERIES = S.STUDY, S.SOP, "SeriesInstanceUID"


def _decode(job):
    study, exam_dir, sops, kw = job
    t0 = time.time()
    order, vol, ok = read_exam(exam_dir, sops, **kw)
    return study, order, vol, ok, time.time() - t0


def load_models(runs, devices):
    import torch
    from pe_tp2.models import Net
    models = []
    for i, r in enumerate(runs):
        ck = torch.load(Path(r) / "model.pt", map_location="cpu", weights_only=True)
        m = Net(ck["model"], pretrained=False)
        m.load_state_dict(ck["state_dict"])
        dev = devices[i % len(devices)]
        m = m.float().to(dev).eval()
        if m.family == "cnn":
            m = m.to(memory_format=torch.channels_last)
        models.append((m, dev))
    return models


def predict_exam(models, vol, ok, batch, offset=1):
    """-> matriz (n_modelos, n_cortes_ok) de probabilidades."""
    import torch
    n = vol.shape[0]
    keep = np.flatnonzero(ok)
    out = np.zeros((len(models), len(keep)), np.float32)
    if not len(keep):
        return out
    i = torch.from_numpy(keep)
    prev, nxt = (i - offset).clamp(min=0), (i + offset).clamp(max=n - 1)
    v_cpu = torch.from_numpy(vol)
    with torch.inference_mode():
        for mi, (m, dev) in enumerate(models):
            v = v_cpu.to(dev, non_blocking=True)
            for s in range(0, len(keep), batch):
                sl = slice(s, s + batch)
                x = torch.stack([v[prev[sl].to(dev)], v[i[sl].to(dev)], v[nxt[sl].to(dev)]], 1)
                x = x.float().div_(255)
                if m.family == "cnn":
                    x = x.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    p = torch.sigmoid(m(x).float())
                out[mi, sl] = p.cpu().numpy()
    return out


def find_root(cfg):
    """Raiz da competição: a do config (ou PE_DATA_ROOT) ou a primeira com train.csv + test.csv."""
    root = Path(data_path(cfg, "."))
    if (root / "test.csv").exists():
        return root
    for depth in range(0, 3):          # sem busca recursiva: a pasta tem ~1,8 milhão de DICOMs
        pat = "/kaggle/input/" + "*/" * depth + "rsna-str-pulmonary-embolism-detection*/test.csv"
        for f in sorted(glob.glob(pat)):
            return Path(f).parent
    raise SystemExit(f"test.csv não encontrado (config: {root})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="pastas (ou padrões glob) com model.pt e exam_val.csv")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--workers", type=int, default=4, help="processos de leitura dos DICOMs (0 = no principal)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="só os N primeiros exames (teste rápido)")
    ap.add_argument("--time-limit-h", type=float, default=8.0)
    ap.add_argument("--priors", default=None, help="priors.json (padrão: <código>/priors.json, se existir)")
    args = ap.parse_args()
    t_start = time.time()
    import torch

    cfg = load_config(args.config)
    t = cfg["tp2"]
    runs = sorted({p for r in args.runs for p in (glob.glob(r) or [r]) if (Path(p) / "model.pt").exists()})
    if not runs:
        raise SystemExit(f"nenhum model.pt em {args.runs}")
    root = Path(args.data_root) if args.data_root else find_root(cfg)
    test = pd.read_csv(root / cfg["full"]["test_csv"])
    sample_f = root / "sample_submission.csv"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pri_f = Path(args.priors) if args.priors else ROOT / "priors.json"
    if pri_f.exists():
        priors = json.loads(pri_f.read_text(encoding="utf-8"))
    else:
        train = pd.read_csv(root / cfg["data"]["train_csv"])
        priors = S.train_priors(train, pd.read_csv(repo_path(cfg["split"]["file"])), t["neg_ratio"])
    cal = S.fit_platt(*S.val_exam_scores(runs, t["topk"])[["pe", "score"]].T.to_numpy())
    n_gpu = torch.cuda.device_count()
    devices = [torch.device(f"cuda:{g}") for g in range(n_gpu)] or [torch.device("cpu")]
    models = load_models(runs, devices)
    print(f"{len(runs)} modelos em {[str(d) for d in devices]} | dados: {root} | "
          f"{test[STUDY].nunique()} exames, {len(test)} imagens", flush=True)

    pre = cfg["preprocess"]
    lvl, wid = pre["windows"][cfg["full"].get("volume_window", "ctpa")]
    kw = dict(size=int(pre["out_size"]), hu_clip=tuple(pre["hu_clip"]), level=lvl, width=wid)
    test_dir = cfg["full"]["test_dir"]
    jobs = [(s, root / test_dir / s / g[SERIES].iloc[0], g[SOP].tolist(), kw)
            for s, g in test.groupby(STUDY, sort=True)]
    if args.limit:
        jobs = jobs[:args.limit]

    k, limit_s = t["topk"], args.time_limit_h * 3600
    slices, per_model, stopped, n_bad, t_dec = [], {i: {} for i in range(len(runs))}, 0, 0, 0.0
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
    pool = ctx.Pool(args.workers) if (args.workers and ctx) else None
    it = pool.imap(_decode, jobs, chunksize=1) if pool else map(_decode, jobs)
    try:
        for j, (study, order, vol, ok, dt) in enumerate(it, 1):
            t_dec += dt
            n_bad += int((~ok).sum())
            probs = predict_exam(models, vol, ok, args.batch, t["offset"])
            sops_ok = [s for s, o in zip(order, ok) if o]
            if sops_ok:
                slices.append(pd.DataFrame({STUDY: study, SOP: sops_ok, "p": probs.mean(0)}))
                for mi in range(len(runs)):
                    per_model[mi][study] = float(np.sort(probs[mi])[::-1][:k].mean())
            el = time.time() - t_start
            if j % 25 == 0 or j == len(jobs):
                eta = el / j * (len(jobs) - j)
                print(f"{j}/{len(jobs)} exames | {el / 60:.1f} min | faltam ~{eta / 60:.0f} min | "
                      f"cortes ilegíveis: {n_bad}", flush=True)
            if el > limit_s and j < len(jobs):
                stopped = len(jobs) - j
                print(f"LIMITE DE TEMPO: {stopped} exames ficam com as prevalências do treino", flush=True)
                break
    finally:
        if pool:
            pool.terminate()
            pool.join()

    sl = pd.concat(slices, ignore_index=True) if slices else pd.DataFrame(columns=[STUDY, SOP, "p"])
    scores = pd.DataFrame(per_model)
    exam_score = scores.mean(axis=1)
    e, im = S.build(sl, exam_score, priors, cal, test[[STUDY, SOP]])
    sample = pd.read_csv(sample_f) if sample_f.exists() and not args.limit else None
    if args.limit:                                  # teste rápido: só os exames processados
        done = set(exam_score.index)
        e = e[e.index.isin(done)]
        im = im[im[STUDY].isin(done)]
    sub, cons = S.finalize(e, im, sample)
    sub.to_csv(out / "submission.csv", index=False)

    # concordância com as predições do treino (mesmos exames públicos), se disponíveis
    parity = {}
    for mi, r in enumerate(runs):
        f = Path(r) / "exam_official_test.csv"
        if f.exists():
            ref = pd.read_csv(f).set_index(STUDY)[f"top{k}"]
            both = pd.concat([scores[mi].rename("dicom"), ref.rename("extracao")], axis=1, join="inner")
            if len(both) > 1:
                parity[Path(r).name] = {"n_exams": len(both),
                                        "pearson": float(both.corr().iloc[0, 1]),
                                        "max_abs_diff": float((both["dicom"] - both["extracao"]).abs().max()),
                                        "mean_abs_diff": float((both["dicom"] - both["extracao"]).abs().mean())}
    rep = {"runs": runs, "priors_from": str(pri_f) if pri_f.exists() else "train.csv", "devices": [str(d) for d in devices], "n_exams": len(jobs),
           "exams_without_prediction": int(len(e) - exam_score.reindex(e.index).notna().sum()),
           "stopped_by_time_limit": stopped, "unreadable_slices": n_bad, "rows": len(sub),
           "checked_against_sample": sample is not None, "consistency": cons,
           "exams_positive": int((e["negative_exam_for_pe"] < 0.5).sum()), "priors": priors, "platt_val": cal,
           "parity_vs_extraction": parity, "decode_cpu_minutes": t_dec / 60,
           "total_minutes": (time.time() - t_start) / 60}
    (out / "submission_report.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k_: v for k_, v in rep.items() if k_ not in ("priors",)}, indent=2, default=float))
    print("gravado:", out / "submission.csv")


if __name__ == "__main__":
    main()
