"""Passo 4 — figuras e tabelas formatadas a partir de results/ (não retreina nada).

  python scripts/04_make_figures.py            # figuras a partir das tabelas
  python scripts/04_make_figures.py --gallery  # + galeria de acertos/erros (precisa dos DICOMs)

Critério da galeria (definido ANTES de olhar as imagens): no conjunto de teste, os k exames
de maior confiança em cada célula da matriz de confusão (VP, VN, FP, FN) segundo o modelo
final; para cada exame mostra-se o corte de maior probabilidade do classificador de corte.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,  # noqa: E402
                             RocCurveDisplay, confusion_matrix)

from pe_baseline.config import data_path, load_config, repo_path  # noqa: E402

FAMILY_PT = {"intensity": "Intensidade", "texture": "Textura", "gradient": "Gradiente",
             "wavelet": "Wavelet", "all": "Todas"}


def fmt(m, s, d=3):
    return f"{m:.{d}f} ± {s:.{d}f}"


def tables_markdown(res: Path, fig: Path):
    md = []
    grid = pd.read_csv(res / "table_exam_cv.csv")
    for metric in ["auc_roc", "auc_pr", "balanced_acc", "sensitivity", "specificity", "f1"]:
        piv = grid.assign(v=[fmt(m, s) for m, s in zip(grid[f"{metric}_mean"], grid[f"{metric}_std"])]) \
            .pivot(index="family", columns="model", values="v")
        md.append(f"### Exame — {metric} (CV aninhada, média ± dp entre dobras)\n\n" + piv.to_markdown() + "\n")
    for name, title in [("table_slice_cv.csv", "Corte — CV"), ("table_slice2exam_cv.csv", "Corte → exame — CV"),
                        ("table_ablations.csv", "Ablações (melhor configuração)"),
                        ("table_test_final.csv", "Teste final (avaliado uma única vez)"),
                        ("table_test_slice.csv", "Teste — nível de corte"),
                        ("table_subgroups.csv", "Análise por subgrupo (teste)"),
                        ("table_importance_groups.csv", "Importância por permutação de grupo (teste)")]:
        p = res / name
        if p.exists():
            df = pd.read_csv(p).drop(columns=["params"], errors="ignore")
            md.append(f"### {title}\n\n" + df.to_markdown(index=False, floatfmt=".3f") + "\n")
    (res / "RESULTS.md").write_text("# Resultados do TP1 (gerado automaticamente)\n\n" + "\n".join(md))

    heat = grid.pivot(index="family", columns="model", values="auc_roc_mean")
    heat = heat.reindex([f for f in FAMILY_PT if f in heat.index])
    plt.figure(figsize=(7, 3.6))
    sns.heatmap(heat.rename(index=FAMILY_PT), annot=True, fmt=".3f", cmap="Blues", vmin=0.5,
                cbar_kws={"label": "AUC-ROC (média CV)"})
    plt.title("AUC-ROC por descritor × modelo (nível de exame)")
    plt.xlabel("modelo"); plt.ylabel("família de descritores")
    plt.tight_layout(); plt.savefig(fig / "fig_grid_auc.png", dpi=200); plt.close()


def curves(res: Path, fig: Path):
    cfgf = json.load(open(res / "final_config.json"))
    pred = pd.read_csv(res / "test_predictions.csv")
    y = pred["pe"]
    fig_, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    series = [(f"score_{cfgf['best_model']}", f"{FAMILY_PT.get(cfgf['best_family'])}+{cfgf['best_model']}"),
              ("slice_top5", f"corte({cfgf['best_slice_model']})→exame top-5"),
              ("score_dummy", "baseline trivial")]
    for col, name in series:
        if col in pred:
            RocCurveDisplay.from_predictions(y, pred[col], name=name, ax=axes[0])
            PrecisionRecallDisplay.from_predictions(y, pred[col], name=name, ax=axes[1])
    axes[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    axes[1].axhline(y.mean(), color="k", ls=":", lw=0.8)
    axes[0].set_title("ROC — conjunto de teste"); axes[1].set_title("Precisão-Revocação — teste")
    plt.tight_layout(); plt.savefig(fig / "fig_roc_pr_test.png", dpi=200); plt.close()

    thr = cfgf["threshold"]
    cm = confusion_matrix(y, (pred[f"score_{cfgf['best_model']}"] >= thr).astype(int))
    ConfusionMatrixDisplay(cm, display_labels=["sem TEP", "TEP"]).plot(cmap="Blues", colorbar=False)
    plt.title("Matriz de confusão — teste (limiar escolhido na CV interna)")
    plt.tight_layout(); plt.savefig(fig / "fig_confusion_test.png", dpi=200); plt.close()

    oof = pd.read_parquet(res / "oof_exam.parquet")
    plt.figure(figsize=(5.5, 4.4))
    for fam in FAMILY_PT:
        d = oof[(oof.family == fam) & (oof.model == cfgf["best_model"])]
        if len(d):
            RocCurveDisplay.from_predictions(d["y"], d["score"], name=FAMILY_PT[fam], ax=plt.gca())
    plt.plot([0, 1], [0, 1], "k:", lw=0.8)
    plt.title(f"ROC fora-da-dobra (dev) por família — {cfgf['best_model']}")
    plt.tight_layout(); plt.savefig(fig / "fig_roc_families_cv.png", dpi=200); plt.close()

    p = res / "table_importance_groups.csv"
    if p.exists():
        imp = pd.read_csv(p)
        plt.figure(figsize=(6, 3.6))
        plt.barh(imp["group"], imp["auc_drop_mean"], xerr=imp["auc_drop_std"], color="#4C72B0")
        plt.gca().invert_yaxis()
        plt.xlabel("queda de AUC-ROC ao permutar o grupo")
        plt.title("Importância por permutação de grupo (teste)")
        plt.tight_layout(); plt.savefig(fig / "fig_importance_groups.png", dpi=200); plt.close()

    p = res / "table_subgroups.csv"
    if p.exists():
        sg = pd.read_csv(p).dropna(subset=["value"])
        plt.figure(figsize=(6.5, 0.32 * len(sg) + 1))
        colors = ["#DD8452" if m == "sensibilidade" else "#4C72B0" for m in sg["metric"]]
        plt.barh([f"{s} (n={n})" for s, n in zip(sg["subgroup"], sg["n"])], sg["value"], color=colors)
        plt.gca().invert_yaxis(); plt.xlim(0, 1)
        plt.xlabel("laranja: sensibilidade | azul: AUC-ROC")
        plt.title("Desempenho por subgrupo (teste)")
        plt.tight_layout(); plt.savefig(fig / "fig_subgroups.png", dpi=200); plt.close()


def gallery(cfg: dict, res: Path, fig: Path, k: int = 3):
    from pe_baseline.dicom_io import read_slice, window
    from pe_baseline.preprocess import compute_masks, resize_hu

    cfgf = json.load(open(res / "final_config.json"))
    pred = pd.read_csv(res / "test_predictions.csv")
    sp = pd.read_parquet(res / "test_slice_predictions.parquet")
    split = pd.read_csv(data_path(cfg, cfg["data"]["train_csv"]),
                        usecols=["StudyInstanceUID", "SeriesInstanceUID"]).drop_duplicates()
    col, thr = f"score_{cfgf['best_model']}", cfgf["threshold"]
    pred["pred"] = (pred[col] >= thr).astype(int)
    cells = {"VP": (1, 1, False), "VN": (0, 0, True), "FP": (0, 1, False), "FN": (1, 0, True)}
    fig_, axes = plt.subplots(4, k, figsize=(3.2 * k, 13))
    pre = cfg["preprocess"]
    for r, (name, (yt, yp, asc)) in enumerate(cells.items()):
        sel = pred[(pred.pe == yt) & (pred.pred == yp)].sort_values(col, ascending=asc).head(k)
        for c in range(k):
            ax = axes[r, c]; ax.axis("off")
            if c >= len(sel):
                continue
            ex = sel.iloc[c]
            s = sp[sp.StudyInstanceUID == ex.StudyInstanceUID].sort_values("score").iloc[-1]
            series = split.loc[split.StudyInstanceUID == ex.StudyInstanceUID, "SeriesInstanceUID"].iloc[0]
            path = data_path(cfg, cfg["data"]["train_dir"], ex.StudyInstanceUID, series, f"{s.SOPInstanceUID}.dcm")
            hu = resize_hu(read_slice(path, tuple(pre["hu_clip"])).hu, pre["out_size"])
            m = compute_masks(hu, pre)
            ax.imshow(window(hu, *pre["windows"]["ctpa"]), cmap="gray")
            ax.contour(m.roi, levels=[0.5], colors="yellow", linewidths=0.6)
            ax.contour(m.defect, levels=[0.5], colors="red", linewidths=0.6)
            ax.set_title(f"{name} | p_exame={ex[col]:.2f}\ncorte p={s.score:.2f} "
                         f"rótulo={int(s.pe_present_on_image)}", fontsize=8)
    plt.suptitle("Galeria (teste): ROI em amarelo, candidatos a falha de enchimento em vermelho", fontsize=10)
    plt.tight_layout(); plt.savefig(fig / "fig_error_gallery.png", dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--gallery", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = repo_path(cfg["results_dir"])
    fig = res / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    tables_markdown(res, fig)
    curves(res, fig)
    if args.gallery:
        gallery(cfg, res, fig)
    print(f"Figuras em {fig} | tabelas em {res / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
