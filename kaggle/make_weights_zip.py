"""Empacota os pesos e as predições de validação de uma configuração (3 sementes) para subir
como Dataset do Kaggle (entrada do notebook de submissão, que roda sem internet).

  python kaggle/make_weights_zip.py --config ft_dinov2_s_llrd_main --roots kaggle_outputs \
      --out kaggle/tp2-pesos-dinov2-llrd.zip

Por semente: model.pt, exam_val.csv (calibração de Platt), exam_official_test.csv (conferência
DICOM -> rede x extração), run.json e metrics.json.
"""
import argparse
import zipfile
from pathlib import Path

KEEP = ("model.pt", "exam_val.csv", "exam_official_test.csv", "run.json", "metrics.json")

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="ft_dinov2_s_llrd_main")
ap.add_argument("--roots", nargs="+", default=["kaggle_outputs"])
ap.add_argument("--out", default="kaggle/tp2-pesos-dinov2-llrd.zip")
a = ap.parse_args()
dirs = {}
for r in a.roots:
    for d in sorted(Path(r).rglob(f"{a.config}_s*")):
        if (d / "model.pt").exists():
            dirs.setdefault(d.name, d)                 # a mesma execução baixada duas vezes: a 1a
if not dirs:
    raise SystemExit(f"nenhuma execução {a.config}_s* com model.pt em {a.roots}")
with zipfile.ZipFile(a.out, "w", compression=zipfile.ZIP_STORED) as z:
    for name, d in sorted(dirs.items()):
        for f in KEEP:
            if (d / f).exists():
                z.write(d / f, arcname=f"{name}/{f}")
        print(name, "<-", d)
print("gravado:", a.out, f"({Path(a.out).stat().st_size / 1e6:.0f} MB)")
