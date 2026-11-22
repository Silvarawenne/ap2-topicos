"""Executor para Kaggle Kernels (enviado com `kaggle kernels push`).

Uso dentro do kernel (o estágio vem da variável STAGE no topo, preenchida por kaggle/push.py):
  smoke        extração de poucos exames (mede tempo real por exame)
  eda          EDA completa (cabeçalhos DICOM de todos os exames)
  extract      extração de características (aceita SHARD/N_SHARDS)
  experiments  experimentos + figuras (lê as características da saída do kernel de extração)

Tudo é escrito em /kaggle/working (vira a saída do kernel).
"""
import glob
import os
import shutil
import subprocess
import sys
import time

STAGE = os.environ.get("STAGE", "__STAGE__")
SHARD = int(os.environ.get("SHARD", "__SHARD__"))
N_SHARDS = int(os.environ.get("N_SHARDS", "__N_SHARDS__"))
LIMIT = os.environ.get("LIMIT", "__LIMIT__")

WORK = "/kaggle/working"
REPO = f"{WORK}/ap2-topicos"
T0 = time.time()


def sh(cmd, check=True, env=None):
    print(f"\n$ {cmd}  [{(time.time() - T0) / 60:.1f} min]", flush=True)
    r = subprocess.run(cmd, shell=True, env={**os.environ, **(env or {})})
    if check and r.returncode != 0:
        raise SystemExit(f"falhou ({r.returncode}): {cmd}")
    return r.returncode


def find_one(pattern, what):
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        raise SystemExit(f"não encontrado: {what} ({pattern})")
    return hits[0]


# 1) código: copia do Dataset privado para uma pasta gravável
cfg_path = find_one("/kaggle/input/**/configs/config.yaml", "código (dataset ap2-topicos-code)")
code_root = os.path.dirname(os.path.dirname(cfg_path))
if not os.path.exists(REPO):
    shutil.copytree(code_root, REPO)
os.chdir(REPO)
sys.path.insert(0, f"{REPO}/src")

# 2) dados da competição
train_csv = find_one("/kaggle/input/**/rsna-str-pulmonary-embolism-detection*/train.csv", "train.csv")
os.environ["PE_DATA_ROOT"] = os.path.dirname(train_csv)
print("código:", code_root, "| dados:", os.environ["PE_DATA_ROOT"], flush=True)

sh(f"{sys.executable} -m pip install -q -r requirements-kaggle.txt", check=False)
sh(f"nproc; free -g | head -2; df -h {WORK} | tail -1", check=False)

if STAGE == "smoke":
    # Teste curto: partição + extração de poucos exames, só para medir tempo e validar DICOM/máscaras.
    sh(f"{sys.executable} scripts/01_make_split_and_sample.py")
    t = time.time()
    sh(f"{sys.executable} -W ignore scripts/02_extract_features.py --limit {LIMIT or 20} "
       f"--chunk-size 10 --out-dir {WORK}/smoke_features")
    print(f"\n### TEMPO extração {LIMIT or 20} exames: {(time.time() - t) / 60:.2f} min", flush=True)
    sh(f"{sys.executable} - <<'EOF'\n"
       "import pandas as pd, glob\n"
       f"df = pd.concat(pd.read_parquet(f) for f in glob.glob('{WORK}/smoke_features/*.parquet'))\n"
       "print('cortes', len(df), '| status', df.status.str.slice(0, 80).value_counts().to_dict())\n"
       "ok = df[df.status == 'ok']\n"
       "print(ok.groupby('pe_present_on_image')[['int_defect_frac','int_defect_per_vessel','int_vessel_frac','int_roi_fallback']].mean())\n"
       "print(ok[[c for c in ok.columns if c.startswith('meta_')]].astype(str).nunique())\n"
       "EOF", check=False)

elif STAGE == "eda":
    # EDA completa (lê o cabeçalho de todos os exames: pode levar bastante tempo).
    sh(f"{sys.executable} scripts/01_make_split_and_sample.py")
    import nbformat
    nb = nbformat.read("notebooks/00_eda.ipynb", 4)
    nb.cells[1].source = f"import os, sys; os.chdir('{REPO}'); sys.path.insert(0, '{REPO}/src')"
    nbformat.write(nb, "notebooks/00_eda_run.ipynb")
    sh("jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=14400 "
       "notebooks/00_eda_run.ipynb --output 00_eda_executed.ipynb", check=False)
    shutil.copytree("results", f"{WORK}/results", dirs_exist_ok=True)
    if os.path.exists("notebooks/00_eda_executed.ipynb"):
        shutil.copy("notebooks/00_eda_executed.ipynb", WORK)

elif STAGE == "extract":
    os.environ["PE_FEATURES_DIR"] = f"{WORK}/features"
    sh(f"{sys.executable} scripts/01_make_split_and_sample.py")
    sh(f"{sys.executable} -W ignore scripts/02_extract_features.py --shard {SHARD} --n-shards {N_SHARDS}")
    shutil.copytree("splits", f"{WORK}/features/splits", dirs_exist_ok=True)

elif STAGE == "experiments":
    feats = [p.rsplit("/slices", 1)[0] for p in glob.glob("/kaggle/input/**/slices", recursive=True)]
    if not feats:
        raise SystemExit("características não encontradas (adicione a saída do kernel de extração)")
    # junta shards (vários kernels de extração) em uma pasta só
    merged = f"{WORK}/features"
    os.makedirs(f"{merged}/slices", exist_ok=True)
    for f in feats:
        for p in glob.glob(f"{f}/slices/*.parquet"):
            dst = f"{merged}/slices/{os.path.basename(p)}"
            if not os.path.exists(dst):
                os.symlink(p, dst)
        if os.path.exists(f"{f}/exams.csv"):
            shutil.copy(f"{f}/exams.csv", merged)
    os.environ["PE_FEATURES_DIR"] = merged
    os.environ["PE_RESULTS_DIR"] = f"{WORK}/results"
    sh(f"{sys.executable} -W ignore scripts/03_run_experiments.py")
    sh(f"{sys.executable} -W ignore scripts/04_make_figures.py --gallery")
else:
    raise SystemExit(f"STAGE desconhecido: {STAGE}")

print(f"\n### FIM ({STAGE}) em {(time.time() - T0) / 60:.1f} min", flush=True)
