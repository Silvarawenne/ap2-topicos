"""Executor para Kaggle Kernels (enviado com `kaggle kernels push`).

Uso dentro do kernel (o estágio vem da variável STAGE no topo, preenchida por kaggle/push.py):
  smoke        extração de poucos exames (mede tempo real por exame)
  eda          EDA completa (cabeçalhos DICOM de todos os exames)
  extract      extração de características (aceita SHARD/N_SHARDS)
  gallery      galeria de acertos/erros do teste (lê results/ do dataset de código + DICOMs)
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

sh(f"timeout 300 {sys.executable} -m pip install -q -r requirements-kaggle.txt", check=False)
sh(f"nproc; free -g | head -2; df -h {WORK} | tail -1", check=False)

if STAGE == "smoke":
    # Diagnóstico curto com cronômetro por etapa (cada linha é impressa na hora).
    sh(f"{sys.executable} -u scripts/01_make_split_and_sample.py | tail -3")
    sh(f"{sys.executable} -u - <<'EOF'\n"
       "import time, glob, os, sys\n"
       "sys.path.insert(0, 'src')\n"
       "import pandas as pd, numpy as np\n"
       "from pe_baseline.config import load_config, data_path\n"
       "from pe_baseline.dicom_io import read_slice\n"
       "from pe_baseline.extract import slice_features\n"
       "cfg = load_config()\n"
       "t = time.time(); tr = pd.read_csv(data_path(cfg, 'train.csv'), nrows=2000); print('ler train.csv (2000 linhas): %.1fs' % (time.time() - t), flush=True)\n"
       "r = tr.iloc[0]\n"
       "d = data_path(cfg, 'train', r.StudyInstanceUID, r.SeriesInstanceUID)\n"
       "t = time.time(); files = sorted(os.listdir(d)); print('listar pasta (%d arquivos): %.2fs' % (len(files), time.time() - t), flush=True)\n"
       "for i in range(3):\n"
       "    t = time.time(); sl = read_slice(d / files[i * 40]); print('ler 1 DICOM #%d: %.3fs shape=%s HU[%.0f,%.0f] %s' % (i, time.time() - t, sl.hu.shape, sl.hu.min(), sl.hu.max(), sl.meta.get('Manufacturer')), flush=True)\n"
       "t = time.time(); f = slice_features(sl.hu, cfg); print('descritores de 1 corte: %.3fs (%d feats)' % (time.time() - t, len(f)), flush=True)\n"
       "t = time.time()\n"
       "for k in range(0, 60, 2): read_slice(d / files[k])\n"
       "print('ler 30 DICOMs em sequência: %.2fs' % (time.time() - t), flush=True)\n"
       "EOF", check=False)
    t = time.time()
    sh(f"{sys.executable} -u -W ignore scripts/02_extract_features.py --limit {LIMIT or 5} "
       f"--chunk-size 5 --n-jobs 4 --out-dir {WORK}/smoke_features", check=False)
    print(f"\n### TEMPO extração {LIMIT or 5} exames: {(time.time() - t) / 60:.2f} min", flush=True)

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
elif STAGE == "gallery":
    os.environ["PE_RESULTS_DIR"] = f"{WORK}/results"
    shutil.copytree("results", f"{WORK}/results", dirs_exist_ok=True)
    sh(f"{sys.executable} -W ignore scripts/04_make_figures.py --gallery")
else:
    raise SystemExit(f"STAGE desconhecido: {STAGE}")

print(f"\n### FIM ({STAGE}) em {(time.time() - T0) / 60:.1f} min", flush=True)
