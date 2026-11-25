"""Testes do TP2 com dados sintéticos (sem internet: pesos aleatórios só aqui).  pytest -q"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from pe_baseline.config import load_config  # noqa: E402
from pe_baseline.data import exam_table  # noqa: E402
from pe_baseline.extract import run_extraction  # noqa: E402
from synthetic import generate  # noqa: E402


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Sintético -> extração completa com volumes -> partição -> config temporário."""
    root = tmp_path_factory.mktemp("tp2")
    data = generate(root / "data", n_exams=24, n_slices=6, size=96, seed=3, indet=2)
    train = pd.read_csv(data / "train.csv").assign(source="train")
    cfg = load_config()
    cfg["data"]["root"] = str(data)
    studies = sorted(train["StudyInstanceUID"].unique())
    # 3 exames fazem o papel do "teste oficial" (sem rótulo)
    off = studies[-3:]
    table = train.copy()
    table.loc[table["StudyInstanceUID"].isin(off), "source"] = "test"
    cfg["full"]["test_dir"] = "train"
    run_extraction(cfg, table, studies, root / "inputs" / "shard0" / "slices", chunk_size=8,
                   n_jobs=1, verbose=False, stride=1, save_volumes=True)
    exams = exam_table(train[~train["StudyInstanceUID"].isin(off)])
    # partição determinística (os UIDs sintéticos são aleatórios): rodízio dentro de cada classe,
    # garantindo positivos e negativos em train/val/test
    split = exams[["StudyInstanceUID", "label3", "pe", "n_slices"]].copy()
    split["split"] = split.groupby("label3").cumcount().map(lambda i: ["train", "val", "train", "test"][i % 4])
    split.to_csv(root / "split.csv", index=False)
    cfg["split"]["file"] = str(root / "split.csv")
    cfg["tp2"].update(inputs=str(root / "inputs"), out_dir=str(root / "runs"), epochs=2,
                      batch_size=4, eval_batch_size=8, num_workers=0, neg_ratio=1, n_bootstrap=50)
    cfg["tp2"]["probe"].update(epochs=2, batch_size=16, weight_decays=[1e-3])
    (root / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    return root, cfg


def test_index_and_counts(world):
    from pe_tp2.index import counts, find_slice_dirs, load_index
    root, cfg = world
    dirs = find_slice_dirs([root / "inputs"])
    assert len(dirs) == 1
    idx = load_index(dirs, root / "split.csv")
    assert set(idx["split"]) == {"train", "val", "test", "official_test"}
    assert (idx.loc[idx["split"] == "official_test", "pe"] == -1).all()
    assert "indeterminate" not in set(idx["label3"])
    # nenhum exame em duas partições
    assert idx.groupby("StudyInstanceUID")["split"].nunique().max() == 1
    c = counts(idx)
    assert c.loc["official_test", "exams"] == 3


def test_epoch_sampling_keeps_all_positives(world):
    from pe_tp2.data import TrainSlices
    from pe_tp2.index import find_slice_dirs, load_index
    root, _ = world
    idx = load_index(find_slice_dirs([root / "inputs"]), root / "split.csv")
    tr = idx[idx["split"] == "train"]
    ds = TrainSlices(tr, neg_ratio=1, seed=0, buffer=4)
    ys = [float(y) for _, y in ds]
    assert sum(ys) == ds.n_pos                                  # todos os positivos
    ds.set_epoch(1)
    p0, p1 = ds.plan(), (ds.set_epoch(0), ds.plan())[1]
    assert [len(a[2]) for a in p0] != [] and p1 is not None
    x, _ = next(iter(ds))
    assert x.shape == (3, 256, 256) and x.dtype == torch.float16


def test_augment_policies_shape():
    from pe_tp2 import augment
    x = torch.randint(0, 255, (3, 256, 256), dtype=torch.uint8)
    for pol in ("none", "main", "strong"):
        y = augment.build(pol)(x)
        assert y.shape == x.shape and y.dtype == torch.float32 and 0 <= y.min() and y.max() <= 1


def test_param_groups_llrd():
    from pe_tp2.models import Net, param_groups
    m = Net("deit_s", pretrained=False)
    g = param_groups(m, 1e-3, 0.05, llrd=0.5)
    lrs = sorted({round(x["lr"], 12) for x in g})
    assert max(lrs) == pytest.approx(1e-3) and min(lrs) < 1e-6   # entrada << cabeça
    n = sum(p.numel() for x in g for p in x["params"])
    assert n == sum(p.numel() for p in m.parameters())


def test_delong_matches_auc():
    from sklearn.metrics import roc_auc_score
    from pe_tp2.stats import delong, paired_bootstrap
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s1 = y + rng.normal(0, 1, 300)
    s2 = y + rng.normal(0, 3, 300)
    r = delong(y, s1, s2)
    assert r["auc1"] == pytest.approx(roc_auc_score(y, s1))
    assert r["p"] < 0.05 and paired_bootstrap(y, s1, s2, 300)["diff"] > 0
    assert delong(y, s1, s1)["p"] == pytest.approx(1.0)


@pytest.mark.parametrize("cmd", [
    ["finetune", "--model", "resnet50", "--strategy", "lpft", "--seeds", "0", "--max-steps", "2"],
    ["finetune", "--model", "dinov2_s", "--strategy", "llrd", "--aug", "strong", "--seeds", "1", "--max-steps", "2"],
    ["frozen", "--model", "resnet50", "--seeds", "0"],
])
def test_script_end_to_end(world, cmd):
    root, _ = world
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "10_tp2.py"), *cmd, "--config",
                        str(root / "cfg.yaml"), "--no-pretrained"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    runs = sorted((root / "runs").iterdir())
    last = max(runs, key=lambda p: p.stat().st_mtime)
    m = __import__("json").loads((last / "metrics.json").read_text())
    assert "test" in m and 0 <= m["test"]["auc_roc"] <= 1
    ex = pd.read_csv(last / "exam_official_test.csv")
    assert len(ex) == 3 and (ex["pe"] == -1).all()
    assert (last / "slices_official_test.parquet").exists()
