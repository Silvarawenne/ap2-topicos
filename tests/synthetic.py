"""Gera um mini conjunto sintético com a MESMA estrutura do RSNA-STR PE (train.csv +
train/<Study>/<Series>/<SOP>.dcm), para testar o pipeline sem os dados reais.

Cada corte: corpo elíptico (~40 HU), dois pulmões (~-850 HU), artéria contrastada
(~350 HU) e, nos cortes positivos, uma falha de enchimento (~60 HU) dentro da artéria.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

COLS = ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "pe_present_on_image",
        "negative_exam_for_pe", "qa_motion", "qa_contrast", "flow_artifact", "rv_lv_ratio_gte_1",
        "rv_lv_ratio_lt_1", "leftsided_pe", "chronic_pe", "true_filling_defect_not_pe",
        "rightsided_pe", "acute_and_chronic_pe", "central_pe", "indeterminate"]


def make_image(rng, size, positive):
    yy, xx = np.mgrid[:size, :size]
    c = size / 2
    img = np.full((size, size), -1000.0)
    body = ((yy - c) / (0.42 * size)) ** 2 + ((xx - c) / (0.48 * size)) ** 2 < 1
    img[body] = 40
    for sx in (-1, 1):
        lung = ((yy - c) / (0.30 * size)) ** 2 + ((xx - c - sx * 0.22 * size) / (0.16 * size)) ** 2 < 1
        img[lung] = -850
    vy, vx, vr = c - 0.05 * size, c + rng.uniform(-0.03, 0.03) * size, 0.06 * size
    vessel = (yy - vy) ** 2 + (xx - vx) ** 2 < vr ** 2
    img[vessel] = 350 + rng.normal(0, 20)
    if positive:
        d = (yy - vy) ** 2 + (xx - vx - vr * 0.3) ** 2 < (vr * 0.5) ** 2
        img[d] = 60
    img += rng.normal(0, 15, img.shape)
    return img


def write_dcm(path, hu, study, series, sop, z, patient):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    meta.MediaStorageSOPInstanceUID = sop
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.SOPInstanceUID = study, series, sop
    ds.PatientID, ds.Modality, ds.Manufacturer = patient, "CT", "SYNTH"
    ds.Rows, ds.Columns = hu.shape
    ds.PixelSpacing = [0.7, 0.7]
    ds.SliceThickness = 1.25
    ds.ImagePositionPatient = [0.0, 0.0, float(z)]
    ds.InstanceNumber = 1
    ds.RescaleSlope, ds.RescaleIntercept = 1, -1024
    ds.WindowCenter, ds.WindowWidth = 40, 400
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
    ds.PixelRepresentation, ds.SamplesPerPixel = 0, 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = np.clip(hu + 1024, 0, 65535).astype(np.uint16).tobytes()
    ds.save_as(str(path), enforce_file_format=True)


def generate(root: Path, n_exams=40, n_slices=12, size=128, seed=0, pos_frac=0.35, indet=6):
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_exams):
        study, series = generate_uid(), generate_uid()
        label = "pe" if e < int(pos_frac * n_exams) else ("indet" if e >= n_exams - indet else "neg")
        d = root / "train" / study / series
        d.mkdir(parents=True, exist_ok=True)
        pos_slices = set(rng.choice(n_slices, size=max(2, n_slices // 4), replace=False)) if label == "pe" else set()
        zs = rng.permutation(n_slices)  # arquivos fora de ordem, como no conjunto real
        for i in range(n_slices):
            sop = generate_uid()
            positive = i in pos_slices
            write_dcm(d / f"{sop}.dcm", make_image(rng, size, positive), study, series, sop,
                      -2.5 * zs[i], f"P{e:04d}")
            r = dict.fromkeys(COLS, 0)
            r.update(StudyInstanceUID=study, SeriesInstanceUID=series, SOPInstanceUID=sop,
                     pe_present_on_image=int(positive), negative_exam_for_pe=int(label == "neg"),
                     indeterminate=int(label == "indet"),
                     central_pe=int(label == "pe" and e % 3 == 0),
                     rightsided_pe=int(label == "pe"), qa_contrast=int(e % 7 == 0))
            rows.append(r)
    pd.DataFrame(rows, columns=COLS).to_csv(root / "train.csv", index=False)
    return root


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n-exams", type=int, default=40)
    ap.add_argument("--n-slices", type=int, default=12)
    a = ap.parse_args()
    generate(Path(a.out), a.n_exams, a.n_slices)
    print("ok", a.out)
