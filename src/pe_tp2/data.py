"""Leitura dos volumes e montagem das entradas 2.5D.

Entrada da rede (3 canais) = cortes [i-a, i, i+a] do volume ordenado por z, todos na janela
CTPA. Motivo: as redes pré-treinadas esperam 3 canais; replicar o mesmo corte desperdiça dois
canais, enquanto os vizinhos dão contexto de continuidade do vaso (um trombo se estende por
vários cortes; um artefato de fluxo/ruído, em geral, não).

Treino: amostragem por época. Todos os exames e todos os cortes do treino são elegíveis; a
cada época entram TODOS os cortes positivos e uma amostra nova de negativos (neg_ratio vezes o
número de positivos), sorteada com a semente (seed, época). O orçamento por época é o mesmo
para todos os modelos. Avaliação: inferência sobre TODOS os cortes dos exames avaliados.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .index import STUDY


class VolumeCache:
    """Guarda os últimos volumes lidos (cada .npz guarda vários exames; ler um = descomprimir)."""

    def __init__(self, size: int = 2):
        self.size, self.data = size, OrderedDict()

    def get(self, path: str, study: str) -> np.ndarray:
        key = (path, study)
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key]
        with np.load(path) as z:
            vol = z[study]
        self.data[key] = vol
        if len(self.data) > self.size:
            self.data.popitem(last=False)
        return vol


def stack25d(vol: np.ndarray, i: int, offset: int = 1) -> np.ndarray:
    n = vol.shape[0]
    return np.stack([vol[max(i - offset, 0)], vol[i], vol[min(i + offset, n - 1)]])


def _worker_share(items: list):
    info = get_worker_info()
    if info is None:
        return items
    return items[info.id::info.num_workers]


def exam_groups(idx: pd.DataFrame) -> list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]]:
    """[(study, vol_path, vol_idx[], rótulo_corte[], posição_no_índice[])] só com cortes válidos."""
    ok = idx[idx["ok"]]
    out = []
    for study, g in ok.groupby(STUDY, sort=True):
        out.append((study, g["vol_path"].iloc[0], g["vol_idx"].to_numpy(int),
                    g["pe_present_on_image"].to_numpy(int), g.index.to_numpy()))
    return out


class TrainSlices(IterableDataset):
    def __init__(self, idx: pd.DataFrame, neg_ratio: float, seed: int, transform=None,
                 offset: int = 1, buffer: int = 1024):
        self.groups = exam_groups(idx)
        self.neg_ratio, self.seed, self.transform = neg_ratio, seed, transform
        self.offset, self.buffer, self.epoch = offset, buffer, 0
        self.n_pos = int(sum((g[3] == 1).sum() for g in self.groups))
        self.n_neg_total = int(sum((g[3] == 0).sum() for g in self.groups))
        self.q = min(1.0, neg_ratio * max(self.n_pos, 1) / max(self.n_neg_total, 1))

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):   # tamanho esperado da época (o sorteio de negativos é Bernoulli)
        return int(self.n_pos + self.q * self.n_neg_total)

    def plan(self) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
        """Seleção da época: igual em todos os workers (mesma semente), depois repartida."""
        rng = np.random.default_rng([self.seed, self.epoch])
        plan = []
        for study, path, vidx, y, _ in self.groups:
            keep = (y == 1) | (rng.random(len(y)) < self.q)
            if keep.any():
                plan.append((study, path, vidx[keep], y[keep]))
        order = rng.permutation(len(plan))
        return [plan[i] for i in order]

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng([self.seed, self.epoch, 1000 + wid])
        cache, buf = VolumeCache(1), []
        for study, path, vidx, y in _worker_share(self.plan()):
            vol = cache.get(path, study)
            for i, lab in zip(vidx, y):
                x = torch.from_numpy(stack25d(vol, int(i), self.offset))
                x = self.transform(x) if self.transform else x.float().div_(255)
                buf.append((x.half(), torch.tensor(float(lab))))
                if len(buf) >= self.buffer:
                    yield buf.pop(int(rng.integers(len(buf))))
        rng.shuffle(buf)
        yield from buf


class EvalSlices(IterableDataset):
    """Todos os cortes válidos dos exames dados; devolve (x, posição no índice)."""

    def __init__(self, idx: pd.DataFrame, offset: int = 1):
        self.groups, self.offset = exam_groups(idx), offset
        self.n = int(sum(len(g[2]) for g in self.groups))

    def __len__(self):
        return self.n

    def __iter__(self):
        cache = VolumeCache(1)
        for study, path, vidx, _, pos in _worker_share(self.groups):
            vol = cache.get(path, study)
            for i, p in zip(vidx, pos):
                x = torch.from_numpy(stack25d(vol, int(i), self.offset)).float().div_(255)
                yield x.half(), int(p)


def loader(ds, batch_size: int, num_workers: int):
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                                       pin_memory=torch.cuda.is_available(),
                                       persistent_workers=False,
                                       prefetch_factor=4 if num_workers else None)


def steps_per_epoch(ds: TrainSlices, batch_size: int) -> int:
    return max(1, math.ceil(len(ds) / batch_size))
