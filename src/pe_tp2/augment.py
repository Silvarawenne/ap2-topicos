"""Políticas de aumento de dados (aplicadas SÓ no treino, por corte 2.5D).

Executadas na GPU, em lote, com parâmetros sorteados por amostra: os workers da CPU só
descomprimem os volumes e montam as pilhas 2.5D (no Kaggle há 4 vCPUs para alimentar 2 GPUs;
o aumento na CPU limitava a vazão).

Justificativa por transformação (angio-TC de tórax, achado = falha de enchimento arterial):
* SEM espelhamento horizontal: troca a lateralidade (coração à direita; o desafio rotula TEP
  esquerda/direita), criando anatomia que não existe no conjunto.
* Rotação pequena (±10°), translação (±5%) e escala (0,9–1,1): imitam variação de
  posicionamento do paciente e de campo de visão; não mudam a presença do trombo (que fica
  dentro da imagem com essas amplitudes).
* Ruído gaussiano leve (σ = 0,02 na escala [0,1], ~14 HU na janela CTPA): imita ruído de
  aquisição/dose. Fica bem abaixo do contraste trombo × lúmen (~150–400 HU), então não apaga
  a falha de enchimento.
* NÃO usamos brilho/contraste/gama na política principal: alterar o contraste pode apagar a
  falha de enchimento (o próprio enunciado alerta). A política alternativa ("strong") inclui
  essas alterações, com deformação elástica e desfoque, para MEDIR esse efeito.
Papel do aumento: regularização. O rebalanceamento de classes é feito pela amostragem por
época (todos os positivos + neg_ratio × negativos), não pelo aumento.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

POLICIES = {
    "none": None,
    "main": dict(degrees=10, translate=0.05, scale=(0.9, 1.1), noise_p=0.5, noise_sigma=0.02),
    # elástica: deslocamentos aleatórios numa grade grossa 8x8 (desvio 3 px), interpolados:
    # deformação suave de tecido mole, sem rasgar vasos
    "strong": dict(degrees=15, translate=0.08, scale=(0.85, 1.15), elastic_p=0.3, elastic_px=3.0,
                   jitter_p=0.5, jitter=0.15, blur_p=0.2, blur_sigma=(0.1, 1.0),
                   noise_p=0.5, noise_sigma=0.03),
}


class GpuAugment(nn.Module):
    """Entrada/saída: float (B, C, H, W) em [0, 1], no dispositivo que for (CPU ou GPU)."""

    def __init__(self, degrees=0.0, translate=0.0, scale=(1.0, 1.0), elastic_p=0.0, elastic_px=0.0,
                 jitter_p=0.0, jitter=0.0, blur_p=0.0, blur_sigma=(0.1, 1.0), noise_p=0.0,
                 noise_sigma=0.0, grid=8):
        super().__init__()
        self.degrees, self.translate, self.scale = degrees, translate, scale
        self.elastic_p, self.elastic_px, self.grid = elastic_p, elastic_px, grid
        self.jitter_p, self.jitter = jitter_p, jitter
        self.blur_p, self.blur_sigma = blur_p, blur_sigma
        self.noise_p, self.noise_sigma = noise_p, noise_sigma

    def _u(self, n, lo, hi, x):
        return torch.empty(n, device=x.device).uniform_(lo, hi)

    def _hit(self, n, p, x):
        return torch.rand(n, device=x.device) < p

    @torch.no_grad()
    def forward(self, x):
        b, _, h, w = x.shape
        # afim (rotação, escala, translação), preenchimento com 0 como RandomAffine
        ang = self._u(b, -self.degrees, self.degrees, x) * math.pi / 180
        s = self._u(b, *self.scale, x)
        t = (torch.rand(b, 2, device=x.device) * 2 - 1) * self.translate * 2   # coord. normalizadas
        cos, sin = torch.cos(ang) / s, torch.sin(ang) / s
        theta = torch.stack([torch.stack([cos, -sin, t[:, 0]], 1),
                             torch.stack([sin, cos, t[:, 1]], 1)], 1)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        if self.elastic_p > 0:
            hit = self._hit(b, self.elastic_p, x).float().view(b, 1, 1, 1)
            d = torch.randn(b, 2, self.grid, self.grid, device=x.device) * (self.elastic_px * 2 / w)
            d = F.interpolate(d, size=(h, w), mode="bicubic", align_corners=False)
            grid = grid + d.permute(0, 2, 3, 1) * hit
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        if self.jitter_p > 0:                      # brilho e contraste
            hit = self._hit(b, self.jitter_p, x).view(b, 1, 1, 1)
            br = self._u(b, 1 - self.jitter, 1 + self.jitter, x).view(b, 1, 1, 1)
            ct = self._u(b, 1 - self.jitter, 1 + self.jitter, x).view(b, 1, 1, 1)
            m = x.mean((1, 2, 3), keepdim=True)
            y = ((x * br - m * br) * ct + m * br).clamp(0, 1)
            x = torch.where(hit, y, x)
        if self.blur_p > 0:                        # desfoque gaussiano 5x5, sigma por amostra
            hit = self._hit(b, self.blur_p, x).view(b, 1, 1, 1)
            sig = self._u(b, *self.blur_sigma, x)
            r = torch.arange(-2, 3, device=x.device, dtype=x.dtype)
            k = torch.exp(-(r[None] ** 2) / (2 * sig[:, None] ** 2))
            k = k / k.sum(1, keepdim=True)                                   # (b, 5)
            c = x.shape[1]
            kk = k.repeat_interleave(c, 0)                                   # (b*c, 5)
            y = x.reshape(1, b * c, h, w)
            y = F.conv2d(F.pad(y, (2, 2, 0, 0), mode="reflect"), kk.view(b * c, 1, 1, 5), groups=b * c)
            y = F.conv2d(F.pad(y, (0, 0, 2, 2), mode="reflect"), kk.view(b * c, 1, 5, 1), groups=b * c)
            x = torch.where(hit, y.view(b, c, h, w), x)
        if self.noise_p > 0:
            hit = self._hit(b, self.noise_p, x).float().view(b, 1, 1, 1)
            x = x + torch.randn_like(x) * self.noise_sigma * hit
        return x.clamp(0, 1)


def build(policy: str):
    """'none' | 'main' | 'strong' -> módulo aplicado ao lote float [0,1] (ou None)."""
    if policy not in POLICIES:
        raise ValueError(f"política de aumento desconhecida: {policy}")
    kw = POLICIES[policy]
    return None if kw is None else GpuAugment(**kw)


def to_float(x: torch.Tensor) -> torch.Tensor:
    """uint8 (vindo dos workers) -> float [0, 1]."""
    return x.float().div_(255) if x.dtype == torch.uint8 else x.float()
