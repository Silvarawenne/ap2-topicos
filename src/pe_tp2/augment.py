"""Políticas de aumento de dados (aplicadas SÓ no treino, por corte 2.5D, nos workers).

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

import torch
from torchvision.transforms import v2


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, sigma: float):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        return (x + torch.randn_like(x) * self.sigma).clamp_(0, 1)


def _to_float():
    return v2.ToDtype(torch.float32, scale=True)


def build(policy: str):
    """'none' | 'main' | 'strong'. Entrada: tensor uint8 (3, H, W); saída: float em [0, 1]."""
    if policy == "none":
        return v2.Compose([_to_float()])
    if policy == "main":
        return v2.Compose([
            v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            _to_float(),
            v2.RandomApply([AddGaussianNoise(0.02)], p=0.5),
        ])
    if policy == "strong":
        return v2.Compose([
            v2.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.85, 1.15)),
            v2.RandomApply([v2.ElasticTransform(alpha=30.0, sigma=6.0)], p=0.3),
            _to_float(),
            v2.RandomApply([v2.ColorJitter(brightness=0.15, contrast=0.15)], p=0.5),
            v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))], p=0.2),
            v2.RandomApply([AddGaussianNoise(0.03)], p=0.5),
        ])
    raise ValueError(f"política de aumento desconhecida: {policy}")
