"""Backbones pré-treinados (nenhum treino a partir de pesos aleatórios) e grupos de parâmetros.

Fatores do desenho (adaptado de Menegola et al., ISBI 2017):
  família:   CNN (ResNet-50)            x  ViT (DeiT-S, DINOv2 ViT-S/14)
  pré-treino: supervisionado ImageNet   x  auto-supervisionado (DINOv2)  x  médico (RadImageNet)
  regime:    congelado + linear          x  ajuste fino
Resolução: volumes em 256 px. ResNet e DeiT recebem 256 (DeiT com codificação de posição
interpolada de 224 pelo timm); o DINOv2 (patch 14) recebe 252 = 18 x 14, o múltiplo de 14 mais
próximo, também com codificação de posição interpolada.
"""
from __future__ import annotations

import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

SPECS = {
    "resnet50": dict(timm="resnet50.tv_in1k", img=256, family="cnn", pretrain="ImageNet-1k (supervisionado)"),
    "deit_s": dict(timm="deit_small_patch16_224.fb_in1k", img=256, family="vit",
                   pretrain="ImageNet-1k (supervisionado)", vit_kw=True),
    "dinov2_s": dict(timm="vit_small_patch14_dinov2.lvd142m", img=252, family="vit",
                     pretrain="LVD-142M (auto-supervisionado, DINOv2)", vit_kw=True),
    # RadImageNet (Mei et al., 2022), ResNet-50 em PyTorch. Arquivo local em RADIMAGENET_WEIGHTS ou,
    # sem ela, baixado do espelho no Hugging Face numa revisão fixa (o SHA-256 vai para o run.json).
    # Pré-processamento do código de treino oficial (pytorch_example.ipynb): (x - 127,5) * 2 / 255,
    # isto é, [0,1] -> [-1,1] (média 0,5, desvio 0,5), sem a normalização do ImageNet.
    "radimagenet_resnet50": dict(timm="resnet50", img=256, family="cnn", pretrain="RadImageNet (TC/RM/US)",
                                 local_weights="RADIMAGENET_WEIGHTS", mean=(0.5,) * 3, std=(0.5,) * 3,
                                 hf=("Lab-Rasool/RadImageNet", "ResNet50.pt",
                                     "1c94f2700886f822c953989255e039352fba6c56")),
}
# filhos de torchvision.resnet50 por índice (o código oficial salva nn.Sequential(children[:9]))
_CHILD = {"0": "conv1", "1": "bn1", "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4"}
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _load_local_resnet(model: nn.Module, path: str) -> dict:
    """Carrega um state_dict de ResNet-50 com nomes no padrão torchvision/timm (conv1, bn1,
    layer1..4), tolerando prefixos ('module.', 'backbone.', 'model.', 'encoder.')."""
    sd = torch.load(path, map_location="cpu", weights_only=True)   # só tensores (sem pickle arbitrário)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    clean = {}
    for k, v in sd.items():
        k = re.sub(r"^(module\.|backbone\.|model\.|encoder\.)+", "", k)
        m = re.match(r"^(\d+)\.(.*)$", k)
        if m:                                       # formato nn.Sequential: '4.0.conv1.weight'
            if m.group(1) not in _CHILD:
                continue
            k = f"{_CHILD[m.group(1)]}.{m.group(2)}"
        if k.startswith(("fc.", "classifier.", "head.")):
            continue
        clean[k] = v
    res = model.load_state_dict(clean, strict=False)
    n_model = len([k for k in model.state_dict() if not k.startswith("fc.")])
    loaded = n_model - len([k for k in res.missing_keys if not k.startswith("fc.")])
    if loaded < 0.95 * n_model:
        raise RuntimeError(f"pesos locais incompatíveis: {loaded}/{n_model} tensores carregados; "
                           f"exemplos faltando: {res.missing_keys[:5]}")
    import hashlib
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    return {"loaded": loaded, "total": n_model, "unexpected": len(res.unexpected_keys), "sha256": sha,
            "path": str(path)}


def _local_weights_path(spec) -> str:
    path = os.environ.get(spec["local_weights"])
    if path:
        return path
    if "hf" in spec:
        from huggingface_hub import hf_hub_download
        repo, fname, rev = spec["hf"]
        return hf_hub_download(repo_id=repo, filename=fname, revision=rev)
    raise RuntimeError(f"defina {spec['local_weights']} com o caminho dos pesos")


class Net(nn.Module):
    """Backbone sem cabeça + cabeça linear (1 logit). Entrada: float [0,1] (B, 3, 256, 256)."""

    def __init__(self, name: str, pretrained: bool = True, drop_path: float = 0.0):
        super().__init__()
        import timm
        spec = SPECS[name]
        kw = dict(num_classes=0)
        if spec.get("vit_kw"):
            kw.update(img_size=spec["img"], drop_path_rate=drop_path)
        use_hub = pretrained and not spec.get("local_weights")
        self.backbone = timm.create_model(spec["timm"], pretrained=use_hub, **kw)
        self.load_info = None
        if pretrained and spec.get("local_weights"):
            self.load_info = _load_local_resnet(self.backbone, _local_weights_path(spec))
        cfg = getattr(self.backbone, "pretrained_cfg", {}) or {}
        mean = spec.get("mean") or cfg.get("mean", IMAGENET_MEAN)
        std = spec.get("std") or cfg.get("std", IMAGENET_STD)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.name, self.img, self.family = name, spec["img"], spec["family"]
        self.head = nn.Linear(self.backbone.num_features, 1)
        if self.family == "cnn":                       # mais rápido com AMP (tensor cores)
            self.backbone = self.backbone.to(memory_format=torch.channels_last)

    def prep(self, x):
        x = x.float()
        if x.shape[-1] != self.img:
            x = F.interpolate(x, size=(self.img, self.img), mode="bilinear", align_corners=False,
                              antialias=True)
        return (x - self.mean) / self.std

    def features(self, x):
        return self.backbone(self.prep(x))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def stage_of(name: str, param_name: str, n_blocks: int) -> int:
    """Profundidade de cada parâmetro (0 = entrada). Cabeça = n_stages - 1."""
    p = param_name
    if p.startswith("head."):
        return n_blocks + 1
    p = p.removeprefix("backbone.")
    if SPECS[name]["family"] == "cnn":
        m = re.match(r"layer(\d)\.", p)
        return int(m.group(1)) if m else (0 if p.startswith(("conv1", "bn1")) else n_blocks)
    m = re.match(r"blocks\.(\d+)\.", p)
    if m:
        return int(m.group(1)) + 1
    if p.startswith(("patch_embed", "cls_token", "pos_embed", "reg_token", "mask_token")):
        return 0
    return n_blocks                                        # norm final / fc_norm


def n_blocks(model: Net) -> int:
    return 4 if model.family == "cnn" else len(model.backbone.blocks)


def param_groups(model: Net, lr: float, weight_decay: float, llrd: float | None = None):
    """Grupos para AdamW: sem weight decay em bias/normalização; com llrd, taxa por camada
    lr * llrd^(profundidade_máxima - profundidade) (taxas discriminativas / LLRD)."""
    nb = n_blocks(model)
    top = nb + 1
    groups = {}
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        depth = stage_of(model.name, pname, nb)
        scale = llrd ** (top - depth) if llrd else 1.0
        no_wd = p.ndim <= 1 or pname.endswith(("cls_token", "pos_embed", "reg_token"))
        key = (depth if llrd else 0, no_wd)
        g = groups.setdefault(key, {"params": [], "lr": lr * scale, "base_lr": lr * scale,
                                    "weight_decay": 0.0 if no_wd else weight_decay})
        g["params"].append(p)
    return list(groups.values())


def set_backbone_trainable(model: Net, trainable: bool):
    for p in model.backbone.parameters():
        p.requires_grad_(trainable)


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
