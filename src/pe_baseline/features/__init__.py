"""Famílias de descritores hand-crafted.

Cada família expõe `extract(hu, masks, cfg) -> dict[str, float]` com prefixo próprio:
  int_   intensidade em HU (histograma, percentis, frações vaso/falha)
  tex_   textura (GLCM/Haralick + LBP)
  grad_  gradiente/bordas (HOG + histograma de Sobel)
  wav_   wavelets (energia por sub-banda)
"""
from . import intensity, texture, gradient, wavelet

FAMILIES = {
    "intensity": intensity,
    "texture": texture,
    "gradient": gradient,
    "wavelet": wavelet,
}
PREFIX = {"intensity": "int_", "texture": "tex_", "gradient": "grad_", "wavelet": "wav_"}
