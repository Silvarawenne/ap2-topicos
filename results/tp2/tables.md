## Configurações (3 sementes; ordenadas pela AUC de validação)

| Configuração | n | AUC val | AUC teste | AUC teste (conjunto, IC 95%) | AUC-PR teste | Sens. | Espec. | min/exec. |
|---|---|---|---|---|---|---|---|---|
| DINOv2 ViT-S/14 · ajuste fino · llrd · aum. main | 3 | 0.834 ± 0.000 | 0.819 ± 0.009 | 0.827 [0.803, 0.851] | 0.760 | 0.630 | 0.857 | 140 |
| ResNet-50 (ImageNet) · ajuste fino · cosine · aum. strong | 3 | 0.825 ± 0.007 | 0.801 ± 0.001 | 0.806 [0.779, 0.831] | 0.739 | 0.631 | 0.831 | 159 |
| ResNet-50 (ImageNet) · ajuste fino · cosine · aum. main | 3 | 0.822 ± 0.009 | 0.791 ± 0.011 | 0.796 [0.768, 0.822] | 0.721 | 0.661 | 0.791 | 132 |
| ResNet-50 (RadImageNet) · ajuste fino · cosine · aum. main | 3 | 0.771 ± 0.009 | 0.739 ± 0.004 | 0.749 [0.720, 0.777] | 0.651 | 0.569 | 0.800 | 135 |
| DeiT-S (ImageNet) · ajuste fino · cosine · aum. main | 3 | 0.766 ± 0.005 | 0.748 ± 0.010 | 0.760 [0.732, 0.788] | 0.655 | 0.575 | 0.793 | 119 |
| ResNet-50 (ImageNet) · ajuste fino · cosine · aum. none | 3 | 0.727 ± 0.034 | 0.701 ± 0.041 | 0.726 [0.697, 0.755] | 0.585 | 0.497 | 0.792 | 244 |
| DeiT-S (ImageNet) · congelado | 3 | 0.653 ± 0.004 | 0.589 ± 0.002 | 0.590 [0.558, 0.622] | 0.401 | 0.479 | 0.640 | 47 |
| DINOv2 ViT-S/14 · congelado | 3 | 0.650 ± 0.004 | 0.586 ± 0.004 | 0.587 [0.555, 0.617] | 0.402 | 0.443 | 0.684 | 67 |
| ResNet-50 (ImageNet) · congelado | 3 | 0.649 ± 0.010 | 0.597 ± 0.004 | 0.598 [0.567, 0.630] | 0.428 | 0.448 | 0.692 | 52 |
| ResNet-50 (RadImageNet) · congelado | 3 | 0.637 ± 0.006 | 0.590 ± 0.002 | 0.592 [0.561, 0.623] | 0.410 | 0.517 | 0.614 | 63 |
| DINOv2 ViT-S/14 · ajuste fino · cosine · aum. main | 3 | 0.629 ± 0.011 | 0.572 ± 0.006 | 0.579 [0.547, 0.611] | 0.386 | 0.525 | 0.591 | 160 |
| DINOv2 ViT-S/14 · ajuste fino · const · aum. main | 3 | 0.566 ± 0.009 | 0.508 ± 0.019 | 0.512 [0.479, 0.543] | 0.318 | 0.565 | 0.457 | 135 |

## Comparações pareadas no teste (conjunto de sementes)

| Comparação | A | B | AUC A | AUC B | Δ [IC 95% bootstrap] | p DeLong | p Holm |
|---|---|---|---|---|---|---|---|
| melhor CNN x melhor ViT (ajuste fino) | ft_resnet50_cosine_strong | ft_dinov2_s_llrd_main | 0.806 | 0.827 | -0.022 [-0.039, -0.006] | 0.011 | 0.023 |
| melhor ImageNet/auto-sup. x RadImageNet (ajuste fino) | ft_dinov2_s_llrd_main | ft_radimagenet_resnet50_cosine_main | 0.827 | 0.749 | +0.078 [+0.056, +0.100] | 2e-12 | 9.9e-12 |
| ajuste fino x congelado: ResNet-50 (ImageNet) | ft_resnet50_cosine_strong | frozen_resnet50 | 0.806 | 0.598 | +0.207 [+0.175, +0.237] | 3.3e-36 | 2.6e-35 |
| ajuste fino x congelado: ResNet-50 (RadImageNet) | ft_radimagenet_resnet50_cosine_main | frozen_radimagenet_resnet50 | 0.749 | 0.592 | +0.158 [+0.126, +0.188] | 4.3e-21 | 2.6e-20 |
| ajuste fino x congelado: DeiT-S (ImageNet) | ft_deit_s_cosine_main | frozen_deit_s | 0.760 | 0.590 | +0.171 [+0.139, +0.204] | 3.8e-24 | 2.6e-23 |
| ajuste fino x congelado: DINOv2 ViT-S/14 | ft_dinov2_s_llrd_main | frozen_dinov2_s | 0.827 | 0.587 | +0.241 [+0.209, +0.273] | 1.7e-46 | 1.5e-45 |
| aumento: principal x nenhum (ResNet-50) | ft_resnet50_cosine_main | ft_resnet50_cosine_none | 0.796 | 0.726 | +0.070 [+0.049, +0.089] | 8.7e-12 | 3.5e-11 |
| aumento: forte x principal (ResNet-50) | ft_resnet50_cosine_strong | ft_resnet50_cosine_main | 0.806 | 0.796 | +0.010 [+0.001, +0.019] | 0.026 | 0.026 |
| ajuste fino: LLRD x cosseno (DINOv2) | ft_dinov2_s_llrd_main | ft_dinov2_s_cosine_main | 0.827 | 0.579 | +0.248 [+0.214, +0.281] | 6.6e-49 | 6.6e-48 |
| ajuste fino: cosseno x constante (DINOv2) | ft_dinov2_s_cosine_main | ft_dinov2_s_const_main | 0.579 | 0.512 | +0.068 [+0.039, +0.095] | 4.3e-06 | 1.3e-05 |
