# TP1: baseline clássico para detecção de embolia pulmonar (RSNA-STR PE 2020)

**Disciplina:** Tópicos Especiais em Sistemas de Informação · **Grupo G2:** Rawenne da Silva Leite,
Wheverson de Abreu Lima, João Victor do Nascimento Silva

Este repositório é o baseline clássico do projeto. Ele usa descritores manuais
(*hand-crafted*) e aprendizado de máquina clássico, **sem redes profundas**, e define o
desempenho mínimo que os modelos do TP2 (transfer learning) terão de superar. O código já
foi feito para o TP2: a partição de avaliação é **fixa, por paciente, sobre o treino oficial
completo**, e o mesmo pipeline roda na amostra (TP1) ou no conjunto completo (TP2) mudando
apenas a configuração.

---

## 1. Dados

| Item | Valor |
|---|---|
| Desafio | RSNA STR Pulmonary Embolism Detection (2020): <https://www.kaggle.com/competitions/rsna-str-pulmonary-embolism-detection> |
| Artigo do conjunto | Colak et al., *The RSNA Pulmonary Embolism CT Dataset*, Radiology: AI, 2021 |
| Modalidade | Angio-TC pulmonar (CTPA), DICOM, 512×512, cerca de 250 cortes por exame |
| Treino oficial | cerca de 7.279 exames (1 série por exame), cerca de 1,79 milhão de cortes |
| Rótulos | por corte (`pe_present_on_image`) e por exame (negativo, indeterminado, lateralidade, central, crônico, razão VD/VE, qualidade) |
| Teste oficial | **sem rótulos públicos**, por isso não é usado (ver §3) |

**Como obter:** crie uma conta no Kaggle, aceite as regras da competição e
(a) use os notebooks em `notebooks/` diretamente no Kaggle (recomendado: os dados ficam em
`/kaggle/input/...` sem download) ou (b) baixe com `kaggle competitions download -c
rsna-str-pulmonary-embolism-detection` e defina `PE_DATA_ROOT=/caminho/para/os/dados`.
**Nenhum dado bruto é versionado aqui.**

## 2. Formulação da tarefa

- **Unidade de amostra:** o exame (`StudyInstanceUID`). Os DICOMs da competição são
  anonimizados (`PatientID` e `Manufacturer` vêm vazios, conferido na EDA nos 7.279 exames),
  então **não é possível verificar** se um mesmo paciente aparece em mais de um exame. Assumimos
  1 exame = 1 paciente, como na descrição do conjunto, e particionamos sempre por exame; isso é
  uma limitação declarada.
- **Tarefa principal (exame):** o exame tem TEP? `pe = 1` se `negative_exam_for_pe = 0` e
  `indeterminate = 0`. Exames *indeterminados* (qualidade insuficiente para laudo) não têm
  verdade binária e ficam fora da amostra do TP1, mas recebem partição.
- **Tarefa auxiliar (corte):** o corte mostra TEP (`pe_present_on_image`)? As probabilidades
  de corte são agregadas por exame. Isso respeita a **hierarquia de rótulos** do desafio
  (exame positivo se e somente se existe ao menos um corte positivo).

## 3. Partição e amostra (a crítica da AP1 anterior foi "poucos dados")

1. **Partição fixa do treino completo** (`splits/split_full_train.csv`): 70% treino, 10%
   validação e 20% **teste**, estratificada pelo rótulo de exame, semente 42. Ela é gerada uma
   única vez e **não muda mais**. O TP2 usa o mesmo conjunto `test`.
2. **Amostra do TP1** (`splits/sample_tp1.csv`), grande mas não completa:

   | Subconjunto | Exames | Origem | Uso |
   |---|---|---|---|
   | dev | **2.400** | `train`+`val` | validação cruzada aninhada (seleção de modelo) |
   | test | **600** | `test` | avaliação final, **uma única vez** |
   | **total** | **3.000 (cerca de 41% dos exames)** | estratificado por classe | |

   Dentro de cada exame, os cortes são ordenados por z e usamos **1 a cada 2**
   (`slice_stride: 2`). Isso dá **cerca de 370 mil cortes** e preserva a cobertura do volume
   inteiro. As contagens exatas por classe saem em `results/sample_counts.csv`.
3. Os tamanhos ficam em `configs/config.yaml` (`sample.*`). Para o TP2, basta usar todos os
   exames e `slice_stride: 1`.

## 4. Pipeline

### 4.1 Pré-processamento (`src/pe_baseline/dicom_io.py`, `preprocess.py`)

| Etapa | Decisão | Por quê |
|---|---|---|
| Unidades | `HU = pixel·RescaleSlope + RescaleIntercept`, recorte em [-1024, 3071] | valores brutos variam entre fabricantes; HU é uma unidade física comparável |
| Ordem dos cortes | por `ImagePositionPatient[2]` (z) | `InstanceNumber` não é confiável entre fabricantes |
| Resolução | 512 → **256 px** (cerca de 1,4 mm/px), bilinear com anti-aliasing | reduz o custo cerca de 4×. Abaixo disso, êmbolos segmentares (poucos mm) começam a desaparecer, então 256 é o limite que adotamos |
| Janelas | CTPA (L100/W700), mediastino (L40/W400), pulmão (L-600/W1500) | a janela CTPA separa contraste arterial (cerca de 250–500 HU) de trombo (cerca de 20–100 HU) |
| ROI torácica | casco convexo dos pulmões (limiar de -400 HU, remoção de ar externo) | contém pulmões e mediastino, onde estão as artérias pulmonares. Exclui mesa, braços e parede |
| Máscara de vaso | ROI ∩ HU ∈ [150, 600], **sem osso** | lúmen arterial opacificado. Osso = componente conexo (HU ≥ 150) com pico local > 800 HU (máximo 2×2 antes da média, que dilui o córtex fino). Sem isso, na TC real o corpo vertebral gerava centenas de falsos candidatos (214 → 1 no corte de teste). Costelas e esterno ainda aparecem como vaso (limitação) |
| Máscara de falha de enchimento | partes moles (0–120 HU) **cercadas** por contraste (buraco no lúmen, ou vizinhança 7×7 com mais de 55% de vaso) | é o sinal radiológico direto de TEP. O critério "cercada" elimina a borda do vaso (volume parcial), que caía na mesma faixa de HU |

### 4.2 Descritores: 4 famílias (`src/pe_baseline/features/`)

| Família | Descritores | Justificativa |
|---|---|---|
| **Intensidade (HU)** | momentos, percentis, histograma em faixas de HU, frações de vaso e de falha, falha/vaso | HU é calibrado: as frações em faixas clínicas medem diretamente contraste e trombo |
| **Textura** | GLCM/Haralick (32 níveis, d = 1, 2, 4, média em 4 ângulos) + LBP uniforme rotação-invariante (P8R1, P16R2) | o LBP é invariante a mudanças monotônicas de intensidade (diferenças de contraste entre aquisições) |
| **Gradiente** | HOG (ROI 128×128, células de 32 px, 324 dims) + histograma de Sobel | geometria das bordas: contorno de vasos e interface trombo/contraste |
| **Wavelet** | Daubechies-2, 2 níveis: energia, média absoluta e desvio por sub-banda | textura multiescala: os detalhes finos respondem a vasos pequenos |

Também são calculadas versões **sem ROI** de intensidade e textura, para a ablação.

**Estratégia volumétrica:** descritores **2D por corte**, agregados por exame. Comparamos
duas formas de agregar (`aggregate.py`):
(i) *pooling de características*: média, desvio, máximo e p90 de cada descritor sobre os
cortes. O máximo e o p90 preservam sinais focais que a média diluiria.
(ii) *pooling de probabilidades*: um classificador de corte, depois o máximo ou a média dos
top-5 por exame. Descritores 3D diretos foram descartados porque diluiriam um trombo que
ocupa poucos dos cerca de 250 cortes.

### 4.3 Modelos (`models.py`)

- **Baseline trivial obrigatório:** `DummyClassifier(prior)`, que sempre prediz a classe
  majoritária (AUC = 0,5).
- Regressão logística L2, SVM RBF, Random Forest, LightGBM e MLP raso (sobre os vetores
  hand-crafted, o que é permitido).
- Padronização, PCA (95% da variância) e SMOTE ficam **dentro do `Pipeline`**: são ajustados
  só no treino de cada dobra.
- Desbalanceamento: `class_weight` (padrão), comparado na ablação com nenhum tratamento e com
  SMOTE.
- No nível de corte (cerca de 300 mil linhas), usamos apenas regressão logística e LightGBM:
  SVM com kernel e RF denso são inviáveis nessa escala.

### 4.4 Protocolo (`evaluate.py`)

- **Validação cruzada aninhada:** 5 dobras externas `StratifiedGroupKFold` (grupo = exame)
  para estimar o desempenho, e 3 internas para `RandomizedSearchCV` (AUC-ROC) e para escolher
  o **limiar** (Youden) sobre predições fora da dobra interna. A dobra externa de teste nunca
  participa de escolha nenhuma.
- As dobras são **as mesmas** para todos os experimentos de exame e de corte
  (`results/cv_folds.csv`), o que torna as comparações pareadas.
- Há uma asserção no código que impede o mesmo exame em treino e teste.
- **Métricas:** AUC-ROC, AUC-PR, sensibilidade, especificidade, F1 e acurácia balanceada
  (média ± desvio entre dobras). A acurácia é registrada só como referência.
- **Teste final:** a melhor configuração da CV (escolhida só pela AUC média da CV) é
  reajustada no `dev` e avaliada **uma vez** no `test`, com **IC 95% por bootstrap em nível de
  exame**.
- **Ablações** sobre a melhor configuração: desbalanceamento (nenhum, pesos, SMOTE), pooling
  (média, média+máx, as 4 estatísticas) e ROI (com ou sem).
- **Análises:**
  - importância por permutação **de grupo** (família e estatística de pooling);
  - sensibilidade por subtipo (central, lateralidade, crônico, VD/VE);
  - AUC por qualidade de exame (`qa_contrast`, `qa_motion`, `flow_artifact`);
  - galeria de VP/VN/FP/FN escolhidos por critério **pré-definido**: os k exames de maior
    confiança em cada célula da matriz de confusão.

## 4.5 O que a TC real mostrou (limitações conhecidas)

- **DICOMs reais:** 94% sem compressão; 474 exames (6,5%) em *JPEG Lossless*, que exigem
  `pylibjpeg` (`requirements-kaggle.txt`). `RescaleIntercept = 0` em 205 exames (2,8%) ainda
  precisa de conferência. Espessura de 0,6 a 5 mm; kVp de 80 a 140; pixel de 0,3 a 0,98 mm.
- **Cortes fora do tórax** (pescoço/abdome, nas pontas do exame) não têm pulmão: a ROI cai para o
  corpo todo (`int_roi_fallback`). Isso acontece em 56% dos cortes no primeiro décimo do exame e
  80% no último. Como o TEP só existe no tórax, `int_roi_fallback` e `z_rank` funcionam como
  atalho para o classificador de corte. Por isso reportamos o desempenho de corte **também só em
  cortes torácicos** e tratamos o nível de exame como a métrica principal.
- **Falha de enchimento:** a máscara é uma heurística de HU e forma, não um detector de trombo;
  ainda marca candidatos no coração (trabéculas) e na borda do esterno.

## 5. Como reproduzir

### No Kaggle (recomendado: dados sem download)

1. `notebooks/00_eda.ipynb`: EDA e geração da partição e da amostra (`splits/`).
2. `notebooks/kaggle_01_extract.ipynb`: extração de características (CPU, cerca de 2–3 h,
   retomável e divisível em *shards*). Use **Save Version** para a saída virar um Dataset.
3. `notebooks/kaggle_02_experiments.ipynb`: adicione esse Dataset como Input e rode. Ele gera
   `results/` (tabelas, `RESULTS.md` e figuras).

Cada notebook clona este repositório (é preciso ligar **Internet** nas configurações) ou copia
o código de um Dataset chamado `ap2-topicos-code`, e depois instala `requirements-kaggle.txt`.

### Local ou outra máquina

```bash
pip install -r requirements.txt
```

```bash
export PE_DATA_ROOT=/caminho/rsna-str-pulmonary-embolism-detection
```

```bash
python scripts/01_make_split_and_sample.py
```

```bash
python scripts/02_extract_features.py
```

```bash
python scripts/03_run_experiments.py
```

```bash
python scripts/04_make_figures.py --gallery
```

### Testes (sem os dados reais: geram DICOMs sintéticos)

```bash
python -m pytest -q
```

**Reprodutibilidade:**
- semente única (`seed: 42`) em `configs/config.yaml`;
- nenhum caminho absoluto no código (os caminhos são relativos à raiz do repositório ou vêm
  de `PE_DATA_ROOT`, `PE_FEATURES_DIR` ou `PE_RESULTS_DIR`);
- versões em `requirements.txt`, e as versões efetivamente usadas ficam em
  `results/environment.txt`;
- cortes ilegíveis são listados em `results/excluded_slices.csv`.

## 6. Estrutura

```
configs/config.yaml          parâmetros (amostra, pré-processamento, CV, modelos)
src/pe_baseline/             biblioteca: leitura DICOM, máscaras, descritores, modelos, protocolo
scripts/01..04_*.py          pipeline em 4 passos (cada tabela/figura sai de um script)
notebooks/                   EDA + notebooks prontos para o Kaggle
splits/                      partição fixa + amostra do TP1 (versionadas)
results/                     tabelas, figuras e RESULTS.md (versionados após a execução)
tests/                       gerador de DICOM sintético + testes do pipeline
```

## 7. Referências das decisões

- Colak E. et al. The RSNA Pulmonary Embolism CT Dataset. *Radiology: Artificial Intelligence*, 3(2), 2021.
- Wittram C. et al. CT angiography of pulmonary embolism: diagnostic criteria and causes of misdiagnosis. *RadioGraphics*, 24(5), 2004.
- Haralick R., Shanmugam K., Dinstein I. Textural features for image classification. *IEEE Trans. SMC*, 3(6), 1973.
- Ojala T., Pietikäinen M., Mäenpää T. Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. *IEEE TPAMI*, 24(7), 2002.
- Dalal N., Triggs B. Histograms of oriented gradients for human detection. *CVPR*, 2005.
- Mallat S. A theory for multiresolution signal decomposition: the wavelet representation. *IEEE TPAMI*, 11(7), 1989.
- Chawla N. et al. SMOTE: synthetic minority over-sampling technique. *JAIR*, 16, 2002.
- Varma S., Simon R. Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*, 7, 2006.
- Saito T., Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, 10(3), 2015.
- Youden W. J. Index for rating diagnostic tests. *Cancer*, 3(1), 1950.
- Breiman L. Random forests. *Machine Learning*, 45, 2001. · Ke G. et al. LightGBM. *NeurIPS*, 2017.
- Huang S.-C. et al. PENet: a scalable deep-learning model for automated diagnosis of pulmonary embolism using volumetric CT imaging. *npj Digital Medicine*, 3, 2020 (referência para a fase profunda do TP2).
