# Verificação das referências (`refs.bib`)

Conferido em 25/11/2026 por busca web (títulos, anos, venues, páginas e DOI). **Isto não substitui a
leitura**: o enunciado trata referência inexistente ou citada sem leitura como falha de integridade.
Antes de entregar, cada integrante deve abrir o DOI/arXiv de cada entrada e ler o trecho que o
artigo cita. Várias páginas de editora não puderam ser abertas daqui (rede restrita); o que veio
de agregador ou índice secundário está marcado abaixo.

Onde a lista completa de autores não foi confirmada, o `.bib` usa `and others` (aparece como "et al.").

| Chave | O que foi confirmado | Fonte | Ressalva |
|---|---|---|---|
| `menegola2017` | 6 autores; ISBI 2017, pp. 297–300; arXiv:1703.07479 | arXiv abs/1703.07479; DBLP `conf/isbi/MenegolaFPBAV17` | DOI 10.1109/ISBI.2017.7950523 veio de um agregador, não do IEEE |
| `colak2021` | *Radiology: AI* 3(2), e200254, 2021; DOI 10.1148/ryai.2021200254 | pubs.rsna.org/doi/10.1148/ryai.2021200254 | autores completos não confirmados |
| `mei2022` | *Radiology: AI* 4(5), e210315, 27/07/2022; DOI 10.1148/ryai.210315; PMID 36204533 | PubMed 36204533; PMC9530758 | volume/número vêm da linha de citação do índice; texto completo não aberto |
| `oquab2024` | arXiv:2304.07193 (Oquab e 25 coautores) | arXiv abs/2304.07193 | **venue/ano (TMLR 2024) vêm de índices secundários (mlanthology, HAL); uma entrada BibTeX de pesquisador lista 2023.** Conferir no OpenReview/TMLR |
| `touvron2021` | 6 autores; ICML 2021, PMLR 139, pp. 10347–10357 | proceedings.mlr.press/v139/touvron21a | — |
| `he2016` | 4 autores; CVPR 2016, pp. 770–778; DOI 10.1109/CVPR.2016.90 | openaccess.thecvf.com (CVPR 2016) | — |
| `selvaraju2017` | ICCV 2017, pp. 618–626; arXiv:1610.02391 | entrada BibTeX do ICCV; arXiv | autores completos não vistos; existe versão estendida no IJCV 2019 (citamos a do ICCV) |
| `sundararajan2017` | 3 autores; ICML 2017, PMLR 70, pp. 3319–3328 | proceedings.mlr.press/v70/sundararajan17a | — |
| `adebayo2018` | 6 autores; NeurIPS 2018, vol. 31 | papers.nips.cc/paper/8160 | páginas não consultadas (omitidas) |
| `kumar2022` | 5 autores; ICLR 2022; arXiv:2202.10054 | arXiv abs/2202.10054; iclr.cc 2022 | — |
| `delong1988` | 3 autores; *Biometrics* 44(3), 837–845, 1988 | Duke Scholars; registros de catálogo | DOI não consultado (omitido) |
| `howard2018` | 2 autores; ACL 2018, pp. 328–339; DOI 10.18653/v1/P18-1031 | aclanthology.org/P18-1031 | — |
| `zech2018` | *PLOS Medicine* 15(11), e1002683, 2018; DOI 10.1371/journal.pmed.1002683 | PMC6219764 | só o primeiro autor confirmado |
| `degrave2021` | *Nature Machine Intelligence* 3, 610–619, 2021; DOI 10.1038/s42256-021-00338-7 | registros AHRQ/medRxiv e imprensa | página da Nature não aberta; só o sobrenome dos 3 autores visto |

Total: 14 referências, todas em veículo com revisão por pares (cumpre ≥ 12, ≥ 8 revisadas).
Obrigatórias do enunciado presentes: Menegola *et al.* (ISBI 2017) e Colak *et al.* (2021).

## O RadImageNet pode ter visto o RSNA PE?

Pergunta do §3 do enunciado ("verifiquem se o modelo médico viu o seu conjunto").

**Confirmado (resumo e registros do artigo de Mei *et al.*, 2022):**
- 1,35 milhão de imagens de 131.872 pacientes, em TC, RM e ultrassom, em várias regiões anatômicas.
- Estudo retrospectivo: pacientes que fizeram exame entre 2005 e 2020 **em uma única instalação de
  radiologia ambulatorial** (um resumo a situa em Nova York; os autores têm afiliação ao Icahn School
  of Medicine at Mount Sinai, a Cornell Medicine e a East River Medical Imaging, em Nova York).
- Os rótulos vêm da interpretação clínica original, por 20 radiologistas certificados.
- O banco de imagens é liberado por solicitação; os modelos pré-treinados são públicos (CC BY 4.0).

**Não confirmado:** o nome da instalação como fonte das imagens (nenhum resultado o afirma
literalmente) e se o texto cita o RSNA PE ou outro conjunto público. **O texto completo (PMC) não pôde
ser aberto daqui.**

**Leitura (inferência, não fato):** como as imagens vêm de uma única instalação ambulatorial e o RSNA PE
é um conjunto público multi-instituição de 2020, não há indicação de sobreposição de exames. Porém,
nenhum trecho dos Métodos foi lido, e não conferimos de quais instituições vêm os exames do RSNA PE.

**A fazer antes de afirmar no artigo:** ler a seção de Métodos de Mei *et al.* e a lista de instituições
contribuintes em Colak *et al.*. No artigo, escrever algo como "não há indicação de sobreposição" e não
"não há sobreposição".
