"""Submissão tardia ao desafio (10 rótulos) a partir das predições binárias das redes.

As redes do TP2 preveem só "TEP no corte"; o exame é a média dos k maiores cortes. O desafio pede
1 rótulo por imagem (pe_present_on_image) e 9 por exame, avaliados por log loss ponderada, com
REGRAS DE CONSISTÊNCIA entre rótulos (submissão que viola é rejeitada). Construção:

1. Calibração (log loss exige probabilidades, não só ordenação):
   * corte: o treino viu os negativos subamostrados com taxa q (todos os positivos + neg_ratio x
     negativos), então as chances do modelo saem infladas por 1/q. Correção de prior (Saerens et
     al., 2002): chance_corrigida = q * chance_modelo.
   * exame: Platt (regressão logística em logit(escore)) ajustada SÓ na validação.
2. Rótulos de exame não modelados: P(indeterminado) = prevalência no treino; P(rótulo | TEP) =
   prevalência entre os exames com TEP do treino (crônico, agudo+crônico, central, esquerdo,
   direito, VD/VE >= 1, VD/VE < 1); P(rótulo) = P(TEP) * P(rótulo | TEP).
3. Consistência: o exame é declarado positivo se P(TEP) > 0,5 (limiar de log loss, não o de
   Youden) e as probabilidades são empurradas o mínimo (para 0,5 +- 1e-4) até satisfazer as regras.

Regras de consistência do desafio (página de avaliação da competição), com "positivo" = alguma
imagem com p > 0,5:
  exame positivo: nem negative_exam_for_pe nem indeterminate > 0,5; exatamente um entre
    rv_lv_ratio_gte_1 e rv_lv_ratio_lt_1 > 0,5; ao menos um entre central/leftsided/rightsided
    > 0,5; no máximo um entre chronic_pe e acute_and_chronic_pe > 0,5.
  exame negativo: exatamente um entre negative_exam_for_pe e indeterminate > 0,5; nenhum dos
    7 rótulos de TEP > 0,5.
A métrica abaixo replica a do desafio para conferência interna; o escore oficial é o do Kaggle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STUDY, SOP = "StudyInstanceUID", "SOPInstanceUID"
PE_LABELS = ["chronic_pe", "acute_and_chronic_pe", "central_pe", "leftsided_pe", "rightsided_pe",
             "rv_lv_ratio_gte_1", "rv_lv_ratio_lt_1"]
EXAM_LABELS = ["negative_exam_for_pe", "indeterminate", *PE_LABELS]
WEIGHTS = {"negative_exam_for_pe": 0.0736196319, "indeterminate": 0.09202453988,
           "chronic_pe": 0.1042944785, "acute_and_chronic_pe": 0.1042944785,
           "central_pe": 0.1877300613, "leftsided_pe": 0.06257668712, "rightsided_pe": 0.06257668712,
           "rv_lv_ratio_gte_1": 0.2346625767, "rv_lv_ratio_lt_1": 0.0782208589}
W_IMAGE = 0.07361963
HI, LO = 0.5 + 1e-4, 0.5 - 1e-4
EPS = 1e-6


# ----------------------------------------------------------------- priors e calibração

def train_priors(train: pd.DataFrame, split: pd.DataFrame, neg_ratio: float) -> dict:
    """Prevalências e taxa de subamostragem q, só com os exames da partição de treino."""
    tr_ids = set(split.loc[split["split"] == "train", STUDY])
    tr = train[train[STUDY].isin(tr_ids)]
    ex = tr.groupby(STUDY)[["indeterminate", "negative_exam_for_pe", *PE_LABELS]].first()
    pos = ex[(ex["negative_exam_for_pe"] == 0) & (ex["indeterminate"] == 0)]
    det = tr[~tr[STUDY].isin(ex.index[ex["indeterminate"] == 1])]      # treino sem indeterminados
    n_pos = int(det["pe_present_on_image"].sum())
    n_neg = int(len(det) - n_pos)
    return {"p_indeterminate": float(ex["indeterminate"].mean()),
            "p_pe_exam": float(len(pos) / max((ex["indeterminate"] == 0).sum(), 1)),
            "p_image": float(n_pos / max(len(det), 1)),
            "given_pe": {c: float(pos[c].mean()) for c in PE_LABELS},
            "q": float(min(1.0, neg_ratio * n_pos / max(n_neg, 1))),
            "n_train_exams": int(len(ex)), "n_train_pe_exams": int(len(pos))}


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def prior_correct(p, q: float):
    """Chance do modelo treinado com negativos subamostrados por q -> chance na população."""
    return _sigmoid(_logit(p) + np.log(max(q, EPS)))


def fit_platt(y, score):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(score).reshape(-1, 1), np.asarray(y))
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    return {"a": a, "b": b}


def apply_platt(cal, score):
    return _sigmoid(cal["a"] * _logit(score) + cal["b"])


# ----------------------------------------------------------------- montagem + consistência

def exam_table(p_pe: pd.Series, priors: dict) -> pd.DataFrame:
    """P(TEP) por exame (entre não indeterminados) -> 9 rótulos de exame."""
    pi = priors["p_indeterminate"]
    ppos = (1 - pi) * p_pe.to_numpy(float)
    out = pd.DataFrame(index=p_pe.index)
    out["negative_exam_for_pe"] = (1 - pi) - ppos
    out["indeterminate"] = pi
    for c in PE_LABELS:
        out[c] = ppos * priors["given_pe"][c]
    return out


def make_consistent(exams: pd.DataFrame, images: pd.DataFrame, positive: pd.Series):
    """Ajuste mínimo para cumprir as regras. exams: índice = exame; images: STUDY, SOP, p;
    positive: bool por exame. Devolve cópias ajustadas."""
    e, im = exams.copy(), images.copy()
    pos = positive.reindex(e.index).fillna(False).to_numpy(bool)
    # imagens: exame positivo precisa de ao menos uma > 0,5; negativo, nenhuma
    im["_pos"] = im[STUDY].map(positive).fillna(False).astype(bool)
    neg_rows = ~im["_pos"]
    im.loc[neg_rows, "p"] = np.minimum(im.loc[neg_rows, "p"], LO)
    top = im[im["_pos"]].groupby(STUDY)["p"].idxmax()
    im.loc[top, "p"] = np.maximum(im.loc[top, "p"], HI)
    im = im.drop(columns="_pos")
    # exames positivos
    P = e.loc[pos].copy()
    P["negative_exam_for_pe"] = np.minimum(P["negative_exam_for_pe"], LO)
    P["indeterminate"] = np.minimum(P["indeterminate"], LO)
    gte = P["rv_lv_ratio_gte_1"] >= P["rv_lv_ratio_lt_1"]
    P["rv_lv_ratio_gte_1"] = np.where(gte, np.maximum(P["rv_lv_ratio_gte_1"], HI), np.minimum(P["rv_lv_ratio_gte_1"], LO))
    P["rv_lv_ratio_lt_1"] = np.where(gte, np.minimum(P["rv_lv_ratio_lt_1"], LO), np.maximum(P["rv_lv_ratio_lt_1"], HI))
    loc = ["central_pe", "leftsided_pe", "rightsided_pe"]
    none = (P[loc] <= 0.5).all(axis=1)
    if none.any():
        best = P.loc[none, loc].idxmax(axis=1)
        for c in loc:
            m = none & (best.reindex(P.index) == c)
            P.loc[m, c] = HI
    both = (P["chronic_pe"] > 0.5) & (P["acute_and_chronic_pe"] > 0.5)
    small = np.where(P["chronic_pe"] < P["acute_and_chronic_pe"], "chronic_pe", "acute_and_chronic_pe")
    for c in ("chronic_pe", "acute_and_chronic_pe"):
        m = both & (small == c)
        P.loc[m, c] = LO
    e.loc[pos] = P
    # exames negativos
    N = e.loc[~pos].copy()
    neg_wins = N["negative_exam_for_pe"] >= N["indeterminate"]
    N["negative_exam_for_pe"] = np.where(neg_wins, np.maximum(N["negative_exam_for_pe"], HI), np.minimum(N["negative_exam_for_pe"], LO))
    N["indeterminate"] = np.where(neg_wins, np.minimum(N["indeterminate"], LO), np.maximum(N["indeterminate"], HI))
    N[PE_LABELS] = np.minimum(N[PE_LABELS], LO)
    e.loc[~pos] = N
    return e, im


def to_long(exams: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    """Formato do desafio: id,label (SOP para imagem; '<exame>_<rótulo>' para exame)."""
    ex = exams[EXAM_LABELS].stack().reset_index()
    ex.columns = [STUDY, "lab", "label"]
    ex["id"] = ex[STUDY] + "_" + ex["lab"]
    return pd.concat([ex[["id", "label"]], images.rename(columns={SOP: "id", "p": "label"})[["id", "label"]]],
                     ignore_index=True)


def build(slices: pd.DataFrame, exam_scores: pd.Series, priors: dict, cal: dict | None,
          sop_by_study: pd.DataFrame | None = None, constant: bool = False):
    """slices: STUDY, SOP, p (probabilidade bruta do modelo); exam_scores: escore bruto (top-k) por
    exame; sop_by_study: TODAS as imagens a prever (STUDY, SOP), para cobrir cortes sem predição.
    constant=True gera a referência 'só prevalências' (sem modelo)."""
    if sop_by_study is None:
        sop_by_study = slices[[STUDY, SOP]]
    im = sop_by_study[[STUDY, SOP]].merge(slices[[SOP, "p"]], on=SOP, how="left")
    studies = pd.Index(sorted(im[STUDY].unique()))
    if constant:
        im["p"] = priors["p_image"]
        p_pe = pd.Series(priors["p_pe_exam"], index=studies)
    else:
        im["p"] = prior_correct(im["p"].to_numpy(float), priors["q"])
        p_pe = pd.Series(apply_platt(cal, exam_scores.reindex(studies).to_numpy(float)), index=studies)
        p_pe = p_pe.fillna(priors["p_pe_exam"])                      # exame sem predição
        fill = im.groupby(STUDY)["p"].transform("mean")              # corte sem predição
        im["p"] = im["p"].fillna(fill).fillna(priors["p_image"])
    exams = exam_table(p_pe, priors)
    exams, im = make_consistent(exams, im, p_pe > 0.5)
    return exams, im


# ----------------------------------------------------------------- verificação e métrica

def check_consistency(exams: pd.DataFrame, images: pd.DataFrame) -> dict:
    """Implementação independente das regras (contagem de exames que violam cada uma)."""
    img_pos = images.groupby(STUDY)["p"].max().reindex(exams.index) > 0.5
    e = exams > 0.5
    P, N = e[img_pos], e[~img_pos]
    return {
        "pos_neg_or_indet": int((P["negative_exam_for_pe"] | P["indeterminate"]).sum()),
        "pos_rv_not_exactly_one": int((P["rv_lv_ratio_gte_1"] == P["rv_lv_ratio_lt_1"]).sum()),
        "pos_no_location": int((~P[["central_pe", "leftsided_pe", "rightsided_pe"]].any(axis=1)).sum()),
        "pos_chronic_and_acute_chronic": int((P["chronic_pe"] & P["acute_and_chronic_pe"]).sum()),
        "neg_not_exactly_one_of_neg_indet": int((N["negative_exam_for_pe"] == N["indeterminate"]).sum()),
        "neg_with_pe_label": int(N[PE_LABELS].any(axis=1).sum()),
    }


def _bce(y, p):
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def weighted_log_loss(exams: pd.DataFrame, images: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Métrica do desafio. truth: linhas de train.csv (STUDY, SOP, pe_present_on_image + rótulos
    de exame) dos exames avaliados."""
    t_img = truth[[STUDY, SOP, "pe_present_on_image"]].merge(images[[SOP, "p"]], on=SOP, how="inner")
    qi = t_img.groupby(STUDY)["pe_present_on_image"].transform("mean")
    img_loss = (W_IMAGE * qi * _bce(t_img["pe_present_on_image"], t_img["p"])).sum()
    img_w = (W_IMAGE * qi).sum()
    t_ex = truth.groupby(STUDY)[EXAM_LABELS].first().reindex(exams.index)
    ex_loss = sum(WEIGHTS[c] * _bce(t_ex[c], exams[c]).sum() for c in EXAM_LABELS)
    ex_w = len(exams) * sum(WEIGHTS.values())
    per_label = {c: float(_bce(t_ex[c], exams[c]).mean()) for c in EXAM_LABELS}
    per_label["pe_present_on_image (ponderado por q_i)"] = float(img_loss / max(img_w, EPS))
    return {"weighted_log_loss": float((img_loss + ex_loss) / (img_w + ex_w)),
            "n_exams": int(len(exams)), "n_images": int(len(t_img)), "per_label": per_label}
