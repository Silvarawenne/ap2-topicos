"""Modelos clássicos e espaços de busca de hiperparâmetros.

Todo pré-processamento ajustado a dados (padronização, PCA, SMOTE) fica DENTRO do
Pipeline e, portanto, é ajustado apenas na partição de treino de cada dobra.

Desbalanceamento (opções comparadas em ablação):
* 'class_weight': pesos inversamente proporcionais à frequência da classe na perda
  (sem criar dados sintéticos; altera a calibração das probabilidades).
* 'smote': sobreamostragem sintética (Chawla et al., 2002) só no treino da dobra; pode
  criar exemplos irreais em alta dimensão e não deve tocar a validação.
* 'none': referência.
"""
from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODELS = ["dummy", "logreg", "svm_rbf", "rf", "lgbm", "mlp"]
SLICE_MODELS = ["dummy", "logreg", "lgbm"]  # kernels/árvores densas inviáveis em ~10^5 cortes


def build(name: str, seed: int, imbalance: str = "class_weight", model_jobs: int = 1):
    """Retorna (pipeline, espaço de busca). Espaço vazio => sem ajuste de hiperparâmetros."""
    cw = "balanced" if imbalance == "class_weight" else None
    steps = [("impute", SimpleImputer(strategy="median"))]
    if imbalance == "smote":
        steps.append(("scale_smote", StandardScaler()))
        steps.append(("smote", SMOTE(random_state=seed)))
    space: dict = {}

    if name == "dummy":
        # Baseline trivial: probabilidade = prevalência do treino; prediz a classe majoritária.
        steps.append(("clf", DummyClassifier(strategy="prior")))
    elif name == "logreg":
        steps += [("scale", StandardScaler()), ("pca", "passthrough"),
                  ("clf", LogisticRegression(class_weight=cw, max_iter=3000, solver="lbfgs"))]
        space = {"clf__C": loguniform(1e-4, 1e1),
                 "pca": ["passthrough", PCA(0.95, random_state=seed)]}
    elif name == "svm_rbf":
        steps += [("scale", StandardScaler()), ("pca", PCA(0.95, random_state=seed)),
                  ("clf", SVC(kernel="rbf", class_weight=cw))]
        space = {"clf__C": loguniform(1e-2, 1e2), "clf__gamma": ["scale", 1e-3, 1e-2]}
    elif name == "rf":
        steps.append(("clf", RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample" if cw else None,
            random_state=seed, n_jobs=model_jobs)))
        space = {"clf__max_depth": [None, 8, 16], "clf__min_samples_leaf": randint(1, 20),
                 "clf__max_features": ["sqrt", 0.1, 0.3]}
    elif name == "lgbm":
        steps.append(("clf", LGBMClassifier(
            n_estimators=400, class_weight=cw, random_state=seed, n_jobs=model_jobs,
            verbose=-1, subsample_freq=1)))
        space = {"clf__learning_rate": loguniform(1e-2, 2e-1), "clf__num_leaves": randint(8, 64),
                 "clf__min_child_samples": randint(5, 100), "clf__subsample": uniform(0.6, 0.4),
                 "clf__colsample_bytree": uniform(0.3, 0.7), "clf__reg_lambda": loguniform(1e-3, 10)}
    elif name == "mlp":
        # MLP raso sobre vetores hand-crafted (permitido pelo enunciado). Sem class_weight
        # no sklearn: usar com imbalance='smote' ou interpretar o limiar ajustado.
        steps += [("scale", StandardScaler()), ("pca", PCA(0.95, random_state=seed)),
                  ("clf", MLPClassifier(max_iter=500, early_stopping=True, random_state=seed))]
        space = {"clf__hidden_layer_sizes": [(64,), (128,), (64, 32)],
                 "clf__alpha": loguniform(1e-5, 1e-1), "clf__learning_rate_init": loguniform(1e-4, 1e-2)}
    else:
        raise ValueError(name)
    return Pipeline(steps), space
