"""Batería de modelos, espacios de búsqueda de Optuna y evaluación con validación temporal."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .caracteristicas import BaseTasaPrevia, preprocesador

# Nombre, familia (a-d del ACA 2) y si el balanceo se pasa como pesos de muestra en fit.
MODELOS = {
    "Base mayoritaria": ("a. Línea base", False),
    "Base estratificada": ("a. Línea base", False),
    "Base tasa previa": ("a. Línea base", False),
    "Regresión logística": ("a. Clásico", False),
    "Lasso logístico (L1)": ("b. Convencional", False),
    "KNN": ("b. Convencional", False),
    "SVM (RBF)": ("b. Convencional", False),
    "Árbol de decisión": ("b. Convencional", False),
    "Random Forest": ("c. Ensamble", False),
    "XGBoost": ("c. Ensamble", True),
    "LightGBM": ("c. Ensamble", False),
    "MLP": ("d. Red neuronal", True),
}
SIN_AJUSTE = ["Base mayoritaria", "Base estratificada", "Base tasa previa"]
AJUSTABLES = [m for m in MODELOS if m not in SIN_AJUSTE]

UMBRAL = ("umbral de decisión", "0.05 a 0.95")  # se agrega a todos los modelos ajustables
ESPACIOS = {  # texto para la tabla del documento: (hiperparámetro, rango)
    "Regresión logística": [("C", "1e-3 a 10, log")],
    "Lasso logístico (L1)": [("C", "1e-3 a 10, log")],
    "KNN": [("n_neighbors", "5 a 151"), ("weights", "uniform, distance"), ("p", "1, 2")],
    "SVM (RBF)": [("C", "1e-2 a 10, log"), ("gamma", "1e-4 a 1, log")],
    "Árbol de decisión": [("max_depth", "2 a 12"), ("min_samples_leaf", "5 a 200"), ("criterion", "gini, entropy")],
    "Random Forest": [("n_estimators", "100 a 400"), ("max_depth", "3 a 16"), ("min_samples_leaf", "1 a 50"), ("max_features", "sqrt, 0.5, 0.8")],
    "XGBoost": [("n_estimators", "100 a 500"), ("max_depth", "2 a 8"), ("learning_rate", "0.01 a 0.3, log"), ("subsample", "0.6 a 1"), ("colsample_bytree", "0.5 a 1"), ("min_child_weight", "1 a 20"), ("reg_lambda", "1e-3 a 10, log")],
    "LightGBM": [("n_estimators", "100 a 500"), ("num_leaves", "8 a 64"), ("learning_rate", "0.01 a 0.3, log"), ("min_child_samples", "10 a 200"), ("subsample", "0.6 a 1"), ("colsample_bytree", "0.5 a 1"), ("reg_lambda", "1e-3 a 10, log")],
    "MLP": [("hidden_layer_sizes", "(32,), (64,), (64, 32), (128, 64)"), ("alpha", "1e-5 a 1e-1, log"), ("learning_rate_init", "1e-4 a 1e-2, log")],
}
CAPAS = {"32": (32,), "64": (64,), "64-32": (64, 32), "128-64": (128, 64)}


def sugerir(trial, nombre: str) -> dict:
    """Hiperparámetros del modelo más el umbral de decisión, que se ajusta igual para todos."""
    return {**_sugerir_modelo(trial, nombre), "umbral": trial.suggest_float("umbral", .05, .95)}


def _sugerir_modelo(trial, nombre: str) -> dict:
    s = trial
    if nombre in ("Regresión logística", "Lasso logístico (L1)"):
        return {"C": s.suggest_float("C", 1e-3, 10, log=True)}
    if nombre == "KNN":
        return {"n_neighbors": s.suggest_int("n_neighbors", 5, 151, step=2), "weights": s.suggest_categorical("weights", ["uniform", "distance"]), "p": s.suggest_categorical("p", [1, 2])}
    if nombre == "SVM (RBF)":
        return {"C": s.suggest_float("C", 1e-2, 10, log=True), "gamma": s.suggest_float("gamma", 1e-4, 1, log=True)}
    if nombre == "Árbol de decisión":
        return {"max_depth": s.suggest_int("max_depth", 2, 12), "min_samples_leaf": s.suggest_int("min_samples_leaf", 5, 200, log=True), "criterion": s.suggest_categorical("criterion", ["gini", "entropy"])}
    if nombre == "Random Forest":
        return {"n_estimators": s.suggest_int("n_estimators", 100, 400, step=50), "max_depth": s.suggest_int("max_depth", 3, 16), "min_samples_leaf": s.suggest_int("min_samples_leaf", 1, 50, log=True), "max_features": s.suggest_categorical("max_features", ["sqrt", 0.5, 0.8])}
    if nombre == "XGBoost":
        return {"n_estimators": s.suggest_int("n_estimators", 100, 500, step=50), "max_depth": s.suggest_int("max_depth", 2, 8), "learning_rate": s.suggest_float("learning_rate", .01, .3, log=True), "subsample": s.suggest_float("subsample", .6, 1), "colsample_bytree": s.suggest_float("colsample_bytree", .5, 1), "min_child_weight": s.suggest_int("min_child_weight", 1, 20), "reg_lambda": s.suggest_float("reg_lambda", 1e-3, 10, log=True)}
    if nombre == "LightGBM":
        return {"n_estimators": s.suggest_int("n_estimators", 100, 500, step=50), "num_leaves": s.suggest_int("num_leaves", 8, 64, log=True), "learning_rate": s.suggest_float("learning_rate", .01, .3, log=True), "min_child_samples": s.suggest_int("min_child_samples", 10, 200, log=True), "subsample": s.suggest_float("subsample", .6, 1), "colsample_bytree": s.suggest_float("colsample_bytree", .5, 1), "reg_lambda": s.suggest_float("reg_lambda", 1e-3, 10, log=True)}
    if nombre == "MLP":
        return {"hidden_layer_sizes": s.suggest_categorical("hidden_layer_sizes", list(CAPAS)), "alpha": s.suggest_float("alpha", 1e-5, 1e-1, log=True), "learning_rate_init": s.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)}
    return {}


def estimador(nombre: str, params: dict | None = None, semilla: int = 42):
    p = dict(params or {})
    if nombre == "Base mayoritaria":
        return DummyClassifier(strategy="most_frequent")
    if nombre == "Base estratificada":
        return DummyClassifier(strategy="stratified", random_state=semilla)
    if nombre == "Regresión logística":
        return LogisticRegression(class_weight="balanced", max_iter=2000, **p)
    if nombre == "Lasso logístico (L1)":
        return LogisticRegression(l1_ratio=1.0, solver="liblinear", class_weight="balanced", max_iter=2000, **p) \
            if _l1_ratio_ok() else LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=2000, **p)
    if nombre == "KNN":
        return KNeighborsClassifier(**p)
    if nombre == "SVM (RBF)":
        return SVC(kernel="rbf", class_weight="balanced", random_state=semilla, **p)
    if nombre == "Árbol de decisión":
        return DecisionTreeClassifier(class_weight="balanced", random_state=semilla, **p)
    if nombre == "Random Forest":
        return RandomForestClassifier(class_weight="balanced_subsample", random_state=semilla, n_jobs=-1, **p)
    if nombre == "XGBoost":
        return XGBClassifier(eval_metric="logloss", tree_method="hist", random_state=semilla, n_jobs=2, verbosity=0, **p)
    if nombre == "LightGBM":
        return LGBMClassifier(class_weight="balanced", subsample_freq=1, random_state=semilla, n_jobs=2, verbose=-1, **p)
    if nombre == "MLP":
        if "hidden_layer_sizes" in p and isinstance(p["hidden_layer_sizes"], str):
            p["hidden_layer_sizes"] = CAPAS[p["hidden_layer_sizes"]]
        return MLPClassifier(early_stopping=True, max_iter=300, random_state=semilla, **p)
    raise KeyError(nombre)


def _l1_ratio_ok():
    import sklearn
    mayor, menor = (int(x) for x in sklearn.__version__.split(".")[:2])
    return (mayor, menor) >= (1, 8)


def pipeline_modelo(nombre: str, variables: list[str], params: dict | None = None, semilla: int = 42):
    """Pipeline completo: preprocesamiento ajustado en entrenamiento + balanceo + modelo."""
    if nombre == "Base tasa previa":
        return BaseTasaPrevia()
    pasos = [("prep", preprocesador(variables, semilla))]
    if nombre == "KNN":  # KNN no admite pesos: se sobremuestrea la clase minoritaria solo en entrenamiento
        pasos.append(("balanceo", RandomOverSampler(random_state=semilla)))
        return ImbPipeline(pasos + [("modelo", estimador(nombre, params, semilla))])
    return Pipeline(pasos + [("modelo", estimador(nombre, params, semilla))])


def ajustar(pipe, nombre: str, X, y):
    if MODELOS[nombre][1]:
        return pipe.fit(X, y, modelo__sample_weight=compute_sample_weight("balanced", y))
    return pipe.fit(X, y)


def puntaje(pipe, X):
    """Puntaje entre 0 y 1. Para la SVM se aplica la función logística a la distancia al hiperplano
    (transformación monótona: no cambia el AUC y permite usar el mismo rango de umbrales)."""
    if isinstance(getattr(pipe, "steps", [[None, None]])[-1][1], SVC):
        return 1 / (1 + np.exp(-pipe.decision_function(X)))
    return pipe.predict_proba(X)[:, 1]


def predecir(pipe, X, umbral):
    score = puntaje(pipe, X)
    if umbral is None:
        return pipe.predict(X), score
    return (score >= umbral).astype(int), score


def metricas(y, pred, score) -> dict:
    una_clase = len(np.unique(y)) < 2
    return {
        "F1-macro": f1_score(y, pred, average="macro", zero_division=0),
        "AUC-ROC": np.nan if una_clase else roc_auc_score(y, score),
        "AUC-PR": np.nan if una_clase else average_precision_score(y, score),
        "MCC": matthews_corrcoef(y, pred),
        "Sensibilidad rechazo": recall_score(y, pred, zero_division=0),
        "Precisión rechazo": precision_score(y, pred, zero_division=0),
    }


def _separar(params):
    p = dict(params or {})
    return p, p.pop("umbral", None)


def evaluar_cv(nombre, variables, params, X, y, fechas, cv, semilla=42) -> pd.DataFrame:
    params, umbral = _separar(params)
    filas = []
    for k, (tr, va) in enumerate(cv.split(X, y, fechas=fechas), start=1):
        pipe = pipeline_modelo(nombre, variables, params, semilla)
        t0 = time.perf_counter()
        ajustar(pipe, nombre, X.iloc[tr], y.iloc[tr])
        seg = time.perf_counter() - t0
        pred, score = predecir(pipe, X.iloc[va], umbral)
        filas.append({"Modelo": nombre, "Partición": k, "n_entrenamiento": len(tr), "n_validación": len(va),
                      "Rechazos en validación": int(y.iloc[va].sum()), **metricas(y.iloc[va], pred, score),
                      "Tiempo de entrenamiento (s)": seg})
    return pd.DataFrame(filas)


def evaluar_prueba(nombre, variables, params, X_tr, y_tr, X_te, y_te, semilla=42):
    params, umbral = _separar(params)
    pipe = pipeline_modelo(nombre, variables, params, semilla)
    t0 = time.perf_counter()
    ajustar(pipe, nombre, X_tr, y_tr)
    seg = time.perf_counter() - t0
    pred, score = predecir(pipe, X_te, umbral)
    return pipe, pred, score, {**metricas(y_te, pred, score), "Tiempo de entrenamiento (s)": seg}
