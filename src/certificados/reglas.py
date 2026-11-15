"""Reglas de limpieza, normalización, deduplicación y enmascaramiento (objetivo específico 2).

Cada regla es un transformador de scikit-learn que recibe y devuelve un DataFrame de pandas.
Las reglas deterministas no aprenden nada en ``fit``; las que sí aprenden (por ejemplo, los
límites IQR de los atípicos) guardan lo aprendido en atributos terminados en guion bajo, de
modo que se ajustan con los datos de referencia y luego se aplican igual a cualquier lote nuevo.

Las reglas que quitan filas (duplicados y estados sin decisión final) solo se usan en el
pipeline de calidad. El pipeline de modelado del objetivo 3 usa únicamente reglas que
conservan las filas, porque scikit-learn exige que X e y conserven el mismo número de filas.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# --- Patrones de datos personales ---------------------------------------------------------
RE_ID = re.compile(r"\d{6,}")
RE_DIR = re.compile(
    r"\b(CRA|CARRERA|KR|CALLE|DIAGONAL|DG|TRANSVERSAL|TV|MZ|MANZANA|CS|CASA|APTO|APARTAMENTO|DTO|LOTE|LT|TORRE)\b.*\d"
    r"|\d+\s*-\s*\d+",
    re.I,
)
MARCA_ID = "[IDENTIFICADOR_ENMASCARADO]"
MARCA_DIR = "[DIRECCION_ENMASCARADA]"


def sin_acentos(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))


def normalizar_valor(valor):
    """Mayúsculas, sin acentos y con espacios simples. Deja intactos los valores no textuales."""
    if not isinstance(valor, str):
        return valor
    return re.sub(r"\s+", " ", sin_acentos(valor).upper()).strip()


def normalizar_comuna(valor) -> str:
    """Lleva la comuna o el corregimiento de Villavicencio a un formato estándar."""
    if not isinstance(valor, str):
        return "Sin clasificar"
    s = normalizar_valor(valor)
    m = re.fullmatch(r"COMUNA ?(\d{1,2})( VILLAVICENCIO)?", s)
    if m and 1 <= int(m.group(1)) <= 10:
        return f"Comuna {int(m.group(1))}"
    m = re.fullmatch(r"CORR?EGIMIENTO ?(?:N ?° ?)?([1-7])(?: - .*)?", s)
    if m:
        return f"Corregimiento {m.group(1)}"
    return "Sin clasificar"


def es_dato_personal(valor) -> str | None:
    """Devuelve 'id', 'dir' o None según el tipo de dato personal aparente."""
    if not isinstance(valor, str):
        return None
    if RE_ID.search(valor):
        return "id"
    if RE_DIR.search(valor):
        return "dir"
    return None


class _Regla(BaseEstimator, TransformerMixin):
    """Base: las reglas deterministas no aprenden nada en fit."""

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columnas_salida_, dtype=object)

    def _salida(self, df):
        self.columnas_salida_ = list(df.columns)
        return df


# --- Reglas que conservan las filas ---------------------------------------------------------
class RenombrarColumnas(_Regla):
    def __init__(self, mapa=None):
        self.mapa = mapa

    def transform(self, X):
        return self._salida(X.rename(columns=self.mapa or {}))


class EnmascaradorDatosPersonales(_Regla):
    """Reemplaza identificadores y direcciones en texto libre por un marcador.

    El conteo de la última ejecución queda en ``reporte_`` (nunca los valores originales).
    """

    def __init__(self, columnas=None):
        self.columnas = columnas

    def transform(self, X):
        df = X.copy()
        self.reporte_ = {}
        for col in self.columnas or []:
            if col not in df.columns:
                continue
            tipo = df[col].map(es_dato_personal)
            serie = df[col].astype("object")
            df[col] = serie.mask(tipo.eq("id"), MARCA_ID).mask(tipo.eq("dir"), MARCA_DIR)
            self.reporte_[col] = {"identificadores": int(tipo.eq("id").sum()), "direcciones": int(tipo.eq("dir").sum())}
        return self._salida(df)


class TiposDeDato(_Regla):
    def __init__(self, fechas=(), numericas=(), booleanas=()):
        self.fechas = fechas
        self.numericas = numericas
        self.booleanas = booleanas

    def transform(self, X):
        df = X.copy()
        for col in self.fechas:
            df[col] = pd.to_datetime(df[col])
        for col in self.numericas:
            df[col] = pd.to_numeric(df[col])
        mapa = {"true": True, "false": False, "True": True, "False": False, True: True, False: False}
        for col in self.booleanas:
            df[col] = df[col].map(mapa)
        return self._salida(df)


class NormalizadorTexto(_Regla):
    """Mayúsculas, sin acentos, espacios simples y correcciones de erratas conocidas."""

    def __init__(self, columnas=(), reemplazos=None):
        self.columnas = columnas
        self.reemplazos = reemplazos

    def transform(self, X):
        df = X.copy()
        for col in self.columnas:
            df[col] = df[col].map(normalizar_valor)
            for viejo, nuevo in (self.reemplazos or {}).get(col, {}).items():
                df[col] = df[col].str.replace(viejo, nuevo, regex=False)
        return self._salida(df)


class NormalizadorCategorias(_Regla):
    """Normaliza el texto y luego unifica categorías equivalentes con un diccionario."""

    def __init__(self, columna=None, equivalencias=None):
        self.columna = columna
        self.equivalencias = equivalencias

    def transform(self, X):
        df = X.copy()
        df[self.columna] = df[self.columna].map(normalizar_valor).replace(self.equivalencias or {})
        return self._salida(df)


class NormalizadorComuna(_Regla):
    def __init__(self, columna="comuna"):
        self.columna = columna

    def transform(self, X):
        df = X.copy()
        df[self.columna] = df[self.columna].map(normalizar_comuna)
        return self._salida(df)


class FaltantesExplicitos(_Regla):
    """Convierte los vacíos en una categoría explícita (no imputa valores)."""

    def __init__(self, valores=None):
        self.valores = valores

    def transform(self, X):
        df = X.copy()
        for col, valor in (self.valores or {}).items():
            df[col] = df[col].fillna(valor)
        return self._salida(df)


class IndicadorIgualdad(_Regla):
    """Crea una columna booleana: ``columna == valor``."""

    def __init__(self, columna=None, valor=None, nombre=None):
        self.columna = columna
        self.valor = valor
        self.nombre = nombre

    def transform(self, X):
        df = X.copy()
        df[self.nombre] = df[self.columna].eq(self.valor)
        return self._salida(df)


class RetirarColumnas(_Regla):
    def __init__(self, columnas=()):
        self.columnas = columnas

    def transform(self, X):
        return self._salida(X.drop(columns=list(self.columnas), errors="ignore"))


class MarcadorAtipicosIQR(BaseEstimator, TransformerMixin):
    """Marca, sin alterar, los valores fuera de [Q1 - k·IQR, Q3 + k·IQR].

    Los límites se aprenden en ``fit``: en el modelado se ajustan solo con el entrenamiento.
    """

    def __init__(self, columna=None, k=1.5, nombre="atipico"):
        self.columna = columna
        self.k = k
        self.nombre = nombre

    def fit(self, X, y=None):
        q1, q3 = X[self.columna].quantile([.25, .75])
        self.limite_inferior_ = q1 - self.k * (q3 - q1)
        self.limite_superior_ = q3 + self.k * (q3 - q1)
        return self

    def transform(self, X):
        df = X.copy()
        df[self.nombre] = ~df[self.columna].between(self.limite_inferior_, self.limite_superior_)
        self.columnas_salida_ = list(df.columns)
        return df

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columnas_salida_, dtype=object)


class Ordenar(_Regla):
    def __init__(self, columna=None):
        self.columna = columna

    def transform(self, X):
        return self._salida(X.sort_values(self.columna).reset_index(drop=True))


# --- Reglas que quitan filas (solo pipeline de calidad) ------------------------------------
class EliminadorDuplicados(_Regla):
    """Quita filas repetidas y conserva la primera aparición.

    ``excluir`` deja fuera de la comparación columnas como un consecutivo (id), que harían
    distintos dos registros idénticos. El número de filas retiradas queda en ``retirados_``.
    """

    def __init__(self, excluir=(), retirar_excluidas=False):
        self.excluir = excluir
        self.retirar_excluidas = retirar_excluidas

    def transform(self, X):
        subset = [c for c in X.columns if c not in set(self.excluir)]
        dup = X.duplicated(subset)
        self.retirados_ = int(dup.sum())
        df = X.loc[~dup].copy()
        if self.retirar_excluidas:
            df = df.drop(columns=list(self.excluir), errors="ignore")
        return self._salida(df)


class FiltroEstadosFinales(_Regla):
    """Conserva los trámites con decisión final y crea la variable objetivo binaria."""

    def __init__(self, columna="estado", finales=(), positivos=(), objetivo="rechazo"):
        self.columna = columna
        self.finales = finales
        self.positivos = positivos
        self.objetivo = objetivo

    def transform(self, X):
        final = X[self.columna].isin(self.finales)
        self.excluidos_ = X.loc[~final, self.columna].value_counts().to_dict()
        df = X.loc[final].copy()
        df[self.objetivo] = df[self.columna].isin(self.positivos).astype(int)
        return self._salida(df)
