"""Ingeniería de características y validación temporal para el modelo de rechazo (objetivo 3).

Dos tipos de variables:

* Variables temporales con rezago (``variables_temporales``): se calculan sobre la base ordenada
  por fecha usando solo días anteriores, con un desfase de seguridad para las que usan el resultado
  del trámite. No aprenden parámetros, así que no producen fuga aunque se calculen antes de partir.
* Transformaciones que aprenden de los datos (frecuencia y codificación del barrio, límites IQR,
  escalado, codificación one-hot): son transformadores que se ajustan dentro del Pipeline, solo con
  el entrenamiento de cada corte.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

try:  # festivos de Colombia; si la librería no está, la variable queda en cero
    import holidays
    _FESTIVOS = holidays.country_holidays("CO", years=range(2023, 2028))
except Exception:  # pragma: no cover
    _FESTIVOS = {}

DESFASE_DIAS = 14  # días que se asume tarda la decisión del trámite antes de conocerse


def _conteo_ventana(diario: pd.Series, desde: int, hasta: int) -> pd.Series:
    """Suma de ``diario`` en la ventana de días [t - desde, t - hasta] (calendario, sin incluir t)."""
    acumulado = diario.cumsum()
    return acumulado.shift(hasta).fillna(0) - acumulado.shift(desde + 1).fillna(0)


def variables_temporales(df: pd.DataFrame, fecha="fecha_ingreso", objetivo="rechazo", m=20) -> pd.DataFrame:
    """Agrega rezagos y ventanas móviles calculados solo con días anteriores a cada solicitud."""
    out = df.copy()
    dia = out[fecha].dt.normalize()
    rango = pd.date_range(dia.min() - pd.Timedelta(days=200), dia.max())
    n_dia = dia.value_counts().reindex(rango, fill_value=0)
    r_dia = out.groupby(dia)[objetivo].sum().reindex(rango, fill_value=0)

    vol7 = _conteo_ventana(n_dia, 7, 1)
    vol30 = _conteo_ventana(n_dia, 30, 1)
    # Tasa de rechazo de la ventana [t-44, t-15]: el desfase evita usar decisiones aún no tomadas.
    d0, d1 = 30 + DESFASE_DIAS, 1 + DESFASE_DIAS
    n_prev, r_prev = _conteo_ventana(n_dia, d0, d1), _conteo_ventana(r_dia, d0, d1)
    tasa_prev = (r_prev / n_prev.replace(0, np.nan))
    out["volumen_7d_previo"] = dia.map(vol7).astype(float)
    out["volumen_30d_previo"] = dia.map(vol30).astype(float)
    out["ratio_volumen_7d_30d"] = out["volumen_7d_previo"] / (out["volumen_30d_previo"] / 30 * 7).replace(0, np.nan)
    out["tasa_rechazo_previa_30d"] = dia.map(tasa_prev).astype(float)

    # Tasa previa del barrio en 90 días con el mismo desfase, suavizada hacia la tasa general previa.
    b0, b1 = 90 + DESFASE_DIAS, 1 + DESFASE_DIAS
    n_g, r_g = _conteo_ventana(n_dia, b0, b1), _conteo_ventana(r_dia, b0, b1)
    tasa_g = r_g / n_g.replace(0, np.nan)
    tasa_barrio = pd.Series(np.nan, index=out.index)
    for barrio, g in out.groupby("barrio"):
        d = g[fecha].dt.normalize()
        nb = d.value_counts().reindex(rango, fill_value=0)
        rb = g.groupby(d)[objetivo].sum().reindex(rango, fill_value=0)
        n_b, r_b = _conteo_ventana(nb, b0, b1), _conteo_ventana(rb, b0, b1)
        prior = tasa_g.fillna(tasa_g.mean())
        suav = (r_b + m * prior) / (n_b + m)
        tasa_barrio.loc[g.index] = d.map(suav).to_numpy()
    out["tasa_rechazo_barrio_previa_90d"] = tasa_barrio.astype(float)
    return out


def variables_calendario(df: pd.DataFrame, fecha="fecha_ingreso") -> pd.DataFrame:
    out = df.copy()
    f = out[fecha]
    out["dia_mes"] = f.dt.day
    out["semana_anio"] = f.dt.isocalendar().week.astype(int)
    out["trimestre"] = f.dt.quarter
    out["es_inicio_mes"] = (f.dt.day <= 5).astype(int)
    out["es_fin_mes"] = (f.dt.day >= 25).astype(int)
    out["mes_sin"], out["mes_cos"] = np.sin(2 * np.pi * f.dt.month / 12), np.cos(2 * np.pi * f.dt.month / 12)
    out["dia_semana_sin"], out["dia_semana_cos"] = np.sin(2 * np.pi * f.dt.day_of_week / 7), np.cos(2 * np.pi * f.dt.day_of_week / 7)
    festivo = f.dt.date.map(lambda d: d in _FESTIVOS)
    out["es_festivo"] = festivo.astype(int)
    out["despues_de_festivo"] = (f - pd.Timedelta(days=1)).dt.date.map(lambda d: d in _FESTIVOS).astype(int)
    return out


def variables_dominio(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    t = out["tiempo_residencia_anios"]
    out["log_tiempo_residencia"] = np.log1p(t)
    out["tiempo_cero"] = t.eq(0).astype(int)
    out["tiempo_decimal"] = (t % 1 != 0).astype(int)
    out["juramento"] = out["juramento"].astype(int)
    return out


def construir_base(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las tres familias deterministas sobre la base limpia ordenada por fecha."""
    df = df.sort_values("fecha_ingreso", kind="stable").reset_index(drop=True)
    return variables_dominio(variables_calendario(variables_temporales(df)))


# --- Transformadores que aprenden dentro del Pipeline --------------------------------------------
class FrecuenciaCategoria(BaseEstimator, TransformerMixin):
    """Proporción de cada categoría en el entrenamiento (0 para categorías nuevas)."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.frecuencias_ = {c: X[c].value_counts(normalize=True).to_dict() for c in X.columns}
        self.columnas_ = list(X.columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columnas_)
        return np.column_stack([X[c].map(self.frecuencias_[c]).fillna(0).to_numpy(float) for c in self.columnas_])

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"{c}_frecuencia" for c in self.columnas_], dtype=object)


class AtipicoIQR(BaseEstimator, TransformerMixin):
    """Indicador de atípico con límites IQR aprendidos en el entrenamiento."""

    def __init__(self, k=1.5):
        self.k = k

    def fit(self, X, y=None):
        x = np.asarray(X, float).ravel()
        q1, q3 = np.percentile(x, [25, 75])
        self.limites_ = (q1 - self.k * (q3 - q1), q3 + self.k * (q3 - q1))
        return self

    def transform(self, X):
        x = np.asarray(X, float).ravel()
        return ((x < self.limites_[0]) | (x > self.limites_[1])).astype(float).reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(["atipico_tiempo"], dtype=object)


# Grupos de variables (sirven para la tabla, la selección y la ablación por grupos)
GRUPOS = {
    "Residencia": ["tiempo_residencia_anios", "log_tiempo_residencia", "tiempo_cero", "tiempo_decimal", "atipico_tiempo"],
    "Solicitante y documento": ["es_menor", "es_extranjero", "es_representacion", "juramento", "tipo_documento"],
    "Barrio": ["barrio_te", "barrio_frecuencia", "tasa_rechazo_barrio_previa_90d"],
    "Calendario": ["mes", "dia_semana", "dia_mes", "semana_anio", "trimestre", "es_inicio_mes", "es_fin_mes",
                   "mes_sin", "mes_cos", "dia_semana_sin", "dia_semana_cos", "es_festivo", "despues_de_festivo"],
    "Rezagos y ventanas": ["volumen_7d_previo", "volumen_30d_previo", "ratio_volumen_7d_30d", "tasa_rechazo_previa_30d"],
}
NUMERICAS = ["tiempo_residencia_anios", "log_tiempo_residencia", "tiempo_cero", "tiempo_decimal", "es_menor",
             "es_extranjero", "es_representacion", "juramento", "mes", "dia_semana", "dia_mes", "semana_anio",
             "trimestre", "es_inicio_mes", "es_fin_mes", "mes_sin", "mes_cos", "dia_semana_sin", "dia_semana_cos",
             "es_festivo", "despues_de_festivo", "volumen_7d_previo", "volumen_30d_previo", "ratio_volumen_7d_30d",
             "tasa_rechazo_previa_30d", "tasa_rechazo_barrio_previa_90d"]


def preprocesador(variables: list[str], semilla: int = 42) -> ColumnTransformer:
    """ColumnTransformer que se ajusta solo con el entrenamiento de cada corte.

    ``variables`` son nombres finales (después de transformar); el preprocesador incluye solo las
    ramas necesarias para producirlas.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    ramas = []
    num = [v for v in NUMERICAS if v in variables]
    if num:
        ramas.append(("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num))
    if "atipico_tiempo" in variables:
        ramas.append(("atipico", AtipicoIQR(), ["tiempo_residencia_anios"]))
    if "tipo_documento" in variables:
        ramas.append(("documento", OneHotEncoder(handle_unknown="ignore", min_frequency=20), ["tipo_documento"]))
    if "barrio_te" in variables:
        # TargetEncoder hace validación cruzada interna al ajustar: evita que cada fila vea su propio resultado.
        ramas.append(("barrio_te", TargetEncoder(target_type="binary", random_state=semilla), ["barrio"]))
    if "barrio_frecuencia" in variables:
        ramas.append(("barrio_frec", FrecuenciaCategoria(), ["barrio"]))
    if "barrio_onehot" in variables:
        ramas.append(("barrio_oh", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30), ["barrio"]))
    if "calidad_solicitante" in variables:
        ramas.append(("calidad", OneHotEncoder(handle_unknown="ignore"), ["calidad_solicitante"]))
    return ColumnTransformer(ramas, remainder="drop", verbose_feature_names_out=False)


# --- Validación temporal por días -------------------------------------------------------------
class TimeSeriesSplitPorDia:
    """TimeSeriesSplit aplicado a los días, no a las filas: un mismo día nunca queda partido
    entre entrenamiento y validación."""

    def __init__(self, n_splits=10, max_train_dias=None):
        self.n_splits = n_splits
        self.max_train_dias = max_train_dias

    def split(self, X, y=None, groups=None, fechas=None):
        from sklearn.model_selection import TimeSeriesSplit

        dias = pd.Series(pd.to_datetime(fechas)).dt.normalize().to_numpy()
        unicos = np.unique(dias)
        tss = TimeSeriesSplit(n_splits=self.n_splits, max_train_size=self.max_train_dias)
        for tr_d, va_d in tss.split(unicos):
            tr = np.flatnonzero(np.isin(dias, unicos[tr_d]))
            va = np.flatnonzero(np.isin(dias, unicos[va_d]))
            yield tr, va

    def get_n_splits(self, *a, **k):
        return self.n_splits


class BaseTasaPrevia(BaseEstimator, ClassifierMixin):
    """Línea base del protocolo: predice rechazo si la tasa de rechazo previa supera la
    tasa media del entrenamiento. Su puntaje es la propia tasa previa."""

    def __init__(self, columna="tasa_rechazo_previa_30d"):
        self.columna = columna

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        x = X[self.columna].to_numpy(float)
        self.relleno_ = float(np.nanmean(y))
        self.umbral_ = float(np.mean(y))
        return self

    def _x(self, X):
        return np.nan_to_num(X[self.columna].to_numpy(float), nan=self.relleno_)

    def predict(self, X):
        return (self._x(X) > self.umbral_).astype(int)

    def predict_proba(self, X):
        p = self._x(X)
        return np.column_stack([1 - p, p])
