"""Indicadores de calidad de datos (ISO/IEC 25012) para medir el efecto de las reglas."""
from __future__ import annotations

import time

import pandas as pd

from .pipelines import PIPELINES, RENOMBRES_CARTAGENA, RENOMBRES_VILLAVICENCIO
from .reglas import es_dato_personal, normalizar_comuna, normalizar_valor

CONFIG = {
    "cartagena": {"renombres": RENOMBRES_CARTAGENA, "geo": "barrio", "excluir_unicidad": [], "texto_libre": ["barrio"]},
    "villavicencio": {"renombres": RENOMBRES_VILLAVICENCIO, "geo": "comuna", "excluir_unicidad": ["id"], "texto_libre": ["barrio", "comuna"]},
}


def _conforme(ciudad: str, valor) -> bool:
    if not isinstance(valor, str):
        return False
    if ciudad == "villavicencio":
        return normalizar_comuna(valor) == valor and valor != "Sin clasificar"
    return normalizar_valor(valor) == valor


def medir(df: pd.DataFrame, ciudad: str) -> dict:
    """Mide los indicadores sobre una tabla con nombres de columna ya armonizados."""
    cfg = CONFIG[ciudad]
    df = df.rename(columns=cfg["renombres"])
    subset = [c for c in df.columns if c not in cfg["excluir_unicidad"]]
    geo = df[cfg["geo"]]
    personales = sum(int(df[c].map(es_dato_personal).notna().sum()) for c in cfg["texto_libre"] if c in df.columns)
    return {
        "Registros": len(df),
        "Completitud (%)": round(float(df.notna().mean().mean() * 100), 2),
        "Unicidad (%)": round(float((~df.duplicated(subset)).mean() * 100), 2) if len(df) else 100.0,
        "Conformidad geográfica (%)": round(float(geo.map(lambda v: _conforme(ciudad, v)).mean() * 100), 2),
        "Categorías geográficas": int(geo.nunique()),
        "Datos personales expuestos": personales,
    }


def aplicar_y_medir(crudo: pd.DataFrame, ciudad: str, pipeline=None) -> tuple[pd.DataFrame, dict]:
    """Aplica el pipeline de calidad y devuelve la base limpia y un reporte antes/después."""
    pipe = pipeline if pipeline is not None else PIPELINES[ciudad]().fit(crudo)
    t0 = time.perf_counter()
    limpio = pipe.transform(crudo)
    segundos = time.perf_counter() - t0
    antes, despues = medir(crudo, ciudad), medir(limpio, ciudad)
    # La unicidad se mide justo después de la deduplicación, con la clave completa (en Cartagena,
    # incluido el código del trámite): después de retirar el código, trámites distintos parecen iguales.
    k = [n for n, _ in pipe.steps].index("duplicados") + 1
    intermedio = pipe[:k].transform(crudo)
    subset = [c for c in intermedio.columns if c not in CONFIG[ciudad]["excluir_unicidad"]]
    despues["Unicidad (%)"] = round(float((~intermedio.duplicated(subset)).mean() * 100), 2) if len(intermedio) else 100.0
    reporte = {"antes": antes, "despues": despues, "segundos": segundos,
               "duplicados_retirados": pipe.named_steps["duplicados"].retirados_}
    return limpio, reporte
