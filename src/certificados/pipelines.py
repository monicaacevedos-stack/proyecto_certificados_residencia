"""Pipelines de calidad de cada fuente, armados con las reglas de ``reglas.py``.

Reproducen el ETL del objetivo 1 (cuaderno ACA1_ETL_EDA) como código reutilizable:
el mismo pipeline se ajusta con los datos de referencia y se aplica a cualquier lote nuevo.
"""
from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from .reglas import (EliminadorDuplicados, EnmascaradorDatosPersonales, FaltantesExplicitos, FiltroEstadosFinales,
                     IndicadorIgualdad, MarcadorAtipicosIQR, NormalizadorCategorias, NormalizadorComuna,
                     NormalizadorTexto, Ordenar, RenombrarColumnas, RetirarColumnas, TiposDeDato, _Regla)

RENOMBRES_CARTAGENA = {
    "c_digo_de_tr_mite": "codigo_tramite", "calidad_en_la_que_act_a": "calidad_solicitante",
    "barrio_corregimiento": "barrio", "alcald_a_local": "alcaldia_local",
    "tipo_de_documento_de_identidad": "tipo_documento", "tiempo_de_residencia_a_o": "tiempo_residencia_anios",
    "juro_que_la_informaci_n": "juramento",
}
RENOMBRES_VILLAVICENCIO = {
    "fecha_y_hora_de_la_solicitud": "fecha_solicitud", "residente_colombiano": "tipo_residente",
    "comuna_o_corregimiento": "comuna", "finalidad_del_certificado": "finalidad", "tramite": "estado",
}
ESTADOS_FINALES = ("Aprobado", "Aprobado Parcial", "Rechazado", "Solicitud Rechazada")
ESTADOS_RECHAZO = ("Rechazado", "Solicitud Rechazada")


def _semestre(fecha: pd.Series) -> pd.Series:
    return fecha.dt.year.astype(str) + "-S" + ((fecha.dt.month > 6).astype(int) + 1).astype(str)


class DerivadasCartagena(_Regla):
    """Calendario e indicadores del tipo de documento y de la calidad del solicitante."""

    def transform(self, X):
        df = X.copy()
        f = df["fecha_ingreso"]
        df["anio"], df["mes"], df["dia_semana"] = f.dt.year, f.dt.month, f.dt.day_of_week
        df["semestre"] = _semestre(f)
        df["es_menor"] = df["tipo_documento"].isin(["Tarjeta de identidad", "Registro civil"]).astype(int)
        df["es_extranjero"] = df["tipo_documento"].eq("Cédula de extranjería").astype(int)
        df["es_representacion"] = df["calidad_solicitante"].str.contains("REPRESENTACION").astype(int)
        return self._salida(df)


class DerivadasVillavicencio(_Regla):
    def transform(self, X):
        df = X.copy()
        f = df["fecha_solicitud"]
        df["anio"], df["mes"], df["hora"], df["dia_semana"] = f.dt.year, f.dt.month, f.dt.hour, f.dt.day_of_week
        df["fuera_horario"] = (~df["hora"].between(7, 17) | (df["dia_semana"] >= 5)).astype(int)
        df["semestre"] = _semestre(f)
        df["es_aprobado"] = df["estado"].eq("Aprobado").astype(int)
        df["es_nino"] = df["grupo_etario"].eq("Niños").astype(int)
        df["es_ppl"] = df["tipo_residente"].str.contains("privadas").astype(int)
        df["es_laboral"] = df["finalidad"].eq("LABORAL").astype(int)
        return self._salida(df)


def pipeline_cartagena() -> Pipeline:
    return Pipeline([
        ("renombrar", RenombrarColumnas(RENOMBRES_CARTAGENA)),
        ("enmascarar", EnmascaradorDatosPersonales(["barrio"])),
        ("tipos", TiposDeDato(fechas=["fecha_ingreso"], numericas=["tiempo_residencia_anios"], booleanas=["juramento"])),
        # Los duplicados se buscan con el código presente: sin él, trámites distintos parecerían la misma fila.
        ("duplicados", EliminadorDuplicados()),
        ("codigo_faltante", IndicadorIgualdad("codigo_tramite", "No Aplica", "codigo_faltante")),
        ("retirar", RetirarColumnas(["alcaldia_local", "codigo_tramite"])),
        ("texto", NormalizadorTexto(["barrio", "calidad_solicitante"], {"barrio": {"URBANZACION": "URBANIZACION"}})),
        ("estados", FiltroEstadosFinales("estado", ESTADOS_FINALES, ESTADOS_RECHAZO, "rechazo")),
        ("atipicos", MarcadorAtipicosIQR("tiempo_residencia_anios", 1.5, "atipico_tiempo")),
        ("derivadas", DerivadasCartagena()),
        ("ordenar", Ordenar("fecha_ingreso")),
    ])


def pipeline_villavicencio() -> Pipeline:
    return Pipeline([
        ("renombrar", RenombrarColumnas(RENOMBRES_VILLAVICENCIO)),
        ("enmascarar", EnmascaradorDatosPersonales(["barrio", "comuna"])),
        ("tipos", TiposDeDato(fechas=["fecha_solicitud"])),
        ("redundante", RetirarColumnas(["estado_del_tramite"])),
        ("duplicados", EliminadorDuplicados(excluir=["id"], retirar_excluidas=True)),
        ("comuna", NormalizadorComuna("comuna")),
        ("finalidad", NormalizadorCategorias("finalidad", {"TRAMITE ANTE EMBAJADA": "ANTE EMBAJADA"})),
        ("faltantes", FaltantesExplicitos({"grupo_etario": "No aplica (PPL)", "estado": "Sin estado"})),
        ("retirar_barrio", RetirarColumnas(["barrio"])),
        ("derivadas", DerivadasVillavicencio()),
        ("ordenar", Ordenar("fecha_solicitud")),
    ])


PIPELINES = {"cartagena": pipeline_cartagena, "villavicencio": pipeline_villavicencio}
