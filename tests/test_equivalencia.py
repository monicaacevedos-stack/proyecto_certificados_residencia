"""El pipeline de scripts debe reproducir exactamente la base limpia del ETL del objetivo 1."""
from pathlib import Path

import pandas as pd
import pytest

from certificados.pipelines import PIPELINES

BASE = Path(__file__).resolve().parents[1]
CASOS = {
    "cartagena": ("data/raw/cartagena_2muq-8ebv_2026-10-31_enmascarado.csv", "data/processed/cartagena_limpio.csv", "fecha_ingreso"),
    "villavicencio": ("data/raw/villavicencio_38t2-j7vd_2026-10-31_enmascarado.csv", "data/processed/villavicencio_limpio.csv", "fecha_solicitud"),
}


@pytest.mark.parametrize("ciudad", CASOS)
def test_pipeline_reproduce_etl_aca1(ciudad):
    crudo_ruta, limpio_ruta, fecha = CASOS[ciudad]
    crudo = pd.read_csv(BASE / crudo_ruta, dtype=str, encoding="utf-8")
    esperado = pd.read_csv(BASE / limpio_ruta, parse_dates=[fecha])
    obtenido = PIPELINES[ciudad]().fit_transform(crudo)
    assert list(obtenido.columns) == list(esperado.columns)
    # Se compara el contenido a través del mismo formato CSV en que se guardó la base del ACA 1.
    obtenido = pd.read_csv(pd.io.common.StringIO(obtenido.to_csv(index=False)), parse_dates=[fecha])
    pd.testing.assert_frame_equal(obtenido, esperado, check_dtype=False)
