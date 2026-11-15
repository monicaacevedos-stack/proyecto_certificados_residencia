"""Pruebas unitarias de las reglas de calidad. Los valores son inventados para la prueba."""
import pandas as pd

from certificados.reglas import (MARCA_DIR, MARCA_ID, EliminadorDuplicados, EnmascaradorDatosPersonales,
                                 FaltantesExplicitos, MarcadorAtipicosIQR, NormalizadorTexto, normalizar_comuna)


def test_normalizar_comuna_variantes():
    assert normalizar_comuna("comuna 5") == "Comuna 5"
    assert normalizar_comuna("COMUNA  05 villavicencio") == "Comuna 5"
    assert normalizar_comuna("Comuna5") == "Comuna 5"
    assert normalizar_comuna("CORREGIMIENTO N° 3") == "Corregimiento 3"
    assert normalizar_comuna("Corregimiento 2 - Pompeya") == "Corregimiento 2"
    assert normalizar_comuna("Comuna 11") == "Sin clasificar"   # no existe
    assert normalizar_comuna("8") == "Sin clasificar"           # no se inventa información
    assert normalizar_comuna(None) == "Sin clasificar"


def test_enmascarar_identificadores_y_direcciones():
    df = pd.DataFrame({"barrio": ["LA ESPERANZA", "BARRIO 1234567890", "CALLE 15 # 22-30", "MZ 4 CASA 12", None]})
    regla = EnmascaradorDatosPersonales(["barrio"])
    out = regla.fit_transform(df)
    assert out["barrio"].tolist()[:4] == ["LA ESPERANZA", MARCA_ID, MARCA_DIR, MARCA_DIR]
    assert regla.reporte_["barrio"] == {"identificadores": 1, "direcciones": 2}
    # Idempotencia: enmascarar de nuevo no cambia nada.
    assert regla.transform(out).equals(out)


def test_normalizar_texto_y_errata():
    df = pd.DataFrame({"barrio": ["  San José  de los Campanos", "URBANZACION LA CAROLINA"]})
    out = NormalizadorTexto(["barrio"], {"barrio": {"URBANZACION": "URBANIZACION"}}).fit_transform(df)
    assert out["barrio"].tolist() == ["SAN JOSE DE LOS CAMPANOS", "URBANIZACION LA CAROLINA"]


def test_duplicados_excluyendo_consecutivo():
    df = pd.DataFrame({"id": [1, 2, 3], "x": ["a", "a", "b"]})
    regla = EliminadorDuplicados(excluir=["id"], retirar_excluidas=True)
    out = regla.fit_transform(df)
    assert regla.retirados_ == 1 and list(out.columns) == ["x"] and len(out) == 2


def test_faltantes_explicitos_no_imputa():
    df = pd.DataFrame({"estado": ["Aprobado", None]})
    assert FaltantesExplicitos({"estado": "Sin estado"}).fit_transform(df)["estado"].tolist() == ["Aprobado", "Sin estado"]


def test_iqr_aprende_en_fit_y_aplica_en_transform():
    entrenamiento = pd.DataFrame({"t": [1, 2, 3, 4, 5, 6, 7, 8]})
    regla = MarcadorAtipicosIQR("t", 1.5, "atipico").fit(entrenamiento)
    nuevo = pd.DataFrame({"t": [4, 100]})
    assert regla.transform(nuevo)["atipico"].tolist() == [False, True]
    assert regla.limite_superior_ == 11.5
