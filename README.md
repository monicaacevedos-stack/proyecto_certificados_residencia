# Certificados de residencia: calidad de datos y rechazo de solicitudes

Trabajo de grado de la Especialización en Analítica de Datos, CUN. Integrantes: Mónica Acevedo Sánchez e Ingrid Vanessa López Tautiva.

Versión **v0.2** (ACA 2): reglas de calidad automatizadas como transformadores de scikit-learn (objetivo 2), e ingeniería de características, batería de modelos con Optuna, ablación y pruebas de robustez para clasificar el rechazo (objetivo 3).

Versión v0.1 (ACA 1): adquisición de datos, ETL, EDA multivariado y protocolo de evaluación.

## Cómo ejecutarlo

En Google Colab, abre el cuaderno y usa *Entorno de ejecución > Ejecutar todo*:

- [ACA1_ETL_EDA.ipynb](https://colab.research.google.com/github/monicaacevedos-stack/proyecto_certificados_residencia/blob/main/notebooks/ACA1_ETL_EDA.ipynb): objetivo 1.
- [ACA2_Calidad_y_Modelado.ipynb](https://colab.research.google.com/github/monicaacevedos-stack/proyecto_certificados_residencia/blob/main/notebooks/ACA2_Calidad_y_Modelado.ipynb): objetivos 2 y 3. Con `REUTILIZAR_BUSQUEDA = True` reutiliza los hiperparámetros guardados en `results/tables/optuna_trials.csv`; con `False` repite la búsqueda completa de Optuna (unos 40 minutos con 2 núcleos).

En un equipo local:

```bash
pip install -r requirements.txt
python -m pytest                      # pruebas de las reglas y equivalencia con el ETL del ACA 1
jupyter nbconvert --to notebook --execute --inplace notebooks/ACA1_ETL_EDA.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/ACA2_Calidad_y_Modelado.ipynb
python scripts/figuras_documento.py   # figuras de método del documento
```

Para limpiar un lote nuevo con las reglas automáticas:

```bash
python scripts/ejecutar_calidad.py --ciudad villavicencio --entrada lote.csv --salida results/lote_limpio.csv \
       --referencia data/raw/villavicencio_38t2-j7vd_2026-10-31_enmascarado.csv
```

Los cuadernos usan semilla fija (42) y trabajan con las copias fechadas de `data/raw`. Si el cuaderno del ACA 1 no las encuentra, descarga los datos de la API de datos.gov.co; una descarga posterior puede dar otras cifras si la fuente cambió.

## Estructura

| Carpeta | Contenido |
|---|---|
| `data/raw` | Copias fechadas y enmascaradas, y un manifiesto de cada fuente |
| `data/processed` | Bases limpias |
| `notebooks` | Cuadernos del ACA 1 (ETL y EDA) y del ACA 2 (calidad y modelado); `aca2_fuente.py` es la fuente del cuaderno del ACA 2 |
| `src/certificados` | Paquete reutilizable: reglas de calidad (`reglas.py`, `pipelines.py`, `calidad.py`), ingeniería de características (`caracteristicas.py`) y modelos (`modelado.py`) |
| `scripts` | `ejecutar_calidad.py` (limpieza de un lote) y `figuras_documento.py` |
| `tests` | Pruebas unitarias de las reglas y de equivalencia con la base del ACA 1 |
| `reports/` | Tablas, figuras y cifras del ACA 1 |
| `results/tables` | Tablas del ACA 2 en CSV, incluidas `metricas_por_particion.csv` (una fila por modelo y corte) y `optuna_trials.csv` |
| `results/figures` | Figuras del ACA 2 |
| `results/experiments.csv` | Una fila por corrida: fecha, modelo, etapa, parámetros, semilla y métricas |
| `results/resultados_aca2.json` | Cifras e interpretaciones calculadas por el cuaderno del ACA 2 |

## Fuentes de datos

- Alcaldía Mayor de Cartagena de Indias (Alcaldía Local 3), [Certificados de Residencia](https://www.datos.gov.co/d/2muq-8ebv).
- Alcaldía de Villavicencio, [Certificados de residencia expedidos en el Municipio de Villavicencio](https://www.datos.gov.co/d/38t2-j7vd).

Ninguna fuente corresponde a Barbosa, Antioquia, cuya página de datos abiertos no publica este trámite.

## Datos personales

La fuente de Villavicencio contiene en texto libre números que parecen identificaciones personales y direcciones. Este repositorio **no guarda esos valores**: se enmascaran antes de escribir cualquier archivo y solo se conservan su conteo y la huella SHA-256 de la descarga. La base limpia tampoco incluye el campo de barrio de Villavicencio. Los resultados se presentan de forma agregada.

## Licencias

- Código: MIT (ver `LICENSE`).
- Datos y bases derivadas: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), con atribución a las alcaldías indicadas. Se enmascararon valores y se normalizaron categorías.
