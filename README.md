# Certificados de residencia: calidad de datos y rechazo de solicitudes

Trabajo de grado de la Especialización en Analítica de Datos, CUN. Integrantes: Mónica Acevedo Sánchez e Ingrid Vanessa López Tautiva.

Versión **v0.1**: adquisición de datos, ETL, EDA multivariado y protocolo de evaluación (ACA 1).

## Cómo ejecutarlo

En Google Colab, abre `notebooks/ACA1_ETL_EDA.ipynb` y usa *Entorno de ejecución > Ejecutar todo*.

En un equipo local:

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/ACA1_ETL_EDA.ipynb
```

El cuaderno usa semilla fija (42) y trabaja con las copias fechadas de `data/raw`. Si no las encuentra, descarga los datos de la API de datos.gov.co; una descarga posterior puede dar otras cifras si la fuente cambió.

## Estructura

| Carpeta | Contenido |
|---|---|
| `data/raw` | Copias fechadas y enmascaradas, y un manifiesto de cada fuente |
| `data/processed` | Bases limpias |
| `notebooks` | Cuaderno de ETL y EDA |
| `reports/figures` | Figuras |
| `reports/tables` | Tablas |
| `reports/resultados.json` | Cifras e interpretaciones calculadas por el cuaderno |

## Fuentes de datos

- Alcaldía Mayor de Cartagena de Indias (Alcaldía Local 3), [Certificados de Residencia](https://www.datos.gov.co/d/2muq-8ebv).
- Alcaldía de Villavicencio, [Certificados de residencia expedidos en el Municipio de Villavicencio](https://www.datos.gov.co/d/38t2-j7vd).

Ninguna fuente corresponde a Barbosa, Antioquia, cuya página de datos abiertos no publica este trámite.

## Datos personales

La fuente de Villavicencio contiene en texto libre números que parecen identificaciones personales y direcciones. Este repositorio **no guarda esos valores**: se enmascaran antes de escribir cualquier archivo y solo se conservan su conteo y la huella SHA-256 de la descarga. La base limpia tampoco incluye el campo de barrio de Villavicencio. Los resultados se presentan de forma agregada.

## Licencias

- Código: MIT (ver `LICENSE`).
- Datos y bases derivadas: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), con atribución a las alcaldías indicadas. Se enmascararon valores y se normalizaron categorías.
