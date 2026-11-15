"""Reglas reutilizables de calidad de datos y de ingeniería de características
para los registros del certificado de residencia (versión v0.2).

Todas las reglas son transformadores compatibles con scikit-learn: heredan de
BaseEstimator y TransformerMixin, así que se combinan en un Pipeline y separan
lo que se aprende de los datos (fit) de lo que se aplica (transform).
"""
__version__ = "0.2.0"
