"""Aplica las reglas de calidad a un lote de registros y guarda la base limpia y su reporte.

Uso:
    python scripts/ejecutar_calidad.py --ciudad villavicencio --entrada lote.csv --salida results/lote_limpio.csv

Si se indica --referencia, el pipeline se ajusta con esa base (por ejemplo, la copia histórica)
y luego se aplica al lote nuevo; así los límites aprendidos no dependen del lote.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from certificados.calidad import aplicar_y_medir  # noqa: E402
from certificados.pipelines import PIPELINES  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ciudad", choices=sorted(PIPELINES), required=True)
    p.add_argument("--entrada", required=True, help="CSV crudo con los nombres de columna de datos.gov.co")
    p.add_argument("--salida", required=True, help="CSV limpio de salida")
    p.add_argument("--referencia", help="CSV crudo de referencia para ajustar el pipeline")
    a = p.parse_args()

    lote = pd.read_csv(a.entrada, dtype=str, encoding="utf-8")
    ref = pd.read_csv(a.referencia, dtype=str, encoding="utf-8") if a.referencia else lote
    pipe = PIPELINES[a.ciudad]().fit(ref)
    limpio, reporte = aplicar_y_medir(lote, a.ciudad, pipe)
    salida = Path(a.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    limpio.to_csv(salida, index=False, encoding="utf-8")
    salida.with_suffix(".reporte.json").write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reporte, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
