"""Figuras de método del documento (v0.2): ruta CRISP-DM y arquitectura de la solución.

Uso: python scripts/figuras_documento.py   (escribe en results/figures)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BASE = Path(__file__).resolve().parents[1]
FIG = BASE / "results/figures"
FIG.mkdir(parents=True, exist_ok=True)
AZUL, AZUL_CLARO, GRIS_F, GRIS_B, TINTA, TINTA2 = "#2a78d6", "#dcebfa", "#f2f1ee", "#c3c2b7", "#0b0b0b", "#52514e"
OSCURO = "#1c4f8f"
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white"})


def ruta_crispdm():
    fases = [("1. Comprensión\ndel problema", "Calidad de datos\ndel certificado\nde residencia", "ACA 1"),
             ("2. Comprensión\nde los datos", "Ficha, diccionario,\nEDA multivariado", "ACA 1"),
             ("3. Preparación\nde los datos", "ETL y reglas en\nscripts (objetivo 2)", "ACA 1 y 2"),
             ("4. Modelado", "Ingeniería de\ncaracterísticas y\n12 modelos", "ACA 2"),
             ("5. Evaluación", "Validación temporal,\nprueba y ablación\n(estadística: ACA 3)", "ACA 2 y 3"),
             ("6. Despliegue", "Tablero y ruta\npara Barbosa", "ACA 3")]
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.set_xlim(0, 6); ax.set_ylim(0, 1); ax.axis("off")
    for i, (t, d, aca) in enumerate(fases):
        hecho = "2" in aca or aca == "ACA 1"
        ax.add_patch(FancyBboxPatch((i + .05, .08), .85, .84, boxstyle="round,pad=0,rounding_size=.04",
                                    fc=AZUL_CLARO if hecho else GRIS_F, ec=AZUL if hecho else GRIS_B, lw=1.2))
        ax.text(i + .475, .72, t, ha="center", va="center", fontsize=9.5, fontweight="bold")
        ax.text(i + .475, .38, d, ha="center", va="center", fontsize=8, color=TINTA2)
        ax.text(i + .475, .14, aca, ha="center", va="center", fontsize=7.5, color=AZUL if hecho else TINTA2, fontweight="bold")
        if i < 5:
            ax.annotate("", xy=(i + 1.05, .5), xytext=(i + .9, .5), arrowprops=dict(arrowstyle="->", color=TINTA2))
    fig.savefig(FIG / "fig00_ruta_crispdm_aca2.png")
    plt.close(fig)


def arquitectura():
    fig, ax = plt.subplots(figsize=(10, 6.6))
    ax.set_xlim(1.8, 11); ax.set_ylim(0, 6.75); ax.axis("off")

    def caja(x, y, w, h, texto, estado="hecho", fs=8.3):
        fc, ec, color, ls = {"hecho": (OSCURO, OSCURO, "white", "-"), "nuevo": (AZUL_CLARO, AZUL, TINTA, "-"),
                             "plan": (GRIS_F, GRIS_B, TINTA2, "-"), "externo": ("white", TINTA2, TINTA2, "--")}[estado]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=.08", fc=fc, ec=ec, lw=1.3, ls=ls))
        ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=fs, color=color)
        return (x, y, w, h)

    def rombo(cx, cy, texto, w=1.5, h=.9):
        ax.add_patch(plt.Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)], fc="white", ec=AZUL, lw=1.3))
        ax.text(cx, cy, texto, ha="center", va="center", fontsize=7.6)

    def flecha(a, b, texto=None, desv=(0, 0)):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, color=TINTA2, lw=1.1))
        if texto:
            ax.text((a[0] + b[0]) / 2 + desv[0], (a[1] + b[1]) / 2 + desv[1], texto, fontsize=7.5, color=TINTA2, ha="center")

    for y, etiqueta in [(6.42, "1. Adquisición"), (4.92, "2. Calidad (objetivos 1 y 2)"), (3.12, "3. Modelado (objetivo 3)"), (1.42, "4. Despliegue (objetivo 4)")]:
        ax.text(1.9, y, etiqueta, fontsize=9.5, fontweight="bold", color=TINTA, va="center")

    caja(1.9, 5.55, 1.7, .7, "API de\ndatos.gov.co")
    rombo(4.6, 5.9, "¿Datos personales\nen texto libre?", w=1.75)
    caja(6.0, 5.55, 1.6, .7, "Enmascarar\nen memoria")
    caja(8.3, 5.55, 2.4, .7, "Copia fechada\n+ manifiesto (SHA-256)")
    flecha((3.6, 5.9), (3.72, 5.9)); flecha((5.47, 5.9), (6.0, 5.9), "sí", (0, .08)); flecha((7.6, 5.9), (8.3, 5.9))
    ax.add_patch(FancyArrowPatch((4.6, 6.35), (9.5, 6.25), connectionstyle="arc3,rad=-.12", arrowstyle="-|>", mutation_scale=11, color=TINTA2, lw=1.1))
    ax.text(7.0, 6.5, "no", fontsize=7.5, color=TINTA2, ha="center")

    caja(1.9, 3.95, 2.6, .8, "Pipelines de reglas\n(transformadores scikit-learn)\nsrc/certificados", "nuevo")
    caja(5.0, 3.95, 2.3, .8, "Pruebas: equivalencia\ncon el ETL e inyección\nde fallas", "nuevo")
    caja(7.8, 3.95, 2.9, .8, "Base limpia + reporte de\ncalidad ISO/IEC 25012\n(diagnóstico y EDA)")
    flecha((9.5, 5.55), (4.25, 4.75)); flecha((4.5, 4.35), (5.0, 4.35)); flecha((7.3, 4.35), (7.8, 4.35))

    caja(1.9, 2.15, 2.2, .8, "Rezagos, ventanas,\ncalendario y dominio\n(solo días anteriores)", "nuevo")
    caja(4.5, 2.15, 2.4, .8, "Partición temporal 80/20\nPipeline + TimeSeriesSplit\npor días + Optuna", "nuevo")
    caja(7.3, 2.15, 1.75, .8, "12 modelos,\nablación y\nrobustez", "nuevo")
    rombo(10.0, 2.55, "¿Supera a la\nlínea base\npor 0,05?", w=1.6, h=1.0)
    flecha((9.2, 3.95), (3.85, 2.95)); flecha((4.1, 2.55), (4.5, 2.55)); flecha((6.9, 2.55), (7.3, 2.55)); flecha((9.05, 2.55), (9.2, 2.55))

    caja(1.9, .45, 2.9, .8, "Tablero Power BI:\nindicadores de calidad", "plan")
    caja(5.3, .45, 2.4, .8, "Predicciones de rechazo\nen el tablero", "plan")
    caja(8.3, .45, 2.4, .8, "Adaptación a Barbosa\n(requiere autorización)", "externo")
    flecha((10.0, 2.05), (6.5, 1.25), "sí", (.3, .05)); flecha((9.6, 2.25), (4.5, 1.25), "no", (-.4, .05))
    flecha((7.7, .85), (8.3, .85))

    ley = [(OSCURO, OSCURO, "Implementado en v0.1"), (AZUL_CLARO, AZUL, "Implementado en v0.2 (ACA 2)"), (GRIS_F, GRIS_B, "Planeado (ACA 3)")]
    for k, (fc, ec, t) in enumerate(ley):
        ax.add_patch(FancyBboxPatch((1.9 + k * 2.9, .02), .25, .17, boxstyle="round,pad=0,rounding_size=.03", fc=fc, ec=ec))
        ax.text(2.22 + k * 2.9, .105, t, fontsize=8, va="center", color=TINTA2)
    fig.savefig(FIG / "fig_arquitectura_v02.png")
    plt.close(fig)


if __name__ == "__main__":
    ruta_crispdm()
    arquitectura()
    print("Figuras escritas en", FIG)
