"""Construye los notebooks .ipynb a partir de la fuente única de contenido.

    python herramientas/construir_notebooks.py

Genera:
- notebooks/01_alumno_exploracion.ipynb    (celdas con TODO, sin las respuestas)
- notebooks/01_docente_solucionario.ipynb  (código resuelto + pauta docente)
- notebooks/00_opcional_waymo_real.ipynb   (datos reales de Waymo, opcional)

Editar los .ipynb a mano es una mala idea: el siguiente build los sobrescribe.
Edita ``herramientas/contenido_semana01.py`` y vuelve a ejecutar este script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from contenido_semana01 import CELDAS, URL_REPO  # noqa: E402
from contenido_waymo import CELDAS_WAYMO  # noqa: E402

DESTINO = RAIZ / "notebooks"

METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "colab": {"provenance": [], "toc_visible": True},
}


def _fuente(texto: str) -> list[str]:
    """Convierte un bloque de texto al formato de nbformat (lista de líneas)."""
    lineas = texto.split("\n")
    return [linea + "\n" for linea in lineas[:-1]] + [lineas[-1]]


def _celda_ipynb(tipo: str, texto: str) -> dict:
    if tipo == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": _fuente(texto)}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _fuente(texto),
    }


def _badge(nombre_archivo: str) -> dict:
    url = f"https://colab.research.google.com/github/{URL_REPO.split('github.com/')[1]}/blob/main/notebooks/{nombre_archivo}"
    return _celda_ipynb(
        "md",
        f"[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})",
    )


def construir(celdas: list[dict], nombre_archivo: str, para_docente: bool) -> Path:
    """Escribe un notebook a partir de la lista de celdas de la fuente única.

    Args:
        celdas: celdas del módulo de contenido.
        nombre_archivo: nombre del .ipynb de salida.
        para_docente: si es False, se omiten las celdas ``solo_docente`` y las
            celdas de código usan su variante con ``TODO``.
    """
    salida = [_badge(nombre_archivo)]
    for celda in celdas:
        if celda.get("solo_docente") and not para_docente:
            continue
        texto = celda["fuente"]
        if celda["tipo"] == "code" and not para_docente and "todo" in celda:
            texto = celda["todo"]
        salida.append(_celda_ipynb(celda["tipo"], texto))

    notebook = {
        "cells": salida,
        "metadata": METADATA,
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    ruta = DESTINO / nombre_archivo
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return ruta


def main() -> None:
    construidos = [
        construir(CELDAS, "01_alumno_exploracion.ipynb", para_docente=False),
        construir(CELDAS, "01_docente_solucionario.ipynb", para_docente=True),
        construir(CELDAS_WAYMO, "00_opcional_waymo_real.ipynb", para_docente=True),
    ]
    for ruta in construidos:
        celdas = json.loads(ruta.read_text(encoding="utf-8"))["cells"]
        codigo = sum(1 for c in celdas if c["cell_type"] == "code")
        print(f"{ruta.relative_to(RAIZ)}: {len(celdas)} celdas ({codigo} de código)")


if __name__ == "__main__":
    main()
