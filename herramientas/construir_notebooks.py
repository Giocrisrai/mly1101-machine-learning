"""Construye los notebooks .ipynb a partir de la fuente única de contenido.

    python herramientas/construir_notebooks.py

Genera, para la Semana 1 completa:

| Actividad   | Notebooks                                  | Fuente                     |
|-------------|--------------------------------------------|----------------------------|
| 1.1 · IL1.1 | ``02_alumno_fuentes`` / ``02_docente_*``   | ``contenido_actividad11``  |
| 1.2 · IL1.2 | ``03_alumno_estructuras`` / ``03_docente_*``| ``contenido_actividad12``  |
| 1.3 · IL1.3 | ``01_alumno_exploracion`` / ``01_docente_*``| ``contenido_semana01``     |
| 1.4 · IL1.4 | ``07_alumno_etica`` / ``07_docente_etica``  | ``contenido_actividad14``  |
| 2.2 · IL2.2 | ``05_alumno_supervisado`` / ``05_docente_*``| ``contenido_actividad22``  |
| 2.3 · IL2.3 | ``06_alumno_no_supervisado`` / ``06_doc_*`` | ``contenido_actividad23``  |
| transversal | ``10_proyecto_equipo_plantilla``           | ``contenido_proyecto``     |
| opcional    | ``04_opcional_kedro_databricks``           | ``contenido_kedro``        |
| opcional    | ``00_opcional_waymo_real``                 | ``contenido_waymo``        |

El número del archivo no coincide con el de la actividad por una razón histórica: el
notebook de EDA se publicó primero como ``01`` y sus enlaces de Colab ya circulan. El
número de actividad está declarado en la primera celda de cada notebook.

Editar los .ipynb a mano es una mala idea: el siguiente build los sobrescribe.
Edita el módulo de contenido correspondiente y vuelve a ejecutar este script.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from contenido_actividad11 import CELDAS_ACT11  # noqa: E402
from contenido_actividad12 import CELDAS_ACT12  # noqa: E402
from contenido_actividad14 import CELDAS_ACT14  # noqa: E402
from contenido_actividad22 import CELDAS_ACT22  # noqa: E402
from contenido_actividad23 import CELDAS_ACT23  # noqa: E402
from contenido_kedro import CELDAS_KEDRO  # noqa: E402
from contenido_proyecto import CELDAS_PROYECTO  # noqa: E402
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


def _dividir_en_todos(texto: str) -> list[str]:
    """Separa un bloque Markdown en trozos, uno por cada encabezado de TODO.

    Motivo: el índice de Google Colab muestra **solo el primer encabezado de
    cada celda**. Si un TODO comparte celda con el texto que lo introduce, no
    aparece en el índice y el alumno no puede saltar a él cuando el docente
    dice "vayan al TODO 5". Separarlos no cambia nada visualmente: Colab apila
    las celdas Markdown una tras otra.

    Solo se divide en los encabezados de TODO, no en todos: la plantilla del
    mini-informe y la introducción deben seguir siendo una sola celda editable.
    """
    lineas = texto.split("\n")
    trozos: list[list[str]] = [[]]
    for numero, linea in enumerate(lineas):
        if numero > 0 and re.match(r"^#{1,3} .*TODO", linea):
            trozos.append([])
        trozos[-1].append(linea)
    return ["\n".join(trozo).strip("\n") for trozo in trozos if "\n".join(trozo).strip()]


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
        if celda["tipo"] == "md":
            # Cada TODO en su propia celda, para que aparezca en el índice de Colab.
            salida.extend(_celda_ipynb("md", trozo) for trozo in _dividir_en_todos(texto))
        else:
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
        # Actividad 1.1 — Fuentes de datos y trabajo colaborativo
        construir(CELDAS_ACT11, "02_alumno_fuentes.ipynb", para_docente=False),
        construir(CELDAS_ACT11, "02_docente_fuentes.ipynb", para_docente=True),
        # Actividad 1.2 — Estructuras de datos y almacenamiento
        construir(CELDAS_ACT12, "03_alumno_estructuras.ipynb", para_docente=False),
        construir(CELDAS_ACT12, "03_docente_estructuras.ipynb", para_docente=True),
        # Actividad 1.4 — Impacto ético, sesgos y privacidad
        construir(CELDAS_ACT14, "07_alumno_etica.ipynb", para_docente=False),
        construir(CELDAS_ACT14, "07_docente_etica.ipynb", para_docente=True),
        # Actividad 1.3 — Análisis exploratorio de datos
        construir(CELDAS, "01_alumno_exploracion.ipynb", para_docente=False),
        construir(CELDAS, "01_docente_solucionario.ipynb", para_docente=True),
        # Actividad 2.2 — Modelamiento supervisado
        construir(CELDAS_ACT22, "05_alumno_supervisado.ipynb", para_docente=False),
        construir(CELDAS_ACT22, "05_docente_supervisado.ipynb", para_docente=True),
        # Actividad 2.3 — Modelamiento no supervisado
        construir(CELDAS_ACT23, "06_alumno_no_supervisado.ipynb", para_docente=False),
        construir(CELDAS_ACT23, "06_docente_no_supervisado.ipynb", para_docente=True),
        # Plantilla del proyecto de equipo (una sola versión, sin solucionario)
        construir(CELDAS_PROYECTO, "10_proyecto_equipo_plantilla.ipynb", para_docente=False),
        # Opcional — Kedro ejecutable y Databricks conceptual
        construir(CELDAS_KEDRO, "04_opcional_kedro_databricks.ipynb", para_docente=False),
        # Opcional — datos reales de Waymo
        construir(CELDAS_WAYMO, "00_opcional_waymo_real.ipynb", para_docente=True),
    ]
    for ruta in construidos:
        celdas = json.loads(ruta.read_text(encoding="utf-8"))["cells"]
        codigo = sum(1 for c in celdas if c["cell_type"] == "code")
        print(f"{ruta.relative_to(RAIZ)}: {len(celdas)} celdas ({codigo} de código)")


if __name__ == "__main__":
    main()
