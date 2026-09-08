"""Ingesta de los datos REALES del Waymo Open Dataset.

Traduce los componentes ``lidar_box`` y ``stats`` al esquema de la asignatura,
para que **el mismo pipeline** de calidad, preprocesamiento y modelamiento pueda
correr sobre datos reales sin cambiar un solo nodo.

La traducción vive en ``src/waymo.py``, no aquí: la usan por igual el notebook 00,
este pipeline y los tests. Este módulo solo la enchufa al grafo.

Los datos reales **no están en el repositorio**: la licencia de Waymo es de uso no
comercial y prohíbe redistribuirlos. Hay que aceptarla y descargarlos con
``herramientas/descargar_waymo.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import waymo


def traducir_waymo(particiones: dict) -> pd.DataFrame:
    """Traduce **todos** los segmentos descargados y los concatena en una tabla.

    Recibe el diccionario de un ``PartitionedDataset``: una entrada por archivo,
    con el identificador de la partición y una función que lo carga. Cada segmento
    trae dos archivos, ``lidar_box`` y ``stats``, que hay que emparejar por carpeta.

    **Se cargan varios segmentos a propósito, no uno.** Con un solo segmento no se
    puede partir el dataset en entrenamiento y prueba sin fuga: todas las
    detecciones comparten contexto, así que cualquier corte deja las dos mitades
    contaminadas. Se descargan con::

        python herramientas/descargar_waymo.py --muestra 40

    (``--lote 8`` arma la tabla de clase; ``--muestra 40`` deja los mismos
    componentes livianos en ``muestra/`` para este pipeline.)

    Las tres traducciones que no son un cambio de nombre están explicadas en
    ``waymo.traducir_esquema``: la velocidad es un vector, el tipo de objeto es un
    entero, y el ``NaN`` de la dificultad significa ``LEVEL_1``, no "falta el dato".
    """
    por_segmento: dict[str, dict[str, pd.DataFrame]] = {}
    for identificador, cargar in sorted(particiones.items()):
        carpeta, _, componente = identificador.rpartition("/")
        if componente not in ("lidar_box", "stats"):
            continue
        por_segmento.setdefault(carpeta, {})[componente] = cargar()

    trozos = [
        waymo.traducir_esquema(piezas["lidar_box"], piezas["stats"])
        for piezas in por_segmento.values()
        if "lidar_box" in piezas and "stats" in piezas
    ]

    if not trozos:
        raise ValueError(
            "No se encontró ningún segmento completo en datos/waymo_real/muestra/. "
            "Descárgalos con: python herramientas/descargar_waymo.py --muestra 40 "
            "(o --lote 8 si solo necesitas la tabla de clase)"
        )
    return pd.concat(trozos, ignore_index=True)


def ensamblar_cajas_camara(particiones: dict) -> pd.DataFrame:
    """Junta los ``camera_box`` de la muestra. Vacío si nadie los bajó.

    No entran al clasificador: son otra tabla (cajas 2D). El pipeline las ve
    para que el alumno distinga fuente tabular de JPEG.
    """
    trozos = []
    for identificador, cargar in sorted(particiones.items()):
        _, _, componente = identificador.rpartition("/")
        if componente == "camera_box":
            trozos.append(cargar())
    if not trozos:
        return pd.DataFrame(
            columns=["key.segment_context_name", "key.camera_name"]
        )
    return pd.concat(trozos, ignore_index=True)


def inventariar_fuentes(raiz: str) -> pd.DataFrame:
    """Qué productos de Waymo hay en disco y cuáles alimentan el modelo."""
    return waymo.inventario_fuentes(Path(raiz))


def leer_metadatos_e2e(raiz: str) -> pd.DataFrame:
    """JSON liviano del producto E2E, si está descargado."""
    return waymo.leer_metadatos_e2e(Path(raiz))
