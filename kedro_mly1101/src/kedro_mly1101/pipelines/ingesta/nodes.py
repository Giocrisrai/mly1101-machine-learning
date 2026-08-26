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
            "Descárgalos con: python herramientas/descargar_waymo.py --muestra 40"
        )
    return pd.concat(trozos, ignore_index=True)


def comparar_con_sintetico(
    reales: pd.DataFrame, sinteticas: pd.DataFrame
) -> pd.DataFrame:
    """Contrasta el dataset real con el sintético, columna a columna.

    Es la comprobación que justifica todo el material: si las proporciones y los
    rangos se parecen, lo aprendido sobre el sintético se traslada. Y donde **no**
    se parecen, la diferencia es el aprendizaje.
    """
    filas = []

    def _añadir(metrica: str, real, sintetico) -> None:
        filas.append({"metrica": metrica, "real": real, "sintetico": sintetico})

    _añadir("filas", len(reales), len(sinteticas))
    _añadir("segmentos", reales["segment_id"].nunique(), sinteticas["segment_id"].nunique())

    for tipo in ["vehicle", "pedestrian", "sign", "cyclist"]:
        pct_real = 100 * (reales["object_type"].str.lower() == tipo).mean()
        pct_sint = 100 * (
            sinteticas["object_type"].str.lower().str.startswith(tipo[:3])
        ).mean()
        _añadir(f"% {tipo}", round(pct_real, 2), round(pct_sint, 2))

    for columna in ["box_length", "speed_mps", "num_lidar_points"]:
        if columna in reales.columns and columna in sinteticas.columns:
            _añadir(
                f"mediana {columna}",
                round(float(reales[columna].median()), 2),
                round(float(pd.to_numeric(sinteticas[columna], errors="coerce").median()), 2),
            )

    pct_dificil_real = 100 * (reales["detection_difficulty"] == "LEVEL_2").mean()
    pct_dificil_sint = 100 * (sinteticas["detection_difficulty"] == "LEVEL_2").mean()
    _añadir("% LEVEL_2", round(pct_dificil_real, 2), round(pct_dificil_sint, 2))

    return pd.DataFrame(filas)
