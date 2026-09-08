"""Registro de pipelines del proyecto.

Cada experiencia de la asignatura suma la suya. El orden entre ellas no lo decide
este archivo: lo decide el grafo.

**Un solo hilo de datos: Perception v2 real.** No hay CSV de pauta ni recorrido
paralelo. ``__default__`` y ``waymo_real`` son el mismo grafo: ingesta de la
muestra local más calidad, preprocesamiento, supervisado, no supervisado y
optimización. ``camera_box`` y el JSON E2E salen de ``ingesta`` y **no** entran
al clasificador (lo fija ``tests/test_pipeline_supervisado.py``). El fotograma
sin JPEG se dibuja en el notebook 14, no en este grafo (34 nodos, sin matplotlib).
Motion / v1 / video E2E no tienen dataset en el catálogo.

Conteos que los tests bloquean: ``ingesta`` = 4, ``__default__`` = ``waymo_real``
= 34 (esas 4 + 30 de análisis).
"""

from __future__ import annotations

from kedro.pipeline import Pipeline

from kedro_mly1101.pipelines.calidad.pipeline import create_pipeline as calidad
from kedro_mly1101.pipelines.ingesta.pipeline import create_pipeline as ingesta
from kedro_mly1101.pipelines.no_supervisado.pipeline import (
    create_pipeline as no_supervisado,
)
from kedro_mly1101.pipelines.optimizacion.pipeline import create_pipeline as optimizacion
from kedro_mly1101.pipelines.preprocesamiento.pipeline import (
    create_pipeline as preprocesamiento,
)
from kedro_mly1101.pipelines.supervisado.pipeline import create_pipeline as supervisado


def register_pipelines() -> dict[str, Pipeline]:
    p_calidad = calidad()
    p_preprocesamiento = preprocesamiento()
    p_supervisado = supervisado()
    p_no_supervisado = no_supervisado()
    p_ingesta = ingesta()
    p_optimizacion = optimizacion()

    analisis = (
        p_calidad + p_preprocesamiento + p_supervisado + p_no_supervisado + p_optimizacion
    )
    completo = p_ingesta + analisis

    return {
        "calidad": p_calidad,
        "preprocesamiento": p_preprocesamiento,
        "supervisado": p_supervisado,
        "no_supervisado": p_no_supervisado,
        "optimizacion": p_optimizacion,
        "ingesta": p_ingesta,
        "waymo_real": completo,
        "__default__": completo,
    }
