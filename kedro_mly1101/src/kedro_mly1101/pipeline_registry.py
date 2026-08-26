"""Registro de pipelines del proyecto.

Cada experiencia de la asignatura suma su pipeline aquí. La de EA3 (no
supervisado) se enchufará en este mismo diccionario, y el ``__default__`` la
encadenará sola.

El orden del ``__default__`` no lo decide esta lista: lo decide el grafo. Que
``supervisado`` corra después de ``preprocesamiento`` no está escrito en ninguna
parte — se deduce de que consume ``detecciones_limpias``, que el otro produce.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline

from kedro_mly1101.pipelines.calidad.pipeline import create_pipeline as calidad
from kedro_mly1101.pipelines.preprocesamiento.pipeline import (
    create_pipeline as preprocesamiento,
)
from kedro_mly1101.pipelines.supervisado.pipeline import create_pipeline as supervisado


def register_pipelines() -> dict[str, Pipeline]:
    pipeline_calidad = calidad()
    pipeline_preprocesamiento = preprocesamiento()
    pipeline_supervisado = supervisado()
    return {
        "calidad": pipeline_calidad,
        "preprocesamiento": pipeline_preprocesamiento,
        "supervisado": pipeline_supervisado,
        "__default__": pipeline_calidad + pipeline_preprocesamiento + pipeline_supervisado,
    }
