"""Registro de pipelines del proyecto.

Cada experiencia de la asignatura suma su pipeline aquí. Hoy hay dos, ambas de la
EA1; las de EA2 (supervisado) y EA3 (no supervisado) se enchufan en este mismo
diccionario cuando existan, y el ``__default__`` las encadena.

El orden del ``__default__`` no lo decide esta lista: lo decide el grafo. Kedro
ejecuta ``preprocesamiento`` y ``calidad`` según sus dependencias de datos.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline

from kedro_mly1101.pipelines.calidad.pipeline import create_pipeline as calidad
from kedro_mly1101.pipelines.preprocesamiento.pipeline import (
    create_pipeline as preprocesamiento,
)


def register_pipelines() -> dict[str, Pipeline]:
    pipeline_calidad = calidad()
    pipeline_preprocesamiento = preprocesamiento()
    return {
        "calidad": pipeline_calidad,
        "preprocesamiento": pipeline_preprocesamiento,
        "__default__": pipeline_calidad + pipeline_preprocesamiento,
    }
