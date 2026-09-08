"""Pipeline de ingesta de datos reales de Waymo.

Produce ``detecciones_reales`` (Perception v2) y deja a la vista las otras
fuentes locales: inventario, ``camera_box`` (tabla 2D) y metadatos E2E.
El resto del grafo se remapea sobre ``detecciones_reales``; no se mezclan
productos (ver ``pipeline_registry.py``).
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from .nodes import (
    comparar_con_sintetico,
    ensamblar_cajas_camara,
    inventariar_fuentes,
    leer_metadatos_e2e,
    traducir_waymo,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=inventariar_fuentes,
                inputs="params:fuentes_waymo.raiz",
                outputs="inventario_fuentes_waymo",
                name="inventariar_fuentes_waymo",
            ),
            node(
                func=traducir_waymo,
                inputs="waymo_muestra",
                outputs="detecciones_reales",
                name="traducir_esquema_de_waymo",
            ),
            node(
                func=ensamblar_cajas_camara,
                inputs="waymo_muestra",
                outputs="cajas_camara_2d",
                name="ensamblar_cajas_camara",
            ),
            node(
                func=leer_metadatos_e2e,
                inputs="params:fuentes_waymo.raiz",
                outputs="metadatos_e2e",
                name="leer_metadatos_e2e",
            ),
            node(
                func=comparar_con_sintetico,
                inputs=["detecciones_reales", "detecciones_crudas"],
                outputs="comparacion_real_vs_sintetico",
                name="comparar_real_contra_sintetico",
            ),
        ]
    )
