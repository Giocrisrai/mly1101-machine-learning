"""Pipeline de ingesta de datos reales de Waymo.

Produce ``detecciones_reales`` (Perception v2) y deja a la vista las otras
fuentes locales: inventario, ``camera_box`` (tabla 2D traducida a píxeles) y
metadatos E2E (479 clusters, no video). El resto del grafo consume
``detecciones_reales``; no se mezclan productos (ver ``pipeline_registry.py``).
Para *ver* un cuadro: notebook 14 etapa F sobre ``cajas_camara_2d``.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from .nodes import (
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
        ]
    )
