"""Pipeline de ingesta de datos reales de Waymo.

Produce ``detecciones_reales``, con el mismo esquema que el CSV sintético. Esa es
la pieza que permite remapear el resto del grafo sobre datos reales sin duplicar
ni un nodo (ver ``pipeline_registry.py``).
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from .nodes import comparar_con_sintetico, traducir_waymo


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=traducir_waymo,
                inputs="waymo_muestra",
                outputs="detecciones_reales",
                name="traducir_esquema_de_waymo",
            ),
            node(
                func=comparar_con_sintetico,
                inputs=["detecciones_reales", "detecciones_crudas"],
                outputs="comparacion_real_vs_sintetico",
                name="comparar_real_contra_sintetico",
            ),
        ]
    )
