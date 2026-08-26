"""Pipeline de aprendizaje no supervisado (EA3).

Consume ``detecciones_limpias``, igual que el supervisado. Los dos parten del
mismo Parquet: es la misma materia prima con dos preguntas distintas.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from .nodes import (
    agrupar,
    contrastar_con_etiqueta,
    elegir_k,
    perfilar_grupos,
    preparar_matriz,
    proyectar_2d,
    resumir_pca,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=preparar_matriz,
                inputs=["detecciones_limpias", "params:agrupamiento"],
                outputs="matriz_escalada",
                name="preparar_y_escalar",
            ),
            node(
                func=elegir_k,
                inputs=["matriz_escalada", "params:agrupamiento"],
                outputs="busqueda_de_k",
                name="buscar_numero_de_grupos",
            ),
            node(
                func=agrupar,
                inputs=["matriz_escalada", "params:agrupamiento"],
                outputs="detecciones_agrupadas",
                name="agrupar_con_kmedias",
            ),
            node(
                func=perfilar_grupos,
                inputs=["detecciones_agrupadas", "params:agrupamiento"],
                outputs="perfil_de_grupos",
                name="perfilar_los_grupos",
            ),
            node(
                func=contrastar_con_etiqueta,
                inputs=["detecciones_agrupadas", "params:agrupamiento"],
                outputs="grupos_vs_etiqueta",
                name="contrastar_grupos_con_la_etiqueta",
            ),
            node(
                func=resumir_pca,
                inputs=["matriz_escalada", "params:agrupamiento"],
                outputs="varianza_pca",
                name="resumir_componentes_principales",
            ),
            node(
                func=proyectar_2d,
                inputs=["matriz_escalada", "params:agrupamiento"],
                outputs="proyeccion_2d",
                name="proyectar_a_dos_componentes",
            ),
        ]
    )
