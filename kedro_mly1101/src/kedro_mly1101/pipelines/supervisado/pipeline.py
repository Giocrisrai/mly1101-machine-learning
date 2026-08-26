"""Pipeline de aprendizaje supervisado (EA2).

Consume ``detecciones_limpias``, la salida del pipeline de preprocesamiento. Esa
dependencia es lo que hace que Kedro ejecute primero la limpieza y después el
modelamiento sin que nadie escriba el orden.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from .nodes import (
    comparar_fuga_de_variable,
    comparar_particiones,
    entrenar,
    evaluar_por_clase,
    matriz_confusion,
    particionar,
    particionar_al_azar,
    preparar_variables,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=preparar_variables,
                inputs=["detecciones_limpias", "params:modelo", "params:fuga"],
                outputs="tabla_modelamiento",
                name="preparar_variables",
            ),
            node(
                func=particionar,
                inputs=["tabla_modelamiento", "params:modelo"],
                outputs="particion_por_segmento",
                name="particionar_por_segmento",
            ),
            node(
                func=entrenar,
                inputs=["particion_por_segmento", "params:modelo"],
                outputs="clasificador",
                name="entrenar_clasificador",
            ),
            node(
                func=evaluar_por_clase,
                inputs=["clasificador", "particion_por_segmento", "params:modelo"],
                outputs="metricas_por_clase",
                name="evaluar_por_clase",
            ),
            node(
                func=matriz_confusion,
                inputs=["clasificador", "particion_por_segmento", "params:modelo"],
                outputs="matriz_confusion",
                name="calcular_matriz_de_confusion",
            ),
            # Los dos nodos siguientes existen para MEDIR el efecto de la fuga,
            # no para producir el modelo final.
            node(
                func=particionar_al_azar,
                inputs=["tabla_modelamiento", "params:modelo"],
                outputs="_particion_al_azar",
                name="particionar_al_azar_para_comparar",
            ),
            node(
                func=comparar_fuga_de_variable,
                inputs=["particion_por_segmento", "params:modelo", "params:fuga"],
                outputs="fuga_de_variable",
                name="medir_fuga_de_variable_derivada",
            ),
            node(
                func=comparar_particiones,
                inputs=[
                    "particion_por_segmento",
                    "_particion_al_azar",
                    "params:modelo",
                ],
                outputs="comparacion_particiones",
                name="comparar_el_efecto_de_la_fuga",
            ),
        ]
    )
