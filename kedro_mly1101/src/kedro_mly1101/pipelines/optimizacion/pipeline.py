"""Pipeline de optimización y ensamble (RA3).

Consume ``particion_por_segmento``, la salida del pipeline supervisado. Toda la
búsqueda y la comparación ocurren **dentro del conjunto de entrenamiento**: el de
prueba se reserva y solo se toca en el nodo que mide, precisamente, el costo de
haberlo mirado antes de tiempo.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from .nodes import (
    analizar_robustez,
    buscar_hiperparametros,
    comparar_ajuste_contra_defecto,
    comparar_ensambles,
    medir_fuga_por_ajustar_en_prueba,
    tabla_de_seleccion,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=buscar_hiperparametros,
                inputs=["particion_por_segmento", "params:modelo", "params:ajuste"],
                outputs="busqueda_hiperparametros",
                name="buscar_hiperparametros",
            ),
            node(
                func=comparar_ajuste_contra_defecto,
                inputs=[
                    "particion_por_segmento",
                    "params:modelo",
                    "params:ajuste",
                    "busqueda_hiperparametros",
                ],
                outputs="ganancia_del_ajuste",
                name="medir_la_ganancia_del_ajuste",
            ),
            node(
                func=medir_fuga_por_ajustar_en_prueba,
                inputs=["particion_por_segmento", "params:modelo", "params:ajuste"],
                outputs="fuga_por_ajuste",
                name="medir_fuga_por_ajustar_en_prueba",
            ),
            node(
                func=comparar_ensambles,
                inputs=["particion_por_segmento", "params:modelo", "params:ajuste"],
                outputs="comparacion_modelos",
                name="comparar_modelos_y_ensambles",
            ),
            node(
                func=analizar_robustez,
                inputs="comparacion_modelos",
                outputs="robustez_modelos",
                name="analizar_robustez",
            ),
            node(
                func=tabla_de_seleccion,
                inputs=["comparacion_modelos", "robustez_modelos", "params:modelo"],
                outputs="seleccion_de_modelo",
                name="sustentar_la_seleccion",
            ),
        ]
    )
