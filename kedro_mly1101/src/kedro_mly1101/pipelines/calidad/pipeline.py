"""Pipeline de diagnóstico de calidad.

No se declara el ORDEN de ejecución, solo las dependencias de cada nodo: qué
recibe y qué produce. Kedro construye el grafo y deduce el orden. Los cuatro
nodos dependen únicamente de ``detecciones_crudas``, así que son independientes
entre sí y pueden correr en paralelo con ``--runner=ParallelRunner``.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from .nodes import auditar_dominio, diagnosticar, medir_desbalance, medir_sesgo


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=diagnosticar,
                inputs="detecciones_crudas",
                outputs="resumen_calidad",
                name="diagnosticar_calidad",
            ),
            node(
                func=medir_desbalance,
                inputs=["detecciones_crudas", "params:mapas_categorias"],
                outputs="desbalance_clases",
                name="medir_desbalance_de_clases",
            ),
            node(
                func=auditar_dominio,
                inputs=["detecciones_crudas", "params:reglas_dominio"],
                outputs="valores_imposibles",
                name="auditar_reglas_de_dominio",
            ),
            node(
                func=medir_sesgo,
                inputs=["detecciones_crudas", "params:sesgo"],
                outputs="nulos_por_grupo",
                name="medir_sesgo_de_faltantes",
            ),
        ]
    )
