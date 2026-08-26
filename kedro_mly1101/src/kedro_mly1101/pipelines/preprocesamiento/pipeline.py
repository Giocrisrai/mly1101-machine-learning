"""Pipeline de limpieza.

A diferencia del de calidad, aquí los nodos **sí** dependen unos de otros: cada
uno recibe la salida del anterior. Kedro lo deduce de los nombres de los datos
intermedios y los ejecuta en ese orden, sin que nadie lo escriba.

Los nombres que empiezan por ``_`` no están en ``catalog.yml``: son datasets de
memoria, existen solo durante la ejecución y no tocan el disco. Declarar en el
catálogo únicamente lo que hay que conservar es lo que evita llenar ``data/`` de
archivos intermedios que nadie vuelve a mirar.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from .nodes import (
    descubrir_faltantes,
    marcar_imposibles,
    normalizar_categorias,
    quitar_duplicados_y_constantes,
    resumir_limpieza,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=normalizar_categorias,
                inputs=["detecciones_crudas", "params:mapas_categorias"],
                outputs="_categorias_normalizadas",
                name="normalizar_categorias",
            ),
            node(
                func=descubrir_faltantes,
                inputs=["_categorias_normalizadas", "params:centinelas"],
                outputs="_faltantes_descubiertos",
                name="descubrir_nulos_ocultos",
            ),
            node(
                func=marcar_imposibles,
                inputs=["_faltantes_descubiertos", "params:reglas_dominio"],
                outputs="_imposibles_marcados",
                name="marcar_valores_imposibles",
            ),
            node(
                func=quitar_duplicados_y_constantes,
                inputs=["_imposibles_marcados", "params:columnas_a_descartar"],
                outputs="detecciones_limpias",
                name="quitar_duplicados_y_constantes",
            ),
            node(
                func=resumir_limpieza,
                inputs=["detecciones_crudas", "detecciones_limpias"],
                outputs="informe_limpieza",
                name="resumir_la_limpieza",
            ),
        ]
    )
