"""Registro de pipelines del proyecto.

Cada experiencia de la asignatura suma la suya. El orden entre ellas no lo decide
este archivo: lo decide el grafo. Que ``supervisado`` corra después de
``preprocesamiento`` no está escrito en ninguna parte — se deduce de que consume
``detecciones_limpias``, que el otro produce.

**El pipeline ``waymo_real`` no duplica ni un nodo.** Reutiliza los mismos de
siempre remapeando su entrada: donde leían el CSV sintético, ahora leen la salida
de la ingesta de Waymo. Esa es, en una línea, la razón de haber separado el
catálogo del análisis.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, pipeline

from kedro_mly1101.pipelines.calidad.pipeline import create_pipeline as calidad
from kedro_mly1101.pipelines.ingesta.pipeline import create_pipeline as ingesta
from kedro_mly1101.pipelines.no_supervisado.pipeline import (
    create_pipeline as no_supervisado,
)
from kedro_mly1101.pipelines.preprocesamiento.pipeline import (
    create_pipeline as preprocesamiento,
)
from kedro_mly1101.pipelines.supervisado.pipeline import create_pipeline as supervisado

# Qué salidas del recorrido sobre datos reales se guardan en disco. Las que no
# están aquí quedan en memoria: existen durante la ejecución y no se persisten.
SALIDAS_REALES = {
    "resumen_calidad": "resumen_calidad_real",
    "valores_imposibles": "valores_imposibles_real",
    "nulos_por_grupo": "nulos_por_grupo_real",
    "detecciones_limpias": "detecciones_limpias_reales",
    "informe_limpieza": "informe_limpieza_real",
    "metricas_por_clase": "metricas_por_clase_real",
    "matriz_confusion": "matriz_confusion_real",
    "fuga_de_variable": "fuga_de_variable_real",
    "perfil_de_grupos": "perfil_de_grupos_real",
    "grupos_vs_etiqueta": "grupos_vs_etiqueta_real",
    "busqueda_de_k": "busqueda_de_k_real",
    "varianza_pca": "varianza_pca_real",
}

# Los parámetros se comparten entre el recorrido sintético y el real: son las
# mismas decisiones. Si hubiera que ajustarlas para datos reales, este es el
# único sitio que habría que tocar.
PARAMETROS_COMPARTIDOS = {
    f"params:{clave}": f"params:{clave}"
    for clave in [
        "mapas_categorias",
        "centinelas",
        "reglas_dominio",
        "columnas_a_descartar",
        "sesgo",
        "modelo",
        "fuga",
        "agrupamiento",
    ]
}


def register_pipelines() -> dict[str, Pipeline]:
    p_calidad = calidad()
    p_preprocesamiento = preprocesamiento()
    p_supervisado = supervisado()
    p_no_supervisado = no_supervisado()
    p_ingesta = ingesta()

    # El recorrido completo sobre el dataset sintético de la asignatura.
    analisis = p_calidad + p_preprocesamiento + p_supervisado + p_no_supervisado

    # El mismo recorrido, sobre datos reales. Cambia la entrada, no los nodos.
    analisis_real = pipeline(
        analisis,
        inputs={"detecciones_crudas": "detecciones_reales"},
        outputs=SALIDAS_REALES,
        parameters=PARAMETROS_COMPARTIDOS,
        namespace="real",
    )

    return {
        "calidad": p_calidad,
        "preprocesamiento": p_preprocesamiento,
        "supervisado": p_supervisado,
        "no_supervisado": p_no_supervisado,
        "ingesta": p_ingesta,
        "waymo_real": p_ingesta + analisis_real,
        "__default__": analisis,
    }
