"""Nodos del diagnóstico de calidad de datos (RA1 · Actividad 1.3).

Cada función es pura: recibe DataFrames y parámetros, devuelve DataFrames. No
imprime, no grafica y no lee ni escribe archivos — de eso se encarga el catálogo.

Todas reutilizan ``src/eda.py``, las mismas funciones que los alumnos usan en el
notebook. Si el diagnóstico del notebook y el del pipeline se separaran, tendríamos
dos verdades sobre los mismos datos.
"""

from __future__ import annotations

import pandas as pd

import eda


def diagnosticar(detecciones: pd.DataFrame) -> pd.DataFrame:
    """Radiografía por columna: dtype, cardinalidad, nulos declarados y ocultos.

    Los nulos ocultos son la parte que ``df.isna().sum()`` no ve: valores como
    ``-1`` o ``"N/D"`` que significan "falta el dato" sin ser ``NaN``.
    """
    return eda.resumen_calidad(detecciones)


def medir_desbalance(detecciones: pd.DataFrame, mapas: dict) -> pd.DataFrame:
    """Frecuencia de cada tipo de objeto, tras unificar variantes de escritura.

    Sin normalizar antes, el conteo reparte a los peatones entre cuatro grafías
    distintas y ninguna cifra sirve.
    """
    tipos = eda.normalizar_categoria(
        detecciones["object_type"], mapa=mapas.get("object_type")
    )
    return eda.resumen_desbalance(tipos)


def auditar_dominio(detecciones: pd.DataFrame, reglas: dict) -> pd.DataFrame:
    """Cuenta las filas que violan cada regla de dominio.

    Las reglas llegan desde ``parameters.yml``: son decisiones del negocio, no del
    código, y por eso viven en configuración. Cada una trae ``condicion`` (qué
    filas son inválidas) y ``columna`` (qué dato marcar); para auditar basta la
    primera.
    """
    condiciones = {nombre: regla["condicion"] for nombre, regla in reglas.items()}
    return eda.valores_imposibles(detecciones, condiciones)


def medir_sesgo(detecciones: pd.DataFrame, sesgo: dict) -> pd.DataFrame:
    """Porcentaje de faltantes de una columna, cruzado por variables de grupo.

    Es lo que distingue un faltante aleatorio de uno con patrón. Si el porcentaje
    cambia mucho entre grupos, eliminar esas filas no es neutral: sesga el dataset
    contra el grupo peor medido.
    """
    return eda.matriz_nulos_por_grupo(detecciones, sesgo["columna"], sesgo["grupos"])
