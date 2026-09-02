"""Nodos de limpieza (RA1 · tabla de decisiones de la Actividad 1.3).

Cada paso corresponde a una fila de la tabla de decisiones que los alumnos
justifican en su informe. Escrita como código, la decisión se ejecuta igual
siempre; escrita solo en Markdown, se interpreta distinto cada vez que alguien
la lee.

Principio que gobierna todo el módulo: **marcar antes que eliminar**. Un valor
imposible se convierte en faltante, no se borra la fila entera, porque el resto
de esa fila sí era válido. Se eliminan únicamente los duplicados exactos, que por
definición no aportan nada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import eda


def normalizar_categorias(detecciones: pd.DataFrame, mapas: dict) -> pd.DataFrame:
    """Unifica las variantes de escritura de las columnas categóricas."""
    limpio = detecciones.copy()
    for columna, mapa in mapas.items():
        if columna in limpio.columns:
            limpio[columna] = eda.normalizar_categoria(limpio[columna], mapa=mapa)
    return limpio


def descubrir_faltantes(detecciones: pd.DataFrame, centinelas: dict) -> pd.DataFrame:
    """Convierte los nulos ocultos en ``NaN`` de verdad.

    Mientras el faltante siga disfrazado de ``-1``, cualquier promedio lo suma
    como si fuera una medición y ``dropna()`` no lo ve. Convertirlo es lo que
    permite que el resto del pipeline lo trate como lo que es.

    También convierte a numérico la marca de tiempo: basta un ``"N/D"`` entre
    40.000 filas para que toda la columna deje de ser numérica.
    """
    limpio = detecciones.copy()
    for columna, centinela in centinelas.items():
        if columna in limpio.columns:
            # ``np.nan`` y no ``pd.NA``: en una columna numérica, ``pd.NA`` la
            # convierte a ``object``, y scikit-learn no sabe qué hacer con eso.
            # El error aparece mucho más tarde, al entrenar, con un TypeError
            # sobre NAType que no dice nada de dónde vino.
            limpio[columna] = eda.a_numerico(limpio[columna].replace(centinela, np.nan))
    if "timestamp_micros" in limpio.columns:
        limpio["timestamp_micros"] = eda.a_numerico(limpio["timestamp_micros"])
    return limpio


def marcar_imposibles(detecciones: pd.DataFrame, reglas: dict) -> pd.DataFrame:
    """Pone en ``NaN`` los valores que violan una regla de dominio.

    Cada regla trae su ``columna`` declarada en ``parameters.yml``. **No se deduce
    del texto de la condición**: una regla que menciona dos columnas
    (``num_lidar_points < 0 or speed_mps > 100``) dejaría la segunda sin limpiar y
    sin avisar.

    Ojo con la distinción que se evalúa en la rúbrica: se marca lo **imposible**
    (una altura de 0 m), no lo **atípico legítimo** (un bus de 15 m). El criterio
    estadístico propone; el conocimiento del dominio dispone.
    """
    limpio = detecciones.copy()
    for nombre, regla in reglas.items():
        columna = regla["columna"]
        if columna not in limpio.columns:
            continue
        malas = limpio.eval(regla["condicion"])
        # ``np.nan``, por el mismo motivo que en ``descubrir_faltantes``: mantener
        # la columna numérica en vez de degradarla a ``object``.
        limpio.loc[malas.fillna(False), columna] = np.nan
    return limpio


def quitar_duplicados_y_constantes(
    detecciones: pd.DataFrame, columnas_a_descartar: list[str]
) -> pd.DataFrame:
    """Elimina duplicados exactos y las columnas sin poder informativo."""
    limpio = detecciones.drop_duplicates().reset_index(drop=True)
    presentes = [c for c in columnas_a_descartar if c in limpio.columns]
    return limpio.drop(columns=presentes)


def resumir_limpieza(
    crudas: pd.DataFrame, limpias: pd.DataFrame
) -> pd.DataFrame:
    """Qué cambió entre el dataset crudo y el limpio, en cifras.

    Existe para que la limpieza sea auditable: quien reciba el Parquet puede ver
    cuánto se descartó sin volver a ejecutar nada. Ninguna transformación debería
    ser invisible aguas abajo.
    """
    filas = [
        {"metrica": "filas", "crudo": len(crudas), "limpio": len(limpias)},
        {"metrica": "columnas", "crudo": crudas.shape[1], "limpio": limpias.shape[1]},
    ]
    for columna in ["object_type", "weather"]:
        if columna in crudas.columns and columna in limpias.columns:
            filas.append(
                {
                    "metrica": f"categorias de {columna}",
                    "crudo": int(crudas[columna].nunique(dropna=True)),
                    "limpio": int(limpias[columna].nunique(dropna=True)),
                }
            )
    filas.append(
        {
            "metrica": "celdas faltantes",
            "crudo": int(crudas.isna().sum().sum()),
            "limpio": int(limpias.isna().sum().sum()),
        }
    )
    tabla = pd.DataFrame(filas)
    tabla["diferencia"] = tabla["limpio"] - tabla["crudo"]
    return tabla
