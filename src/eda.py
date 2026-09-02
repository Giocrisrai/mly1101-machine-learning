"""Utilidades de diagnóstico de calidad de datos para la EA1 (MLY1101).

Funciones puras y sin efectos secundarios: reciben un DataFrame o una Serie y
devuelven un objeto de pandas. No imprimen ni grafican, para que puedan usarse
tanto en los notebooks como en los tests.

Las usan los tres notebooks del repositorio, incluido el de datos reales de
Waymo: el esquema es el mismo, así que el diagnóstico también.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Valores que suelen representar un dato faltante sin ser NaN.
CENTINELAS_HABITUALES = [-1, -999, 0, "", " ", "N/D", "NA", "n/a", "?", "sin dato"]


def resumen_calidad(
    df: pd.DataFrame, centinelas: list | None = None, max_ejemplos: int = 3
) -> pd.DataFrame:
    """Radiografía por columna: tipo, cardinalidad, nulos y nulos ocultos.

    Args:
        df: DataFrame a diagnosticar.
        centinelas: valores a contar como faltantes además de NaN. Si es None,
            se usa ``CENTINELAS_HABITUALES``.
        max_ejemplos: cuántos valores de muestra incluir por columna.

    Returns:
        DataFrame con una fila por columna de ``df`` y las columnas:
        ``dtype``, ``n_unicos``, ``pct_unicos``, ``n_nulos``, ``pct_nulos``,
        ``n_centinelas``, ``pct_faltante_total``, ``ejemplos``.

        ``pct_faltante_total`` = 100 * (n_nulos + n_centinelas) / len(df)
    """
    if centinelas is None:
        centinelas = CENTINELAS_HABITUALES
    n = len(df)
    filas = []
    for columna in df.columns:
        serie = df[columna]
        n_nulos = int(serie.isna().sum())
        n_centinelas = int(serie.isin(centinelas).sum())
        n_unicos = int(serie.nunique(dropna=True))
        ejemplos = serie.dropna().unique()[:max_ejemplos].tolist()
        filas.append(
            {
                "dtype": str(serie.dtype),
                "n_unicos": n_unicos,
                "pct_unicos": round(100 * n_unicos / n, 2) if n else 0.0,
                "n_nulos": n_nulos,
                "pct_nulos": round(100 * n_nulos / n, 2) if n else 0.0,
                "n_centinelas": n_centinelas,
                "pct_faltante_total": (
                    round(100 * (n_nulos + n_centinelas) / n, 2) if n else 0.0
                ),
                "ejemplos": ejemplos,
            }
        )
    return pd.DataFrame(filas, index=df.columns)


def detectar_outliers_iqr(serie: pd.Series, k: float = 1.5) -> pd.Series:
    """Marca outliers por el criterio del rango intercuartil.

    Es outlier todo x tal que:

        x < Q1 - k*IQR   ó   x > Q3 + k*IQR,   con IQR = Q3 - Q1

    Args:
        serie: valores numéricos (los NaN se marcan como False).
        k: factor del criterio. 1.5 = outlier moderado, 3.0 = extremo.

    Returns:
        Serie booleana con el mismo índice que ``serie``.
    """
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    return (serie < q1 - k * iqr) | (serie > q3 + k * iqr)


def limites_iqr(serie: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Devuelve la tupla (límite inferior, límite superior) del criterio IQR."""
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def detectar_outliers_zscore(serie: pd.Series, umbral: float = 3.0) -> pd.Series:
    """Marca outliers por puntaje z: |z| > umbral, con z = (x - media) / sigma.

    Advertencia pedagógica: la media y sigma se contaminan con los propios
    outliers, así que este criterio es menos robusto que el IQR.
    """
    sigma = serie.std()
    if sigma == 0 or pd.isna(sigma):
        return pd.Series(False, index=serie.index)
    return ((serie - serie.mean()) / sigma).abs() > umbral


def normalizar_categoria(serie: pd.Series, mapa: dict | None = None) -> pd.Series:
    """Unifica variantes de una categoría de texto.

    Aplica, en orden: quitar espacios sobrantes, pasar a minúsculas y, si se
    entrega ``mapa``, traducir cada valor a su forma canónica.

    Args:
        serie: columna de texto.
        mapa: diccionario ``{valor_normalizado: valor_canonico}``. Los valores
            que no estén en el mapa se dejan tal como quedaron tras normalizar.
    """
    limpia = serie.astype("string").str.strip().str.lower()
    if mapa:
        limpia = limpia.replace(mapa)
    return limpia


def reporte_duplicados(df: pd.DataFrame, llave: list[str]) -> pd.DataFrame:
    """Compara duplicados exactos contra duplicados lógicos según una llave.

    Un duplicado lógico es una fila que repite la ``llave`` de otra pero difiere
    en alguna otra columna: ``drop_duplicates()`` no lo elimina, y por eso hay
    que buscarlo aparte.

    Returns:
        DataFrame de una fila con las columnas ``filas``, ``dup_exactos``,
        ``dup_por_llave`` y ``dup_logicos``, donde
        ``dup_logicos = dup_por_llave - dup_exactos``.
    """
    dup_exactos = int(df.duplicated().sum())
    dup_llave = int(df.duplicated(subset=llave).sum())
    return pd.DataFrame(
        [
            {
                "filas": len(df),
                "dup_exactos": dup_exactos,
                "dup_por_llave": dup_llave,
                "dup_logicos": dup_llave - dup_exactos,
            }
        ]
    )


def matriz_nulos_por_grupo(
    df: pd.DataFrame, columna: str, grupos: list[str]
) -> pd.DataFrame:
    """Porcentaje de nulos de ``columna`` cruzado por una o dos variables.

    Sirve para distinguir nulos aleatorios (MCAR) de nulos con patrón (MAR /
    MNAR): si el porcentaje cambia mucho entre grupos, el nulo no es aleatorio
    y eliminar esas filas introduce sesgo.
    """
    aux = df.assign(_nulo=df[columna].isna())
    if len(grupos) == 1:
        return (aux.groupby(grupos[0], dropna=False)["_nulo"].mean() * 100).round(2).to_frame("pct_nulos")
    return (
        aux.pivot_table(index=grupos[0], columns=grupos[1], values="_nulo", aggfunc="mean")
        .mul(100)
        .round(2)
    )


def valores_imposibles(df: pd.DataFrame, reglas: dict[str, str]) -> pd.DataFrame:
    """Cuenta las filas que violan reglas de dominio expresadas como consultas.

    Args:
        df: datos a revisar.
        reglas: ``{nombre_de_la_regla: expresion_para_df.query}``. La expresión
            describe las filas **inválidas**. Ejemplo:
            ``{"altura nula": "box_height <= 0"}``

    Returns:
        DataFrame con columnas ``regla``, ``n_filas`` y ``pct``.
    """
    filas = []
    for nombre, expresion in reglas.items():
        n_malas = len(df.query(expresion))
        filas.append(
            {
                "regla": nombre,
                "n_filas": n_malas,
                "pct": round(100 * n_malas / len(df), 3) if len(df) else 0.0,
            }
        )
    return pd.DataFrame(filas).sort_values("n_filas", ascending=False).reset_index(drop=True)


def a_numerico(serie: pd.Series) -> pd.Series:
    """Convierte a numérico dejando NaN donde la conversión falla.

    Equivale a ``pd.to_numeric(serie, errors="coerce")``; existe como función
    con nombre para que en el notebook quede explícito qué se está haciendo con
    los valores corruptos: no se pierden filas, se marcan como faltantes.
    """
    return pd.to_numeric(serie, errors="coerce")


def resumen_desbalance(serie: pd.Series) -> pd.DataFrame:
    """Frecuencia absoluta y relativa de una variable categórica.

    Incluye la razón de desbalance respecto de la clase mayoritaria
    (``ratio_vs_mayoritaria``), útil para anticipar problemas en la Actividad 2.2.
    """
    conteo = serie.value_counts(dropna=False)
    return pd.DataFrame(
        {
            "n": conteo,
            "pct": (100 * conteo / len(serie)).round(2),
            "ratio_vs_mayoritaria": (conteo / conteo.max()).round(4),
        }
    )


def perfil_numerico(df: pd.DataFrame, columnas: list[str] | None = None) -> pd.DataFrame:
    """describe() ampliado con asimetría, curtosis y conteo de outliers IQR."""
    if columnas is None:
        columnas = df.select_dtypes(include=np.number).columns.tolist()
    base = df[columnas].describe().T
    base["asimetria"] = df[columnas].skew()
    base["curtosis"] = df[columnas].kurtosis()
    base["n_outliers_iqr"] = [int(detectar_outliers_iqr(df[c]).sum()) for c in columnas]
    return base.round(3)
