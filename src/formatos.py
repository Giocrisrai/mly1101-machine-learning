"""Comparación medida de formatos de almacenamiento (Act 1.2, IL1.2).

La Actividad 1.2 afirma que Parquet comprime mejor que CSV y que Excel es el
peor candidato para volumen. Este módulo permite **medirlo** en vez de citarlo:
escribe el mismo DataFrame en varios formatos y devuelve una tabla con el peso
en disco, el tiempo de escritura, el tiempo de lectura y si los tipos de dato
sobrevivieron al viaje de ida y vuelta.

Ese último punto es el que más cuesta ver y el que más caro se paga: un CSV no
guarda tipos, solo texto. Una columna que salió como ``int64`` vuelve como
``int64`` solo si tiene suerte; si tenía nulos o un ``"N/D"``, vuelve como
``object`` y el pipeline se rompe aguas abajo.

Funciones puras salvo por la escritura de archivos en la carpeta que se les
indica: no imprimen ni grafican.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

# Orden en que se prueban los formatos. El nombre es la clave que usa el resto
# del módulo y el que aparece en la tabla de resultados.
FORMATOS_POR_DEFECTO = ("csv", "json", "excel", "parquet")

# Límite duro del formato .xlsx, por si alguien intenta exportar el dataset
# completo: 1.048.576 filas y 16.384 columnas.
LIMITE_FILAS_EXCEL = 1_048_576


class FormatoNoDisponible(RuntimeError):
    """El formato necesita una dependencia que no está instalada."""


def _escribir(df: pd.DataFrame, formato: str, ruta: Path) -> None:
    if formato == "csv":
        df.to_csv(ruta, index=False)
    elif formato == "json":
        df.to_json(ruta, orient="records", date_format="iso")
    elif formato == "excel":
        if len(df) > LIMITE_FILAS_EXCEL:
            raise FormatoNoDisponible(
                f"Excel admite hasta {LIMITE_FILAS_EXCEL:,} filas y el DataFrame tiene "
                f"{len(df):,}. Usa una muestra."
            )
        try:
            df.to_excel(ruta, index=False)
        except ImportError as error:  # openpyxl ausente
            raise FormatoNoDisponible(f"Falta el motor de Excel: {error}") from error
    elif formato == "parquet":
        try:
            df.to_parquet(ruta, index=False)
        except ImportError as error:  # pyarrow / fastparquet ausentes
            raise FormatoNoDisponible(f"Falta el motor de Parquet: {error}") from error
    else:
        raise ValueError(f"Formato desconocido: {formato!r}")


def _leer(formato: str, ruta: Path) -> pd.DataFrame:
    if formato == "csv":
        return pd.read_csv(ruta)
    if formato == "json":
        return pd.DataFrame(json.loads(ruta.read_text(encoding="utf-8")))
    if formato == "excel":
        return pd.read_excel(ruta)
    if formato == "parquet":
        return pd.read_parquet(ruta)
    raise ValueError(f"Formato desconocido: {formato!r}")


def dtypes_conservados(original: pd.DataFrame, releido: pd.DataFrame) -> bool:
    """¿El DataFrame releído tiene exactamente los mismos dtypes que el original?

    Compara columna a columna. Si el orden o el conjunto de columnas cambió,
    devuelve False: eso también es una pérdida de fidelidad.
    """
    if list(original.columns) != list(releido.columns):
        return False
    return all(original[c].dtype == releido[c].dtype for c in original.columns)


def columnas_con_dtype_cambiado(
    original: pd.DataFrame, releido: pd.DataFrame
) -> pd.DataFrame:
    """Detalle de qué columnas cambiaron de tipo al releer.

    Returns:
        DataFrame con columnas ``columna``, ``dtype_original`` y
        ``dtype_releido``, solo para las que cambiaron. Vacío si no cambió
        ninguna.
    """
    comunes = [c for c in original.columns if c in releido.columns]
    filas = [
        {
            "columna": c,
            "dtype_original": str(original[c].dtype),
            "dtype_releido": str(releido[c].dtype),
        }
        for c in comunes
        if original[c].dtype != releido[c].dtype
    ]
    return pd.DataFrame(filas, columns=["columna", "dtype_original", "dtype_releido"])


def medir_formatos(
    df: pd.DataFrame,
    carpeta: str | Path,
    formatos: tuple[str, ...] = FORMATOS_POR_DEFECTO,
    omitir_no_disponibles: bool = True,
) -> pd.DataFrame:
    """Escribe ``df`` en cada formato y mide peso, tiempos y fidelidad de tipos.

    Args:
        df: datos a exportar. Para Excel conviene una muestra: escribir decenas
            de miles de filas con ``openpyxl`` puede tardar minutos.
        carpeta: dónde dejar los archivos. Se crea si no existe.
        formatos: cuáles probar, de ``FORMATOS_POR_DEFECTO``.
        omitir_no_disponibles: si True, un formato sin su dependencia instalada
            se salta en silencio; si False, se propaga ``FormatoNoDisponible``.

    Returns:
        DataFrame ordenado de menor a mayor peso, con las columnas:

        ``formato``, ``archivo``, ``bytes``, ``mb``, ``veces_vs_csv``,
        ``seg_escritura``, ``seg_lectura``, ``conserva_dtypes``.

        ``veces_vs_csv`` es el peso relativo al CSV (1,00 = igual que el CSV).
        Si el CSV no se midió, la columna queda en NaN.
    """
    destino = Path(carpeta)
    destino.mkdir(parents=True, exist_ok=True)

    extensiones = {"csv": ".csv", "json": ".json", "excel": ".xlsx", "parquet": ".parquet"}
    filas = []
    for formato in formatos:
        ruta = destino / f"muestra{extensiones[formato]}"

        inicio = time.perf_counter()
        try:
            _escribir(df, formato, ruta)
        except FormatoNoDisponible:
            if omitir_no_disponibles:
                continue
            raise
        seg_escritura = time.perf_counter() - inicio

        inicio = time.perf_counter()
        releido = _leer(formato, ruta)
        seg_lectura = time.perf_counter() - inicio

        tam = ruta.stat().st_size
        filas.append(
            {
                "formato": formato,
                "archivo": ruta.name,
                "bytes": tam,
                "mb": round(tam / 1024**2, 3),
                "seg_escritura": round(seg_escritura, 3),
                "seg_lectura": round(seg_lectura, 3),
                "conserva_dtypes": dtypes_conservados(df, releido),
            }
        )

    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return tabla
    referencia = tabla.loc[tabla["formato"] == "csv", "bytes"]
    tabla["veces_vs_csv"] = (
        (tabla["bytes"] / referencia.iloc[0]).round(2) if not referencia.empty else pd.NA
    )
    orden = [
        "formato",
        "archivo",
        "bytes",
        "mb",
        "veces_vs_csv",
        "seg_escritura",
        "seg_lectura",
        "conserva_dtypes",
    ]
    return tabla[orden].sort_values("bytes").reset_index(drop=True)


def memoria_lista_python(valores: list) -> int:
    """Bytes que ocupa realmente una lista de Python, incluidos sus elementos.

    ``sys.getsizeof(lista)`` solo mide el arreglo de punteros, no los objetos
    apuntados: para una lista de un millón de flotantes se queda corto por un
    factor de tres. Aquí se suman ambos, que es la comparación honesta contra
    ``ndarray.nbytes``.

    Nota: los enteros pequeños y los booleanos están internados por CPython, así
    que este cálculo los cuenta varias veces y sobreestima. Para flotantes, que
    es el caso del notebook, la aproximación es buena.
    """
    import sys

    return sys.getsizeof(valores) + sum(sys.getsizeof(v) for v in valores)
