"""Lectura de datos desde fuentes de distinta naturaleza (Act 1.1, IL1.1).

La Actividad 1.1 sostiene que un dataset no siempre llega como un CSV limpio:
puede venir de una base relacional, de una API que responde JSON anidado o de
texto libre escrito por una persona. Este módulo entrega las piezas mínimas
para que esa afirmación se pueda **comprobar en clase** sobre el mismo dataset
de detecciones, en vez de quedarse en una tabla comparativa.

Funciones puras y sin efectos secundarios visibles, igual que ``src/eda.py``:
no imprimen ni grafican, para que sirvan tanto en los notebooks como en los
tests. La única excepción declarada es ``a_sqlite``, que crea una conexión en
memoria; el llamador decide cuándo cerrarla.
"""

from __future__ import annotations

import re
import sqlite3

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Fuente estructurada: base de datos relacional
# ---------------------------------------------------------------------------

def a_sqlite(
    df: pd.DataFrame, tabla: str = "detecciones", conexion: sqlite3.Connection | None = None
) -> sqlite3.Connection:
    """Carga un DataFrame en una tabla de SQLite y devuelve la conexión.

    Sirve para que la clase consulte con SQL exactamente los mismos datos que
    tiene en pandas, sin instalar ningún motor: ``sqlite3`` es parte de la
    biblioteca estándar de Python y funciona igual en Colab que en local.

    Args:
        df: datos a cargar.
        tabla: nombre de la tabla destino.
        conexion: conexión existente. Si es None, se crea una en memoria
            (``:memory:``), que desaparece al cerrarla.

    Returns:
        La conexión, con la tabla ya escrita.
    """
    if conexion is None:
        conexion = sqlite3.connect(":memory:")
    df.to_sql(tabla, conexion, if_exists="replace", index=False)
    return conexion


def consultar(conexion: sqlite3.Connection, sql: str) -> pd.DataFrame:
    """Ejecuta una consulta SQL y devuelve el resultado como DataFrame.

    Existe como función con nombre para que en el notebook quede explícito que
    ``pd.read_sql`` es el puente entre los dos mundos: la consulta la resuelve
    el motor, el resultado vuelve a ser un DataFrame de pandas.
    """
    return pd.read_sql(sql, conexion)


# ---------------------------------------------------------------------------
# Fuente semiestructurada: JSON anidado
# ---------------------------------------------------------------------------

def contexto_por_segmento(df: pd.DataFrame) -> list[dict]:
    """Arma el contexto de cada segmento como registros JSON **anidados**.

    Reproduce la forma en que llega el componente ``stats`` del Waymo Open
    Dataset v2: un registro por segmento, con las condiciones de grabación
    agrupadas bajo una clave y un conteo por tipo de objeto en una lista. Es
    decir, la estructura jerárquica típica de una respuesta de API, que pandas
    no puede convertir en tabla con ``pd.DataFrame`` a secas.

    Returns:
        Lista de diccionarios con la forma::

            {
              "segment_id": "seg_0063",
              "condiciones": {"weather": "sunny", "time_of_day": "Day"},
              "n_detecciones": 512,
              "objetos": [{"tipo": "VEHICLE", "n": 380}, ...]
            }
    """
    registros = []
    for segmento, grupo in df.groupby("segment_id", sort=True):
        conteo = grupo["object_type"].value_counts()
        registros.append(
            {
                "segment_id": str(segmento),
                "condiciones": {
                    "weather": _primero_no_nulo(grupo["weather"]),
                    "time_of_day": _primero_no_nulo(grupo["time_of_day"]),
                },
                "n_detecciones": int(len(grupo)),
                "objetos": [
                    {"tipo": str(tipo), "n": int(n)} for tipo, n in conteo.items()
                ],
            }
        )
    return registros


def _primero_no_nulo(serie: pd.Series):
    """Primer valor no nulo de una serie, o None si todos lo son."""
    sin_nulos = serie.dropna()
    return None if sin_nulos.empty else sin_nulos.iloc[0]


def aplanar_contexto(registros: list[dict]) -> pd.DataFrame:
    """Convierte los registros anidados en una tabla plana.

    Usa ``pd.json_normalize``, que aplana los diccionarios anidados creando
    columnas con notación de punto (``condiciones.weather``). La lista
    ``objetos`` **no** se aplana aquí: una lista dentro de una celda necesita
    ``record_path``, y ese es justamente el ejercicio del notebook.

    Returns:
        DataFrame con una fila por segmento y las columnas ``segment_id``,
        ``n_detecciones``, ``condiciones.weather``, ``condiciones.time_of_day``
        y ``objetos``.
    """
    return pd.json_normalize(registros)


def aplanar_objetos(registros: list[dict]) -> pd.DataFrame:
    """Expande la lista ``objetos`` a una fila por (segmento, tipo de objeto).

    Muestra el otro modo de ``json_normalize``: ``record_path`` elige la lista
    que se convierte en filas y ``meta`` arrastra los campos del nivel padre.

    Returns:
        DataFrame con columnas ``tipo``, ``n`` y ``segment_id``.
    """
    return pd.json_normalize(registros, record_path="objetos", meta=["segment_id"])


# ---------------------------------------------------------------------------
# Fuente no estructurada: texto libre
# ---------------------------------------------------------------------------

PATRON_SEGMENTO = re.compile(r"seg_\d{4}")

PLANTILLAS_PARTE = [
    "Turno {turno}. El operador reporta lluvia intensa durante {segmento}; "
    "varias detecciones quedaron marcadas como difíciles.",
    "Revisión de calidad: en {segmento} el LiDAR perdió sincronía por unos "
    "instantes. Revisar marcas de tiempo antes de usar el tramo.",
    "Incidente menor en {segmento}: un ciclista apareció por la derecha y el "
    "sistema tardó en clasificarlo. Sin daños.",
    "Mantención del sensor tras {segmento}. Se recomienda no mezclar este "
    "tramo con los posteriores hasta recalibrar.",
    "Nota del turno {turno}: {segmento} se grabó de noche con neblina. "
    "Calidad visual pobre, pero el LiDAR respondió bien.",
]


def generar_partes_incidente(
    df: pd.DataFrame, n: int = 12, semilla: int = 42
) -> list[str]:
    """Genera partes de incidente en **texto libre** que mencionan segmentos reales.

    Son el ejemplo de dato no estructurado del notebook: prosa escrita por una
    persona, sin esquema, donde el dato útil (qué segmento está comprometido)
    está enterrado en la frase. Para la misma semilla el resultado es idéntico.

    Args:
        df: dataset del que se toman los ``segment_id`` que se mencionarán.
        n: cuántos partes generar.
        semilla: semilla del generador aleatorio.

    Returns:
        Lista de ``n`` cadenas de texto.
    """
    rng = np.random.default_rng(semilla)
    segmentos = np.sort(df["segment_id"].dropna().unique())
    elegidos = rng.choice(segmentos, size=n, replace=False)
    plantillas = rng.integers(0, len(PLANTILLAS_PARTE), size=n)
    return [
        PLANTILLAS_PARTE[int(p)].format(segmento=str(s), turno=int(i % 3) + 1)
        for i, (s, p) in enumerate(zip(elegidos, plantillas))
    ]


def extraer_segmentos(texto: str) -> list[str]:
    """Extrae los identificadores de segmento mencionados en un texto libre.

    Busca el patrón ``seg_`` seguido de cuatro dígitos. Devuelve las
    apariciones en orden y sin repetir, para que el resultado sea determinista
    y se pueda cruzar con el DataFrame.
    """
    vistos: list[str] = []
    for hallazgo in PATRON_SEGMENTO.findall(texto):
        if hallazgo not in vistos:
            vistos.append(hallazgo)
    return vistos


def segmentos_comprometidos(partes: list[str]) -> pd.DataFrame:
    """Tabla de segmentos mencionados en una lista de partes de incidente.

    Es el paso que convierte texto no estructurado en algo cruzable con el
    dataset: una tabla con el segmento y cuántos partes lo mencionan.

    Returns:
        DataFrame con columnas ``segment_id`` y ``n_menciones``, ordenado de
        más a menos menciones y, a igualdad, por identificador.
    """
    menciones: list[str] = []
    for parte in partes:
        menciones.extend(extraer_segmentos(parte))
    if not menciones:
        return pd.DataFrame(columns=["segment_id", "n_menciones"])
    conteo = pd.Series(menciones).value_counts()
    tabla = conteo.rename_axis("segment_id").reset_index(name="n_menciones")
    return tabla.sort_values(
        ["n_menciones", "segment_id"], ascending=[False, True]
    ).reset_index(drop=True)
