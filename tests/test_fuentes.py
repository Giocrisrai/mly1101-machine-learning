"""Tests de la lectura desde fuentes heterogéneas de src/fuentes.py (Act 1.1)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import fuentes  # noqa: E402


@pytest.fixture
def df() -> pd.DataFrame:
    """Mini-dataset con la misma forma que las detecciones reales."""
    return pd.DataFrame(
        {
            "segment_id": ["seg_0001", "seg_0001", "seg_0002", "seg_0002", "seg_0003"],
            "object_type": ["VEHICLE", "PEDESTRIAN", "VEHICLE", "VEHICLE", "CYCLIST"],
            "weather": [None, "sunny", "rain", "rain", "fog"],
            "time_of_day": ["Day", "Day", "Night", "Night", "Dawn"],
            "speed_mps": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


# --- Fuente estructurada: SQL ------------------------------------------------

def test_a_sqlite_permite_recuperar_las_mismas_filas(df: pd.DataFrame) -> None:
    conexion = fuentes.a_sqlite(df)
    recuperado = fuentes.consultar(conexion, "SELECT * FROM detecciones")
    assert len(recuperado) == len(df)
    assert list(recuperado.columns) == list(df.columns)
    conexion.close()


def test_consulta_sql_agregada_coincide_con_groupby(df: pd.DataFrame) -> None:
    """SQL y pandas deben dar el mismo resultado: es el punto del ejercicio."""
    conexion = fuentes.a_sqlite(df)
    por_sql = fuentes.consultar(
        conexion,
        "SELECT object_type, COUNT(*) AS n FROM detecciones "
        "GROUP BY object_type ORDER BY object_type",
    )
    por_pandas = (
        df.groupby("object_type").size().reset_index(name="n").sort_values("object_type")
    )
    assert por_sql["n"].tolist() == por_pandas["n"].tolist()
    conexion.close()


def test_a_sqlite_acepta_nombre_de_tabla_propio(df: pd.DataFrame) -> None:
    conexion = fuentes.a_sqlite(df, tabla="cajas")
    assert len(fuentes.consultar(conexion, "SELECT * FROM cajas")) == 5
    conexion.close()


def test_a_sqlite_reemplaza_la_tabla_en_una_conexion_existente(df: pd.DataFrame) -> None:
    conexion = fuentes.a_sqlite(df)
    fuentes.a_sqlite(df.head(2), conexion=conexion)
    assert len(fuentes.consultar(conexion, "SELECT * FROM detecciones")) == 2
    conexion.close()


# --- Fuente semiestructurada: JSON anidado -----------------------------------

def test_contexto_por_segmento_arma_un_registro_por_segmento(df: pd.DataFrame) -> None:
    registros = fuentes.contexto_por_segmento(df)
    assert len(registros) == df["segment_id"].nunique()
    assert set(registros[0]) == {"segment_id", "condiciones", "n_detecciones", "objetos"}


def test_contexto_por_segmento_esta_realmente_anidado(df: pd.DataFrame) -> None:
    """Si no estuviera anidado, json_normalize no tendría nada que hacer."""
    registros = fuentes.contexto_por_segmento(df)
    assert isinstance(registros[0]["condiciones"], dict)
    assert isinstance(registros[0]["objetos"], list)


def test_contexto_ignora_los_nulos_al_elegir_el_clima(df: pd.DataFrame) -> None:
    """seg_0001 tiene weather None en su primera fila y 'sunny' en la segunda."""
    registros = {r["segment_id"]: r for r in fuentes.contexto_por_segmento(df)}
    assert registros["seg_0001"]["condiciones"]["weather"] == "sunny"


def test_aplanar_contexto_crea_columnas_con_notacion_de_punto(df: pd.DataFrame) -> None:
    plano = fuentes.aplanar_contexto(fuentes.contexto_por_segmento(df))
    assert "condiciones.weather" in plano.columns
    assert "condiciones.time_of_day" in plano.columns
    assert len(plano) == df["segment_id"].nunique()


def test_aplanar_objetos_expande_la_lista_a_filas(df: pd.DataFrame) -> None:
    objetos = fuentes.aplanar_objetos(fuentes.contexto_por_segmento(df))
    assert list(objetos.columns) == ["tipo", "n", "segment_id"]
    # Una fila por par (segmento, tipo de objeto) presente en el dataset.
    assert len(objetos) == len(df.groupby(["segment_id", "object_type"]))
    assert objetos["n"].sum() == len(df)


# --- Fuente no estructurada: texto libre -------------------------------------

def test_extraer_segmentos_encuentra_el_patron() -> None:
    texto = "Incidente en seg_0042 y luego en seg_0007; se repite seg_0042."
    assert fuentes.extraer_segmentos(texto) == ["seg_0042", "seg_0007"]


def test_extraer_segmentos_sin_menciones_devuelve_lista_vacia() -> None:
    assert fuentes.extraer_segmentos("El turno terminó sin novedad.") == []


def test_generar_partes_es_determinista(df: pd.DataFrame) -> None:
    a = fuentes.generar_partes_incidente(df, n=3, semilla=7)
    b = fuentes.generar_partes_incidente(df, n=3, semilla=7)
    assert a == b
    assert fuentes.generar_partes_incidente(df, n=3, semilla=8) != a


def test_los_partes_mencionan_segmentos_que_existen_en_el_dataset(df: pd.DataFrame) -> None:
    """Sin esto, el cruce del notebook devolvería siempre cero filas."""
    partes = fuentes.generar_partes_incidente(df, n=3, semilla=42)
    existentes = set(df["segment_id"])
    for parte in partes:
        assert set(fuentes.extraer_segmentos(parte)) <= existentes


def test_segmentos_comprometidos_cuenta_y_ordena() -> None:
    partes = [
        "Falla en seg_0002.",
        "Otra vez seg_0002 y además seg_0001.",
        "Sin novedad.",
    ]
    tabla = fuentes.segmentos_comprometidos(partes)
    assert tabla["segment_id"].tolist() == ["seg_0002", "seg_0001"]
    assert tabla["n_menciones"].tolist() == [2, 1]


def test_segmentos_comprometidos_sin_menciones_devuelve_tabla_vacia() -> None:
    tabla = fuentes.segmentos_comprometidos(["Turno sin novedad."])
    assert tabla.empty
    assert list(tabla.columns) == ["segment_id", "n_menciones"]
