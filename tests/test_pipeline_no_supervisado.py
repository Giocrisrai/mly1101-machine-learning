"""Tests del pipeline de aprendizaje no supervisado (RA2 · Act. 2.3)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "kedro_mly1101" / "src"))

from kedro_mly1101.pipelines.no_supervisado import nodes as ns  # noqa: E402

CONFIG = {
    "variables": ["box_length", "box_height", "num_lidar_points"],
    "etiqueta_de_contraste": "object_type",
    "k_a_probar": [2, 3, 4],
    "k": 3,
    "muestra_silueta": 300,
    "semilla": 42,
}


@pytest.fixture
def limpias() -> pd.DataFrame:
    """Tres nubes bien separadas: la estructura existe y debe encontrarse."""
    generador = np.random.default_rng(0)
    grupos = [
        ("vehicle", 4.5, 1.7, 400),
        ("pedestrian", 0.9, 1.7, 60),
        ("sign", 0.4, 0.6, 25),
    ]
    filas = []
    for tipo, largo, alto, puntos in grupos:
        for _ in range(200):
            filas.append(
                {
                    "object_type": tipo,
                    "box_length": float(generador.normal(largo, 0.15)),
                    "box_height": float(generador.normal(alto, 0.1)),
                    "num_lidar_points": float(generador.normal(puntos, 10)),
                }
            )
    return pd.DataFrame(filas)


# --- Preparación y escalado --------------------------------------------------

def test_preparar_escala_las_variables(limpias: pd.DataFrame) -> None:
    """Sin escalar, num_lidar_points (cientos) aplastaria a box_height (~1,7)."""
    matriz = ns.preparar_matriz(limpias, CONFIG)
    for columna in CONFIG["variables"]:
        assert abs(matriz[columna].mean()) < 1e-9
        assert abs(matriz[columna].std(ddof=0) - 1.0) < 1e-9


def test_preparar_conserva_la_etiqueta_pero_no_la_escala(limpias: pd.DataFrame) -> None:
    """La etiqueta viaja para contrastar despues; NO entra en el agrupamiento."""
    matriz = ns.preparar_matriz(limpias, CONFIG)
    assert "object_type" in matriz.columns
    assert set(matriz["object_type"]) == set(limpias["object_type"])
    assert "object_type" not in CONFIG["variables"]


def test_preparar_descarta_filas_incompletas(limpias: pd.DataFrame) -> None:
    con_huecos = limpias.copy()
    con_huecos.loc[:4, "box_length"] = np.nan
    assert len(ns.preparar_matriz(con_huecos, CONFIG)) == len(limpias) - 5


# --- Elección de k -----------------------------------------------------------

def test_elegir_k_devuelve_una_fila_por_k(limpias: pd.DataFrame) -> None:
    tabla = ns.elegir_k(ns.preparar_matriz(limpias, CONFIG), CONFIG)
    assert tabla["k"].tolist() == CONFIG["k_a_probar"]
    assert set(tabla.columns) == {"k", "inercia", "silueta"}


def test_la_inercia_siempre_baja_al_anadir_grupos(limpias: pd.DataFrame) -> None:
    """Por eso la inercia sola no decide nada: siempre premia mas grupos."""
    tabla = ns.elegir_k(ns.preparar_matriz(limpias, CONFIG), CONFIG)
    assert tabla["inercia"].is_monotonic_decreasing


def test_con_tres_nubes_separadas_la_silueta_prefiere_tres(limpias: pd.DataFrame) -> None:
    tabla = ns.elegir_k(ns.preparar_matriz(limpias, CONFIG), CONFIG)
    assert tabla.loc[tabla["silueta"].idxmax(), "k"] == 3


# --- Agrupamiento e interpretación -------------------------------------------

def test_agrupar_asigna_un_grupo_a_cada_fila(limpias: pd.DataFrame) -> None:
    matriz = ns.preparar_matriz(limpias, CONFIG)
    agrupada = ns.agrupar(matriz, CONFIG)
    assert len(agrupada) == len(matriz)
    assert agrupada["grupo"].nunique() == CONFIG["k"]


def test_agrupar_es_reproducible(limpias: pd.DataFrame) -> None:
    matriz = ns.preparar_matriz(limpias, CONFIG)
    assert ns.agrupar(matriz, CONFIG)["grupo"].equals(ns.agrupar(matriz, CONFIG)["grupo"])


def test_perfilar_describe_cada_grupo(limpias: pd.DataFrame) -> None:
    agrupada = ns.agrupar(ns.preparar_matriz(limpias, CONFIG), CONFIG)
    perfil = ns.perfilar_grupos(agrupada, CONFIG)
    assert len(perfil) == CONFIG["k"]
    assert {"grupo", "n", "pct"} <= set(perfil.columns)
    assert abs(perfil["pct"].sum() - 100) < 0.1


def test_contrastar_da_porcentajes_por_grupo(limpias: pd.DataFrame) -> None:
    agrupada = ns.agrupar(ns.preparar_matriz(limpias, CONFIG), CONFIG)
    cruce = ns.contrastar_con_etiqueta(agrupada, CONFIG)
    assert len(cruce) == CONFIG["k"]
    porcentajes = cruce.drop(columns=["grupo"])
    assert np.allclose(porcentajes.sum(axis=1), 100, atol=0.1)


def test_con_nubes_separadas_cada_grupo_recupera_un_tipo(limpias: pd.DataFrame) -> None:
    """Sobre datos construidos para separarse, el agrupamiento debe recuperarlos.

    Que aqui salga limpio es lo que permite afirmar con base que sobre los datos
    REALES no salga limpio: el nodo funciona, lo que no se separa es el mundo.
    """
    agrupada = ns.agrupar(ns.preparar_matriz(limpias, CONFIG), CONFIG)
    cruce = ns.contrastar_con_etiqueta(agrupada, CONFIG).drop(columns=["grupo"])
    assert (cruce.max(axis=1) > 90).all()


# --- Reducción de dimensionalidad --------------------------------------------

def test_resumir_pca_acumula_hasta_uno(limpias: pd.DataFrame) -> None:
    matriz = ns.preparar_matriz(limpias, CONFIG)
    tabla = ns.resumir_pca(matriz, CONFIG)
    assert len(tabla) == len(CONFIG["variables"])
    assert tabla["varianza_explicada"].is_monotonic_decreasing
    assert abs(tabla["varianza_acumulada"].iloc[-1] - 1.0) < 1e-6


def test_proyectar_devuelve_dos_componentes(limpias: pd.DataFrame) -> None:
    matriz = ns.preparar_matriz(limpias, CONFIG)
    proyectada = ns.proyectar_2d(matriz, CONFIG)
    assert {"componente_1", "componente_2"} <= set(proyectada.columns)
    assert len(proyectada) == len(matriz)


def test_la_proyeccion_es_serializable_a_parquet(limpias: pd.DataFrame, tmp_path) -> None:
    """Fijado tras un fallo real: la varianza iba en df.attrs y Parquet reventaba."""
    proyectada = ns.proyectar_2d(ns.preparar_matriz(limpias, CONFIG), CONFIG)
    destino = tmp_path / "p.parquet"
    proyectada.to_parquet(destino, index=False)
    assert "varianza_explicada_2d" in pd.read_parquet(destino).columns
