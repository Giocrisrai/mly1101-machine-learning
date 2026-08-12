"""Tests de las utilidades de diagnóstico de src/eda.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import eda  # noqa: E402


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "categoria": ["a", "A ", " a", "b", None, "b"],
            "medida": [1.0, 2.0, 2.5, 3.0, np.nan, 1000.0],
            "puntos": [10, -1, 20, 30, -1, 40],
            "constante": ["v1"] * 6,
        }
    )


def test_resumen_calidad_cuenta_nulos_y_centinelas(df: pd.DataFrame) -> None:
    resumen = eda.resumen_calidad(df, centinelas=[-1])
    assert resumen.loc["medida", "n_nulos"] == 1
    assert resumen.loc["puntos", "n_centinelas"] == 2
    assert resumen.loc["puntos", "pct_faltante_total"] == pytest.approx(33.33, abs=0.01)
    assert resumen.loc["constante", "n_unicos"] == 1


def test_resumen_calidad_tiene_una_fila_por_columna(df: pd.DataFrame) -> None:
    assert list(eda.resumen_calidad(df).index) == list(df.columns)


def test_detectar_outliers_iqr_marca_el_valor_extremo(df: pd.DataFrame) -> None:
    marcados = eda.detectar_outliers_iqr(df["medida"])
    assert marcados.iloc[5]
    assert not marcados.iloc[0]
    assert not marcados.iloc[4], "un NaN no es un outlier"


def test_limites_iqr_es_coherente_con_la_deteccion() -> None:
    serie = pd.Series([1, 2, 3, 4, 5, 100])
    inferior, superior = eda.limites_iqr(serie)
    marcados = eda.detectar_outliers_iqr(serie)
    assert marcados.equals((serie < inferior) | (serie > superior))


def test_zscore_devuelve_falso_si_no_hay_varianza() -> None:
    serie = pd.Series([5.0, 5.0, 5.0])
    assert not eda.detectar_outliers_zscore(serie).any()


def test_normalizar_categoria_unifica_variantes(df: pd.DataFrame) -> None:
    normalizada = eda.normalizar_categoria(df["categoria"])
    assert normalizada.dropna().tolist() == ["a", "a", "a", "b", "b"]


def test_normalizar_categoria_aplica_el_mapa() -> None:
    serie = pd.Series(["lluvia", "RAIN ", "sunny"])
    normalizada = eda.normalizar_categoria(serie, {"lluvia": "rain"})
    assert normalizada.tolist() == ["rain", "rain", "sunny"]


def test_reporte_duplicados_separa_exactos_de_logicos() -> None:
    df = pd.DataFrame(
        {
            "id": ["x", "x", "y", "y"],
            "valor": [1, 1, 2, 3],  # x: duplicado exacto; y: duplicado lógico
        }
    )
    reporte = eda.reporte_duplicados(df, ["id"]).iloc[0]
    assert reporte["dup_exactos"] == 1
    assert reporte["dup_por_llave"] == 2
    assert reporte["dup_logicos"] == 1


def test_matriz_nulos_por_grupo_con_una_variable() -> None:
    df = pd.DataFrame({"g": ["a", "a", "b", "b"], "v": [np.nan, np.nan, 1.0, 2.0]})
    matriz = eda.matriz_nulos_por_grupo(df, "v", ["g"])
    assert matriz.loc["a", "pct_nulos"] == 100.0
    assert matriz.loc["b", "pct_nulos"] == 0.0


def test_matriz_nulos_por_grupo_con_dos_variables() -> None:
    df = pd.DataFrame(
        {
            "g1": ["a", "a", "b", "b"],
            "g2": ["x", "y", "x", "y"],
            "v": [np.nan, 1.0, 2.0, 3.0],
        }
    )
    matriz = eda.matriz_nulos_por_grupo(df, "v", ["g1", "g2"])
    assert matriz.loc["a", "x"] == 100.0
    assert matriz.loc["b", "y"] == 0.0


def test_valores_imposibles_cuenta_violaciones(df: pd.DataFrame) -> None:
    reporte = eda.valores_imposibles(df, {"puntos negativos": "puntos < 0"})
    assert reporte.loc[0, "n_filas"] == 2
    assert reporte.loc[0, "pct"] == pytest.approx(33.333, abs=0.01)


def test_a_numerico_marca_lo_no_convertible() -> None:
    serie = pd.Series(["1", "2", "N/D"])
    convertida = eda.a_numerico(serie)
    assert convertida.isna().sum() == 1
    assert convertida.iloc[0] == 1


def test_resumen_desbalance_calcula_la_razon() -> None:
    serie = pd.Series(["a"] * 90 + ["b"] * 10)
    resumen = eda.resumen_desbalance(serie)
    assert resumen.loc["a", "pct"] == 90.0
    assert resumen.loc["b", "ratio_vs_mayoritaria"] == pytest.approx(0.1111, abs=0.001)


def test_perfil_numerico_incluye_outliers(df: pd.DataFrame) -> None:
    perfil = eda.perfil_numerico(df, ["medida"])
    assert "n_outliers_iqr" in perfil.columns
    assert perfil.loc["medida", "n_outliers_iqr"] == 1
