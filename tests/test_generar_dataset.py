"""Verifica que el dataset de la EA1 sea reproducible y contenga sus defectos.

Si un test de este archivo falla, el solucionario del docente quedó desalineado
con los datos que reciben los alumnos.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from generar_dataset import (  # noqa: E402
    CATALOGO_DEFECTOS,
    COLUMNAS,
    escribir_csv,
    generar_dataset,
)

FILAS_PRUEBA = 8_000
LLAVE = ["segment_id", "timestamp_micros", "id_interno"]


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    """Dataset pequeño generado con la semilla de la asignatura."""
    return generar_dataset(FILAS_PRUEBA, semilla=42)


@pytest.fixture(scope="module")
def df_publicado() -> pd.DataFrame:
    """El CSV que efectivamente está en el repositorio, leído como lo lee el alumno."""
    ruta = RAIZ / "datos" / "crudos" / "detecciones_waymo_like.csv"
    if not ruta.exists():
        pytest.skip("falta el CSV publicado; generar con python src/generar_dataset.py")
    return pd.read_csv(ruta)


# --- Contrato del generador -------------------------------------------------


def test_columnas_y_orden(df: pd.DataFrame) -> None:
    assert list(df.columns) == COLUMNAS


def test_cantidad_de_filas_incluye_duplicados(df: pd.DataFrame) -> None:
    # 1.2% de duplicados exactos + 0.5% de duplicados lógicos.
    assert FILAS_PRUEBA < len(df) <= int(FILAS_PRUEBA * 1.02)


def test_es_reproducible_byte_a_byte(tmp_path: Path) -> None:
    hashes = []
    for nombre in ("a.csv", "b.csv"):
        ruta = escribir_csv(generar_dataset(2_000, semilla=42), tmp_path / nombre)
        hashes.append(hashlib.sha256(ruta.read_bytes()).hexdigest())
    assert hashes[0] == hashes[1]


def test_otra_semilla_da_otro_dataset() -> None:
    a = generar_dataset(2_000, semilla=42)
    b = generar_dataset(2_000, semilla=7)
    assert not a["box_center_x"].equals(b["box_center_x"])


# --- Los diez defectos del catálogo ----------------------------------------


def test_catalogo_tiene_diez_defectos() -> None:
    assert len(CATALOGO_DEFECTOS) == 10


def test_defecto_1_timestamp_corrupto(df_publicado: pd.DataFrame) -> None:
    assert df_publicado["timestamp_micros"].dtype == object
    assert (df_publicado["timestamp_micros"] == "N/D").sum() > 0
    # El resto debe ser convertible: el defecto afecta a pocas filas.
    convertidos = pd.to_numeric(df_publicado["timestamp_micros"], errors="coerce")
    assert convertidos.isna().mean() < 0.01


def test_defecto_2_nulos_ocultos(df: pd.DataFrame) -> None:
    proporcion = (df["num_lidar_points"] == -1).mean()
    assert 0.02 < proporcion < 0.045
    # El nulo está oculto: pandas no lo ve como faltante.
    assert df["num_lidar_points"].isna().sum() == 0


def test_defecto_3_categorias_de_clima(df: pd.DataFrame) -> None:
    valores = set(df["weather"].dropna().unique())
    assert {"sunny", "Sunny", "SUNNY", "soleado"} <= valores
    assert {"rain", "RAIN ", " rain", "lluvia"} <= valores
    assert 0.03 < df["weather"].isna().mean() < 0.07
    # Al normalizar, las 11 variantes colapsan a 3 categorías reales.
    canonicas = {"sunny": "sunny", "soleado": "sunny", "rain": "rain", "lluvia": "rain",
                 "fog": "fog", "niebla": "fog"}
    normalizado = df["weather"].astype("string").str.strip().str.lower().replace(canonicas)
    assert set(normalizado.dropna().unique()) == {"sunny", "rain", "fog"}


def test_defecto_3_variantes_conservan_espacios_en_el_csv(
    df_publicado: pd.DataFrame,
) -> None:
    """El CSV debe preservar los espacios: son parte del ejercicio de limpieza."""
    valores = set(df_publicado["weather"].dropna().unique())
    assert "RAIN " in valores
    assert " rain" in valores


def test_defecto_4_categorias_de_object_type(df: pd.DataFrame) -> None:
    valores = set(df["object_type"].unique())
    assert {"PEDESTRIAN", "Pedestrian", "PEATON", "Ped"} <= valores


def test_defecto_5_duplicados_exactos_y_logicos(df: pd.DataFrame) -> None:
    exactos = int(df.duplicated().sum())
    por_llave = int(df.duplicated(subset=LLAVE).sum())
    assert exactos > 0
    assert por_llave > exactos, "deben existir duplicados lógicos, no solo exactos"


def test_defecto_6_outliers_imposibles(df: pd.DataFrame) -> None:
    assert df["speed_mps"].max() > 150, "debe haber velocidades imposibles"
    assert (df["box_height"] == 0).sum() > 0
    assert (df["box_length"] < 0).sum() > 0


def test_defecto_7_outliers_legitimos(df: pd.DataFrame) -> None:
    buses = df[(df["box_length"] > 12) & (df["box_length"] < 18.5)]
    assert len(buses) > 0
    # Un bus es coherente: si es largo, también es alto y ancho.
    assert buses["box_height"].min() > 2.5
    assert buses["box_width"].min() > 2.2


def test_defecto_8_nulos_mnar_tienen_patron(df: pd.DataFrame) -> None:
    tasa = df.groupby("detection_difficulty")["speed_mps"].apply(lambda s: s.isna().mean())
    assert tasa["LEVEL_2"] > 5 * tasa["LEVEL_1"], "el nulo debe depender de la dificultad"
    dificiles = df[df["detection_difficulty"] == "LEVEL_2"]
    por_hora = dificiles.groupby("time_of_day")["speed_mps"].apply(lambda s: s.isna().mean())
    assert por_hora["Night"] > por_hora["Day"], "de noche deben faltar más velocidades"


def test_defecto_9_desbalance_de_clases(df: pd.DataFrame) -> None:
    proporcion = (df["object_type"] == "CYCLIST").mean()
    assert 0.01 < proporcion < 0.035


def test_defecto_10_columnas_sin_valor_predictivo(df: pd.DataFrame) -> None:
    assert df["sensor_version"].nunique() == 1
    assert df["id_interno"].nunique() / len(df) > 0.95


# --- Coherencia física de lo que NO es defecto ------------------------------


def test_las_dimensiones_validas_son_plausibles(df: pd.DataFrame) -> None:
    validas = df[(df["box_length"] > 0) & (df["box_height"] > 0)]
    assert validas["box_length"].between(0.1, 20).all()
    assert validas["box_height"].between(0.1, 5).all()


def test_las_senales_no_se_mueven(df: pd.DataFrame) -> None:
    senales = df[(df["object_type"] == "SIGN") & df["speed_mps"].notna()]
    # Solo los outliers imposibles inyectados pueden romper esta regla.
    assert (senales["speed_mps"] > 1).mean() < 0.02


def test_menos_puntos_laser_a_mayor_distancia(df: pd.DataFrame) -> None:
    reales = df[df["num_lidar_points"] > 0].copy()
    reales["distancia"] = np.sqrt(reales["box_center_x"] ** 2 + reales["box_center_y"] ** 2)
    correlacion = reales["distancia"].corr(reales["num_lidar_points"], method="spearman")
    assert correlacion < -0.5, "la relación distancia/puntos debe ser descubrible"
