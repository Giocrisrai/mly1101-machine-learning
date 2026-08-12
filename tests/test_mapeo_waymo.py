"""Verifica el mapeo del esquema real de Waymo v2 al esquema de la clase.

Estos tests se **saltan** si no hay datos reales descargados. Para activarlos:

    gcloud auth login          # con la cuenta que aceptó los términos de Waymo
    python herramientas/descargar_waymo.py

Los datos quedan en ``datos/waymo_real/`` (ignorado por git: la licencia de
Waymo no permite redistribuirlos).

Sirven para dos cosas:

1. Confirmar que los nombres de columna del notebook opcional siguen siendo los
   del dataset real (Waymo puede cambiar el esquema entre versiones).
2. Confirmar que las relaciones que el dataset sintético reproduce (distancia
   vs. puntos láser, coherencia dimensional por tipo de objeto) existen también
   en los datos reales. Si no existieran, el dataset de clase estaría enseñando
   algo falso.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "herramientas"))

DIRECTORIO = RAIZ / "datos" / "waymo_real"
CAJAS = DIRECTORIO / "lidar_box.parquet"
STATS = DIRECTORIO / "stats.parquet"

LB = "[LiDARBoxComponent]"
ST = "[StatsComponent]"

# Las mismas equivalencias que usa notebooks/00_opcional_waymo_real.ipynb.
COLUMNAS_CAJAS = [
    "key.segment_context_name",
    "key.frame_timestamp_micros",
    "key.laser_object_id",
    f"{LB}.type",
    f"{LB}.box.center.x",
    f"{LB}.box.center.y",
    f"{LB}.box.center.z",
    f"{LB}.box.size.x",
    f"{LB}.box.size.y",
    f"{LB}.box.size.z",
    f"{LB}.num_lidar_points_in_box",
    f"{LB}.difficulty_level.detection",
]

COLUMNAS_STATS = [
    "key.segment_context_name",
    "key.frame_timestamp_micros",
    f"{ST}.weather",
    f"{ST}.time_of_day",
    f"{ST}.location",
]

SIN_DATOS = "faltan los datos reales de Waymo; ver el docstring de este archivo"


@pytest.fixture(scope="module")
def cajas() -> pd.DataFrame:
    if not CAJAS.exists():
        pytest.skip(SIN_DATOS)
    return pd.read_parquet(CAJAS)


@pytest.fixture(scope="module")
def stats() -> pd.DataFrame:
    if not STATS.exists():
        pytest.skip(SIN_DATOS)
    return pd.read_parquet(STATS)


# --- El esquema sigue siendo el que documenta el notebook -------------------


def test_lidar_box_tiene_las_columnas_esperadas(cajas: pd.DataFrame) -> None:
    faltan = [c for c in COLUMNAS_CAJAS if c not in cajas.columns]
    assert not faltan, f"el esquema de Waymo cambió; faltan: {faltan}"


def test_stats_tiene_las_columnas_esperadas(stats: pd.DataFrame) -> None:
    faltan = [c for c in COLUMNAS_STATS if c not in stats.columns]
    assert not faltan, f"el esquema de Waymo cambió; faltan: {faltan}"


def test_la_velocidad_es_un_vector(cajas: pd.DataFrame) -> None:
    """En Waymo v2 la velocidad son componentes x/y/z, no una rapidez escalar."""
    assert f"{LB}.speed.x" in cajas.columns
    assert f"{LB}.speed.y" in cajas.columns
    assert f"{LB}.speed" not in cajas.columns


def test_el_tipo_de_objeto_es_entero_entre_0_y_4(cajas: pd.DataFrame) -> None:
    tipos = set(cajas[f"{LB}.type"].dropna().unique())
    assert tipos <= {0, 1, 2, 3, 4}, f"valores inesperados de type: {tipos}"


def test_la_dificultad_es_entera_1_o_2(cajas: pd.DataFrame) -> None:
    niveles = set(cajas[f"{LB}.difficulty_level.detection"].dropna().unique())
    assert niveles <= {0, 1, 2}, f"valores inesperados de difficulty: {niveles}"


def test_stats_usa_las_categorias_documentadas(stats: pd.DataFrame) -> None:
    horas = set(stats[f"{ST}.time_of_day"].dropna().unique())
    assert horas <= {"Day", "Night", "Dawn/Dusk"}, f"time_of_day inesperado: {horas}"
    climas = set(stats[f"{ST}.weather"].dropna().unique())
    assert climas <= {"sunny", "rain", "Sunny", "Rain"}, f"weather inesperado: {climas}"


# --- La unión que hace el notebook funciona --------------------------------


def test_la_union_por_llave_encuentra_correspondencia(
    cajas: pd.DataFrame, stats: pd.DataFrame
) -> None:
    llave = ["key.segment_context_name", "key.frame_timestamp_micros"]
    unido = cajas[llave].merge(stats[llave + [f"{ST}.weather"]], on=llave, how="left")
    sin_clima = unido[f"{ST}.weather"].isna().mean()
    assert sin_clima < 0.05, f"{sin_clima:.1%} de las cajas quedó sin clima tras la unión"


# --- Las relaciones que el dataset sintético reproduce existen de verdad ----


def test_menos_puntos_laser_a_mayor_distancia(cajas: pd.DataFrame) -> None:
    """La relación que el dataset de clase reproduce debe existir en los datos reales."""
    distancia = np.sqrt(cajas[f"{LB}.box.center.x"] ** 2 + cajas[f"{LB}.box.center.y"] ** 2)
    puntos = cajas[f"{LB}.num_lidar_points_in_box"]
    correlacion = distancia.corr(puntos, method="spearman")
    assert correlacion < -0.3, f"correlación real {correlacion:.3f}: revisar el dataset sintético"


def test_las_dimensiones_son_coherentes_por_tipo(cajas: pd.DataFrame) -> None:
    """Un peatón real mide en torno a 1,7 m: el sintético usa esa misma escala."""
    peatones = cajas[cajas[f"{LB}.type"] == 2]
    if peatones.empty:
        pytest.skip("este segmento no tiene peatones")
    alto_mediano = peatones[f"{LB}.box.size.z"].median()
    assert 1.4 < alto_mediano < 2.1, f"altura mediana de peatón inesperada: {alto_mediano:.2f} m"


def test_el_desbalance_de_clases_es_real(cajas: pd.DataFrame) -> None:
    """Los ciclistas son minoría también en los datos reales (justifica el 2 % sintético)."""
    proporciones = cajas[f"{LB}.type"].value_counts(normalize=True)
    ciclistas = proporciones.get(4, 0.0)
    vehiculos = proporciones.get(1, 0.0)
    assert ciclistas < vehiculos, "se esperaba que los ciclistas fueran minoría"
