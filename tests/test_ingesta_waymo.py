"""Tests de la traducción del esquema real de Waymo y de su ingesta.

Los que necesitan datos reales se **saltan** si no están descargados, igual que
``tests/test_mapeo_waymo.py``: la licencia de Waymo prohíbe redistribuirlos, así
que no pueden vivir en el repositorio.

Los que no los necesitan usan Parquet sintéticos con el esquema real, verificado
contra un archivo de verdad el 2026-08-26.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "kedro_mly1101" / "src"))

from kedro_mly1101.pipelines.ingesta import nodes as ingesta  # noqa: E402

import waymo  # noqa: E402

MUESTRA_REAL = RAIZ / "datos" / "waymo_real" / "muestra"
LB = "[LiDARBoxComponent]"
ST = "[StatsComponent]"


@pytest.fixture
def cajas() -> pd.DataFrame:
    """Dos detecciones con el esquema REAL de lidar_box."""
    return pd.DataFrame(
        {
            "key.segment_context_name": ["seg_a", "seg_a"],
            "key.frame_timestamp_micros": [1000, 1000],
            "key.laser_object_id": ["obj1", "obj2"],
            f"{LB}.type": [1, 4],
            f"{LB}.box.center.x": [10.0, -5.0],
            f"{LB}.box.center.y": [2.0, 3.0],
            f"{LB}.box.center.z": [0.9, 0.8],
            f"{LB}.box.size.x": [4.2, 1.8],
            f"{LB}.box.size.y": [1.8, 0.6],
            f"{LB}.box.size.z": [1.7, 1.6],
            f"{LB}.speed.x": [3.0, 0.0],
            f"{LB}.speed.y": [4.0, 0.0],
            f"{LB}.num_lidar_points_in_box": [441, 97],
            f"{LB}.difficulty_level.detection": [np.nan, 2.0],
        }
    )


@pytest.fixture
def stats() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "key.segment_context_name": ["seg_a"],
            "key.frame_timestamp_micros": [1000],
            f"{ST}.time_of_day": ["Day"],
            f"{ST}.weather": ["sunny"],
            f"{ST}.location": ["location_sf"],
        }
    )


# --- Las tres traducciones que no son un cambio de nombre --------------------

def test_la_velocidad_es_el_modulo_del_vector(cajas, stats) -> None:
    """speed.x=3 y speed.y=4 dan 5, no 3. Quedarse con speed.x es un error mudo."""
    df = waymo.traducir_esquema(cajas, stats)
    assert df.loc[0, "speed_mps"] == pytest.approx(5.0)
    assert "speed_x" not in df.columns


def test_el_tipo_de_objeto_se_traduce_del_entero(cajas, stats) -> None:
    df = waymo.traducir_esquema(cajas, stats)
    assert df["object_type"].tolist() == ["vehicle", "cyclist"]


def test_el_nan_de_la_dificultad_significa_level_1(cajas, stats) -> None:
    """El reverso del defecto de la Act. 1.3: aquí un faltante disfraza un valor.

    Waymo solo rellena la dificultad cuando la detección es difícil. Tratar ese
    NaN como dato faltante borraría la mayoría de las filas y dejaría el dataset
    con una sola clase.
    """
    df = waymo.traducir_esquema(cajas, stats)
    assert df["detection_difficulty"].tolist() == ["LEVEL_1", "LEVEL_2"]
    assert df["detection_difficulty"].isna().sum() == 0


# --- Estructura del resultado -----------------------------------------------

def test_el_resultado_usa_el_esquema_de_la_clase(cajas, stats) -> None:
    df = waymo.traducir_esquema(cajas, stats)
    esperadas = {
        "segment_id", "timestamp_micros", "id_interno", "object_type",
        "box_center_x", "box_center_y", "box_center_z",
        "box_length", "box_width", "box_height",
        "speed_mps", "num_lidar_points", "detection_difficulty",
        "time_of_day", "weather",
    }
    assert esperadas <= set(df.columns)


def test_el_contexto_se_pega_por_segmento_y_marca_de_tiempo(cajas, stats) -> None:
    df = waymo.traducir_esquema(cajas, stats)
    assert df["time_of_day"].tolist() == ["Day", "Day"]
    assert df["location"].tolist() == ["location_sf", "location_sf"]


def test_la_traduccion_no_pierde_ni_duplica_detecciones(cajas, stats) -> None:
    assert len(waymo.traducir_esquema(cajas, stats)) == len(cajas)


def test_una_columna_ausente_no_rompe_la_traduccion(cajas, stats) -> None:
    """Waymo puede cambiar su esquema; la traducción no debe reventar por eso."""
    df = waymo.traducir_esquema(cajas.drop(columns=[f"{LB}.speed.y"]), stats)
    assert len(df) == 2
    assert "speed_mps" not in df.columns


# --- Ingesta de varias particiones ------------------------------------------

def test_la_ingesta_concatena_los_segmentos(cajas, stats) -> None:
    otras = cajas.assign(**{"key.segment_context_name": "seg_b"})
    otros_stats = stats.assign(**{"key.segment_context_name": "seg_b"})
    particiones = {
        "seg_a/lidar_box": lambda: cajas,
        "seg_a/stats": lambda: stats,
        "seg_b/lidar_box": lambda: otras,
        "seg_b/stats": lambda: otros_stats,
    }
    df = ingesta.traducir_waymo(particiones)
    assert len(df) == 4
    assert set(df["segment_id"]) == {"seg_a", "seg_b"}


def test_la_ingesta_ignora_los_segmentos_incompletos(cajas, stats) -> None:
    """En la muestra descargada hay carpetas con stats pero sin lidar_box."""
    particiones = {
        "seg_a/lidar_box": lambda: cajas,
        "seg_a/stats": lambda: stats,
        "seg_b/stats": lambda: stats,          # sin cajas: se salta
    }
    assert len(ingesta.traducir_waymo(particiones)) == 2


def test_sin_ningun_segmento_el_error_dice_que_hacer() -> None:
    with pytest.raises(ValueError, match="descargar_waymo.py --muestra"):
        ingesta.traducir_waymo({"solo/stats": lambda: pd.DataFrame()})


def test_comparar_con_sintetico_produce_una_tabla(cajas, stats) -> None:
    reales = waymo.traducir_esquema(cajas, stats)
    sinteticas = pd.read_csv(RAIZ / "datos" / "crudos" / "detecciones_waymo_like.csv")
    tabla = ingesta.comparar_con_sintetico(reales, sinteticas)
    assert set(tabla.columns) == {"metrica", "real", "sintetico"}
    assert "% cyclist" in set(tabla["metrica"])


# --- Contra los datos reales, si están descargados ---------------------------

@pytest.mark.skipif(
    not MUESTRA_REAL.exists(),
    reason="requiere: python herramientas/descargar_waymo.py --muestra 40",
)
def test_la_traduccion_funciona_sobre_un_segmento_real() -> None:
    ruta = next(MUESTRA_REAL.glob("*/lidar_box.parquet"), None)
    if ruta is None:
        pytest.skip("no hay segmentos con lidar_box descargados")

    df = waymo.traducir_esquema(
        pd.read_parquet(ruta), pd.read_parquet(ruta.parent / "stats.parquet")
    )
    assert len(df) > 0
    assert set(df["detection_difficulty"]) <= {"LEVEL_1", "LEVEL_2"}
    assert set(df["object_type"]) <= {"unknown", "vehicle", "pedestrian", "sign", "cyclist"}
    assert (df["speed_mps"] >= 0).all()


@pytest.mark.skipif(
    not MUESTRA_REAL.exists(),
    reason="requiere: python herramientas/descargar_waymo.py --muestra 40",
)
def test_hay_varios_segmentos_para_poder_partir_sin_fuga() -> None:
    """Con un solo segmento no se puede partir en entrenamiento y prueba sin fuga."""
    segmentos = list(MUESTRA_REAL.glob("*/lidar_box.parquet"))
    if not segmentos:
        pytest.skip("no hay segmentos con lidar_box descargados")
    assert len(segmentos) >= 2, (
        "el pipeline supervisado necesita varios segmentos: "
        "python herramientas/descargar_waymo.py --muestra 40"
    )
