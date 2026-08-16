"""Tests del análisis de sesgo de muestreo.

No requieren datos descargados: usan un DataFrame construido a mano con la
misma forma que devuelve ``cargar_muestra()``. Así la lógica del análisis queda
verificada aunque no haya credenciales de Waymo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "herramientas"))

from analizar_sesgo_waymo import (  # noqa: E402
    calidad_por_condicion,
    composicion_por_condicion,
    generar_informe,
    tabla_segmentos,
)


@pytest.fixture
def muestra() -> pd.DataFrame:
    """Dos segmentos: uno diurno con ciclistas, uno nocturno sin ellos."""
    dia = pd.DataFrame({
        "segmento": ["seg_dia"] * 10,
        "tipo": ["vehicle"] * 5 + ["pedestrian"] * 3 + ["cyclist"] * 2,
        "puntos": [100] * 10,
        "dificil": [False] * 8 + [True] * 2,
        "weather": ["sunny"] * 10,
        "time_of_day": ["Day"] * 10,
        "location": ["location_sf"] * 10,
    })
    noche = pd.DataFrame({
        "segmento": ["seg_noche"] * 10,
        "tipo": ["vehicle"] * 8 + ["pedestrian"] * 2,
        "puntos": [20] * 9 + [0],
        "dificil": [True] * 6 + [False] * 4,
        "weather": ["rain"] * 10,
        "time_of_day": ["Night"] * 10,
        "location": ["location_phx"] * 10,
    })
    return pd.concat([dia, noche], ignore_index=True)


def test_tabla_segmentos_resume_una_fila_por_segmento(muestra: pd.DataFrame) -> None:
    tabla = tabla_segmentos(muestra)
    assert len(tabla) == 2
    assert set(tabla["segmento"]) == {"seg_dia", "seg_noche"}
    assert tabla.set_index("segmento").loc["seg_dia", "detecciones"] == 10


def test_composicion_por_condicion_suma_cien_por_grupo(muestra: pd.DataFrame) -> None:
    tabla = composicion_por_condicion(muestra, "time_of_day")
    for momento in tabla.index:
        assert tabla.loc[momento].sum() == pytest.approx(100.0, abs=0.01)


def test_composicion_detecta_la_ausencia_de_ciclistas_de_noche(muestra: pd.DataFrame) -> None:
    tabla = composicion_por_condicion(muestra, "time_of_day")
    assert tabla.loc["Day", "cyclist"] == pytest.approx(20.0)
    assert tabla.loc["Night", "cyclist"] == pytest.approx(0.0)


def test_calidad_por_condicion_detecta_la_degradacion_nocturna(muestra: pd.DataFrame) -> None:
    tabla = calidad_por_condicion(muestra, "time_of_day")
    assert tabla.loc["Night", "puntos_medianos"] < tabla.loc["Day", "puntos_medianos"]
    assert tabla.loc["Night", "pct_dificiles"] > tabla.loc["Day", "pct_dificiles"]
    assert tabla.loc["Night", "pct_sin_puntos"] == pytest.approx(10.0)


def test_el_informe_incluye_todas_las_secciones(muestra: pd.DataFrame) -> None:
    informe = generar_informe(muestra)
    for seccion in ["Segmentos por momento del día", "Segmentos por clima",
                    "Segmentos por ubicación", "Composición de objetos por clima",
                    "Calidad de la detección", "Peatones + ciclistas"]:
        assert seccion in informe, f"falta la sección: {seccion}"


def test_el_informe_reporta_el_tamano_de_la_muestra(muestra: pd.DataFrame) -> None:
    assert "Muestra: 2 segmentos, 20 detecciones" in generar_informe(muestra)
