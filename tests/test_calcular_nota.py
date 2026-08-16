"""Tests de la conversión de puntajes de rúbrica a nota (escala 1,0–7,0)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "herramientas"))

from calcular_nota import (  # noqa: E402
    INDICADORES,
    PUNTAJE_MAXIMO,
    evaluar,
    nota,
    puntaje_ponderado,
)

TODOS = list(INDICADORES)


def puntajes(valor: float) -> dict[str, float]:
    """Mismo puntaje en los cinco indicadores."""
    return {clave: valor for clave in TODOS}


# --- Los pesos de la rúbrica -----------------------------------------------


def test_los_pesos_suman_uno() -> None:
    assert sum(INDICADORES.values()) == pytest.approx(1.0)


def test_il2_es_el_indicador_de_mayor_peso() -> None:
    """La cuantificación de problemas de calidad es el núcleo de la sesión."""
    assert max(INDICADORES, key=INDICADORES.get) == "IL2"


def test_puntaje_ponderado_con_puntaje_uniforme() -> None:
    assert puntaje_ponderado(puntajes(3)) == pytest.approx(3.0)


def test_puntaje_ponderado_pondera_de_verdad() -> None:
    # Solo IL2 (30 %) en nivel 4, el resto en 0.
    parcial = {**puntajes(0), "IL2": 4}
    assert puntaje_ponderado(parcial) == pytest.approx(1.2)


def test_falla_si_falta_un_indicador() -> None:
    incompleto = puntajes(3)
    del incompleto["IL4"]
    with pytest.raises(ValueError, match="faltan indicadores"):
        puntaje_ponderado(incompleto)


def test_falla_con_puntaje_fuera_de_rango() -> None:
    with pytest.raises(ValueError, match="entre 0 y 4"):
        puntaje_ponderado({**puntajes(3), "IL1": 5})


# --- La escala de notas ----------------------------------------------------


def test_los_tres_puntos_de_anclaje_con_exigencia_60() -> None:
    assert nota(0.0) == 1.0                    # sin evidencia
    assert nota(0.6 * PUNTAJE_MAXIMO) == 4.0   # 60 % del puntaje: aprueba justo
    assert nota(PUNTAJE_MAXIMO) == 7.0         # todo destacado


def test_la_escala_es_monotona() -> None:
    valores = [nota(p / 10) for p in range(0, 41)]
    assert valores == sorted(valores)


def test_nivel_logrado_en_todo_aprueba_con_holgura() -> None:
    """Puntaje 3 (Logrado) en los cinco indicadores debe dar una nota cómoda."""
    resultado = evaluar(puntajes(3))
    assert resultado["nota"] == pytest.approx(5.1, abs=0.05)
    assert resultado["aprueba"]


def test_nivel_en_desarrollo_en_todo_reprueba() -> None:
    """Puntaje 2 (En desarrollo) queda bajo el 60 % de exigencia."""
    resultado = evaluar(puntajes(2))
    assert resultado["porcentaje"] == pytest.approx(50.0)
    assert not resultado["aprueba"]


def test_una_exigencia_menor_sube_la_nota() -> None:
    assert nota(2.0, exigencia=0.50) == 4.0
    assert nota(2.0, exigencia=0.60) < 4.0


def test_exigencia_invalida() -> None:
    with pytest.raises(ValueError, match="entre 0 y 1"):
        nota(2.0, exigencia=1.5)


def test_evaluar_devuelve_el_detalle_completo() -> None:
    resultado = evaluar(puntajes(4))
    assert resultado == {
        "puntaje_ponderado": 4.0,
        "porcentaje": 100.0,
        "nota": 7.0,
        "aprueba": True,
    }


def test_caso_realista_mezclado() -> None:
    """Buen diagnóstico pero decisiones flojas: debe aprobar sin destacar."""
    caso = {"IL1": 3, "IL2": 4, "IL3": 2, "IL4": 3, "IL5": 3}
    resultado = evaluar(caso)
    # 3(0,20) + 4(0,30) + 2(0,25) + 3(0,15) + 3(0,10) = 3,05
    assert resultado["puntaje_ponderado"] == pytest.approx(3.05)
    assert resultado["nota"] == pytest.approx(5.2, abs=0.05)
