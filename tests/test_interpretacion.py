"""Tests de las utilidades de interpretación (Act. 2.4 · IL2.4).

Fijan el contrato de la pauta: una métrica no es un conocimiento para la
organización hasta que se traduce a frecuencia, a costo y a una decisión.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import interpretacion  # noqa: E402


def test_por_cada_cien_usa_las_cifras_de_la_actividad_22() -> None:
    """456 encontradas y 676 perdidas (pauta 2.2) son 40 y 60 de cada 100."""
    resumen = interpretacion.por_cada_cien(encontrados=456, perdidos=676)
    assert resumen["encuentra"] == 40
    assert resumen["pierde"] == 60
    assert resumen["encuentra"] + resumen["pierde"] == 100


def test_leer_errores_toma_filas_reales_y_columnas_predichas() -> None:
    matriz = pd.DataFrame(
        [[8737, 385], [676, 456]],
        index=["LEVEL_1", "LEVEL_2"],
        columns=["LEVEL_1", "LEVEL_2"],
    )
    errores = interpretacion.leer_errores(matriz, positiva="LEVEL_2")
    assert errores["tp"] == 456
    assert errores["fn"] == 676
    assert errores["fp"] == 385


def test_costo_esperado_penaliza_el_error_mas_caro() -> None:
    # Un FN (confiar en una detección mala) cuesta 10 veces un FP (ser prudente).
    assert interpretacion.costo_esperado(n_fn=676, n_fp=385, costo_fn=10, costo_fp=1) == 7145
    assert interpretacion.costo_esperado(n_fn=0, n_fp=100, costo_fn=10, costo_fp=1) == 100


def test_frase_para_la_organizacion_no_habla_de_f1() -> None:
    frase = interpretacion.frase_para_la_organizacion(encuentra=40, pierde=60, sujeto="detecciones difíciles")
    assert "40" in frase and "60" in frase
    assert "f1" not in frase.casefold()
    assert "exactitud" not in frase.casefold()


def test_metricas_minimas_de_regresion_no_son_r2() -> None:
    minimas = interpretacion.metricas_minimas("regresion")
    assert "mae" in minimas
    assert "rmse" in minimas
    assert "r2" not in minimas


def test_metricas_minimas_de_clasificacion_no_son_la_exactitud() -> None:
    minimas = interpretacion.metricas_minimas("clasificacion")
    assert "recall" in minimas
    assert "f1" in minimas
    assert "exactitud" not in minimas


def test_reportar_solo_r2_no_basta() -> None:
    assert interpretacion.reporta_solo_promedio(["r2"])
    assert interpretacion.reporta_solo_promedio(["exactitud"])
    assert not interpretacion.reporta_solo_promedio(["mae", "rmse"])
    assert not interpretacion.reporta_solo_promedio(["recall", "f1"])


def test_error_en_unidades_queda_en_la_escala_original() -> None:
    real = np.array([100.0, 200.0, 300.0])
    predicho = np.array([110.0, 190.0, 330.0])
    errores = interpretacion.error_en_unidades(real, predicho)
    assert errores["mae"] == pytest.approx(50.0 / 3)
    assert errores["rmse"] == pytest.approx(np.sqrt((10**2 + 10**2 + 30**2) / 3))


def test_el_notebook_alumno_no_filtra_la_pauta() -> None:
    ruta = RAIZ / "notebooks" / "13_alumno_interpretacion.ipynb"
    assert ruta.exists(), "regenerar con python herramientas/construir_notebooks.py"
    texto = ruta.read_text(encoding="utf-8")
    assert "Pauta docente" not in texto
    assert "TODO 1" in texto
