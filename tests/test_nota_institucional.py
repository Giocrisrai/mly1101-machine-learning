"""Escala institucional de las parciales y el EFT (% de logro 100/80/60/30/0)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "herramientas"))

from calcular_nota import (  # noqa: E402
    INSTRUMENTOS,
    NIVELES_LOGRO,
    evaluar_csv,
    evaluar_institucional,
    logro_ponderado,
    nota,
)


def test_los_cuatro_instrumentos_suman_cien() -> None:
    for nombre, pesos in INSTRUMENTOS.items():
        assert sum(pesos.values()) == pytest.approx(1.0), nombre


def test_ep1_cubre_ie1_a_ie4() -> None:
    assert list(INSTRUMENTOS["ep1"]) == ["IE1", "IE2", "IE3", "IE4"]
    assert INSTRUMENTOS["ep1"]["IE3"] == pytest.approx(0.40)


def test_ep2_cubre_ie5_a_ie8() -> None:
    assert list(INSTRUMENTOS["ep2"]) == ["IE5", "IE6", "IE7", "IE8"]


def test_ep3_cubre_ie9_a_ie12() -> None:
    assert list(INSTRUMENTOS["ep3"]) == ["IE9", "IE10", "IE11", "IE12"]
    assert INSTRUMENTOS["ep3"]["IE10"] == pytest.approx(0.30)
    assert INSTRUMENTOS["ep3"]["IE12"] == pytest.approx(0.30)


def test_eft_son_doce_indicadores() -> None:
    assert list(INSTRUMENTOS["eft"]) == [f"IE{i}" for i in range(1, 13)]
    assert INSTRUMENTOS["eft"]["IE1"] == pytest.approx(0.05)
    assert INSTRUMENTOS["eft"]["IE12"] == pytest.approx(0.10)


def test_logro_uniforme_sesenta_es_aprobacion() -> None:
    niveles = {clave: 60 for clave in INSTRUMENTOS["ep1"]}
    assert logro_ponderado("ep1", niveles) == pytest.approx(60.0)
    resultado = evaluar_institucional("ep1", niveles)
    assert resultado["nota"] == 4.0
    assert resultado["aprueba"]


def test_logro_cien_es_siete() -> None:
    niveles = {clave: 100 for clave in INSTRUMENTOS["eft"]}
    resultado = evaluar_institucional("eft", niveles)
    assert resultado["logro"] == pytest.approx(100.0)
    assert resultado["nota"] == 7.0


def test_solo_niveles_de_la_pauta() -> None:
    with pytest.raises(ValueError, match="nivel"):
        logro_ponderado("ep1", {"IE1": 70, "IE2": 60, "IE3": 60, "IE4": 60})


def test_falla_si_falta_un_ie() -> None:
    with pytest.raises(ValueError, match="faltan"):
        logro_ponderado("ep1", {"IE1": 80, "IE2": 80, "IE3": 80})


def test_ep1_realista_aprueba() -> None:
    """IE3 (calidad, 40 %) en 80 y el resto en 60 → 68 % → nota > 4."""
    niveles = {"IE1": 60, "IE2": 60, "IE3": 80, "IE4": 60}
    resultado = evaluar_institucional("ep1", niveles)
    assert resultado["logro"] == pytest.approx(68.0)
    assert resultado["nota"] == nota(0.68 * 4.0)
    assert resultado["aprueba"]


def test_niveles_admitidos() -> None:
    assert NIVELES_LOGRO == (0, 30, 60, 80, 100)


def test_csv_ep1_con_columnas_ie(tmp_path: Path) -> None:
    ruta = tmp_path / "ep1.csv"
    ruta.write_text("nombre,IE1,IE2,IE3,IE4\nAna,80,60,100,60\n", encoding="utf-8")
    filas = evaluar_csv(ruta, instrumento="ep1")
    esperado = evaluar_institucional(
        "ep1", {"IE1": 80, "IE2": 60, "IE3": 100, "IE4": 60}
    )
    assert filas[0]["nombre"] == "Ana"
    assert filas[0]["logro"] == esperado["logro"]
    assert filas[0]["nota"] == esperado["nota"]
    assert filas[0]["aprueba"]


def test_csv_detecta_ep1_si_no_pasan_instrumento(tmp_path: Path) -> None:
    ruta = tmp_path / "curso.csv"
    ruta.write_text("nombre,IE1,IE2,IE3,IE4\nBeto,60,60,60,60\n", encoding="utf-8")
    filas = evaluar_csv(ruta)
    assert filas[0]["instrumento"] == "ep1"
    assert filas[0]["nota"] == 4.0


def test_csv_formativo_con_il_sigue_funcionando(tmp_path: Path) -> None:
    ruta = tmp_path / "formativo.csv"
    ruta.write_text("nombre,IL1,IL2,IL3,IL4,IL5\nCami,4,4,4,4,4\n", encoding="utf-8")
    filas = evaluar_csv(ruta)
    assert filas[0]["nota"] == 7.0
    assert filas[0].get("instrumento") is None
