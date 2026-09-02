"""Tests de las utilidades de CRISP-DM (Act. 2.1 · IL2.1).

Fijan el contrato de la pauta: las seis fases en orden, el mapa del curso
(para que nadie vuelva a poner el no supervisado en el RA3) y lo que hace
válida una carta de proyecto.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import crispdm  # noqa: E402


def test_hay_seis_fases_en_el_orden_canonico() -> None:
    assert crispdm.FASES == (
        "comprension_del_negocio",
        "comprension_de_los_datos",
        "preparacion_de_los_datos",
        "modelado",
        "evaluacion",
        "despliegue",
    )


def test_el_ciclo_no_es_una_cascada() -> None:
    """CRISP-DM es iterativo: de evaluacion se puede volver a negocio o a datos."""
    assert "evaluacion" in crispdm.RETORNOS
    assert "comprension_del_negocio" in crispdm.RETORNOS["evaluacion"]
    assert "comprension_de_los_datos" in crispdm.RETORNOS["evaluacion"]


def test_el_mapa_del_curso_pone_supervisado_y_no_supervisado_en_el_ra2() -> None:
    mapa = crispdm.mapa_del_curso()
    por_actividad = {fila["actividad"]: fila for fila in mapa}
    assert por_actividad["2.2"]["ra"] == "RA2"
    assert por_actividad["2.3"]["ra"] == "RA2"
    assert por_actividad["2.3"]["fase"] == "modelado"
    assert por_actividad["3.1"]["ra"] == "RA3"
    assert por_actividad["3.1"]["fase"] == "evaluacion"


def test_el_ra1_cubre_datos_y_preparacion_no_el_modelado() -> None:
    fases_ra1 = {fila["fase"] for fila in crispdm.mapa_del_curso() if fila["ra"] == "RA1"}
    assert "comprension_de_los_datos" in fases_ra1
    assert "preparacion_de_los_datos" in fases_ra1
    assert "modelado" not in fases_ra1


def test_un_criterio_sin_cifra_ni_metrica_no_es_medible() -> None:
    assert not crispdm.es_criterio_medible("el modelo tiene que ser bueno")
    assert not crispdm.es_criterio_medible("mejorar la percepción del vehículo")


def test_un_criterio_con_metrica_o_umbral_es_medible() -> None:
    assert crispdm.es_criterio_medible("F1-macro ≥ 0,70 en detecciones difíciles")
    assert crispdm.es_criterio_medible("recall de LEVEL_2 de al menos 0,60")
    assert crispdm.es_criterio_medible("RMSE menor que 20.000 en el conjunto de prueba")


def test_empezar_por_el_algoritmo_se_detecta() -> None:
    assert crispdm.empieza_por_el_algoritmo("qué modelo usamos, random forest o red neuronal")
    assert crispdm.empieza_por_el_algoritmo("vamos a probar XGBoost")
    assert not crispdm.empieza_por_el_algoritmo(
        "¿en qué condiciones el sensor deja de ser confiable?"
    )


def test_validar_carta_reporta_campos_faltantes() -> None:
    faltantes = crispdm.validar_carta({})
    for campo in crispdm.CAMPOS_CARTA:
        assert campo in faltantes


def test_validar_carta_rechaza_criterio_vago_y_pregunta_de_algoritmo() -> None:
    carta = {
        "pregunta_de_negocio": "qué algoritmo conviene",
        "criterio_de_exito": "que prediga bien",
        "fuentes": "CSV de detecciones",
        "riesgos": "desbalance",
        "proxima_fase": "modelado",
    }
    problemas = crispdm.validar_carta(carta)
    assert "criterio_de_exito" in problemas
    assert "pregunta_de_negocio" in problemas


def test_validar_carta_acepta_una_carta_completa() -> None:
    carta = {
        "pregunta_de_negocio": "¿Se puede anticipar cuándo una detección LiDAR no es confiable?",
        "criterio_de_exito": "F1 de LEVEL_2 ≥ 0,60 en un split por segmento",
        "fuentes": "detecciones_waymo_like.csv (sintético, semilla 42)",
        "riesgos": "desbalance de ciclistas (~2 %) y nulos MNAR de speed_mps de noche",
        "proxima_fase": "modelado",
    }
    assert crispdm.validar_carta(carta) == []


def test_el_notebook_alumno_no_filtra_la_pauta() -> None:
    ruta = RAIZ / "notebooks" / "12_alumno_crispdm.ipynb"
    assert ruta.exists(), "regenerar con python herramientas/construir_notebooks.py"
    texto = ruta.read_text(encoding="utf-8")
    assert "Pauta docente" not in texto
    assert "TODO 1" in texto
