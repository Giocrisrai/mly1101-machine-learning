"""Tests del pipeline de optimización y ensamble (RA3).

El test que importa es ``test_la_busqueda_nunca_toca_el_conjunto_de_prueba``: es la
garantía que separa un ajuste honesto de uno que se autoengaña, y el tipo de error que
no produce ningún síntoma si se rompe.
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

from kedro_mly1101.pipelines.optimizacion import nodes as opt  # noqa: E402
from kedro_mly1101.pipelines.supervisado import nodes as sup  # noqa: E402

CONFIG = {
    "objetivo": "detection_difficulty",
    "grupo": "segment_id",
    "variables": ["box_length", "box_width", "speed_mps"],
    "proporcion_prueba": 0.25,
    "semilla": 42,
    "n_arboles": 20,
    "profundidad_maxima": 6,
}
AJUSTE = {
    "metrica": "f1_macro",
    "n_pliegues": 3,
    "n_combinaciones": 4,
    "n_estimators": [10, 20],
    "max_depth": [3, 6, None],
    "min_samples_leaf": [1, 10],
    "max_features": ["sqrt"],
}
FUGA = {"variable": "num_lidar_points"}


@pytest.fixture
def marcada() -> pd.DataFrame:
    """30 segmentos de 40 detecciones, ya particionados por grupo.

    La etiqueta depende de las VARIABLES, no solo de ``num_lidar_points``:
    reproduce la relacion del dataset real --los objetos mas grandes reciben mas
    puntos laser y son mas faciles de detectar-- para que los modelos tengan algo
    que aprender.

    Sin esa dependencia el fixture no sirve: ningun modelo puede superar al
    baseline y varios tests dejan de comprobar lo que dicen comprobar.
    """
    generador = np.random.default_rng(0)
    filas = []
    for s in range(30):
        for _ in range(40):
            largo = float(generador.normal(4.0, 1.5))
            ancho = float(generador.normal(1.8, 0.4))
            # Mas grande -> mas puntos laser -> deteccion mas facil.
            puntos = max(1.0, 40 * largo * ancho + generador.normal(0, 60))
            filas.append(
                {
                    "segment_id": f"seg_{s:04d}",
                    "detection_difficulty": "LEVEL_2" if puntos < 220 else "LEVEL_1",
                    "box_length": largo,
                    "box_width": ancho,
                    "speed_mps": float(abs(generador.normal(5.0, 3.0))),
                    "num_lidar_points": float(puntos),
                }
            )
    tabla = sup.preparar_variables(pd.DataFrame(filas), CONFIG, FUGA)
    return sup.particionar(tabla, CONFIG)


# --- La garantía central -----------------------------------------------------

def test_la_busqueda_nunca_toca_el_conjunto_de_prueba(marcada, monkeypatch) -> None:
    """El ajuste solo puede ver el entrenamiento.

    Se comprueba por construcción: ``_partes`` filtra por particion. Si alguien
    cambiara eso, el ajuste se optimizaria contra la prueba y la metrica final
    dejaria de estimar nada.
    """
    X, y, grupos = opt._partes(marcada, CONFIG)
    esperado = (marcada["particion"] == "entrenamiento").sum()
    assert len(X) == esperado
    assert len(y) == esperado
    assert set(grupos) == set(marcada.loc[marcada["particion"] == "entrenamiento", "segment_id"])
    # Ningun segmento de prueba se cuela.
    segmentos_prueba = set(marcada.loc[marcada["particion"] == "prueba", "segment_id"])
    assert set(grupos) & segmentos_prueba == set()


# --- 3.1 Ajuste --------------------------------------------------------------

def test_la_busqueda_devuelve_una_fila_por_combinacion(marcada) -> None:
    tabla = opt.buscar_hiperparametros(marcada, CONFIG, AJUSTE)
    assert len(tabla) == AJUSTE["n_combinaciones"]
    assert tabla["rank_test_score"].min() == 1
    assert {"mean_test_score", "std_test_score"} <= set(tabla.columns)


def test_la_busqueda_viene_ordenada_por_rango(marcada) -> None:
    tabla = opt.buscar_hiperparametros(marcada, CONFIG, AJUSTE)
    assert tabla["rank_test_score"].is_monotonic_increasing


def test_comparar_ajuste_reporta_ganancia_y_ruido(marcada) -> None:
    busqueda = opt.buscar_hiperparametros(marcada, CONFIG, AJUSTE)
    tabla = opt.comparar_ajuste_contra_defecto(marcada, CONFIG, AJUSTE, busqueda)
    assert len(tabla) == 2
    assert tabla.loc[0, "ganancia"] == 0.0
    assert {"media", "desv_entre_pliegues", "ganancia", "ganancia_supera_el_ruido"} <= set(
        tabla.columns
    )


def test_la_fuga_por_ajuste_reporta_las_tres_magnitudes(marcada) -> None:
    """Optimismo, margen de la trampa y brecha validacion/prueba son cosas distintas."""
    tabla = opt.medir_fuga_por_ajustar_en_prueba(marcada, CONFIG, AJUSTE)
    criterios = set(tabla["criterio"].dropna())
    assert any("validación cruzada" in c for c in criterios)
    assert any("mirando la prueba" in c for c in criterios)
    assert any("margen de la trampa" in c for c in criterios)
    assert "brecha_validacion_vs_prueba" in tabla.columns


def test_max_depth_none_no_rompe_la_tabla(marcada) -> None:
    """Fijado tras un fallo real: ``None`` en una columna de enteros da NaN."""
    tabla = opt.medir_fuga_por_ajustar_en_prueba(marcada, CONFIG, AJUSTE)
    assert "sin límite" in set(tabla["max_depth"])
    assert tabla["max_depth"].notna().all()


# --- 3.2 Ensamble ------------------------------------------------------------

def test_la_comparacion_incluye_baseline_y_ensamble(marcada) -> None:
    """Sin baseline la comparacion no significa nada."""
    tabla = opt.comparar_ensambles(marcada, CONFIG, AJUSTE)
    assert {"baseline", "ensamble_votacion"} <= set(tabla["modelo"])
    assert tabla["media"].is_monotonic_decreasing


def test_la_comparacion_reporta_costo_ademas_de_metrica(marcada) -> None:
    """La pregunta del RA3 no es si mejora, es si mejora lo suficiente."""
    tabla = opt.comparar_ensambles(marcada, CONFIG, AJUSTE)
    assert "segundos" in tabla.columns
    # >= 0 y no > 0: sobre un fixture pequeno los modelos rapidos redondean a 0,0.
    # Lo que se comprueba es que la columna existe y se mide, no su magnitud.
    assert (tabla["segundos"] >= 0).all()
    assert tabla["segundos"].notna().all()


def test_los_modelos_de_arbol_le_ganan_al_baseline(marcada) -> None:
    """Sobre datos con senal aprendible, el bosque debe superar al trivial.

    Si esto falla, o el fixture no tiene senal o algo se rompio en la cadena:
    en ambos casos el resto de los tests deja de significar lo que dice.
    """
    tabla = opt.comparar_ensambles(marcada, CONFIG, AJUSTE).set_index("modelo")
    assert tabla.loc["bosque_aleatorio", "media"] > tabla.loc["baseline", "media"]


def test_la_regresion_logistica_tolera_faltantes(marcada) -> None:
    """Los arboles admiten NaN desde sklearn 1.4; la regresion logistica no.

    Por eso lleva un SimpleImputer delante. Sin el, la comparacion revienta con
    'Input X contains NaN' en mitad de la validacion cruzada.
    """
    con_huecos = marcada.copy()
    con_huecos.loc[con_huecos.index[:50], "box_length"] = np.nan
    tabla = opt.comparar_ensambles(con_huecos, CONFIG, AJUSTE)
    assert tabla.loc[tabla["modelo"] == "regresion_logistica", "media"].notna().all()


# --- 3.3 Robustez y selección ------------------------------------------------

def test_robustez_compara_contra_el_ruido_del_mejor(marcada) -> None:
    comparacion = opt.comparar_ensambles(marcada, CONFIG, AJUSTE)
    robustez = opt.analizar_robustez(comparacion)
    assert len(robustez) == len(comparacion)
    # El mejor no es distinguible de si mismo.
    assert robustez.loc[0, "diferencia_vs_mejor"] == 0.0
    assert not robustez.loc[0, "distinguible_del_mejor"]


def test_el_baseline_si_es_distinguible_del_mejor(marcada) -> None:
    """Si esto fallara, el modelo no le estaria ganando a responder siempre lo mismo."""
    comparacion = opt.comparar_ensambles(marcada, CONFIG, AJUSTE)
    robustez = opt.analizar_robustez(comparacion).set_index("modelo")
    assert robustez.loc["baseline", "distinguible_del_mejor"]


def test_la_tabla_de_seleccion_trae_las_cuatro_dimensiones(marcada) -> None:
    """El IL3.4 pide sustentar, no reportar el maximo."""
    comparacion = opt.comparar_ensambles(marcada, CONFIG, AJUSTE)
    robustez = opt.analizar_robustez(comparacion)
    seleccion = opt.tabla_de_seleccion(comparacion, robustez, CONFIG)
    assert {
        "media",
        "desv_entre_pliegues",
        "distinguible_del_mejor",
        "veces_mas_lento",
        "interpretabilidad",
    } <= set(seleccion.columns)
    assert seleccion["interpretabilidad"].notna().all()


def test_las_pipelines_del_ra3_se_registran() -> None:
    from kedro_mly1101.pipeline_registry import register_pipelines

    pipelines = register_pipelines()
    assert "optimizacion" in pipelines
    assert len(pipelines["optimizacion"].nodes) == 6
    # Consume la particion que produce el pipeline supervisado.
    assert "particion_por_segmento" in pipelines["optimizacion"].inputs()
