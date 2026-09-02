"""Tests del pipeline de aprendizaje supervisado (RA2 · Act. 2.2).

El test que importa es ``test_ningun_segmento_queda_en_las_dos_particiones``: es
la garantía que el pipeline entero existe para dar, y el tipo de error que no
produce ningún síntoma si se rompe.
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
FUGA = {"variable": "num_lidar_points"}


@pytest.fixture
def limpias() -> pd.DataFrame:
    """40 segmentos de 25 detecciones, con la dificultad ligada a los puntos láser."""
    generador = np.random.default_rng(0)
    n_segmentos, por_segmento = 40, 25
    filas = []
    for s in range(n_segmentos):
        for _ in range(por_segmento):
            puntos = int(generador.integers(5, 300))
            filas.append(
                {
                    "segment_id": f"seg_{s:04d}",
                    "detection_difficulty": "LEVEL_2" if puntos < 60 else "LEVEL_1",
                    "box_length": float(generador.normal(4.0, 1.0)),
                    "box_width": float(generador.normal(1.8, 0.3)),
                    "speed_mps": float(abs(generador.normal(5.0, 3.0))),
                    "num_lidar_points": puntos,
                }
            )
    return pd.DataFrame(filas)


# --- Preparación -------------------------------------------------------------

def test_preparar_descarta_las_filas_sin_etiqueta(limpias: pd.DataFrame) -> None:
    con_huecos = limpias.copy()
    con_huecos.loc[:9, "detection_difficulty"] = None
    tabla = sup.preparar_variables(con_huecos, CONFIG, FUGA)
    assert len(tabla) == len(limpias) - 10
    assert tabla["detection_difficulty"].notna().all()


def test_preparar_arrastra_la_variable_excluida_por_fuga(limpias: pd.DataFrame) -> None:
    """El modelo no la usa, pero el nodo que mide la fuga la necesita."""
    tabla = sup.preparar_variables(limpias, CONFIG, FUGA)
    assert "num_lidar_points" in tabla.columns
    assert "num_lidar_points" not in CONFIG["variables"]


def test_preparar_conserva_la_llave_de_agrupacion(limpias: pd.DataFrame) -> None:
    tabla = sup.preparar_variables(limpias, CONFIG, FUGA)
    assert "segment_id" in tabla.columns


# --- Partición: el núcleo ----------------------------------------------------

def test_ningun_segmento_queda_en_las_dos_particiones(limpias: pd.DataFrame) -> None:
    """La garantía que justifica todo el pipeline.

    Si esto se rompe, nada falla: el modelo simplemente saca mejor nota de la que
    merece. Por eso tiene que estar fijado por un test.
    """
    marcada = sup.particionar(sup.preparar_variables(limpias, CONFIG, FUGA), CONFIG)
    entrena = set(marcada.loc[marcada["particion"] == "entrenamiento", "segment_id"])
    prueba = set(marcada.loc[marcada["particion"] == "prueba", "segment_id"])
    assert entrena & prueba == set()
    assert entrena and prueba


def test_la_particion_al_azar_si_rompe_los_segmentos(limpias: pd.DataFrame) -> None:
    """Sin esto, la comparación de fuga no probaría nada."""
    marcada = sup.particionar_al_azar(
        sup.preparar_variables(limpias, CONFIG, FUGA), CONFIG
    )
    entrena = set(marcada.loc[marcada["particion"] == "entrenamiento", "segment_id"])
    prueba = set(marcada.loc[marcada["particion"] == "prueba", "segment_id"])
    assert len(entrena & prueba) > 0


def test_la_particion_no_pierde_ni_duplica_filas(limpias: pd.DataFrame) -> None:
    tabla = sup.preparar_variables(limpias, CONFIG, FUGA)
    marcada = sup.particionar(tabla, CONFIG)
    assert len(marcada) == len(tabla)
    assert set(marcada["particion"]) == {"entrenamiento", "prueba"}


def test_la_particion_es_reproducible(limpias: pd.DataFrame) -> None:
    tabla = sup.preparar_variables(limpias, CONFIG, FUGA)
    a = sup.particionar(tabla, CONFIG)["particion"]
    b = sup.particionar(tabla, CONFIG)["particion"]
    assert a.equals(b)


def test_la_proporcion_de_prueba_es_aproximadamente_la_pedida(limpias: pd.DataFrame) -> None:
    """Aproximada, no exacta: se cortan segmentos enteros, no filas sueltas."""
    marcada = sup.particionar(sup.preparar_variables(limpias, CONFIG, FUGA), CONFIG)
    proporcion = (marcada["particion"] == "prueba").mean()
    assert 0.15 < proporcion < 0.35


# --- Entrenamiento y evaluación ---------------------------------------------

def test_el_modelo_solo_ve_las_variables_declaradas(limpias: pd.DataFrame) -> None:
    """Ni la etiqueta, ni el segmento, ni la variable excluida por fuga."""
    marcada = sup.particionar(sup.preparar_variables(limpias, CONFIG, FUGA), CONFIG)
    modelo = sup.entrenar(marcada, CONFIG)
    assert list(modelo.feature_names_in_) == CONFIG["variables"]
    assert "segment_id" not in modelo.feature_names_in_
    assert "num_lidar_points" not in modelo.feature_names_in_


def test_evaluar_devuelve_una_fila_por_clase(limpias: pd.DataFrame) -> None:
    marcada = sup.particionar(sup.preparar_variables(limpias, CONFIG, FUGA), CONFIG)
    modelo = sup.entrenar(marcada, CONFIG)
    metricas = sup.evaluar_por_clase(modelo, marcada, CONFIG)
    clases = set(metricas["clase"])
    assert {"LEVEL_1", "LEVEL_2"} <= clases
    assert "macro avg" in clases          # el promedio que NO hay que mirar solo
    assert {"precision", "recall", "f1-score", "support"} <= set(metricas.columns)


def test_la_matriz_de_confusion_es_cuadrada_y_suma_la_prueba(limpias: pd.DataFrame) -> None:
    marcada = sup.particionar(sup.preparar_variables(limpias, CONFIG, FUGA), CONFIG)
    modelo = sup.entrenar(marcada, CONFIG)
    matriz = sup.matriz_confusion(modelo, marcada, CONFIG)
    assert matriz.shape[0] == matriz.shape[1]
    assert list(matriz.index) == list(matriz.columns)
    assert matriz.to_numpy().sum() == (marcada["particion"] == "prueba").sum()


# --- Las dos mediciones de fuga ---------------------------------------------

def test_incluir_la_variable_derivada_infla_la_metrica(limpias: pd.DataFrame) -> None:
    """La fuga que SÍ se manifiesta en este dataset.

    La etiqueta se deriva de ``num_lidar_points``, así que incluirla entre las
    predictoras sube la métrica sin que el modelo sirva para nada más.
    """
    marcada = sup.particionar(sup.preparar_variables(limpias, CONFIG, FUGA), CONFIG)
    tabla = sup.comparar_fuga_de_variable(marcada, CONFIG, FUGA)
    assert len(tabla) == 2
    assert tabla.loc[0, "inflacion_f1_macro"] == 0.0
    assert tabla.loc[1, "f1_macro"] > tabla.loc[0, "f1_macro"]


def test_comparar_particiones_reporta_los_segmentos_compartidos(limpias: pd.DataFrame) -> None:
    """La fuga por agrupación existe estructuralmente aunque no mueva la métrica."""
    tabla = sup.preparar_variables(limpias, CONFIG, FUGA)
    comparacion = sup.comparar_particiones(
        sup.particionar(tabla, CONFIG), sup.particionar_al_azar(tabla, CONFIG), CONFIG
    )
    assert comparacion.loc[0, "segmentos_compartidos"] == 0
    assert comparacion.loc[1, "segmentos_compartidos"] > 0
    assert comparacion.loc[0, "clase_minoritaria"] == "LEVEL_2"


def test_el_grafo_completo_encadena_las_cinco_pipelines() -> None:
    """supervisado consume lo que produce preprocesamiento, sin decírselo a nadie."""
    from kedro_mly1101.pipeline_registry import register_pipelines

    pipelines = register_pipelines()
    assert {
        "calidad",
        "preprocesamiento",
        "supervisado",
        "no_supervisado",
        "optimizacion",
        "ingesta",
        "waymo_real",
        "__default__",
    } == set(pipelines)

    assert "detecciones_limpias" in pipelines["supervisado"].inputs()
    assert "detecciones_limpias" in pipelines["no_supervisado"].inputs()

    completo = pipelines["__default__"]
    externas = {e for e in completo.inputs() if not e.startswith("params:")}
    assert externas == {"detecciones_crudas"}       # solo el CSV crudo entra de fuera
    assert len(completo.nodes) == 30


def test_el_recorrido_real_reutiliza_los_mismos_nodos() -> None:
    """waymo_real no duplica nodos: remapea la entrada del grafo de siempre.

    Si alguien copiara y pegara los nodos para los datos reales, este test seguiria
    pasando en numero pero las dos versiones se desincronizarian. Lo que se fija
    aqui es que la entrada del analisis viene de la ingesta, no del CSV.
    """
    from kedro_mly1101.pipeline_registry import register_pipelines

    pipelines = register_pipelines()
    real = pipelines["waymo_real"]

    # 24 nodos de analisis + 2 de ingesta.
    assert len(real.nodes) == len(pipelines["__default__"].nodes) + 2

    externas = {e for e in real.inputs() if not e.startswith("params:")}
    assert "waymo_muestra" in externas          # los Parquet reales
    assert "detecciones_reales" not in externas  # la produce la ingesta, no entra de fuera


def test_con_un_solo_segmento_el_error_explica_por_que(limpias: pd.DataFrame) -> None:
    """El fallo real al correr el pipeline sobre UN segmento de Waymo.

    No se hace un apaño cayendo a una particion al azar: eso seria justo la mala
    practica que el material ensena a evitar. Se falla, y se explica que hacer.
    """
    un_segmento = limpias.assign(segment_id="seg_0000")
    tabla = sup.preparar_variables(un_segmento, CONFIG, FUGA)
    with pytest.raises(ValueError, match="descargar_waymo.py --muestra"):
        sup.particionar(tabla, CONFIG)
