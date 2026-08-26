"""Tests del proyecto Kedro de kedro_mly1101/.

Los nodos de Kedro son funciones normales de Python, así que se prueban sin
levantar nada: ni catálogo, ni runner, ni sesión. Esa es justamente una de las
ventajas del pipeline sobre el notebook que se explica en el notebook 04.

Estos tests son el contrato entre el pipeline y la tabla de decisiones de la EA1:
si alguien cambia una decisión de limpieza, aquí se entera de qué pauta quedó
desalineada.
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

from kedro_mly1101.pipelines.calidad import nodes as calidad  # noqa: E402
from kedro_mly1101.pipelines.preprocesamiento import nodes as prep  # noqa: E402


@pytest.fixture
def sucio() -> pd.DataFrame:
    """Mini-dataset que reproduce los defectos que el pipeline debe tratar."""
    df = pd.DataFrame(
        {
            "segment_id": ["seg_0001"] * 6,
            "timestamp_micros": ["100", "200", "N/D", "400", "500", "500"],
            # det_4 se repite a proposito: las dos ultimas filas son un
            # duplicado exacto, y esa es la unica clase que se elimina.
            "id_interno": ["det_0", "det_1", "det_2", "det_3", "det_4", "det_4"],
            "object_type": ["VEHICLE", "PEATON", "Ped", "PEDESTRIAN", "CYCLIST", "CYCLIST"],
            "weather": ["sunny", "lluvia", "RAIN ", "Sunny", "niebla", "niebla"],
            "time_of_day": ["Day", "Day", "Night", "Night", "Day", "Day"],
            "detection_difficulty": ["LEVEL_1"] * 6,
            "box_height": [1.7, 0.0, 1.8, 1.6, -1.0, -1.0],
            "box_length": [4.4, 0.9, -2.0, 0.8, 1.9, 1.9],
            "box_width": [1.6, 0.7, 0.8, 0.7, 0.6, 0.6],
            "speed_mps": [2.0, np.nan, 3.0, 340.0, 5.0, 5.0],
            "num_lidar_points": [40, -1, 30, 25, -1, -1],
            "sensor_version": ["v2.0.1"] * 6,
        }
    )
    return df


MAPAS = {
    "object_type": {"peaton": "pedestrian", "ped": "pedestrian"},
    "weather": {"lluvia": "rain", "soleado": "sunny", "niebla": "fog"},
}
REGLAS = {
    "altura nula o negativa": {"condicion": "box_height <= 0", "columna": "box_height"},
    "largo negativo": {"condicion": "box_length < 0", "columna": "box_length"},
    "velocidad superior a 100 m/s": {"condicion": "speed_mps > 100", "columna": "speed_mps"},
    "puntos laser negativos": {"condicion": "num_lidar_points < 0", "columna": "num_lidar_points"},
}


# --- Nodos de calidad --------------------------------------------------------

def test_diagnosticar_detecta_los_nulos_ocultos(sucio: pd.DataFrame) -> None:
    """El centinela -1 debe contarse como faltante aunque no sea NaN."""
    resumen = calidad.diagnosticar(sucio)
    assert resumen.loc["num_lidar_points", "n_centinelas"] == 3
    assert resumen.loc["num_lidar_points", "n_nulos"] == 0


def test_diagnosticar_marca_la_columna_constante(sucio: pd.DataFrame) -> None:
    resumen = calidad.diagnosticar(sucio)
    assert resumen.loc["sensor_version", "n_unicos"] == 1


def test_medir_desbalance_unifica_antes_de_contar(sucio: pd.DataFrame) -> None:
    """Sin normalizar habría 5 categorías; con el mapa, 3."""
    tabla = calidad.medir_desbalance(sucio, MAPAS)
    assert set(tabla.index) == {"vehicle", "pedestrian", "cyclist"}
    assert tabla.loc["pedestrian", "n"] == 3


def test_auditar_dominio_cuenta_cada_regla(sucio: pd.DataFrame) -> None:
    tabla = calidad.auditar_dominio(sucio, REGLAS).set_index("regla")
    assert tabla.loc["altura nula o negativa", "n_filas"] == 3
    assert tabla.loc["largo negativo", "n_filas"] == 1
    assert tabla.loc["velocidad superior a 100 m/s", "n_filas"] == 1
    assert tabla.loc["puntos laser negativos", "n_filas"] == 3


def test_medir_sesgo_cruza_por_grupo(sucio: pd.DataFrame) -> None:
    tabla = calidad.medir_sesgo(sucio, {"columna": "speed_mps", "grupos": ["time_of_day"]})
    assert tabla.loc["Day", "pct_nulos"] == pytest.approx(25.0)
    assert tabla.loc["Night", "pct_nulos"] == pytest.approx(0.0)


# --- Nodos de preprocesamiento ----------------------------------------------

def test_normalizar_unifica_las_variantes(sucio: pd.DataFrame) -> None:
    limpio = prep.normalizar_categorias(sucio, MAPAS)
    assert set(limpio["object_type"].unique()) == {"vehicle", "pedestrian", "cyclist"}
    assert set(limpio["weather"].unique()) == {"sunny", "rain", "fog"}


def test_normalizar_no_pierde_filas(sucio: pd.DataFrame) -> None:
    assert len(prep.normalizar_categorias(sucio, MAPAS)) == len(sucio)


def test_descubrir_faltantes_convierte_el_centinela(sucio: pd.DataFrame) -> None:
    limpio = prep.descubrir_faltantes(sucio, {"num_lidar_points": -1})
    assert limpio["num_lidar_points"].isna().sum() == 3
    assert (limpio["num_lidar_points"].dropna() > 0).all()


def test_descubrir_faltantes_arregla_la_marca_de_tiempo(sucio: pd.DataFrame) -> None:
    """El "N/D" impedía que la columna fuera numérica."""
    assert sucio["timestamp_micros"].dtype == object
    limpio = prep.descubrir_faltantes(sucio, {})
    assert pd.api.types.is_numeric_dtype(limpio["timestamp_micros"])
    assert limpio["timestamp_micros"].isna().sum() == 1


def test_marcar_imposibles_no_elimina_filas(sucio: pd.DataFrame) -> None:
    """El principio del modulo: marcar, no borrar. El resto de la fila era valido."""
    limpio = prep.marcar_imposibles(sucio, REGLAS)
    assert len(limpio) == len(sucio)
    assert limpio["box_height"].isna().sum() == 3
    assert limpio["box_length"].isna().sum() == 1


def test_marcar_imposibles_respeta_los_atipicos_legitimos() -> None:
    """Un bus de 15 m es atipico y legitimo: ninguna regla debe tocarlo."""
    buses = pd.DataFrame({"box_length": [4.4, 15.0, 18.0], "box_height": [1.6, 3.2, 3.4]})
    limpio = prep.marcar_imposibles(
        buses, {"largo negativo": {"condicion": "box_length < 0", "columna": "box_length"}}
    )
    assert limpio["box_length"].notna().all()
    assert limpio["box_length"].max() == 18.0


def test_quitar_duplicados_y_constantes(sucio: pd.DataFrame) -> None:
    limpio = prep.quitar_duplicados_y_constantes(sucio, ["sensor_version"])
    assert len(limpio) == len(sucio) - 1          # una fila exactamente repetida
    assert "sensor_version" not in limpio.columns


def test_quitar_constantes_ignora_columnas_ausentes(sucio: pd.DataFrame) -> None:
    """Pedir descartar algo que no esta no puede reventar el pipeline."""
    limpio = prep.quitar_duplicados_y_constantes(sucio, ["no_existe"])
    assert limpio.shape[1] == sucio.shape[1]


def test_resumir_limpieza_reporta_el_aumento_de_faltantes(sucio: pd.DataFrame) -> None:
    """Contraintuitivo y correcto: al limpiar, los faltantes SUBEN.

    No aparecen faltantes nuevos: los que estaban disfrazados de -1 o de "N/D"
    pasan a contarse. Si esta cifra bajara, seria senal de que se eliminaron
    filas en vez de marcarlas.
    """
    limpio = prep.descubrir_faltantes(sucio, {"num_lidar_points": -1})
    informe = prep.resumir_limpieza(sucio, limpio).set_index("metrica")
    assert informe.loc["celdas faltantes", "diferencia"] > 0


def test_el_pipeline_completo_es_coherente(sucio: pd.DataFrame) -> None:
    """Encadena los cuatro nodos como lo hace pipeline.py y revisa el resultado."""
    paso = prep.normalizar_categorias(sucio, MAPAS)
    paso = prep.descubrir_faltantes(paso, {"num_lidar_points": -1})
    paso = prep.marcar_imposibles(paso, REGLAS)
    limpio = prep.quitar_duplicados_y_constantes(paso, ["sensor_version"])

    assert set(limpio["object_type"].unique()) == {"vehicle", "pedestrian", "cyclist"}
    assert "sensor_version" not in limpio.columns
    assert (limpio["box_height"].dropna() > 0).all()
    assert (limpio["speed_mps"].dropna() <= 100).all()


def test_las_pipelines_de_la_ea1_se_registran_y_su_grafo_es_valido() -> None:
    """Que las dos pipelines de la EA1 construyan sin ciclos ni cabos sueltos.

    El grafo COMPLETO (con la de EA2 encadenada) se comprueba en
    ``tests/test_pipeline_supervisado.py``; aqui solo la parte de datos.
    """
    from kedro_mly1101.pipeline_registry import register_pipelines

    pipelines = register_pipelines()
    assert {"calidad", "preprocesamiento"} <= set(pipelines)

    ea1 = pipelines["calidad"] + pipelines["preprocesamiento"]
    assert len(ea1.nodes) == 9

    # Lo unico que espera de fuera son los datos crudos y los parametros.
    entradas = {e for e in ea1.inputs() if not e.startswith("params:")}
    assert entradas == {"detecciones_crudas"}

    # ``outputs()`` devuelve solo las salidas LIBRES: ``detecciones_limpias`` no
    # esta ahi porque la consume ``resumir_la_limpieza``. Que sea intermedia no le
    # quita valor: esta en el catalogo, asi que se persiste igual.
    producidos = {salida for nodo in ea1.nodes for salida in nodo.outputs}
    assert "detecciones_limpias" in producidos
    assert "informe_limpieza" in ea1.outputs()


def test_una_regla_con_dos_columnas_marca_la_columna_declarada() -> None:
    """El defecto que motivó declarar la columna: deducirla del texto fallaba.

    Con ``columna`` deducida de la condicion, esta regla habria marcado
    ``num_lidar_points`` y dejado ``speed_mps`` sucia sin avisar.
    """
    df = pd.DataFrame({"num_lidar_points": [10, -1, 20], "speed_mps": [1.0, 2.0, 340.0]})
    regla = {
        "sensor fuera de rango": {
            "condicion": "num_lidar_points < 0 or speed_mps > 100",
            "columna": "speed_mps",
        }
    }
    limpio = prep.marcar_imposibles(df, regla)
    assert limpio["speed_mps"].isna().sum() == 2      # las dos filas que violan la regla
    assert limpio["num_lidar_points"].notna().all()   # esta columna no se tocó
