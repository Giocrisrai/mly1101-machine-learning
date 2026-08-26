"""Tests de la comparación de formatos de src/formatos.py (Act 1.2)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import formatos  # noqa: E402


@pytest.fixture
def df() -> pd.DataFrame:
    """DataFrame con tipos que el CSV **no** sabe conservar."""
    base = pd.DataFrame(
        {
            "segment_id": [f"seg_{i:04d}" for i in range(200)],
            "object_type": ["VEHICLE", "PEDESTRIAN", "CYCLIST", "SIGN"] * 50,
            "speed_mps": [float(i) / 3 for i in range(200)],
            "num_lidar_points": list(range(200)),
        }
    )
    base["object_type"] = base["object_type"].astype("category")
    base["speed_mps"] = base["speed_mps"].astype("float32")
    return base


@pytest.fixture
def df_grande() -> pd.DataFrame:
    """Mismo esquema que ``df`` pero con volumen: 20.000 filas.

    Parquet comprime por columna, así que necesita repetición para lucirse. Es
    el mismo motivo por el que en el notebook la medición se hace sobre una
    muestra grande y no sobre ``df.head()``.
    """
    base = pd.DataFrame(
        {
            "segment_id": [f"seg_{i % 800:04d}" for i in range(20_000)],
            "object_type": ["VEHICLE", "PEDESTRIAN", "CYCLIST", "SIGN"] * 5_000,
            "speed_mps": [float(i) / 3 for i in range(20_000)],
            "num_lidar_points": list(range(20_000)),
        }
    )
    base["object_type"] = base["object_type"].astype("category")
    base["speed_mps"] = base["speed_mps"].astype("float32")
    return base


def test_medir_formatos_devuelve_una_fila_por_formato(df, tmp_path) -> None:
    tabla = formatos.medir_formatos(df, tmp_path, formatos=("csv", "json"))
    assert tabla["formato"].tolist() == sorted(
        ["csv", "json"], key=lambda f: tabla.set_index("formato").loc[f, "bytes"]
    )
    assert len(tabla) == 2


def test_la_tabla_trae_todas_las_columnas_que_usa_el_notebook(df, tmp_path) -> None:
    tabla = formatos.medir_formatos(df, tmp_path, formatos=("csv",))
    assert list(tabla.columns) == [
        "formato",
        "archivo",
        "bytes",
        "mb",
        "veces_vs_csv",
        "seg_escritura",
        "seg_lectura",
        "conserva_dtypes",
    ]


def test_los_archivos_quedan_escritos_en_la_carpeta(df, tmp_path) -> None:
    formatos.medir_formatos(df, tmp_path, formatos=("csv", "json"))
    assert (tmp_path / "muestra.csv").exists()
    assert (tmp_path / "muestra.json").exists()


def test_la_tabla_viene_ordenada_de_menor_a_mayor_peso(df, tmp_path) -> None:
    tabla = formatos.medir_formatos(df, tmp_path, formatos=("csv", "json"))
    assert tabla["bytes"].is_monotonic_increasing


def test_veces_vs_csv_es_uno_para_el_propio_csv(df, tmp_path) -> None:
    tabla = formatos.medir_formatos(df, tmp_path, formatos=("csv", "json"))
    fila_csv = tabla.set_index("formato").loc["csv"]
    assert fila_csv["veces_vs_csv"] == 1.0


def test_sin_csv_la_columna_de_referencia_queda_nula(df, tmp_path) -> None:
    tabla = formatos.medir_formatos(df, tmp_path, formatos=("json",))
    assert pd.isna(tabla["veces_vs_csv"]).all()


def test_el_csv_pierde_los_tipos_y_el_parquet_no(df, tmp_path) -> None:
    """El hallazgo central de la actividad, verificado."""
    pytest.importorskip("pyarrow")
    tabla = formatos.medir_formatos(df, tmp_path, formatos=("csv", "parquet"))
    por_formato = tabla.set_index("formato")["conserva_dtypes"]
    assert por_formato["csv"] is False or not por_formato["csv"]
    assert por_formato["parquet"]


def test_el_parquet_pesa_menos_que_el_csv_cuando_hay_volumen(df_grande, tmp_path) -> None:
    """La ventaja de Parquet es real, pero solo se ve con suficientes filas."""
    pytest.importorskip("pyarrow")
    tabla = formatos.medir_formatos(
        df_grande, tmp_path, formatos=("csv", "parquet")
    ).set_index("formato")
    assert tabla.loc["parquet", "bytes"] < tabla.loc["csv", "bytes"]


def test_con_pocas_filas_el_parquet_puede_pesar_mas(df, tmp_path) -> None:
    """Matiz que el notebook debe declarar: el encabezado de Parquet no es gratis.

    Con 200 filas, los metadatos del formato columnar (esquema, estadísticas por
    grupo de filas) pesan más que la compresión que consiguen. Si la clase mide
    con una muestra minúscula y ve que Parquet "pierde", la explicación es esta,
    no un error suyo.
    """
    pytest.importorskip("pyarrow")
    tabla = formatos.medir_formatos(df, tmp_path, formatos=("csv", "parquet")).set_index(
        "formato"
    )
    assert tabla.loc["parquet", "bytes"] > tabla.loc["csv", "bytes"]


def test_dtypes_conservados_detecta_la_igualdad(df) -> None:
    assert formatos.dtypes_conservados(df, df.copy())


def test_dtypes_conservados_detecta_un_cambio_de_tipo(df) -> None:
    otro = df.copy()
    otro["speed_mps"] = otro["speed_mps"].astype("float64")
    assert not formatos.dtypes_conservados(df, otro)


def test_dtypes_conservados_detecta_columnas_faltantes(df) -> None:
    assert not formatos.dtypes_conservados(df, df.drop(columns=["speed_mps"]))


def test_columnas_con_dtype_cambiado_lista_solo_las_que_cambiaron(df) -> None:
    otro = df.copy()
    otro["speed_mps"] = otro["speed_mps"].astype("float64")
    detalle = formatos.columnas_con_dtype_cambiado(df, otro)
    assert detalle["columna"].tolist() == ["speed_mps"]
    assert detalle.iloc[0]["dtype_original"] == "float32"
    assert detalle.iloc[0]["dtype_releido"] == "float64"


def test_columnas_con_dtype_cambiado_vacio_si_no_hubo_cambios(df) -> None:
    detalle = formatos.columnas_con_dtype_cambiado(df, df.copy())
    assert detalle.empty
    assert list(detalle.columns) == ["columna", "dtype_original", "dtype_releido"]


def test_formato_desconocido_es_un_error_explicito(df, tmp_path) -> None:
    with pytest.raises(KeyError):
        formatos.medir_formatos(df, tmp_path, formatos=("avro",))


def test_excel_rechaza_un_dataframe_que_excede_su_limite_de_filas(tmp_path) -> None:
    """No se escriben dos millones de filas: se comprueba el guardarraíl."""
    class FingeSerMuyLargo(pd.DataFrame):
        @property
        def _constructor(self):
            return FingeSerMuyLargo

        def __len__(self) -> int:
            return formatos.LIMITE_FILAS_EXCEL + 1

    largo = FingeSerMuyLargo({"a": [1, 2, 3]})
    with pytest.raises(formatos.FormatoNoDisponible, match="Excel admite hasta"):
        formatos._escribir(largo, "excel", tmp_path / "x.xlsx")


def test_memoria_lista_python_supera_a_la_del_ndarray() -> None:
    """La afirmación del PPT 1.2, medida."""
    import numpy as np

    valores = [float(i) for i in range(5_000)]
    arreglo = np.array(valores, dtype="float64")
    assert formatos.memoria_lista_python(valores) > arreglo.nbytes
