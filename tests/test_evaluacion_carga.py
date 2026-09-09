"""Carga de los casos oficiales (Telco / Housing / Spotify) para las evaluaciones.

Sin CSV no hay fallback sintético: el código debe fallar y decir cómo obtener
el zip de coordinación.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import casos  # noqa: E402


def test_caso_desconocido_no_es_waymo() -> None:
    with pytest.raises(ValueError, match="telco"):
        casos.cargar_caso("waymo")


def test_cargar_caso_falla_sin_csv(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="preparar_casos_oficiales"):
        casos.cargar_caso("telco", raiz=tmp_path)


def test_cargar_caso_lee_un_csv_en_la_ruta_oficial(tmp_path: Path) -> None:
    carpeta = tmp_path / "datos" / "evaluaciones" / "telco"
    carpeta.mkdir(parents=True)
    (carpeta / "Telco_Customer_Churn_Dataset.csv").write_text(
        "customerID,Churn\n0001,Yes\n", encoding="utf-8"
    )
    tabla = casos.cargar_caso("telco", raiz=tmp_path)
    assert list(tabla.columns) == ["customerID", "Churn"]
    assert len(tabla) == 1


def test_catalogo_son_los_tres_casos_oficiales() -> None:
    assert set(casos.CASOS) == {"telco", "housing", "spotify"}
    assert casos.CASOS["telco"]["objetivo"] == "Churn"
    assert casos.CASOS["housing"]["objetivo"] == "SalePrice"
    assert casos.CASOS["spotify"]["objetivo"] == "popularity"


def test_cargar_telco_oficial_si_esta_en_disco() -> None:
    ruta = RAIZ / "datos" / "evaluaciones" / "telco" / "Telco_Customer_Churn_Dataset.csv"
    if not ruta.exists():
        pytest.skip("sin zip institucional en datos/evaluaciones/")
    tabla = casos.cargar_caso("telco", raiz=RAIZ)
    assert tabla.shape == (7043, 21)
    assert "Churn" in tabla.columns


def test_nombres_csv_alineados_con_el_unpacker() -> None:
    from preparar_casos_oficiales import CASOS as del_zip

    for nombre, meta in casos.CASOS.items():
        assert meta["csv"] == del_zip[nombre]["csv"]


def test_notebook_alumno_no_filtra_la_pauta() -> None:
    ruta = RAIZ / "notebooks" / "15_alumno_evaluacion.ipynb"
    assert ruta.exists(), "regenerar con python herramientas/construir_notebooks.py"
    texto = ruta.read_text(encoding="utf-8")
    assert "Pauta docente" not in texto
    assert "cargar_caso" in texto
    assert "TODO 1" in texto
    assert "matriz_xy" in texto
    assert "SystemExit" not in texto


def test_matriz_telco_quita_id_y_pasa_totalcharges_a_numero() -> None:
    tabla = pd.DataFrame(
        {
            "customerID": ["a", "b"],
            "tenure": [0, 10],
            "MonthlyCharges": [20.0, 50.0],
            "TotalCharges": [" ", "200.5"],
            "gender": ["Female", "Male"],
            "Churn": ["No", "Yes"],
        }
    )
    X, y, grupo = casos.matriz_xy(tabla, "telco")
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert y.tolist() == [0, 1]
    assert float(X["TotalCharges"].iloc[0]) == 0.0
    assert X.isna().sum().sum() == 0
    assert grupo is None


def test_matriz_housing_quita_identificadores() -> None:
    tabla = pd.DataFrame(
        {
            "Order": [1, 2],
            "PID": [11, 22],
            "Lot Frontage": [70.0, None],
            "Neighborhood": ["NAmes", "NAmes"],
            "SalePrice": [120000, 180000],
        }
    )
    X, y, grupo = casos.matriz_xy(tabla, "housing")
    assert "PID" not in X.columns
    assert "Order" not in X.columns
    assert "SalePrice" not in X.columns
    assert y.tolist() == [120000, 180000]
    assert X.isna().sum().sum() == 0


def test_matriz_spotify_no_deja_popularity_ni_ids_en_x() -> None:
    tabla = pd.DataFrame(
        {
            "Unnamed: 0": [0, 1],
            "track_id": ["t1", "t2"],
            "artists": ["A", "B"],
            "album_name": ["Alb1", "Alb1"],
            "track_name": ["s1", "s2"],
            "popularity": [10, 90],
            "danceability": [0.2, 0.8],
            "energy": [0.1, 0.9],
            "track_genre": ["rock", "rock"],
        }
    )
    X, y, grupo = casos.matriz_xy(tabla, "spotify")
    assert "popularity" not in X.columns
    assert "track_id" not in X.columns
    assert "Unnamed: 0" not in X.columns
    assert y.tolist() == [10, 90]
    assert grupo is not None
    assert list(grupo) == ["Alb1", "Alb1"]


def test_muestra_recorta_sin_inventar_filas() -> None:
    tabla = pd.DataFrame(
        {
            "Unnamed: 0": range(20),
            "track_id": [f"t{i}" for i in range(20)],
            "artists": ["A"] * 20,
            "album_name": [f"alb{i // 5}" for i in range(20)],
            "track_name": [f"s{i}" for i in range(20)],
            "popularity": list(range(20)),
            "danceability": [0.5] * 20,
            "energy": [0.5] * 20,
            "track_genre": ["pop"] * 20,
        }
    )
    X, y, _grupo = casos.matriz_xy(tabla, "spotify", muestra=8, semilla=0)
    assert len(X) == 8
    assert len(y) == 8
