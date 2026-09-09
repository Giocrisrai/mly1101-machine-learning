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
    assert "mkdir" in texto
    docente = (RAIZ / "notebooks" / "15_docente_evaluacion.ipynb").read_text(encoding="utf-8")
    assert "álbumes enteros" in docente
    assert "make_pipeline" in docente
    assert "GroupKFold" in docente


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


def test_spotify_album_nan_no_rompe_el_split() -> None:
    tabla = pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2],
            "track_id": ["t1", "t2", "t3"],
            "artists": ["A", "A", "A"],
            "album_name": ["Alb1", None, "Alb2"],
            "track_name": ["s1", "s2", "s3"],
            "popularity": [10, 20, 30],
            "danceability": [0.2, 0.3, 0.4],
            "energy": [0.1, 0.2, 0.3],
            "track_genre": ["rock", "rock", "rock"],
        }
    )
    X, y, grupo = casos.matriz_xy(tabla, "spotify")
    assert grupo is not None
    assert int(grupo.isna().sum()) == 0
    assert len(X) == 2
    from sklearn.model_selection import GroupShuffleSplit

    next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=0).split(X, y, grupo))


def test_muestra_telco_corta_por_fila() -> None:
    tabla = pd.DataFrame(
        {
            "customerID": list("abcde"),
            "tenure": [1, 2, 3, 4, 5],
            "MonthlyCharges": [10.0] * 5,
            "TotalCharges": ["10", "20", "30", "40", "50"],
            "gender": ["Female"] * 5,
            "Churn": ["No", "Yes", "No", "Yes", "No"],
        }
    )
    X, y, grupo = casos.matriz_xy(tabla, "telco", muestra=3, semilla=0)
    assert len(X) == 3
    assert len(y) == 3
    assert grupo is None


def test_muestra_spotify_no_parte_albumes() -> None:
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
    X, y, grupo = casos.matriz_xy(tabla, "spotify", muestra=8, semilla=0)
    assert 5 <= len(X) <= 10
    assert len(y) == len(X)
    assert grupo is not None
    for album in grupo.unique():
        assert int((grupo == album).sum()) == 5


def test_muestra_conserva_el_primer_album_aunque_supere_n() -> None:
    tabla = pd.DataFrame(
        {
            "Unnamed: 0": range(6),
            "track_id": [f"t{i}" for i in range(6)],
            "artists": ["A"] * 6,
            "album_name": ["unico"] * 6,
            "track_name": [f"s{i}" for i in range(6)],
            "popularity": list(range(6)),
            "danceability": [0.5] * 6,
            "energy": [0.5] * 6,
            "track_genre": ["pop"] * 6,
        }
    )
    X, y, grupo = casos.matriz_xy(tabla, "spotify", muestra=3, semilla=0)
    assert len(X) == 6
    assert grupo is not None
    assert grupo.nunique() == 1


@pytest.mark.parametrize("nombre,forma", [("telco", (7043, 30)), ("housing", (2930, 276))])
def test_matriz_oficial_sin_nan_si_esta_el_csv(nombre: str, forma: tuple[int, int]) -> None:
    ruta = casos.ruta_del_caso(nombre, RAIZ)
    if not ruta.exists():
        pytest.skip(f"sin zip institucional ({nombre})")
    X, y, _grupo = casos.matriz_xy(casos.cargar_caso(nombre, RAIZ), nombre)
    assert X.shape == forma
    assert X.isna().sum().sum() == 0
    assert str(casos.CASOS[nombre]["objetivo"]) not in X.columns
    assert len(y) == forma[0]


def test_spotify_oficial_muestra_no_parte_albumes() -> None:
    ruta = casos.ruta_del_caso("spotify", RAIZ)
    if not ruta.exists():
        pytest.skip("sin zip institucional (spotify)")
    tabla = casos.cargar_caso("spotify", RAIZ)
    X, y, grupo = casos.matriz_xy(tabla, "spotify", muestra=8000, semilla=42)
    assert grupo is not None
    assert X.isna().sum().sum() == 0
    assert "popularity" not in X.columns
    assert len(X) <= 8000 or grupo.nunique() == 1
    originales = tabla["album_name"].value_counts()
    for album, n in grupo.value_counts().items():
        assert int(n) == int(originales[album])


def test_spotify_oficial_grupo_sin_nan_en_tabla_completa() -> None:
    ruta = casos.ruta_del_caso("spotify", RAIZ)
    if not ruta.exists():
        pytest.skip("sin zip institucional (spotify)")
    tabla = casos.cargar_caso("spotify", RAIZ)
    X, y, grupo = casos.matriz_xy(tabla, "spotify")
    assert grupo is not None
    assert int(grupo.isna().sum()) == 0
    assert X.shape == (113999, 127)
    from sklearn.model_selection import GroupShuffleSplit

    next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(X, y, grupo))
