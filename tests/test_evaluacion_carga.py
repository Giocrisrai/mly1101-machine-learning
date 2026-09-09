"""Carga de los casos oficiales (Telco / Housing / Spotify) para las evaluaciones.

Sin CSV no hay fallback sintético: el código debe fallar y decir cómo obtener
el zip de coordinación.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
