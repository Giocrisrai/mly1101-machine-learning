"""El mapa de integraciones no puede volver a decir que Databricks es conceptual o el EFT falta."""

from __future__ import annotations

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def test_existe_el_mapa_unico() -> None:
    texto = (RAIZ / "docs" / "integraciones.md").read_text(encoding="utf-8")
    for fragmento in (
        "Google Colab",
        "GitHub Actions",
        "AWS Academy",
        "Databricks Free Edition",
        "MessageError",
        "datos/evaluaciones",
        "LIST 0 filas",
        "Instrumento Duoc",
    ):
        assert fragmento in texto, fragmento
    evals = (RAIZ / "docs" / "evaluaciones.md").read_text(encoding="utf-8")
    assert "forma inicial" in evals
    assert "Perception v2" in evals


def test_kedro_readme_no_deja_evaluaciones_pendientes() -> None:
    texto = (RAIZ / "kedro_mly1101" / "README.md").read_text(encoding="utf-8")
    assert "Lo que sigue pendiente son las evaluaciones" not in texto
    assert "docs/evaluaciones.md" in texto


def test_contenido_kedro_ya_no_es_conceptual() -> None:
    texto = (RAIZ / "herramientas" / "contenido_kedro.py").read_text(encoding="utf-8")
    assert "conceptual por diseño" not in texto
    assert "notebook **15**" in texto
    assert "⏳" not in texto


def test_notebook_04_no_marca_eft_pendiente() -> None:
    ruta = RAIZ / "notebooks" / "04_opcional_kedro_databricks.ipynb"
    assert ruta.exists(), "regenerar con python herramientas/construir_notebooks.py"
    texto = ruta.read_text(encoding="utf-8")
    assert "⏳" not in texto
    assert "conceptual por diseño" not in texto


def test_auth_colab_documentada_en_el_notebook_14() -> None:
    fuente = (RAIZ / "herramientas" / "contenido_waymo_buckets.py").read_text(
        encoding="utf-8"
    )
    assert "MessageError" in fuente
    assert "Chrome" in fuente


def test_spec_antigua_ya_no_deja_databricks_conceptual() -> None:
    texto = (
        RAIZ / "docs" / "superpowers" / "specs" / "2026-08-26-mly1101-actividades-11-12-design.md"
    ).read_text(encoding="utf-8")
    assert "conceptual por diseño" not in texto
    assert "cae al CSV local" not in texto
