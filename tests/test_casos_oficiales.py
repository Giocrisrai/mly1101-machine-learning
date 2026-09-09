"""Tests del desempaquetado de los casos oficiales (Telco / Housing / Spotify)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "herramientas"))

from preparar_casos_oficiales import (  # noqa: E402
    CASOS,
    preparar,
)


def _zip_parciales_minimo(tmp: Path) -> Path:
    """Un zip con la misma jerarquía que ``EV PARCIALES MLY1101`` pero CSV chicos."""
    raiz = tmp / "EV PARCIALES MLY1101" / "PARCIAL 1"
    for caso, meta in CASOS.items():
        carpeta = raiz / meta["zip_interno"].replace(".zip", "")
        # no: the official layout is PARCIAL 1/Telco_Dataset.zip containing Telco_Dataset/file.csv
    dest = tmp / "EV PARCIALES MLY1101.zip"
    with zipfile.ZipFile(dest, "w") as externo:
        for caso, meta in CASOS.items():
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as interno:
                interno.writestr(
                    f"{meta['carpeta_zip']}/{meta['csv']}",
                    "col_a,col_b\n1,2\n",
                )
            externo.writestr(
                f"EV PARCIALES MLY1101/PARCIAL 1/{meta['zip_interno']}",
                buf.getvalue(),
            )
    return dest


def test_catalogo_tiene_los_tres_casos() -> None:
    assert set(CASOS) == {"telco", "housing", "spotify"}


def test_preparar_desempaqueta_los_tres_csv(tmp_path: Path) -> None:
    zip_oficial = _zip_parciales_minimo(tmp_path)
    destino = tmp_path / "datos" / "evaluaciones"
    rutas = preparar(destino, zip_oficial)
    assert set(rutas) == {"telco", "housing", "spotify"}
    for caso, ruta in rutas.items():
        assert ruta.exists()
        assert ruta.name == CASOS[caso]["csv"]
        assert ruta.read_text(encoding="utf-8").startswith("col_a")


def test_falla_si_no_hay_zip(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="EV PARCIALES"):
        preparar(tmp_path / "out", tmp_path / "no-existe.zip")
