"""Carga de los casos oficiales de las evaluaciones (Telco / Housing / Spotify).

Funciones puras: leen un CSV y devuelven un DataFrame. No imprimen ni grafican.

Los archivos **no** viven en git: salen de ``EV PARCIALES MLY1101.zip``
(coordinación) vía ``herramientas/preparar_casos_oficiales.py``. Si el CSV no
está, se levanta ``FileNotFoundError``. No hay filas sintéticas.

Las Act. 1.1–3.3 y ``kedro run`` siguen en Perception v2. Este módulo es solo
para las parciales y el EFT.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAIZ_POR_DEFECTO = Path(__file__).resolve().parents[1]

# Formas medidas 2026-09-09 sobre el zip institucional. No redondear de memoria.
CASOS: dict[str, dict[str, str | int]] = {
    "telco": {
        "csv": "Telco_Customer_Churn_Dataset.csv",
        "objetivo": "Churn",
        "tipo": "clasificacion",
        "filas": 7043,
        "columnas": 21,
    },
    "housing": {
        "csv": "Ames_Iowa_Housing_Dataset.csv",
        "objetivo": "SalePrice",
        "tipo": "regresion",
        "filas": 2930,
        "columnas": 82,
    },
    "spotify": {
        "csv": "Spotify_Tracks_Dataset.csv",
        "objetivo": "popularity",
        "tipo": "regresion",
        "filas": 114000,
        "columnas": 21,
    },
}


def ruta_del_caso(nombre: str, raiz: Path | None = None) -> Path:
    """Ruta gitignored ``datos/evaluaciones/{caso}/{csv}``."""
    if nombre not in CASOS:
        conocidos = ", ".join(sorted(CASOS))
        raise ValueError(
            f"caso desconocido: {nombre!r}. Los oficiales son {conocidos} "
            "(no Waymo: las evaluaciones no mezclan el hilo de las actividades)."
        )
    base = Path(raiz) if raiz is not None else RAIZ_POR_DEFECTO
    return base / "datos" / "evaluaciones" / nombre / str(CASOS[nombre]["csv"])


def cargar_caso(nombre: str, raiz: Path | None = None) -> pd.DataFrame:
    """Lee el CSV oficial del caso.

    Raises:
        ValueError: si el nombre no es telco, housing o spotify.
        FileNotFoundError: si falta el archivo (sin fallback).
    """
    ruta = ruta_del_caso(nombre, raiz)
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No está {ruta}.\n"
            "  Corre: uv run python herramientas/preparar_casos_oficiales.py\n"
            "  El zip lo entrega la coordinación (EV PARCIALES MLY1101).\n"
            "  No hay CSV sintético: sin el archivo oficial no hay evaluación."
        )
    return pd.read_csv(ruta)
