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


# Identificadores / texto que no pueden entrar a X (fuga, cardinalidad o ID).
_IDS: dict[str, tuple[str, ...]] = {
    "telco": ("customerID",),
    "housing": ("Order", "PID"),
    "spotify": ("Unnamed: 0", "track_id", "artists", "album_name", "track_name"),
}
_GRUPO: dict[str, str | None] = {
    "telco": None,
    "housing": None,
    "spotify": "album_name",
}


def matriz_xy(
    tabla: pd.DataFrame,
    nombre: str,
    muestra: int | None = None,
    semilla: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    """Arma ``X``, ``y`` y un grupo opcional para el split.

    No inventa filas: parte de ``tabla`` ya cargada. Imputación mínima del
    esqueleto (mediana / ``faltante``): el equipo la justifica en la EP1.

    Args:
        tabla: CSV oficial (o un recorte de prueba).
        nombre: ``telco``, ``housing`` o ``spotify``.
        muestra: si se indica, recorta a esas filas (esqueleto Colab / Spotify).
        semilla: para el recorte.

    Returns:
        ``X`` numérico sin NaN, ``y`` (0/1 en Telco; continuo en los otros) y
        ``grupo`` (álbum en Spotify; ``None`` si el split puede ser por fila).
    """
    if nombre not in CASOS:
        ruta_del_caso(nombre)

    trabajo = tabla.copy()
    objetivo = str(CASOS[nombre]["objetivo"])
    if objetivo not in trabajo.columns:
        raise ValueError(f"{nombre}: falta la columna objetivo {objetivo!r}")

    if nombre == "telco" and "TotalCharges" in trabajo.columns:
        cobros = pd.to_numeric(trabajo["TotalCharges"], errors="coerce")
        if "tenure" in trabajo.columns:
            cobros = cobros.mask(trabajo["tenure"] == 0, 0.0)
        trabajo["TotalCharges"] = cobros

    if nombre == "telco":
        y = (trabajo[objetivo].astype(str) == "Yes").astype(int)
    else:
        y = pd.to_numeric(trabajo[objetivo], errors="coerce")

    nombre_grupo = _GRUPO[nombre]
    grupo = trabajo[nombre_grupo].copy() if nombre_grupo and nombre_grupo in trabajo.columns else None

    drop = {objetivo, *_IDS[nombre]}
    X = trabajo.drop(columns=[c for c in drop if c in trabajo.columns])

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]) or pd.api.types.is_bool_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].astype("string").fillna("faltante")

    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0)

    if muestra is not None:
        if muestra < 1:
            raise ValueError("muestra debe ser ≥ 1")
        n = min(muestra, len(X))
        idx = X.sample(n=n, random_state=semilla).index
        X = X.loc[idx]
        y = y.loc[idx]
        if grupo is not None:
            grupo = grupo.loc[idx]

    y.name = objetivo
    if grupo is not None:
        grupo.name = nombre_grupo
    return X, y, grupo

