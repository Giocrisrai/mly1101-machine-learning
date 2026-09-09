"""Desempaqueta Telco / Housing / Spotify a ``datos/evaluaciones/`` (gitignored).

Los CSV oficiales **no** se versionan: vienen en ``EV PARCIALES MLY1101.zip``
(coordinación) y en el anexo del EFT. Este script los deja en disco para las
parciales y el EFT, igual que ``descargar_waymo.py`` deja Perception v2 fuera
de git.

Uso:

    uv run python herramientas/preparar_casos_oficiales.py
    uv run python herramientas/preparar_casos_oficiales.py \\
        --zip ~/Downloads/EV\\ PARCIALES\\ MLY1101.zip.zip

Cifras medidas 2026-09-09 (no redondear de memoria):

| Caso | Archivo | Filas × cols |
|---|---|---|
| Telco Churn | ``Telco_Customer_Churn_Dataset.csv`` | 7.043 × 21 |
| House Prices | ``Ames_Iowa_Housing_Dataset.csv`` | 2.930 × 82 |
| Spotify Tracks | ``Spotify_Tracks_Dataset.csv`` | 114.000 × 21 |
"""

from __future__ import annotations

import argparse
import os
import tempfile
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DESTINO_POR_DEFECTO = RAIZ / "datos" / "evaluaciones"

# Layout del zip institucional (PARCIAL 1 = PARCIAL 2 = PARCIAL 3 = anexo EFT).
CASOS: dict[str, dict[str, str]] = {
    "telco": {
        "zip_interno": "Telco_Dataset.zip",
        "carpeta_zip": "Telco_Dataset",
        "csv": "Telco_Customer_Churn_Dataset.csv",
    },
    "housing": {
        "zip_interno": "Housing_Dataset.zip",
        "carpeta_zip": "Housing_Dataset",
        "csv": "Ames_Iowa_Housing_Dataset.csv",
    },
    "spotify": {
        "zip_interno": "Spotify_Dataset.zip",
        "carpeta_zip": "Spotify_Dataset",
        "csv": "Spotify_Tracks_Dataset.csv",
    },
}

CANDIDATOS_ZIP = (
    "EV PARCIALES MLY1101.zip.zip",
    "EV PARCIALES MLY1101.zip",
)


def localizar_zip(explicit: Path | None = None) -> Path:
    """Busca el zip de coordinación. ``EV_PARCIALES_ZIP`` gana si está definido."""
    if explicit is not None:
        return explicit
    env = os.environ.get("EV_PARCIALES_ZIP")
    if env:
        return Path(env).expanduser()
    descargas = Path.home() / "Downloads"
    for nombre in CANDIDATOS_ZIP:
        candidato = descargas / nombre
        if candidato.is_file():
            return candidato
    raise FileNotFoundError(
        "No está el zip de las parciales.\n"
        "  Deja `EV PARCIALES MLY1101.zip` (o `.zip.zip`) en ~/Downloads\n"
        "  o pasa --zip / exporta EV_PARCIALES_ZIP.\n"
        "  No se inventan filas: sin el zip oficial no hay evaluación."
    )


def preparar(destino: Path, zip_oficial: Path) -> dict[str, Path]:
    """Copia los tres CSV a ``destino/{telco,housing,spotify}/``.

    Returns:
        Ruta de cada CSV desempaquetado.

    Raises:
        FileNotFoundError: si falta el zip o un caso dentro.
    """
    if not zip_oficial.is_file():
        raise FileNotFoundError(
            f"No está el zip de las parciales: {zip_oficial}\n"
            "  Busca `EV PARCIALES MLY1101.zip` en Descargas (coordinación)."
        )
    destino.mkdir(parents=True, exist_ok=True)
    rutas: dict[str, Path] = {}
    with tempfile.TemporaryDirectory() as tmp:
        temporal = Path(tmp)
        with zipfile.ZipFile(zip_oficial) as externo:
            internos = {
                caso: _miembro_zip(externo.namelist(), meta["zip_interno"])
                for caso, meta in CASOS.items()
            }
            for caso, miembro in internos.items():
                if miembro is None:
                    raise FileNotFoundError(
                        f"{zip_oficial} no contiene {CASOS[caso]['zip_interno']}"
                    )
                bruto = temporal / f"{caso}.zip"
                bruto.write_bytes(externo.read(miembro))
                with zipfile.ZipFile(bruto) as interno:
                    csv_miembro = _miembro_zip(interno.namelist(), CASOS[caso]["csv"])
                    if csv_miembro is None:
                        raise FileNotFoundError(
                            f"{CASOS[caso]['zip_interno']} no trae {CASOS[caso]['csv']}"
                        )
                    salida = destino / caso / CASOS[caso]["csv"]
                    salida.parent.mkdir(parents=True, exist_ok=True)
                    salida.write_bytes(interno.read(csv_miembro))
                    rutas[caso] = salida
    return rutas


def _miembro_zip(nombres: list[str], sufijo: str) -> str | None:
    for nombre in nombres:
        if nombre.endswith(sufijo) and not nombre.startswith("__MACOSX"):
            return nombre
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", type=Path, help="ruta al zip institucional")
    parser.add_argument(
        "--destino",
        type=Path,
        default=DESTINO_POR_DEFECTO,
        help="carpeta gitignored (por defecto datos/evaluaciones)",
    )
    args = parser.parse_args()
    zip_oficial = localizar_zip(args.zip)
    rutas = preparar(args.destino, zip_oficial)
    print(f"Origen: {zip_oficial}")
    for caso, ruta in rutas.items():
        print(f"  {caso}: {ruta}  ({ruta.stat().st_size} bytes)")
    print("No lo subas a git: datos/evaluaciones/ está en .gitignore.")


if __name__ == "__main__":
    main()
