"""Analiza el sesgo de muestreo del Waymo Open Dataset sobre varios segmentos.

Un solo segmento no sirve para estudiar el sesgo: el clima, la hora del día y
la ubicación son propiedades del segmento completo y no varían dentro de él.
Este script cruza muchos segmentos y responde:

- ¿Qué fracción de la conducción registrada es nocturna? ¿Con lluvia?
- ¿Cambia la composición de objetos según la condición?
- ¿Los ciclistas y peatones están igual de representados de noche que de día?
- ¿Cambia la calidad de la detección (puntos láser, dificultad) según condición?

Requisito previo:

    python herramientas/descargar_waymo.py --muestra 40

Uso:

    python herramientas/analizar_sesgo_waymo.py
    python herramientas/analizar_sesgo_waymo.py --markdown docs/sesgo_waymo.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

MUESTRA = RAIZ / "datos" / "waymo_real" / "muestra"
LB = "[LiDARBoxComponent]"
ST = "[StatsComponent]"
TIPOS = {0: "unknown", 1: "vehicle", 2: "pedestrian", 3: "sign", 4: "cyclist"}


def cargar_muestra(directorio: Path = MUESTRA) -> pd.DataFrame:
    """Une todos los segmentos descargados en una sola tabla de detecciones.

    Returns:
        DataFrame con una fila por detección y las columnas de condición
        (``weather``, ``time_of_day``, ``location``) ya propagadas.
    """
    if not directorio.exists():
        raise SystemExit(
            f"No existe {directorio.relative_to(RAIZ)}.\n"
            "  Descarga primero:  python herramientas/descargar_waymo.py --muestra 40"
        )

    piezas = []
    for carpeta in sorted(directorio.iterdir()):
        cajas_ruta = carpeta / "lidar_box.parquet"
        stats_ruta = carpeta / "stats.parquet"
        if not (cajas_ruta.exists() and stats_ruta.exists()):
            continue
        cajas = pd.read_parquet(cajas_ruta)
        stats = pd.read_parquet(stats_ruta)

        detecciones = pd.DataFrame({
            "segmento": carpeta.name,
            "timestamp": cajas["key.frame_timestamp_micros"],
            "tipo": cajas[f"{LB}.type"].map(TIPOS),
            "largo": cajas[f"{LB}.box.size.x"],
            "alto": cajas[f"{LB}.box.size.z"],
            "x": cajas[f"{LB}.box.center.x"],
            "y": cajas[f"{LB}.box.center.y"],
            "puntos": cajas[f"{LB}.num_lidar_points_in_box"],
            "dificil": cajas[f"{LB}.difficulty_level.detection"].notna(),
        })
        condicion = stats.iloc[0]
        detecciones["weather"] = str(condicion[f"{ST}.weather"]).strip().lower()
        detecciones["time_of_day"] = str(condicion[f"{ST}.time_of_day"]).strip()
        detecciones["location"] = str(condicion[f"{ST}.location"]).strip()
        piezas.append(detecciones)

    if not piezas:
        raise SystemExit(f"No se encontraron segmentos completos en {directorio}")
    return pd.concat(piezas, ignore_index=True)


def tabla_segmentos(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por segmento con su condición y su tamaño."""
    return (
        df.groupby("segmento")
        .agg(clima=("weather", "first"), momento=("time_of_day", "first"),
             lugar=("location", "first"), detecciones=("tipo", "size"))
        .reset_index()
    )


def composicion_por_condicion(df: pd.DataFrame, condicion: str) -> pd.DataFrame:
    """Porcentaje de cada tipo de objeto dentro de cada valor de la condición."""
    tabla = pd.crosstab(df[condicion], df["tipo"], normalize="index") * 100
    return tabla.round(2)


def calidad_por_condicion(df: pd.DataFrame, condicion: str) -> pd.DataFrame:
    """Indicadores de calidad de la detección según la condición."""
    return (
        df.groupby(condicion)
        .agg(detecciones=("tipo", "size"),
             puntos_medianos=("puntos", "median"),
             pct_dificiles=("dificil", lambda s: 100 * s.mean()),
             pct_sin_puntos=("puntos", lambda s: 100 * (s == 0).mean()))
        .round(2)
    )


def _bloque(titulo: str, tabla: pd.DataFrame) -> str:
    return f"\n{titulo}\n{'-' * len(titulo)}\n{tabla.to_string()}\n"


def generar_informe(df: pd.DataFrame) -> str:
    """Arma el informe completo como texto."""
    segmentos = tabla_segmentos(df)
    n_seg, n_det = len(segmentos), len(df)

    detecciones_fmt = f"{n_det:,}".replace(",", ".")
    partes = [
        f"Muestra: {n_seg} segmentos, {detecciones_fmt} detecciones",
        _bloque("Segmentos por momento del día", segmentos["momento"].value_counts().to_frame("segmentos")),
        _bloque("Segmentos por clima", segmentos["clima"].value_counts().to_frame("segmentos")),
        _bloque("Segmentos por ubicación", segmentos["lugar"].value_counts().to_frame("segmentos")),
        _bloque("Composición de objetos por momento del día (%)", composicion_por_condicion(df, "time_of_day")),
        _bloque("Composición de objetos por clima (%)", composicion_por_condicion(df, "weather")),
        _bloque("Calidad de la detección por momento del día", calidad_por_condicion(df, "time_of_day")),
    ]

    # Conclusión cuantitativa sobre los usuarios vulnerables de la vía.
    vulnerables = df["tipo"].isin(["pedestrian", "cyclist"])
    por_momento = df.assign(vulnerable=vulnerables).groupby("time_of_day")["vulnerable"].mean() * 100
    partes.append(_bloque("Peatones + ciclistas sobre el total (%)", por_momento.round(2).to_frame("pct")))

    return "\n".join(partes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--markdown", type=Path, help="además, escribe el informe en un archivo")
    args = parser.parse_args()

    df = cargar_muestra()
    informe = generar_informe(df)
    print(informe)

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(
            "# Sesgo de muestreo en el Waymo Open Dataset\n\n"
            "Generado por `herramientas/analizar_sesgo_waymo.py`.\n\n"
            "```\n" + informe + "\n```\n",
            encoding="utf-8",
        )
        print(f"\nInforme escrito en {args.markdown}")


if __name__ == "__main__":
    main()
