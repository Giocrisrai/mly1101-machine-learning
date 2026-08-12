"""Generador del dataset de trabajo de la EA1 (MLY1101).

Produce un CSV de detecciones de objetos con el mismo esquema del componente
``lidar_box`` de Waymo Open Dataset v2, contaminado a propósito con diez
defectos de calidad documentados (ver ``CATALOGO_DEFECTOS``).

El dataset es **sintético**: no contiene datos reales de Waymo ni de personas.
Se usa porque permite garantizar que cada problema que la clase debe descubrir
está efectivamente presente, y porque su esquema es intercambiable con los
datos reales (ver ``notebooks/00_opcional_waymo_real.ipynb``).

Determinismo: para una misma semilla, el archivo generado es idéntico byte a
byte. Los tests de ``tests/test_generar_dataset.py`` dependen de esa propiedad.

Uso:
    python src/generar_dataset.py
    python src/generar_dataset.py --filas 40000 --semilla 42 --salida datos/crudos/detecciones_waymo_like.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEMILLA_POR_DEFECTO = 42
FILAS_POR_DEFECTO = 40_000
RUTA_POR_DEFECTO = Path("datos/crudos/detecciones_waymo_like.csv")

# Columnas en el orden en que aparecen en el CSV.
COLUMNAS = [
    "segment_id",
    "timestamp_micros",
    "id_interno",
    "object_type",
    "box_center_x",
    "box_center_y",
    "box_center_z",
    "box_length",
    "box_width",
    "box_height",
    "speed_mps",
    "num_lidar_points",
    "weather",
    "time_of_day",
    "detection_difficulty",
    "sensor_version",
]

CATALOGO_DEFECTOS = {
    1: "timestamp_micros contiene 'N/D' -> la columna se lee como object",
    2: "num_lidar_points usa -1 como nulo oculto (~3%)",
    3: "weather con categorias inconsistentes ('sunny', 'Sunny', 'RAIN ', 'lluvia') y nulos",
    4: "object_type con categorias inconsistentes ('PEDESTRIAN', 'Pedestrian', 'PEATON', 'Ped')",
    5: "duplicados exactos (~1.2%) y duplicados logicos (~0.5%, misma llave con valores distintos)",
    6: "outliers imposibles: speed_mps hasta ~340, box_height == 0, box_length negativo",
    7: "outliers legitimos: buses con box_length entre 12 y 18 m (no se deben eliminar)",
    8: "nulos MNAR en speed_mps, concentrados en detection_difficulty == LEVEL_2 y de noche",
    9: "desbalance de clases: CYCLIST ~2% de las filas",
    10: "sensor_version constante e id_interno de cardinalidad casi unica: no son features",
}

# Dimensiones tipicas por tipo de objeto: (largo, ancho, alto) en metros,
# como (media, desviacion).
_DIMENSIONES = {
    "VEHICLE": ((4.6, 0.7), (1.9, 0.2), (1.6, 0.25)),
    "PEDESTRIAN": ((0.9, 0.15), (0.8, 0.12), (1.72, 0.12)),
    "CYCLIST": ((1.8, 0.2), (0.75, 0.1), (1.65, 0.12)),
    "SIGN": ((0.4, 0.1), (0.4, 0.1), (2.1, 0.4)),
}

# Velocidad tipica por tipo de objeto en m/s: (media, desviacion).
_VELOCIDADES = {
    "VEHICLE": (9.5, 5.0),
    "PEDESTRIAN": (1.3, 0.5),
    "CYCLIST": (4.5, 1.6),
    "SIGN": (0.0, 0.0),
}

_TIPOS = ["VEHICLE", "PEDESTRIAN", "SIGN", "CYCLIST"]
_PROBS_TIPOS = [0.62, 0.28, 0.08, 0.02]  # defecto 9: CYCLIST minoritario


def _generar_base(n_filas: int, rng: np.random.Generator) -> pd.DataFrame:
    """Genera las ``n_filas`` limpias, antes de inyectar cualquier defecto."""
    n_segmentos = max(20, n_filas // 260)
    segmento = rng.integers(0, n_segmentos, size=n_filas)

    tipo = rng.choice(_TIPOS, size=n_filas, p=_PROBS_TIPOS)

    largo = np.empty(n_filas)
    ancho = np.empty(n_filas)
    alto = np.empty(n_filas)
    velocidad = np.empty(n_filas)
    for nombre in _TIPOS:
        mascara = tipo == nombre
        n = int(mascara.sum())
        (l_mu, l_sd), (a_mu, a_sd), (h_mu, h_sd) = _DIMENSIONES[nombre]
        largo[mascara] = rng.normal(l_mu, l_sd, n)
        ancho[mascara] = rng.normal(a_mu, a_sd, n)
        alto[mascara] = rng.normal(h_mu, h_sd, n)
        v_mu, v_sd = _VELOCIDADES[nombre]
        velocidad[mascara] = np.abs(rng.normal(v_mu, v_sd, n)) if v_sd else 0.0

    # Las colas de la normal pueden dar dimensiones absurdas: se acotan a un
    # mínimo físico. Los valores imposibles se inyectan después, a propósito.
    largo = np.clip(largo, 0.20, None)
    ancho = np.clip(ancho, 0.20, None)
    alto = np.clip(alto, 0.30, None)

    # Coordenadas en el sistema del vehiculo: x hacia adelante, y lateral.
    centro_x = rng.normal(18.0, 22.0, n_filas)
    centro_y = rng.normal(0.0, 12.0, n_filas)
    centro_z = rng.normal(0.9, 0.4, n_filas)

    # A mayor distancia, menos puntos laser: n ~ Poisson(lambda / (1 + d/12)^2).
    distancia = np.sqrt(centro_x**2 + centro_y**2)
    lam = 900.0 / (1.0 + distancia / 12.0) ** 2
    puntos = rng.poisson(np.clip(lam, 1.0, None)) + 1

    dificultad = np.where(
        (puntos < 25) | (rng.random(n_filas) < 0.08), "LEVEL_2", "LEVEL_1"
    )

    clima = rng.choice(["sunny", "rain", "fog"], size=n_filas, p=[0.70, 0.21, 0.09])
    hora = rng.choice(
        ["Day", "Night", "Dawn/Dusk"], size=n_filas, p=[0.70, 0.20, 0.10]
    )

    # Un timestamp base por segmento, con incrementos de ~100 ms dentro de cada uno.
    base_segmento = 1_691_000_000_000_000 + segmento.astype(np.int64) * 25_000_000_000
    marca = base_segmento + rng.integers(0, 200, size=n_filas) * 100_000

    df = pd.DataFrame(
        {
            "segment_id": [f"seg_{s:04d}" for s in segmento],
            "timestamp_micros": marca,
            "id_interno": [f"det_{i:07d}" for i in range(n_filas)],
            "object_type": tipo,
            "box_center_x": centro_x,
            "box_center_y": centro_y,
            "box_center_z": centro_z,
            "box_length": largo,
            "box_width": ancho,
            "box_height": alto,
            "speed_mps": velocidad,
            "num_lidar_points": puntos,
            "weather": clima,
            "time_of_day": hora,
            "detection_difficulty": dificultad,
            "sensor_version": "v2.0.1",  # defecto 10: columna constante
        }
    )
    return df.sort_values(["segment_id", "timestamp_micros"]).reset_index(drop=True)


def _inyectar_outliers_legitimos(df: pd.DataFrame, rng: np.random.Generator) -> None:
    """Defecto 7: buses reales, largos de 12 a 18 m. No son errores."""
    vehiculos = df.index[df["object_type"] == "VEHICLE"].to_numpy()
    buses = rng.choice(vehiculos, size=max(1, int(0.015 * len(df))), replace=False)
    df.loc[buses, "box_length"] = rng.uniform(12.0, 18.0, len(buses))
    df.loc[buses, "box_width"] = rng.uniform(2.4, 2.9, len(buses))
    df.loc[buses, "box_height"] = rng.uniform(2.9, 3.6, len(buses))


def _inyectar_outliers_imposibles(df: pd.DataFrame, rng: np.random.Generator) -> None:
    """Defecto 6: valores fisicamente imposibles, producto de fallas del sensor."""
    n = len(df)
    veloces = rng.choice(n, size=max(1, int(0.004 * n)), replace=False)
    df.loc[veloces, "speed_mps"] = rng.uniform(180.0, 340.0, len(veloces))

    sin_alto = rng.choice(n, size=max(1, int(0.003 * n)), replace=False)
    df.loc[sin_alto, "box_height"] = 0.0

    largo_negativo = rng.choice(n, size=max(1, int(0.002 * n)), replace=False)
    df.loc[largo_negativo, "box_length"] = -df.loc[largo_negativo, "box_length"].abs()


def _inyectar_nulos_mnar(df: pd.DataFrame, rng: np.random.Generator) -> None:
    """Defecto 8: nulos en speed_mps que NO son aleatorios.

    P(nulo) = 0.35 si detection_difficulty == LEVEL_2 y time_of_day == Night
            = 0.10 si detection_difficulty == LEVEL_2
            = 0.004 en el resto

    El alumno debe notar que borrar esas filas sesgaria el dataset contra las
    detecciones difíciles y nocturnas.
    """
    dificil = df["detection_difficulty"] == "LEVEL_2"
    noche = df["time_of_day"] == "Night"
    prob = np.where(dificil & noche, 0.35, np.where(dificil, 0.10, 0.004))
    df.loc[rng.random(len(df)) < prob, "speed_mps"] = np.nan


def _inyectar_nulos_ocultos(df: pd.DataFrame, rng: np.random.Generator) -> None:
    """Defecto 2: num_lidar_points usa -1 en lugar de un nulo explicito."""
    faltantes = rng.random(len(df)) < 0.03
    df.loc[faltantes, "num_lidar_points"] = -1


def _inyectar_categorias_inconsistentes(
    df: pd.DataFrame, rng: np.random.Generator
) -> None:
    """Defectos 3 y 4: la misma categoria escrita de varias formas, y nulos."""
    variantes_clima = {
        "sunny": (["sunny", "Sunny", "SUNNY", "soleado"], [0.72, 0.12, 0.10, 0.06]),
        "rain": (["rain", "RAIN ", " rain", "lluvia"], [0.70, 0.14, 0.10, 0.06]),
        "fog": (["fog", "Fog", "niebla"], [0.78, 0.14, 0.08]),
    }
    for canonica, (variantes, probs) in variantes_clima.items():
        indices = df.index[df["weather"] == canonica].to_numpy()
        elegidas = rng.choice(variantes, size=len(indices), p=probs)
        df.loc[indices, "weather"] = elegidas
    df.loc[rng.random(len(df)) < 0.05, "weather"] = np.nan

    peatones = df.index[df["object_type"] == "PEDESTRIAN"].to_numpy()
    variantes_peaton = rng.choice(
        ["PEDESTRIAN", "Pedestrian", "PEATON", "Ped"],
        size=len(peatones),
        p=[0.80, 0.09, 0.07, 0.04],
    )
    df.loc[peatones, "object_type"] = variantes_peaton


def _inyectar_timestamps_corruptos(df: pd.DataFrame, rng: np.random.Generator) -> None:
    """Defecto 1: 'N/D' en timestamp_micros fuerza el dtype object al leer el CSV."""
    df["timestamp_micros"] = df["timestamp_micros"].astype(object)
    corruptos = rng.choice(len(df), size=max(1, int(0.0015 * len(df))), replace=False)
    df.loc[corruptos, "timestamp_micros"] = "N/D"


def _inyectar_duplicados(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Defecto 5: duplicados exactos y duplicados logicos.

    - Exactos: la fila completa repetida (los detecta ``df.duplicated()``).
    - Logicos: misma llave (segment_id, timestamp_micros, id_interno) con
      valores distintos en las mediciones. ``drop_duplicates()`` no los ve.
    """
    n = len(df)
    exactos = df.loc[rng.choice(n, size=int(0.012 * n), replace=False)].copy()

    logicos = df.loc[rng.choice(n, size=int(0.005 * n), replace=False)].copy()
    ruido = rng.normal(0.0, 0.4, len(logicos))
    logicos["box_center_x"] = logicos["box_center_x"] + ruido
    logicos["num_lidar_points"] = np.maximum(
        1, logicos["num_lidar_points"] + rng.integers(-8, 9, len(logicos))
    )

    completo = pd.concat([df, exactos, logicos], ignore_index=True)
    # Se mezcla para que los duplicados no queden contiguos y sean menos obvios.
    orden = rng.permutation(len(completo))
    return completo.iloc[orden].reset_index(drop=True)


def generar_dataset(
    n_filas: int = FILAS_POR_DEFECTO, semilla: int = SEMILLA_POR_DEFECTO
) -> pd.DataFrame:
    """Devuelve el DataFrame sucio, listo para escribir a CSV.

    Args:
        n_filas: filas limpias antes de agregar duplicados (el resultado tiene
            aproximadamente ``1.017 * n_filas`` filas).
        semilla: semilla del generador aleatorio.
    """
    rng = np.random.default_rng(semilla)
    df = _generar_base(n_filas, rng)
    _inyectar_outliers_legitimos(df, rng)
    _inyectar_outliers_imposibles(df, rng)
    _inyectar_nulos_mnar(df, rng)
    _inyectar_nulos_ocultos(df, rng)
    _inyectar_categorias_inconsistentes(df, rng)
    _inyectar_timestamps_corruptos(df, rng)
    df = _inyectar_duplicados(df, rng)
    return df[COLUMNAS]


def escribir_csv(df: pd.DataFrame, ruta: Path) -> Path:
    """Escribe el CSV con formato estable (3 decimales) y devuelve la ruta."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False, float_format="%.3f")
    return ruta


def _parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--filas", type=int, default=FILAS_POR_DEFECTO)
    parser.add_argument("--semilla", type=int, default=SEMILLA_POR_DEFECTO)
    parser.add_argument("--salida", type=Path, default=RUTA_POR_DEFECTO)
    return parser.parse_args()


def main() -> None:
    args = _parsear_argumentos()
    df = generar_dataset(args.filas, args.semilla)
    ruta = escribir_csv(df, args.salida)
    print(f"Dataset generado: {ruta}  ({len(df):,} filas x {len(df.columns)} columnas)")
    print("Defectos inyectados:")
    for numero, descripcion in CATALOGO_DEFECTOS.items():
        print(f"  {numero:2d}. {descripcion}")


if __name__ == "__main__":
    main()
