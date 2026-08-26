"""Descarga de componentes del Waymo Open Dataset v2, en Colab o en local.

Existe porque **las credenciales funcionan distinto en cada entorno**:

- En **Google Colab**, ``auth.authenticate_user()`` autentica a Python pero
  **no** al CLI: ``gsutil`` responde *"You are attempting to access protected
  data with no configured credentials"*. Hay que usar el cliente Python
  ``google.cloud.storage``.
- En **local**, tras ``gcloud auth login`` funciona ``gsutil``, y el cliente
  Python en cambio pediría credenciales por defecto (``gcloud auth
  application-default login``), que es un paso extra.

Poner esto en un módulo y no en una celda del notebook tiene una razón: así se
puede probar. ``tests/test_waymo_descarga.py`` verifica la lógica de selección
de entorno sin necesidad de credenciales.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

BUCKET = "waymo_open_dataset_v_2_0_1"
SPLIT = "training"
COMPONENTES_LIVIANOS = ("lidar_box", "stats")


def en_colab() -> bool:
    """True si el código se está ejecutando dentro de Google Colab."""
    return "google.colab" in sys.modules


def ruta_gcs(componente: str, segmento: str) -> str:
    """Ruta del objeto dentro del bucket, sin el prefijo gs://."""
    return f"{SPLIT}/{componente}/{segmento}.parquet"


def _descargar_con_cliente_python(componente: str, segmento: str, destino: Path) -> None:
    """Vía de Colab: usa las credenciales de ``auth.authenticate_user()``.

    El proyecto que se indica solo se usa para facturación; para **leer** un
    objeto sirve cualquier nombre, y las cuentas personales no tienen proyecto
    por defecto.

    Raises:
        RuntimeError: si el bucket responde 403, con el nombre de la cuenta que
            se está usando. Es el error más frecuente y el más confuso: Colab
            suele estar abierto con una cuenta personal, mientras que los
            términos de Waymo se aceptaron con otra.
    """
    from google.api_core import exceptions
    from google.cloud import storage

    bucket = storage.Client(project="mly1101").bucket(BUCKET)
    try:
        bucket.blob(ruta_gcs(componente, segmento)).download_to_filename(str(destino))
    except exceptions.Forbidden as error:
        destino.unlink(missing_ok=True)
        raise RuntimeError(
            "Google Cloud respondió 403: la cuenta con la que está abierto este entorno no "
            "tiene acceso al Waymo Open Dataset.\n\n"
            "  Causa habitual: Colab está abierto con una cuenta de Google distinta de la que "
            "aceptó los términos en https://waymo.com/open/download/\n\n"
            "  Solución: cambia de cuenta en Colab (avatar arriba a la derecha) y usa la misma "
            "con la que te registraste en Waymo, o acepta los términos con esta cuenta.\n\n"
            f"  Mensaje original: {error}"
        ) from error


def _descargar_con_gsutil(componente: str, segmento: str, destino: Path) -> None:
    """Vía local: usa las credenciales de ``gcloud auth login``."""
    gsutil = shutil.which("gsutil")
    if not gsutil:
        raise RuntimeError(
            "No se encontró gsutil. Instálalo con: brew install --cask google-cloud-sdk"
        )
    resultado = subprocess.run(
        [gsutil, "cp", f"gs://{BUCKET}/{ruta_gcs(componente, segmento)}", str(destino)],
        capture_output=True,
        text=True,
    )
    if resultado.returncode != 0:
        raise RuntimeError(f"gsutil falló: {resultado.stderr.strip()[:300]}")


def descargar(componente: str, segmento: str, carpeta: Path, forzar: bool = False) -> Path:
    """Descarga un componente de un segmento y devuelve la ruta local.

    Elige automáticamente la vía que funciona en el entorno actual. Si el
    archivo ya existe, no lo vuelve a bajar salvo que ``forzar`` sea True.

    Args:
        componente: ``"lidar_box"`` o ``"stats"``.
        segmento: nombre del segmento, sin extensión.
        carpeta: directorio de destino; se crea si no existe.
        forzar: vuelve a descargar aunque el archivo ya esté.

    Raises:
        RuntimeError: si la descarga falla, con el mensaje del proveedor.
    """
    carpeta.mkdir(parents=True, exist_ok=True)
    destino = carpeta / f"{componente}.parquet"

    if destino.exists() and not forzar:
        return destino

    if en_colab():
        _descargar_con_cliente_python(componente, segmento, destino)
    else:
        _descargar_con_gsutil(componente, segmento, destino)
    return destino


def descargar_segmento(segmento: str, carpeta: Path) -> dict[str, Path]:
    """Descarga los dos componentes livianos de un segmento.

    Returns:
        ``{"lidar_box": ruta, "stats": ruta}``
    """
    rutas = {}
    for componente in COMPONENTES_LIVIANOS:
        ruta = descargar(componente, segmento, carpeta)
        tamano = ruta.stat().st_size / 1024**2
        print(f"{componente}: {tamano:.2f} MB  ({ruta})")
        rutas[componente] = ruta
    return rutas


# ===========================================================================
# Traducción del esquema real de Waymo v2 al esquema de la asignatura
# ===========================================================================
#
# Vivía dentro del notebook 00 como texto de celda. Aquí es una función pura,
# así que la usan por igual el notebook, el pipeline de Kedro y los tests.
# El esquema fue verificado contra un Parquet real el 2026-08-26.

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_LB = "[LiDARBoxComponent]"
_ST = "[StatsComponent]"

EQUIVALENCIAS_CAJAS = {
    "key.segment_context_name": "segment_id",
    "key.frame_timestamp_micros": "timestamp_micros",
    "key.laser_object_id": "id_interno",
    f"{_LB}.type": "object_type",
    f"{_LB}.box.center.x": "box_center_x",
    f"{_LB}.box.center.y": "box_center_y",
    f"{_LB}.box.center.z": "box_center_z",
    f"{_LB}.box.size.x": "box_length",
    f"{_LB}.box.size.y": "box_width",
    f"{_LB}.box.size.z": "box_height",
    f"{_LB}.speed.x": "speed_x",
    f"{_LB}.speed.y": "speed_y",
    f"{_LB}.num_lidar_points_in_box": "num_lidar_points",
    f"{_LB}.difficulty_level.detection": "detection_difficulty",
}

EQUIVALENCIAS_STATS = {
    "key.segment_context_name": "segment_id",
    "key.frame_timestamp_micros": "timestamp_micros",
    f"{_ST}.time_of_day": "time_of_day",
    f"{_ST}.weather": "weather",
    f"{_ST}.location": "location",
}

# El entero que Waymo usa para el tipo de objeto.
TIPOS_DE_OBJETO = {0: "unknown", 1: "vehicle", 2: "pedestrian", 3: "sign", 4: "cyclist"}


def _renombrar(datos: pd.DataFrame, equivalencias: dict) -> pd.DataFrame:
    """Se queda con las columnas conocidas y les pone el nombre de la clase."""
    presentes = {k: v for k, v in equivalencias.items() if k in datos.columns}
    return datos[list(presentes)].rename(columns=presentes)


def traducir_esquema(cajas: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Convierte los componentes reales de Waymo v2 al esquema de la asignatura.

    Tres traducciones no son un simple cambio de nombre, y son justamente las que
    conviene mirar en clase:

    1. **La velocidad es un vector.** Waymo entrega ``speed.x`` y ``speed.y`` por
       separado; la rapidez es su módulo, ``√(x² + y²)``. Quedarse solo con
       ``speed.x`` es un error silencioso: da valores plausibles y equivocados.

    2. **El tipo de objeto es un entero**, no una cadena. Hay que traducirlo con
       ``TIPOS_DE_OBJETO``, y existe el valor ``0`` (*unknown*), que el dataset
       sintético no tiene.

    3. **El ``NaN`` de la dificultad NO es un dato faltante.** Waymo solo rellena
       ``difficulty_level.detection`` cuando la detección es difícil (valor ``2``);
       si está vacío significa ``LEVEL_1``. En el segmento verificado son 15.356
       ``NaN`` de 18.633 filas: tratarlos como faltantes borraría el 82 % de los
       datos y dejaría un dataset con una sola clase.

       Es el reverso exacto del defecto que se estudia en la Actividad 1.3, donde
       un ``-1`` disfraza un faltante. Aquí un faltante disfraza un valor.

    Args:
        cajas: contenido de ``lidar_box.parquet``.
        stats: contenido de ``stats.parquet``.

    Returns:
        DataFrame con el esquema de ``detecciones_waymo_like.csv``, más la columna
        ``location``, que existe en los datos reales y no en el sintético.
    """
    tabla = _renombrar(cajas, EQUIVALENCIAS_CAJAS).merge(
        _renombrar(stats, EQUIVALENCIAS_STATS),
        on=["segment_id", "timestamp_micros"],
        how="left",
    )

    # 1. La rapidez es el módulo del vector velocidad.
    if {"speed_x", "speed_y"} <= set(tabla.columns):
        tabla["speed_mps"] = np.sqrt(tabla["speed_x"] ** 2 + tabla["speed_y"] ** 2)
        tabla = tabla.drop(columns=["speed_x", "speed_y"])

    # 2. El tipo de objeto llega como entero.
    if "object_type" in tabla.columns:
        tabla["object_type"] = (
            tabla["object_type"].map(TIPOS_DE_OBJETO).fillna("desconocido")
        )

    # 3. El NaN de la dificultad significa LEVEL_1, no "falta el dato".
    if "detection_difficulty" in tabla.columns:
        tabla["detection_difficulty"] = np.where(
            tabla["detection_difficulty"] == 2, "LEVEL_2", "LEVEL_1"
        )

    return tabla.reset_index(drop=True)
