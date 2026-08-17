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
