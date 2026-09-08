"""Valida conexión a los buckets GCS de Waymo y el tratamiento de un fragmento real.

No sube nada al repositorio. Lista cada producto de ``waymo.CATALOGO_BUCKETS``,
baja un archivo que quepa en clase y lo abre (parquet / json / tfrecord).

    uv run python herramientas/validar_buckets_waymo.py           # GCS + disco
    uv run python herramientas/validar_buckets_waymo.py --local   # solo lo ya bajado

Si gcloud pide reautenticación:

    gcloud auth login
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import pandas as pd  # noqa: E402

import waymo  # noqa: E402

DESTINO = RAIZ / "datos" / "waymo_real"


def _imprimir_objetos(objetos: list[dict], limite: int = 5) -> None:
    for objeto in objetos[:limite]:
        print(f"    {objeto['bytes'] / 1024**2:10.2f} MB  {objeto['nombre']}")
    if len(objetos) > limite:
        print(f"    … {len(objetos) - limite} más")


def _tratar_local() -> int:
    """Valida el manejo de los parquet v2 que ya están en disco."""
    cajas = DESTINO / "lidar_box.parquet"
    stats = DESTINO / "stats.parquet"
    if not cajas.exists() or not stats.exists():
        print("LOCAL  no hay lidar_box.parquet + stats.parquet en datos/waymo_real/")
        return 1

    resumen_cajas = waymo.resumir_fragmento(cajas)
    resumen_stats = waymo.resumir_fragmento(stats)
    real = waymo.traducir_esquema(pd.read_parquet(cajas), pd.read_parquet(stats))
    numericas = [
        c for c in ("box_length", "speed_mps", "num_lidar_points") if c in real.columns
    ]
    mal_tipo = [c for c in numericas if real[c].dtype == object]
    print("LOCAL  Perception v2 ya descargado")
    print(f"       lidar_box {resumen_cajas['filas']}×{resumen_cajas['columnas']}")
    print(f"       stats     {resumen_stats['filas']}×{resumen_stats['columnas']}")
    print(f"       traducido {real.shape}  tipos={real['object_type'].nunique()}")
    if mal_tipo:
        print(f"       ERROR columnas numéricas degradadas a object: {mal_tipo}")
        return 1
    print("       tratamiento OK (numéricas siguen float, esquema de la clase)")
    return 0


def _validar_producto(clave: str) -> int:
    info = waymo.producto(clave)
    print(f"\n{clave}  gs://{info['bucket']}/{info['prefijo_muestra']}")
    try:
        objetos = waymo.listar_objetos(info["bucket"], info["prefijo_muestra"], limite=6)
    except RuntimeError as error:
        print(f"    CONEXIÓN FALLÓ\n    {error}")
        return 1

    if not objetos:
        print("    listado vacío (prefijo incorrecto o sin permiso)")
        return 1

    print(f"    conexión OK · {len(objetos)} objetos")
    _imprimir_objetos(objetos)

    try:
        ruta = waymo.descargar_muestra(clave, DESTINO)
    except RuntimeError as error:
        print(f"    descarga omitida: {error}")
        return 0

    resumen = waymo.resumir_fragmento(ruta)
    print(f"    tratado {ruta.name}: {resumen}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local",
        action="store_true",
        help="no habla con GCS: solo valida lidar_box+stats ya descargados",
    )
    args = parser.parse_args()
    DESTINO.mkdir(parents=True, exist_ok=True)

    fallos = _tratar_local()
    if args.local:
        return fallos

    print("\nGCS  listar + un fragmento por producto (tope "
          f"{waymo.TAMANO_MAXIMO_CLASE_MB:.0f} MB)")
    for clave in waymo.CATALOGO_BUCKETS:
        fallos += _validar_producto(clave)
    if fallos:
        print(f"\n{fallos} comprobación(es) fallaron.")
    else:
        print("\nLos cuatro buckets respondieron y el tratamiento local está OK.")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
