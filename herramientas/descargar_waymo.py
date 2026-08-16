"""Descarga un segmento del Waymo Open Dataset v2 para verificar el mapeo.

Baja solo los dos componentes livianos que necesita el análisis de la EA1
(``lidar_box`` y ``stats``) de un único segmento de conducción. **No** baja
imágenes ni nubes de puntos, que son los componentes pesados.

Requisitos previos (una sola vez):

    brew install --cask google-cloud-sdk
    gcloud auth login          # con la cuenta que aceptó los términos en
                               # https://waymo.com/open/download/

Uso:

    python herramientas/descargar_waymo.py                 # primer segmento disponible
    python herramientas/descargar_waymo.py --segmento NOMBRE
    python herramientas/descargar_waymo.py --listar        # solo listar segmentos
    python herramientas/descargar_waymo.py --muestra 40    # varios segmentos completos
    python herramientas/descargar_waymo.py --censo-stats   # stats de los 798 segmentos

Los archivos quedan en ``datos/waymo_real/``, que está en .gitignore: la
licencia de Waymo es de uso no comercial y no permite redistribuirlos.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "datos" / "waymo_real"
BUCKET = "gs://waymo_open_dataset_v_2_0_1/training"
COMPONENTES = ["lidar_box", "stats"]


def _ejecutar(comando: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(comando, capture_output=True, text=True)


def _comprobar_requisitos() -> str:
    """Devuelve la ruta de gsutil o termina con un mensaje accionable."""
    gsutil = shutil.which("gsutil")
    if not gsutil:
        sys.exit(
            "No se encontró gsutil.\n"
            "  Instálalo con:  brew install --cask google-cloud-sdk\n"
            "  Si ya está instalado, abre una terminal nueva para que entre en el PATH."
        )
    cuentas = _ejecutar(["gcloud", "auth", "list", "--format=value(account)"])
    if not cuentas.stdout.strip():
        sys.exit(
            "No hay ninguna cuenta autenticada en gcloud.\n"
            "  Ejecuta:  gcloud auth login\n"
            "  Usa la MISMA cuenta con la que aceptaste los términos en\n"
            "  https://waymo.com/open/download/"
        )
    print(f"Cuenta activa: {cuentas.stdout.strip().splitlines()[0]}")
    return gsutil


def listar_segmentos(gsutil: str, cantidad: int = 5) -> list[str]:
    """Lista los primeros segmentos disponibles del componente lidar_box."""
    resultado = _ejecutar([gsutil, "ls", f"{BUCKET}/lidar_box/"])
    if resultado.returncode != 0:
        error = resultado.stderr.strip()
        if "AccessDenied" in error or "401" in error:
            sys.exit(
                "Acceso denegado al bucket de Waymo.\n"
                "  Suele significar que la cuenta autenticada no es la misma con la que\n"
                "  aceptaste los términos en https://waymo.com/open/download/\n\n"
                f"  Respuesta de gsutil:\n  {error}"
            )
        sys.exit(f"No se pudo listar el bucket:\n{error}")
    rutas = [linea for linea in resultado.stdout.splitlines() if linea.endswith(".parquet")]
    return [Path(ruta).stem for ruta in rutas[:cantidad]]


def descargar(gsutil: str, segmento: str) -> None:
    """Copia lidar_box y stats del segmento indicado a datos/waymo_real/."""
    DESTINO.mkdir(parents=True, exist_ok=True)
    for componente in COMPONENTES:
        origen = f"{BUCKET}/{componente}/{segmento}.parquet"
        destino = DESTINO / f"{componente}.parquet"
        print(f"Descargando {componente}…")
        resultado = _ejecutar([gsutil, "cp", origen, str(destino)])
        if resultado.returncode != 0:
            sys.exit(f"Falló la descarga de {componente}:\n{resultado.stderr.strip()}")
        print(f"  {destino.relative_to(RAIZ)}  ({destino.stat().st_size / 1024**2:.1f} MB)")
    (DESTINO / "SEGMENTO.txt").write_text(segmento + "\n", encoding="utf-8")


def descargar_muestra(gsutil: str, segmentos: list[str], solo_stats: bool = False) -> Path:
    """Descarga varios segmentos para el análisis de sesgo de muestreo.

    Cada segmento queda en ``datos/waymo_real/muestra/{nombre}/``. Se usa desde
    ``herramientas/analizar_sesgo_waymo.py``: un solo segmento no sirve para
    estudiar el sesgo, porque el clima y la hora son propiedades del segmento
    completo y no varían dentro de él.

    Args:
        solo_stats: baja únicamente el componente ``stats`` (~23 KB por
            segmento en vez de ~1 MB). Suficiente para caracterizar clima,
            hora y ubicación sobre muchos segmentos a bajo costo.
    """
    muestra = DESTINO / "muestra"
    for numero, segmento in enumerate(segmentos, start=1):
        carpeta = muestra / segmento
        carpeta.mkdir(parents=True, exist_ok=True)
        requeridos = ["stats"] if solo_stats else COMPONENTES
        pendientes = [c for c in requeridos if not (carpeta / f"{c}.parquet").exists()]
        if not pendientes:
            print(f"[{numero}/{len(segmentos)}] {segmento[:28]}… ya estaba")
            continue
        for componente in pendientes:
            resultado = _ejecutar(
                [gsutil, "cp", f"{BUCKET}/{componente}/{segmento}.parquet",
                 str(carpeta / f"{componente}.parquet")]
            )
            if resultado.returncode != 0:
                print(f"   ⚠️ falló {componente} de {segmento}: {resultado.stderr.strip()[:90]}")
        peso = sum(f.stat().st_size for f in carpeta.glob("*.parquet")) / 1024**2
        print(f"[{numero}/{len(segmentos)}] {segmento[:28]}…  {peso:.2f} MB")
    return muestra


def descargar_censo_stats(gsutil: str) -> Path:
    """Descarga el componente ``stats`` de **todos** los segmentos de training.

    Son ~798 segmentos de ~23 KB, unos 18 MB en total. Con ``gsutil -m`` y un
    comodín se copian en paralelo en una sola llamada, así que tarda minutos y
    no horas.

    A diferencia de ``--muestra N``, esto no es un muestreo: es el censo del
    split de entrenamiento. Elimina la pregunta de si la muestra era
    representativa, que es la objeción obvia a cualquier cifra de sesgo.
    """
    destino = DESTINO / "censo_stats"
    destino.mkdir(parents=True, exist_ok=True)
    print("Copiando stats de todos los segmentos de training (~18 MB, en paralelo)…")
    resultado = _ejecutar([gsutil, "-m", "cp", f"{BUCKET}/stats/*.parquet", str(destino)])
    archivos = list(destino.glob("*.parquet"))
    if not archivos:
        sys.exit(f"No se descargó nada:\n{resultado.stderr.strip()[:400]}")
    peso = sum(f.stat().st_size for f in archivos) / 1024**2
    print(f"{len(archivos)} segmentos · {peso:.1f} MB en {destino.relative_to(RAIZ)}")
    return destino


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--segmento", help="nombre del segmento (sin .parquet)")
    parser.add_argument("--listar", action="store_true", help="solo listar segmentos disponibles")
    parser.add_argument(
        "--muestra",
        type=int,
        metavar="N",
        help="descarga N segmentos a datos/waymo_real/muestra/ para el análisis de sesgo",
    )
    parser.add_argument(
        "--solo-stats",
        action="store_true",
        help="con --muestra, baja solo el componente stats (~23 KB por segmento)",
    )
    parser.add_argument(
        "--censo-stats",
        action="store_true",
        help="baja el stats de TODOS los segmentos de training (~798, ~18 MB)",
    )
    args = parser.parse_args()

    gsutil = _comprobar_requisitos()

    if args.censo_stats:
        descargar_censo_stats(gsutil)
        print("\nListo. Ahora puedes ejecutar:  python herramientas/analizar_sesgo_waymo.py")
        return

    cantidad = args.muestra if args.muestra else 5
    disponibles = listar_segmentos(gsutil, cantidad=cantidad)

    if args.listar:
        print(f"\nSegmentos disponibles (primeros {len(disponibles)}):")
        for nombre in disponibles:
            print("  ", nombre)
        return

    if args.muestra:
        peso = "~23 KB" if args.solo_stats else "~1 MB"
        print(f"\nDescargando {len(disponibles)} segmentos ({peso} cada uno)…")
        muestra = descargar_muestra(gsutil, disponibles, solo_stats=args.solo_stats)
        print(f"\nListo. Ahora puedes ejecutar:  python herramientas/analizar_sesgo_waymo.py")
        print(f"Datos en: {muestra.relative_to(RAIZ)}")
        return

    segmento = args.segmento or disponibles[0]
    print(f"\nSegmento: {segmento}")
    descargar(gsutil, segmento)
    print("\nListo. Ahora puedes ejecutar:  pytest tests/test_mapeo_waymo.py -v")


if __name__ == "__main__":
    main()
