"""Tests de la selección de entorno para descargar datos de Waymo.

No requieren credenciales ni red: sustituyen las dos vías de descarga por
dobles y verifican que se elija la correcta. La razón de que esto exista es
que el error que motivó el módulo (gsutil sin credenciales en Colab) fue
invisible hasta ejecutarlo en Colab; con estos tests, al menos la lógica de
decisión queda fijada.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import waymo  # noqa: E402

SEGMENTO = "10023947602400723454_1120_000_1140_000"


def test_ruta_gcs_apunta_al_split_de_entrenamiento() -> None:
    assert waymo.ruta_gcs("lidar_box", SEGMENTO) == f"training/lidar_box/{SEGMENTO}.parquet"


def test_en_colab_es_falso_fuera_de_colab() -> None:
    assert waymo.en_colab() is False


def test_los_componentes_livianos_son_los_dos_del_analisis() -> None:
    assert waymo.COMPONENTES_LIVIANOS == ("lidar_box", "stats")


def test_en_colab_usa_el_cliente_python(monkeypatch, tmp_path: Path) -> None:
    """En Colab NO se puede usar gsutil: no hereda las credenciales."""
    llamadas = []
    monkeypatch.setattr(waymo, "en_colab", lambda: True)
    monkeypatch.setattr(waymo, "_descargar_con_cliente_python",
                        lambda c, s, d: (llamadas.append("python"), d.write_bytes(b"x")))
    monkeypatch.setattr(waymo, "_descargar_con_gsutil",
                        lambda c, s, d: llamadas.append("gsutil"))

    waymo.descargar("stats", SEGMENTO, tmp_path)
    assert llamadas == ["python"]


def test_en_local_usa_gsutil(monkeypatch, tmp_path: Path) -> None:
    llamadas = []
    monkeypatch.setattr(waymo, "en_colab", lambda: False)
    monkeypatch.setattr(waymo, "_descargar_con_cliente_python",
                        lambda c, s, d: llamadas.append("python"))
    monkeypatch.setattr(waymo, "_descargar_con_gsutil",
                        lambda c, s, d: (llamadas.append("gsutil"), d.write_bytes(b"x")))

    waymo.descargar("stats", SEGMENTO, tmp_path)
    assert llamadas == ["gsutil"]


def test_no_vuelve_a_descargar_si_ya_existe(monkeypatch, tmp_path: Path) -> None:
    llamadas = []
    monkeypatch.setattr(waymo, "en_colab", lambda: False)
    monkeypatch.setattr(waymo, "_descargar_con_gsutil", lambda c, s, d: llamadas.append("gsutil"))
    (tmp_path / "stats.parquet").write_bytes(b"ya estaba")

    ruta = waymo.descargar("stats", SEGMENTO, tmp_path)
    assert llamadas == []
    assert ruta.read_bytes() == b"ya estaba"


def test_forzar_vuelve_a_descargar(monkeypatch, tmp_path: Path) -> None:
    llamadas = []
    monkeypatch.setattr(waymo, "en_colab", lambda: False)
    monkeypatch.setattr(waymo, "_descargar_con_gsutil",
                        lambda c, s, d: (llamadas.append("gsutil"), d.write_bytes(b"nuevo")))
    (tmp_path / "stats.parquet").write_bytes(b"viejo")

    waymo.descargar("stats", SEGMENTO, tmp_path, forzar=True)
    assert llamadas == ["gsutil"]


def test_crea_la_carpeta_de_destino(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(waymo, "en_colab", lambda: False)
    monkeypatch.setattr(waymo, "_descargar_con_gsutil", lambda c, s, d: d.write_bytes(b"x"))
    destino = tmp_path / "nueva" / "carpeta"

    waymo.descargar("stats", SEGMENTO, destino)
    assert destino.exists()


def test_el_error_de_gsutil_se_propaga_con_mensaje(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(waymo, "shutil", type("f", (), {"which": staticmethod(lambda x: None)}))
    with pytest.raises(RuntimeError, match="google-cloud-sdk"):
        waymo._descargar_con_gsutil("stats", SEGMENTO, tmp_path / "s.parquet")


def test_descargar_segmento_baja_los_dos_componentes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(waymo, "en_colab", lambda: False)
    monkeypatch.setattr(waymo, "_descargar_con_gsutil", lambda c, s, d: d.write_bytes(b"x" * 10))

    rutas = waymo.descargar_segmento(SEGMENTO, tmp_path)
    assert set(rutas) == {"lidar_box", "stats"}
    assert all(r.exists() for r in rutas.values())


def test_el_403_se_traduce_a_un_mensaje_sobre_la_cuenta(monkeypatch, tmp_path: Path) -> None:
    """El error más confuso de Colab: la cuenta abierta no es la registrada en Waymo.

    Se salta si no está instalado el extra ``waymo`` (``uv sync --extra waymo``):
    la traducción del error solo se puede probar con las excepciones reales del
    cliente de Google Cloud.
    """
    # Se guarda por google.cloud.storage y NO por google.api_core: este ultimo
    # llega de rebote como dependencia de kedro, asi que estar presente no
    # significa que el extra `waymo` este instalado.
    pytest.importorskip(
        "google.cloud.storage",
        reason="requiere el extra waymo: uv sync --extra waymo",
    )
    from google.api_core import exceptions

    class BlobFalso:
        def download_to_filename(self, ruta):
            raise exceptions.Forbidden("giocrisrai@gmail.com does not have storage.objects.get")

    class BucketFalso:
        def blob(self, ruta):
            return BlobFalso()

    class ClienteFalso:
        def __init__(self, project=None):
            pass

        def bucket(self, nombre):
            return BucketFalso()

    import google.cloud.storage as gcs

    monkeypatch.setattr(gcs, "Client", ClienteFalso)

    with pytest.raises(RuntimeError, match="aceptó los términos"):
        waymo._descargar_con_cliente_python("stats", SEGMENTO, tmp_path / "s.parquet")
