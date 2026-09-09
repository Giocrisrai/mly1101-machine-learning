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

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import waymo  # noqa: E402

SEGMENTO = "10023947602400723454_1120_000_1140_000"


def test_ruta_gcs_apunta_al_split_de_entrenamiento() -> None:
    assert waymo.ruta_gcs("lidar_box", SEGMENTO) == f"training/lidar_box/{SEGMENTO}.parquet"


def test_en_colab_es_falso_fuera_de_colab() -> None:
    assert waymo.en_colab() is False


def test_encontrar_raiz_desde_el_repo_o_notebooks() -> None:
    assert waymo.encontrar_raiz(RAIZ) == RAIZ
    assert waymo.encontrar_raiz(RAIZ / "notebooks") == RAIZ


def test_exigir_detecciones_reales_recupera_si_el_cwd_es_el_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(RAIZ)
    if not (RAIZ / "datos" / "waymo_real" / "detecciones_reales.parquet").exists():
        pytest.skip("sin parquet local")
    ruta = waymo.exigir_detecciones_reales(RAIZ.parent)
    assert ruta.exists()


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
    monkeypatch.setattr(waymo, "usar_cliente_python", lambda: False)
    monkeypatch.setattr(waymo, "_descargar_con_cliente_python",
                        lambda c, s, d: llamadas.append("python"))
    monkeypatch.setattr(waymo, "_descargar_con_gsutil",
                        lambda c, s, d: (llamadas.append("gsutil"), d.write_bytes(b"x")))

    waymo.descargar("stats", SEGMENTO, tmp_path)
    assert llamadas == ["gsutil"]


def test_en_local_con_adc_usa_el_cliente_python(monkeypatch, tmp_path: Path) -> None:
    """ADC autentica al cliente Python aunque `gcloud auth list` esté vacío."""
    llamadas = []
    monkeypatch.setattr(waymo, "en_colab", lambda: False)
    monkeypatch.setattr(waymo, "usar_cliente_python", lambda: True)
    monkeypatch.setattr(
        waymo,
        "_descargar_con_cliente_python",
        lambda c, s, d: (llamadas.append("python"), d.write_bytes(b"x")),
    )
    monkeypatch.setattr(waymo, "_descargar_con_gsutil", lambda c, s, d: llamadas.append("gsutil"))

    waymo.descargar("stats", SEGMENTO, tmp_path)
    assert llamadas == ["python"]


def test_exigir_credenciales_acepta_adc(monkeypatch) -> None:
    monkeypatch.setattr(waymo, "en_colab", lambda: False)
    monkeypatch.setattr(waymo, "hay_adc", lambda: True)
    monkeypatch.setattr(waymo, "hay_cuenta_gcloud", lambda: False)
    assert waymo.exigir_credenciales_gcs() == "python"


def test_adc_gana_aunque_gcloud_liste_una_cuenta(monkeypatch) -> None:
    """El token de `gcloud auth login` puede estar caducado y ADC vigente."""
    monkeypatch.setattr(waymo, "en_colab", lambda: False)
    monkeypatch.setattr(waymo, "hay_adc", lambda: True)
    monkeypatch.setattr(waymo, "hay_cuenta_gcloud", lambda: True)
    assert waymo.usar_cliente_python() is True
    assert waymo.exigir_credenciales_gcs() == "python"


def test_exigir_credenciales_falla_sin_login_ni_adc(monkeypatch) -> None:
    monkeypatch.setattr(waymo, "en_colab", lambda: False)
    monkeypatch.setattr(waymo, "hay_adc", lambda: False)
    monkeypatch.setattr(waymo, "hay_cuenta_gcloud", lambda: False)
    with pytest.raises(RuntimeError, match="application-default login"):
        waymo.exigir_credenciales_gcs()


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
    monkeypatch.setattr(waymo, "usar_cliente_python", lambda: False)
    monkeypatch.setattr(waymo, "_descargar_con_gsutil",
                        lambda c, s, d: (llamadas.append("gsutil"), d.write_bytes(b"nuevo")))
    (tmp_path / "stats.parquet").write_bytes(b"viejo")

    waymo.descargar("stats", SEGMENTO, tmp_path, forzar=True)
    assert llamadas == ["gsutil"]


def test_crea_la_carpeta_de_destino(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(waymo, "en_colab", lambda: False)
    monkeypatch.setattr(waymo, "usar_cliente_python", lambda: False)
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
    monkeypatch.setattr(waymo, "usar_cliente_python", lambda: False)
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


# ---------------------------------------------------------------------------
# Catálogo de productos GCS y descarga genérica (Motion, E2E, Perception v1)
# ---------------------------------------------------------------------------

BUCKETS_ESPERADOS = {
    "percepcion_v2": "waymo_open_dataset_v_2_0_1",
    "percepcion_v1": "waymo_open_dataset_v_1_4_3",
    "motion": "waymo_open_dataset_motion_v_1_3_1",
    "e2e_camara": "waymo_open_dataset_end_to_end_camera_v_1_0_0",
}

CAMPOS_OBLIGATORIOS = (
    "bucket",
    "formato",
    "prefijo_muestra",
    "consola",
    "para_que",
    "pagina",
    "en_clase",
    "tamano_medido",
    "sustituto",
)


def test_el_catalogo_incluye_los_cuatro_productos_de_waymo() -> None:
    """Los tres buckets que el alumno ve en la consola GCS, más el v2 del curso."""
    for clave, bucket in BUCKETS_ESPERADOS.items():
        assert waymo.CATALOGO_BUCKETS[clave]["bucket"] == bucket


def test_las_imagenes_no_entran_en_el_lote_de_clase() -> None:
    assert waymo.COMPONENTES_V2["camera_image"]["uso"] == "no"
    assert waymo.COMPONENTES_V2["lidar"]["uso"] == "no"
    assert set(waymo.COMPONENTES_LIVIANOS) == {"lidar_box", "stats"}


def test_solo_v2_y_el_json_e2e_entran_al_lote_de_clase() -> None:
    """v1/Motion/video E2E superan TAMANO_MAXIMO_CLASE_MB; AWS no cambia el tope."""
    assert waymo.CATALOGO_BUCKETS["percepcion_v2"]["en_clase"] == "tabla"
    assert waymo.CATALOGO_BUCKETS["e2e_camara"]["en_clase"] == "json"
    assert waymo.CATALOGO_BUCKETS["percepcion_v1"]["en_clase"] == "listar"
    assert waymo.CATALOGO_BUCKETS["motion"]["en_clase"] == "listar"
    assert waymo.TAMANO_MAXIMO_CLASE_MB == 250.0


def test_recorte_de_un_frame_deja_un_instante_y_una_camara() -> None:
    camara = pd.DataFrame(
        {
            "segment_id": ["s1", "s1", "s1", "s1"],
            "timestamp_micros": [10, 10, 10, 20],
            "camara": ["FRONT", "FRONT", "SIDE_LEFT", "FRONT"],
            "object_type": ["vehicle", "pedestrian", "sign", "cyclist"],
            "box_center_x_px": [100.0, 200.0, 50.0, 3.0],
            "box_center_y_px": [80.0, 90.0, 40.0, 1.0],
            "box_width_px": [40.0, 20.0, 10.0, 4.0],
            "box_height_px": [20.0, 40.0, 10.0, 4.0],
        }
    )
    frame = waymo.recorte_de_un_frame(camara, timestamp_micros=10)
    assert len(frame) == 2
    assert set(frame["object_type"]) == {"vehicle", "pedestrian"}
    assert set(frame["camara"]) == {"FRONT"}
    assert waymo.tamano_del_lienzo(pd.DataFrame()) == (1920, 1280)
    rects = waymo.rectangulos_del_frame(frame)
    vehiculo = rects.loc[rects["object_type"] == "vehicle"].iloc[0]
    assert vehiculo["x0"] == 80.0
    assert vehiculo["y0"] == 70.0
    assert vehiculo["ancho"] == 40.0
    assert vehiculo["alto"] == 20.0


def test_v1_motion_y_video_e2e_declaran_sustituto_no_el_tfrecord() -> None:
    """Los videos no se resuelven: cada producto dice qué usar en su lugar."""
    v1 = waymo.que_hacer_con_el_producto("percepcion_v1")
    motion = waymo.que_hacer_con_el_producto("motion")
    e2e = waymo.que_hacer_con_el_producto("e2e_camara")
    assert "parquet" in v1
    assert "vehicle_pose" in motion
    assert "1,56 GB" in e2e
    assert "479" in e2e
    assert waymo.CATALOGO_BUCKETS["percepcion_v2"]["en_clase"] == "tabla"


def test_e2e_muestra_es_el_json_liviano_no_una_carpeta() -> None:
    """En la página de descarga el E2E son tfrecord de ~1,6 GB; el JSON sí cabe."""
    prefijo = waymo.CATALOGO_BUCKETS["e2e_camara"]["prefijo_muestra"]
    assert prefijo.endswith(".json")
    assert "val_sequence" in prefijo
    assert prefijo != "val_sequence"


def test_los_tutoriales_oficiales_apuntan_a_colab_de_waymo() -> None:
    for clave, meta in waymo.TUTORIALES_OFICIALES.items():
        assert meta["colab"].startswith(
            "https://colab.research.google.com/github/waymo-research/"
        )
        assert meta["que"]
    assert "faq" in waymo.FAQ_WAYMO
    assert "tutorial" in waymo.MUESTRAS_OFICIALES_GITHUB
    assert waymo.VIEWER_PARQUET_V2["url"] == "https://egolens.org"
    assert "egolens" in waymo.VIEWER_PARQUET_V2["repo"]
    assert waymo.PAGINA_DESCARGA.startswith("https://waymo.com/open/download")
    for producto in waymo.CATALOGO_BUCKETS.values():
        assert producto["pagina"] == waymo.PAGINA_DESCARGA


def test_los_dieciseis_tutoriales_oficiales_estan_en_la_guia() -> None:
    """El repo Waymo tiene 16 notebooks; MLY1101 solo enlaza 4. No copiar el resto."""
    assert len(waymo.TUTORIALES_REPO_OFICIAL) == 16
    assert waymo.TUTORIALES_EN_CLASE <= set(waymo.TUTORIALES_REPO_OFICIAL)
    assert len(waymo.TUTORIALES_EN_CLASE) == 4
    guia = (RAIZ / "docs" / "productos_waymo.md").read_text(encoding="utf-8")
    for nombre in waymo.TUTORIALES_REPO_OFICIAL:
        assert f"`{nombre}`" in guia, nombre
    fuente = (RAIZ / "herramientas" / "contenido_waymo_buckets.py").read_text(
        encoding="utf-8"
    )
    assert "productos_waymo.md" in fuente
    assert "16" in fuente


def test_descargar_camera_box_pide_el_parquet_2d_no_la_imagen(
    monkeypatch, tmp_path: Path
) -> None:
    visto: list[tuple] = []

    def fake(componente: str, segmento: str, carpeta: Path, forzar: bool = False) -> Path:
        visto.append((componente, segmento))
        destino = carpeta / f"{componente}.parquet"
        destino.write_bytes(b"ok")
        return destino

    monkeypatch.setattr(waymo, "descargar", fake)
    ruta = waymo.descargar_camera_box("seg_x", tmp_path)
    assert visto == [("camera_box", "seg_x")]
    assert ruta.name == "camera_box.parquet"


def test_cada_producto_declara_formato_prefijo_y_enlace_de_consola() -> None:
    for clave, producto in waymo.CATALOGO_BUCKETS.items():
        for campo in CAMPOS_OBLIGATORIOS:
            assert producto[campo], f"{clave} sin {campo}"
        assert producto["consola"].startswith("https://console.cloud.google.com/storage/browser/")
        assert producto["bucket"] in producto["consola"]


def test_producto_desconocido_lista_las_claves_disponibles() -> None:
    with pytest.raises(KeyError, match="percepcion_v2"):
        waymo.producto("no_existe")


def test_listar_en_colab_usa_el_cliente_python(monkeypatch) -> None:
    llamadas = []
    monkeypatch.setattr(waymo, "en_colab", lambda: True)
    monkeypatch.setattr(
        waymo,
        "_listar_con_cliente_python",
        lambda b, p, n: llamadas.append(("python", b, p, n)) or [{"nombre": "a", "bytes": 1}],
    )
    monkeypatch.setattr(
        waymo,
        "_listar_con_gsutil",
        lambda *a: llamadas.append("gsutil"),
    )

    objetos = waymo.listar_objetos("waymo_open_dataset_v_2_0_1", "training/stats/", limite=3)
    assert llamadas == [("python", "waymo_open_dataset_v_2_0_1", "training/stats/", 3)]
    assert objetos[0]["nombre"] == "a"


def test_listar_en_local_usa_gsutil(monkeypatch) -> None:
    llamadas = []
    monkeypatch.setattr(waymo, "en_colab", lambda: False)
    monkeypatch.setattr(waymo, "usar_cliente_python", lambda: False)
    monkeypatch.setattr(waymo, "_listar_con_cliente_python", lambda *a: llamadas.append("python"))
    monkeypatch.setattr(
        waymo,
        "_listar_con_gsutil",
        lambda b, p, n: llamadas.append(("gsutil", b, p, n)) or [],
    )

    waymo.listar_objetos("waymo_open_dataset_motion_v_1_3_1", "uncompressed/", limite=5)
    assert llamadas == [("gsutil", "waymo_open_dataset_motion_v_1_3_1", "uncompressed/", 5)]


def test_listar_con_gsutil_parsea_nombre_y_tamano(monkeypatch) -> None:
    salida = (
        "      23456  2025-03-01T12:00:00Z  "
        "gs://waymo_open_dataset_v_2_0_1/training/stats/abc.parquet\n"
        "                                 gs://waymo_open_dataset_v_2_0_1/training/stats/\n"
        "TOTAL: 2 objects, 23456 bytes\n"
    )
    monkeypatch.setattr(waymo, "shutil", type("f", (), {"which": staticmethod(lambda x: "/usr/bin/gsutil")}))
    monkeypatch.setattr(
        waymo.subprocess,
        "run",
        lambda *a, **k: type("r", (), {"returncode": 0, "stdout": salida, "stderr": ""})(),
    )

    objetos = waymo._listar_con_gsutil("waymo_open_dataset_v_2_0_1", "training/stats/", 8)
    assert objetos == [
        {
            "nombre": "training/stats/abc.parquet",
            "bytes": 23456,
            "gs": "gs://waymo_open_dataset_v_2_0_1/training/stats/abc.parquet",
        }
    ]


def test_descargar_objeto_en_colab_usa_el_cliente_python(monkeypatch, tmp_path: Path) -> None:
    llamadas = []
    monkeypatch.setattr(waymo, "en_colab", lambda: True)
    monkeypatch.setattr(
        waymo,
        "_copiar_blob_python",
        lambda b, n, d: (llamadas.append(("python", b, n)), d.write_bytes(b"x")),
    )
    monkeypatch.setattr(waymo, "_copiar_blob_gsutil", lambda *a: llamadas.append("gsutil"))
    monkeypatch.setattr(waymo, "_tamano_blob", lambda *a: 1024)

    ruta = waymo.descargar_objeto(
        "waymo_open_dataset_v_1_4_3",
        "val_sequence_name_to_scenario_cluster.json",
        tmp_path,
    )
    assert llamadas == [
        ("python", "waymo_open_dataset_v_1_4_3", "val_sequence_name_to_scenario_cluster.json")
    ]
    assert ruta.name == "val_sequence_name_to_scenario_cluster.json"
    assert ruta.exists()


def test_descargar_objeto_rechaza_archivos_grandes(monkeypatch, tmp_path: Path) -> None:
    """Un tfrecord de Perception v1 o Motion pesa cientos de MB: no se baja por accidente."""
    monkeypatch.setattr(waymo, "en_colab", lambda: False)
    monkeypatch.setattr(waymo, "_tamano_blob", lambda *a: 800 * 1024 * 1024)
    monkeypatch.setattr(waymo, "_copiar_blob_gsutil", lambda *a: (_ for _ in ()).throw(AssertionError("no debía copiar")))

    with pytest.raises(RuntimeError, match="800"):
        waymo.descargar_objeto(
            "waymo_open_dataset_v_1_4_3",
            "individual_files/training/segment-x.tfrecord",
            tmp_path,
        )


def test_descargar_objeto_permite_forzar_un_archivo_grande(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(waymo, "en_colab", lambda: False)
    monkeypatch.setattr(waymo, "usar_cliente_python", lambda: False)
    monkeypatch.setattr(waymo, "_tamano_blob", lambda *a: 800 * 1024 * 1024)
    monkeypatch.setattr(waymo, "_copiar_blob_gsutil", lambda b, n, d: d.write_bytes(b"ok"))

    ruta = waymo.descargar_objeto(
        "waymo_open_dataset_v_1_4_3",
        "individual_files/training/segment-x.tfrecord",
        tmp_path,
        tamano_maximo_mb=None,
    )
    assert ruta.read_bytes() == b"ok"


def _objetos_de_muestra() -> list[dict]:
    return [
        {"nombre": "grande.tfrecord", "bytes": 800 * 1024 * 1024, "gs": "gs://b/grande.tfrecord"},
        {"nombre": "chico.parquet", "bytes": 1_024_000, "gs": "gs://b/chico.parquet"},
        {"nombre": "mediano.json", "bytes": 80 * 1024 * 1024, "gs": "gs://b/mediano.json"},
    ]


def test_elegir_muestra_se_queda_con_el_archivo_mas_chico_bajo_el_tope() -> None:
    elegido = waymo.elegir_muestra(_objetos_de_muestra(), tamano_maximo_mb=50)
    assert elegido["nombre"] == "chico.parquet"


def test_elegir_muestra_falla_si_todo_supera_el_tope() -> None:
    grandes = [o for o in _objetos_de_muestra() if o["bytes"] > 100 * 1024 * 1024]
    with pytest.raises(RuntimeError, match="800"):
        waymo.elegir_muestra(grandes, tamano_maximo_mb=50)


def test_elegir_muestra_falla_si_el_listado_esta_vacio() -> None:
    with pytest.raises(RuntimeError, match="vacío"):
        waymo.elegir_muestra([])


def test_descargar_muestra_usa_el_catalogo_y_baja_el_mas_chico(monkeypatch, tmp_path: Path) -> None:
    """Así trabajan los alumnos: listar el prefijo del producto y copiar un fragmento real."""
    llamadas = []

    def listar(bucket, prefijo, limite=8):
        llamadas.append(("listar", bucket, prefijo, limite))
        return [
            {"nombre": f"{prefijo}a.parquet", "bytes": 2_000_000, "gs": "gs://x/a"},
            {"nombre": f"{prefijo}b.parquet", "bytes": 500_000, "gs": "gs://x/b"},
        ]

    def copiar(bucket, blob, carpeta, forzar=False, tamano_maximo_mb=None):
        llamadas.append(("copiar", bucket, blob, tamano_maximo_mb))
        destino = carpeta / Path(blob).name
        destino.write_bytes(b"real")
        return destino

    monkeypatch.setattr(waymo, "listar_objetos", listar)
    monkeypatch.setattr(waymo, "descargar_objeto", copiar)

    ruta = waymo.descargar_muestra("percepcion_v2", tmp_path)
    producto = waymo.producto("percepcion_v2")
    assert llamadas[0] == ("listar", producto["bucket"], producto["prefijo_muestra"], 12)
    assert llamadas[1][0] == "copiar"
    assert llamadas[1][1] == producto["bucket"]
    assert llamadas[1][2].endswith("b.parquet")
    assert ruta.read_bytes() == b"real"


def test_el_reauth_de_gcloud_se_traduce_a_un_mensaje_accionable(monkeypatch) -> None:
    """La cuenta aparece en `gcloud auth list` pero el token ya no sirve."""
    monkeypatch.setattr(waymo, "shutil", type("f", (), {"which": staticmethod(lambda x: "/usr/bin/gsutil")}))
    monkeypatch.setattr(
        waymo.subprocess,
        "run",
        lambda *a, **k: type(
            "r",
            (),
            {
                "returncode": 1,
                "stdout": "",
                "stderr": "Reauthentication required.\ncannot prompt during non-interactive execution.",
            },
        )(),
    )
    with pytest.raises(RuntimeError, match="gcloud auth login"):
        waymo._listar_con_gsutil("waymo_open_dataset_v_2_0_1", "training/stats/", 1)


def test_resumir_fragmento_parquet_cuenta_filas_y_numericas(tmp_path: Path) -> None:
    tabla = pd.DataFrame({"box_length": [1.0, 2.0], "object_type": ["vehicle", "sign"]})
    ruta = tmp_path / "lidar_box.parquet"
    tabla.to_parquet(ruta)
    resumen = waymo.resumir_fragmento(ruta)
    assert resumen["formato"] == "parquet"
    assert resumen["filas"] == 2
    assert resumen["columnas"] == 2
    assert resumen["numericas"] == 1


def test_resumir_fragmento_json_cuenta_claves(tmp_path: Path) -> None:
    ruta = tmp_path / "meta.json"
    ruta.write_text('{"a": 1, "b": [2, 3]}', encoding="utf-8")
    resumen = waymo.resumir_fragmento(ruta)
    assert resumen["formato"] == "json"
    assert resumen["claves"] == ["a", "b"]


def test_informe_analitica_deja_claro_el_desbalance_y_las_cajas_vacias() -> None:
    tabla = pd.DataFrame(
        {
            "segment_id": ["s1", "s1", "s2"],
            "object_type": ["vehicle", "cyclist", "vehicle"],
            "detection_difficulty": ["LEVEL_1", "LEVEL_2", "LEVEL_1"],
            "num_lidar_points": [10, 0, 4],
        }
    )
    informe = waymo.informe_analitica(tabla)
    assert informe["filas"] == 3
    assert informe["segmentos"] == 2
    assert informe["tipos"]["vehicle"] == 2
    assert informe["cajas_sin_puntos"] == 1
    assert "proyecto" in informe["siguiente"]


def test_preparar_lote_concatena_segmentos_en_un_parquet(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        waymo,
        "listar_objetos",
        lambda *a, **k: [
            {"nombre": "training/lidar_box/aa.parquet", "bytes": 10, "gs": "gs://x/aa"},
            {"nombre": "training/lidar_box/bb.parquet", "bytes": 10, "gs": "gs://x/bb"},
        ],
    )

    def fake_descargar(segmento: str, carpeta: Path) -> dict[str, Path]:
        carpeta.mkdir(parents=True, exist_ok=True)
        lb = "[LiDARBoxComponent]"
        st = "[StatsComponent]"
        cajas = pd.DataFrame(
            {
                "key.segment_context_name": [segmento],
                "key.frame_timestamp_micros": [1000],
                "key.laser_object_id": ["o1"],
                f"{lb}.type": [1],
                f"{lb}.box.center.x": [1.0],
                f"{lb}.box.center.y": [0.0],
                f"{lb}.box.center.z": [0.5],
                f"{lb}.box.size.x": [4.0],
                f"{lb}.box.size.y": [2.0],
                f"{lb}.box.size.z": [1.5],
                f"{lb}.speed.x": [0.0],
                f"{lb}.speed.y": [0.0],
                f"{lb}.num_lidar_points_in_box": [9],
                f"{lb}.difficulty_level.detection": [float("nan")],
            }
        )
        stats = pd.DataFrame(
            {
                "key.segment_context_name": [segmento],
                "key.frame_timestamp_micros": [1000],
                f"{st}.time_of_day": ["Day"],
                f"{st}.weather": ["sunny"],
                f"{st}.location": ["location_sf"],
            }
        )
        rutas = {
            "lidar_box": carpeta / "lidar_box.parquet",
            "stats": carpeta / "stats.parquet",
        }
        cajas.to_parquet(rutas["lidar_box"])
        stats.to_parquet(rutas["stats"])
        return rutas

    monkeypatch.setattr(waymo, "descargar_segmento", fake_descargar)

    tabla, informe, ruta = waymo.preparar_lote(tmp_path, n=2)
    assert set(tabla["segment_id"]) == {"aa", "bb"}
    assert informe["filas"] == 2
    assert ruta.name == "detecciones_reales.parquet"
    assert pd.read_parquet(ruta).shape[0] == 2


def test_cargar_o_preparar_reusa_el_parquet_si_ya_existe(monkeypatch, tmp_path: Path) -> None:
    tabla = pd.DataFrame(
        {
            "segment_id": ["z"],
            "object_type": ["sign"],
            "detection_difficulty": ["LEVEL_1"],
            "num_lidar_points": [1],
        }
    )
    destino = tmp_path / "detecciones_reales.parquet"
    tabla.to_parquet(destino)
    monkeypatch.setattr(
        waymo,
        "preparar_lote",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no debía volver a bajar")),
    )
    leida, informe, ruta = waymo.cargar_o_preparar(tmp_path, n=8)
    assert ruta == destino
    assert informe["filas"] == 1
    assert leida["object_type"].iloc[0] == "sign"


def _escribir_segmento_falso(carpeta: Path, nombre: str, tipo: int = 1) -> None:
    destino = carpeta / "muestra" / nombre
    destino.mkdir(parents=True, exist_ok=True)
    lb = "[LiDARBoxComponent]"
    st = "[StatsComponent]"
    pd.DataFrame(
        {
            "key.segment_context_name": [nombre],
            "key.frame_timestamp_micros": [1000],
            "key.laser_object_id": ["o1"],
            f"{lb}.type": [tipo],
            f"{lb}.box.center.x": [1.0],
            f"{lb}.box.center.y": [0.0],
            f"{lb}.box.center.z": [0.5],
            f"{lb}.box.size.x": [4.0],
            f"{lb}.box.size.y": [2.0],
            f"{lb}.box.size.z": [1.5],
            f"{lb}.speed.x": [0.0],
            f"{lb}.speed.y": [0.0],
            f"{lb}.num_lidar_points_in_box": [9],
            f"{lb}.difficulty_level.detection": [float("nan")],
        }
    ).to_parquet(destino / "lidar_box.parquet")
    pd.DataFrame(
        {
            "key.segment_context_name": [nombre],
            "key.frame_timestamp_micros": [1000],
            f"{st}.time_of_day": ["Day"],
            f"{st}.weather": ["sunny"],
            f"{st}.location": ["location_sf"],
        }
    ).to_parquet(destino / "stats.parquet")


def test_ensamblar_muestra_concatena_segmentos_completos(tmp_path: Path) -> None:
    _escribir_segmento_falso(tmp_path, "seg_a", tipo=1)
    _escribir_segmento_falso(tmp_path, "seg_b", tipo=2)
    tabla = waymo.ensamblar_muestra(tmp_path / "muestra")
    assert set(tabla["segment_id"]) == {"seg_a", "seg_b"}
    assert len(tabla) == 2


def test_cargar_o_preparar_prefiere_la_muestra_al_parquet_de_un_segmento(
    tmp_path: Path,
) -> None:
    """El parquet suelto de un segmento no sirve para partir train/test."""
    viejo = pd.DataFrame(
        {
            "segment_id": ["viejo"],
            "object_type": ["sign"],
            "detection_difficulty": ["LEVEL_1"],
            "num_lidar_points": [1],
        }
    )
    viejo.to_parquet(tmp_path / "detecciones_reales.parquet")
    _escribir_segmento_falso(tmp_path, "aa")
    _escribir_segmento_falso(tmp_path, "bb")
    tabla, informe, _ruta = waymo.cargar_o_preparar(tmp_path, n=8)
    assert informe["segmentos"] == 2
    assert set(tabla["segment_id"]) == {"aa", "bb"}


def test_partir_por_grupo_no_reparte_un_segmento_en_los_dos_lados() -> None:
    tabla = pd.DataFrame(
        {
            "segment_id": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
            "x": range(12),
        }
    )
    marcada = waymo.partir_por_grupo(tabla, test_size=0.34, semilla=0)
    train = set(marcada.loc[marcada["particion"] == "entrenamiento", "segment_id"])
    test = set(marcada.loc[marcada["particion"] == "prueba", "segment_id"])
    assert train.isdisjoint(test)
    assert train | test == {"a", "b", "c"}
    assert len(test) >= 1


def test_partir_por_grupo_falla_con_un_solo_segmento() -> None:
    tabla = pd.DataFrame({"segment_id": ["solo"] * 5, "x": range(5)})
    with pytest.raises(ValueError, match="al menos 2"):
        waymo.partir_por_grupo(tabla)


def test_inventario_muestra_lista_componentes_sin_inventar_imagenes(
    tmp_path: Path,
) -> None:
    _escribir_segmento_falso(tmp_path, "aa")
    inventario = waymo.inventario_muestra(tmp_path / "muestra")
    assert set(inventario["componente"]) == {"lidar_box", "stats"}
    assert inventario["completo"].all()
    assert "camera_image" not in set(inventario["componente"])


def test_inventario_fuentes_declara_que_entra_al_modelo_y_que_no(
    tmp_path: Path,
) -> None:
    _escribir_segmento_falso(tmp_path, "aa")
    (tmp_path / "muestra" / "aa" / "camera_box.parquet").write_bytes(b"x" * 1000)
    (tmp_path / "val_sequence_name_to_scenario_cluster.json").write_text(
        '{"seq1": "cluster_a"}', encoding="utf-8"
    )
    inv = waymo.inventario_fuentes(tmp_path).set_index("fuente")
    assert bool(inv.loc["percepcion_v2", "entra_al_modelo"])
    assert not bool(inv.loc["camera_box", "entra_al_modelo"])
    assert not bool(inv.loc["e2e_camara", "entra_al_modelo"])
    assert not bool(inv.loc["camera_image", "entra_al_modelo"])
    assert int(inv.loc["percepcion_v2", "archivos"]) == 1
    assert int(inv.loc["camera_box", "archivos"]) == 1
    assert int(inv.loc["e2e_camara", "archivos"]) == 1
    assert int(inv.loc["camera_image", "archivos"]) == 0


def test_descargar_tablas_chicas_pide_tres_parquet_no_jpeg(
    monkeypatch, tmp_path: Path
) -> None:
    visto: list[str] = []

    def fake(componente: str, segmento: str, carpeta: Path, forzar: bool = False) -> Path:
        visto.append(componente)
        destino = carpeta / f"{componente}.parquet"
        destino.write_bytes(b"ok")
        return destino

    monkeypatch.setattr(waymo, "descargar", fake)
    rutas = waymo.descargar_tablas_chicas("seg_x", tmp_path)
    assert visto == list(waymo.COMPONENTES_MANIPULABLES)
    assert set(rutas) == set(waymo.COMPONENTES_MANIPULABLES)
    assert "camera_image" not in visto


def test_traducir_camera_box_nombres_en_pixeles_y_nan_es_level1() -> None:
    crudo = pd.DataFrame(
        {
            "key.segment_context_name": ["s1", "s1"],
            "key.frame_timestamp_micros": [1, 2],
            "key.camera_name": [1, 2],
            "key.camera_object_id": ["a", "b"],
            "[CameraBoxComponent].type": [1, 2],
            "[CameraBoxComponent].box.center.x": [10.0, 20.0],
            "[CameraBoxComponent].box.center.y": [5.0, 6.0],
            "[CameraBoxComponent].box.size.x": [30.0, 40.0],
            "[CameraBoxComponent].box.size.y": [15.0, 16.0],
            "[CameraBoxComponent].difficulty_level.detection": [np.nan, 2.0],
        }
    )
    tabla = waymo.traducir_camera_box(crudo)
    assert list(tabla["object_type"]) == ["vehicle", "pedestrian"]
    assert list(tabla["detection_difficulty"]) == ["LEVEL_1", "LEVEL_2"]
    assert list(tabla["camara"]) == ["FRONT", "FRONT_LEFT"]
    assert "box_center_x_px" in tabla.columns
    assert "box_center_x" not in tabla.columns


def test_traducir_pose_extrae_xyz_de_la_matriz() -> None:
    matriz = [0] * 16
    matriz[3], matriz[7], matriz[11] = 10.0, 20.0, 3.0
    crudo = pd.DataFrame(
        {
            "key.segment_context_name": ["s1"],
            "key.frame_timestamp_micros": [1],
            "[VehiclePoseComponent].world_from_vehicle.transform": [matriz],
        }
    )
    pose = waymo.traducir_pose_vehiculo(crudo)
    assert pose.loc[0, ["pos_x", "pos_y", "pos_z"]].tolist() == [10.0, 20.0, 3.0]


def test_comparar_conteos_no_une_filas_y_deja_sign_solo_en_lidar() -> None:
    lidar = pd.DataFrame({"object_type": ["vehicle", "vehicle", "sign"]})
    camara = pd.DataFrame({"object_type": ["vehicle", "pedestrian"]})
    tabla = waymo.comparar_conteos_por_tipo(lidar, camara).set_index("object_type")
    assert int(tabla.loc["vehicle", "lidar_n"]) == 2
    assert int(tabla.loc["vehicle", "camara_n"]) == 1
    assert int(tabla.loc["sign", "camara_n"]) == 0
    assert int(tabla.loc["pedestrian", "lidar_n"]) == 0


def test_completar_tablas_chicas_solo_segmentos_completos(
    monkeypatch, tmp_path: Path
) -> None:
    bueno = tmp_path / "aa"
    incompleto = tmp_path / "bb"
    bueno.mkdir()
    incompleto.mkdir()
    (bueno / "lidar_box.parquet").write_bytes(b"x")
    (bueno / "stats.parquet").write_bytes(b"x")
    (incompleto / "stats.parquet").write_bytes(b"x")
    visto: list[str] = []

    def fake(segmento: str, carpeta, **kwargs):
        visto.append(segmento)
        return {}

    monkeypatch.setattr(waymo, "descargar_tablas_chicas", fake)
    nombres = waymo.completar_tablas_chicas(tmp_path, limite=8)
    assert nombres == ["aa"]
    assert visto == ["aa"]
    ficha = waymo.ficha_tabla(pd.DataFrame())
    assert list(ficha.columns) == ["columna", "dtype", "nulos_pct", "n_unicos"]


def test_ensamblar_componente_concatena(tmp_path: Path) -> None:
    a = tmp_path / "aa"
    b = tmp_path / "bb"
    a.mkdir()
    b.mkdir()
    pd.DataFrame({"x": [1]}).to_parquet(a / "vehicle_pose.parquet")
    pd.DataFrame({"x": [2]}).to_parquet(b / "vehicle_pose.parquet")
    tabla = waymo.ensamblar_componente(tmp_path, "vehicle_pose")
    assert sorted(tabla["x"].tolist()) == [1, 2]
    tabla = waymo.ensamblar_camera_box(tmp_path / "muestra")
    assert tabla.empty


def test_leer_metadatos_e2e_vacio_si_no_hay_json(tmp_path: Path) -> None:
    tabla = waymo.leer_metadatos_e2e(tmp_path)
    assert tabla.empty


def test_leer_metadatos_e2e_tabula_el_json(tmp_path: Path) -> None:
    (tmp_path / "val_sequence_name_to_scenario_cluster.json").write_text(
        '{"aaa": "urban", "bbb": "highway"}', encoding="utf-8"
    )
    tabla = waymo.leer_metadatos_e2e(tmp_path)
    assert set(tabla.columns) == {"secuencia", "cluster"}
    assert set(tabla["secuencia"]) == {"aaa", "bbb"}


def test_cargar_tabla_curso_exige_el_lote_real(tmp_path: Path) -> None:
    real_dir = tmp_path / "datos" / "waymo_real"
    real_dir.mkdir(parents=True)
    tabla = pd.DataFrame({"object_type": ["vehicle"], "num_lidar_points": [3]})
    tabla.to_parquet(real_dir / "detecciones_reales.parquet")
    df, origen, ruta = waymo.cargar_tabla_curso(tmp_path)
    assert origen == "real"
    assert len(df) == 1
    assert ruta.name == "detecciones_reales.parquet"


def test_cargar_tabla_curso_falla_si_no_hay_lote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="descargar_waymo.py --muestra"):
        waymo.cargar_tabla_curso(tmp_path)


def test_leer_tabla_distingue_parquet_de_csv(tmp_path: Path) -> None:
    parquet = tmp_path / "a.parquet"
    csv = tmp_path / "b.csv"
    pd.DataFrame({"x": [1]}).to_parquet(parquet)
    pd.DataFrame({"x": [2]}).to_csv(csv, index=False)
    assert waymo.leer_tabla(parquet)["x"].iloc[0] == 1
    assert waymo.leer_tabla(csv)["x"].iloc[0] == 2
