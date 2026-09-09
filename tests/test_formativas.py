"""Las formativas 1–3 existen como cuestionario de sala (sin dataset extra)."""

from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
PARQUET_V2 = RAIZ / "datos" / "waymo_real" / "detecciones_reales.parquet"
FUENTE_ETICA = RAIZ / "herramientas" / "contenido_actividad14.py"


def test_formativas_tienen_al_menos_ocho_items_y_pauta() -> None:
    for numero in (1, 2, 3):
        alumno = (RAIZ / "docs" / f"formativa_{numero}.md").read_text(encoding="utf-8")
        pauta = (RAIZ / "docs" / f"formativa_{numero}_pauta.md").read_text(
            encoding="utf-8"
        )
        items = sum(1 for linea in alumno.splitlines() if linea.startswith("### "))
        assert items >= 8, f"formativa {numero} tiene {items} ítems"
        assert "Pauta docente" in pauta
        assert "No proyectar" in pauta


def test_guiones_de_las_tres_parciales() -> None:
    for nombre in ("guion_ep1.md", "guion_ep2.md", "guion_ep3.md", "guion_eft.md"):
        texto = (RAIZ / "docs" / nombre).read_text(encoding="utf-8")
        assert "defensa individual" in texto.lower() or "Defensa" in texto
        assert "Telco" in texto


def test_etica_autochequeo_usa_por_momento() -> None:
    fuente = FUENTE_ETICA.read_text(encoding="utf-8")
    assert 'por_momento["pct_nulos"].max()' in fuente
    assert "nulos_por_momento" not in fuente


def test_etica_autochequeo_reidentifica_con_id_interno() -> None:
    """El lote v2 es denso: tipo no aísla; el tracking id sí."""
    fuente = FUENTE_ETICA.read_text(encoding="utf-8")
    assert "(tres == 1).mean() > 0.5" not in fuente
    assert 'groupby(["segment_id", "timestamp_micros", "id_interno"])' in fuente
    assert "(con_id == 1).mean() == 1" in fuente


def test_etica_y_rubrica_no_citan_el_lote_de_153_segmentos() -> None:
    etica = FUENTE_ETICA.read_text(encoding="utf-8")
    rubrica = (RAIZ / "docs" / "rubrica_ra1.md").read_text(encoding="utf-8")
    for texto in (etica, rubrica):
        assert "75,5 %" not in texto
        assert "1,93 %" not in texto
        assert "| `segment_id` | 153 |" not in texto
    assert "8,2 %" in etica
    assert "100,0 %" in etica
    assert "id_interno" in etica
    assert "0 nulos" in rubrica or "0 %" in rubrica


def test_supervisado_autochequeo_no_exige_ganancia_f1_del_lote_viejo() -> None:
    """En v2 el bosque pierde en exactitud y el F1-macro apenas se mueve (+0,03)."""
    fuente = (RAIZ / "herramientas" / "contenido_actividad22.py").read_text(
        encoding="utf-8"
    )
    assert "ganancia_f1 > 0.15" not in fuente
    assert "ganancia_exactitud < 0" in fuente
    assert "ganancia_f1 < 0.08" in fuente


def test_ajuste_autochequeo_exige_que_la_ganancia_supere_el_ruido() -> None:
    """Perception v2: +0,0789 supera el 0,0424 de desviación entre pliegues."""
    fuente = (RAIZ / "herramientas" / "contenido_actividad31.py").read_text(
        encoding="utf-8"
    )
    assert "abs(delta) < ruido" not in fuente
    assert "delta > ruido" in fuente


def test_ensamble_autochequeo_no_exige_perder_contra_el_bosque() -> None:
    """En v2 el ensamble y el bosque no se distinguen; el signo no está garantizado."""
    fuente = (RAIZ / "herramientas" / "contenido_actividad32.py").read_text(
        encoding="utf-8"
    )
    assert 'assert diferencia < 0' not in fuente
    assert "abs(diferencia) < mejor_solo" in fuente
    assert 'idxmax()' in fuente
    assert 'isin(["bosque_aleatorio", "ensamble_votacion"])' not in fuente


def test_ra3_no_cita_el_ranking_del_lote_de_153() -> None:
    """En v2 gana gradient boosting 0,594; bosque 0,6909 era el lote viejo."""
    for relativo in (
        "herramientas/contenido_actividad33.py",
        "herramientas/contenido_actividad32.py",
        "docs/rubrica_ra3.md",
        "README.md",
    ):
        texto = (RAIZ / relativo).read_text(encoding="utf-8")
        assert "0,6909" not in texto, relativo
        assert "0,6869" not in texto, relativo
        assert "−0,0006" not in texto, relativo
    rubrica = (RAIZ / "docs" / "rubrica_ra3.md").read_text(encoding="utf-8")
    assert "0,594" in rubrica
    assert "0,5938" in rubrica


def test_pautas_no_citan_exactitud_del_csv_sintetico() -> None:
    """89,65 % / 88,96 % / F1 0,46 eran el lote sintético."""
    for relativo in (
        "README.md",
        "herramientas/contenido_actividad21.py",
        "herramientas/contenido_actividad22.py",
        "herramientas/contenido_actividad24.py",
        "herramientas/contenido_kedro.py",
    ):
        texto = (RAIZ / relativo).read_text(encoding="utf-8")
        assert "89,65" not in texto, relativo
        assert "88,96" not in texto, relativo
    kedro = (RAIZ / "herramientas" / "contenido_kedro.py").read_text(encoding="utf-8")
    assert "153 segmentos a caballo" not in kedro
    nodos = (
        RAIZ
        / "kedro_mly1101"
        / "src"
        / "kedro_mly1101"
        / "pipelines"
        / "supervisado"
        / "nodes.py"
    ).read_text(encoding="utf-8")
    assert "cerca de cero" not in nodos
    assert "99,98" not in nodos
    ajuste = (RAIZ / "herramientas" / "contenido_actividad31.py").read_text(
        encoding="utf-8"
    )
    assert "0,6964" not in ajuste
    assert "0,5893" in ajuste


@pytest.mark.skipif(
    not PARQUET_V2.exists(),
    reason="requiere Perception v2 en datos/waymo_real/",
)
def test_lote_v2_id_interno_es_unico_por_frame() -> None:
    df = pd.read_parquet(PARQUET_V2)
    tres = df.groupby(["segment_id", "timestamp_micros", "object_type"]).size()
    con_id = df.groupby(["segment_id", "timestamp_micros", "id_interno"]).size()
    assert (tres == 1).mean() < 0.15
    assert (con_id == 1).mean() == 1.0
    assert round(100 * (tres == 1).mean(), 1) == 8.2


def test_setup_de_notebooks_no_asume_cwd_en_notebooks() -> None:
    """En Cursor el kernel arranca en la raíz del repo, no en notebooks/."""
    for ruta in (RAIZ / "herramientas").glob("contenido_*.py"):
        texto = ruta.read_text(encoding="utf-8")
        assert 'RAIZ = Path("..").resolve()' not in texto, ruta.name
