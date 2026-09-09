"""Las formativas 1–3 existen como cuestionario de sala (sin dataset extra)."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


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
