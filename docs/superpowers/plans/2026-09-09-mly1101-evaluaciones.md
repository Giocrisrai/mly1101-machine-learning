# Evaluaciones oficiales — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Material sumativo (formativas, EP1–EP3, EFT) sobre Telco / Housing / Spotify, sin mezclar Waymo y sin versionar los CSV oficiales.

**Architecture:** Los datos salen del zip institucional a `datos/evaluaciones/` (gitignored). La nota usa IE1–IE12. Las plantillas de notebook salen de `herramientas/contenido_evaluacion.py` como el resto del curso. Kedro 34 nodos no se duplica.

**Tech Stack:** Python, pandas < 3, pytest, el generador de notebooks existente.

**Spec:** `docs/superpowers/specs/2026-09-09-mly1101-evaluaciones-oficiales-design.md`

## Global Constraints

- Español en código y material.
- Sin CSV sintético ni filas inventadas: si falta el zip, el código **falla**.
- No commitear `datos/evaluaciones/` ni PDFs/docx Duoc.
- RA2 = supervisado **y** no supervisado; RA3 = hiperparámetros / ensamble / CV.
- pandas fijado < 3.0.

---

### Task 1: Carga de casos + calculadora institucional

**Files:**
- Done: `herramientas/preparar_casos_oficiales.py`
- Done: `herramientas/calcular_nota.py` (`--instrumento`)
- Done: `tests/test_casos_oficiales.py`, `tests/test_nota_institucional.py`
- Done: `.gitignore`, `docs/evaluaciones.md`

- [x] **Step 1:** Tests de unpack y de IE oficiales
- [x] **Step 2:** Implementación mínima
- [x] **Step 3:** `uv run pytest tests/test_nota_institucional.py tests/test_casos_oficiales.py`

---

### Task 2: Plantilla de notebook de evaluación

**Files:**
- Create: `herramientas/contenido_evaluacion.py`
- Modify: `herramientas/construir_notebooks.py`
- Create: `tests/test_evaluacion_carga.py` (falla sin CSV; lee Telco si está)

- [x] **Step 1:** Test que `cargar_caso("telco")` levanta FileNotFoundError sin el CSV
- [x] **Step 2:** Función pura en `src/casos.py` (sin print)
- [x] **Step 3:** Notebook alumno/docente: elegir caso, EDA mínimo, recordatorio EP1 vs EP2
- [x] **Step 4:** Regenerar con `construir_notebooks.py`

---

### Task 3: Formativas 1–3

**Files:**
- Create: `docs/formativa_1.md`, `docs/formativa_2.md`, `docs/formativa_3.md` (ítems + pauta docente)

- [x] **Step 1:** 8–10 ítems por formativa, 1 h, sin dataset extra
- [x] **Step 2:** Pauta en bloque `solo_docente` o archivo `_pauta`

---

### Task 4: CSV institucional para el curso

**Files:**
- Modify: `herramientas/calcular_nota.py` (`--csv` con IE)
- Modify: `docs/ejemplo_notas.csv` (no sustituir el de D1–D5; añadir `docs/ejemplo_notas_ep1.csv`)

- [x] **Step 1:** Test de CSV con IE1–IE4
- [x] **Step 2:** CLI

---

Hecho en 2026-09-09 (noche): Tasks 1–4. Guiones de EP2/EP3 y plantilla extra de modelado
siguen fuera: la EP2/EP3 se rinden sobre el mismo notebook 15 + el método de las Act. 2.x/3.x.
No se copió el notebook institucional Telco.
