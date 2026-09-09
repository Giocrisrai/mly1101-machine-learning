# Especificación — Evaluaciones oficiales MLY1101 (formativas, parciales, EFT)

- **Fecha:** 2026-09-09
- **Autor:** Giocrisrai Godoy Bonillo (docente)
- **Estado:** diseño listo para implementar; datasets **no** entran a git
- **Fuentes oficiales (no versionar):** `EV PARCIALES MLY1101.zip`, `EFT_MLY1101_*.zip`
  (coordinación, 2026-07-22/23)

> Las Act. 1.1–3.3 y `kedro run` son Perception v2 (el hilo que elegimos). Telco /
> Housing / Spotify son el instrumento Duoc de las parciales y el EFT —la forma inicial
> de tarea—, no un segundo curso. El método se traslada el día de la evaluación.

---

## 1. Problema

El material de actividades está completo. Faltaba el instrumento **sumativo**: tres parciales
(30 / 40 / 30 → 60 % final) y el EFT (40 %), más tres formativas de 1 h. El programa y las
pautas institucionales ya existen; este repositorio no las sustituye. Lo que falta es el
andamiaje docente (carga de casos, calculadora, plantillas) para que el alumno no improvisé
un repo paralelo y el docente no calcule notas a mano.

## 2. Decisiones

| Decisión | Elección | Descartado | Motivo |
|---|---|---|---|
| Datasets | Los CSV del zip institucional, gitignored | Subir Telco/Housing/Spotify a GitHub | Coordinación los entrega; Spotify pesa ~20 MB; no redistribuir el paquete Duoc |
| Carga | `herramientas/preparar_casos_oficiales.py` | Pedir al alumno que “busque el CSV en Kaggle” | El enunciado usa **estos** archivos (`Telco_Customer_Churn_Dataset.csv`, Ames, Spotify Tracks) |
| Rúbrica de nota | Calculadora con IE oficiales (100/80/60/30/0) | Reusar D1–D5 de `rubrica_ra1.md` | El instrumento sumativo no es esa pauta |
| Notebooks de evaluación | Plantilla de equipo + README del caso, **después** de esta spec | Duplicar las 11 actividades sobre Telco | El EFT pide un proyecto, no otro taller Waymo |
| Formativas 1–3 | Cuestionario de 1 h en sala, sin dataset extra | Mini-parcial con notebook | El programa las llama cuestionario; no hay zip propio |
| Notebook institucional Telco | Solo en el zip del EFT (docente) | Copiarlo al repo público | Es pauta Duoc, no CC BY-NC-SA |

## 3. Casos (medidos 2026-09-09)

Tras `uv run python herramientas/preparar_casos_oficiales.py`:

| Caso | Archivo | Forma | Problema (enunciado) |
|---|---|---|---|
| **A** Telco | `datos/evaluaciones/telco/Telco_Customer_Churn_Dataset.csv` | **7.043 × 21** | Clasificación: abandono (`Churn`) |
| **B** Housing | `datos/evaluaciones/housing/Ames_Iowa_Housing_Dataset.csv` | **2.930 × 82** | Regresión: precio (Ames, Iowa) |
| **C** Spotify | `datos/evaluaciones/spotify/Spotify_Tracks_Dataset.csv` | **114.000 × 21** | Popularidad; cuidado con fuga (`popularity` / ranking) |

Un equipo elige **un** caso y lo mantiene en EP1 → EP2 → EP3 → EFT.

## 4. Instrumentos y pesos (pautas 2026)

Escala de logro: 100 / 80 / 60 / 30 / 0. Conversión a nota: exigencia 60 %
(`nota(logro/100 · 4)` en `calcular_nota.py`).

### EP1 · Comprensión y preparación (5 h sala · 30 % de las parciales)

Presentación 10 min, grupal con **defensa individual** y preguntas cruzadas.

| IE | Qué pide | Peso |
|---|---|---|
| IE1 | Fuentes y herramientas colaborativas | 10 % |
| IE2 | Manipulación y preparación en Python | 30 % |
| IE3 | EDA y calidad | 40 % |
| IE4 | Sesgos, ética y privacidad | 20 % |

Entrega: informe Markdown + notebook ejecutable + datos + carpeta `data/ notebooks/ models/ images/ README.md`. Alcance: problema, KPIs, EDA, preparación, CRISP-DM. **Sin** modelos todavía.

### EP2 · Construcción e interpretación (40 % de las parciales)

| IE | Qué pide | Peso |
|---|---|---|
| IE5 | Metodología (CRISP-DM) | 20 % |
| IE6 | Dos modelos **supervisados** (clasificación o regresión según el caso) | 30 % |
| IE7 | Una técnica **no supervisada** | 30 % |
| IE8 | Interpretación de métricas → negocio | 20 % |

### EP3 · Optimización y comparación (30 % de las parciales)

| IE | Qué pide | Peso |
|---|---|---|
| IE9 | Hiperparámetros | 20 % |
| IE10 | Ensamble | 30 % |
| IE11 | Generalización (validación cruzada) | 20 % |
| IE12 | Selección y justificación de la solución | 30 % |

### EFT · 12 h · 40 % de la nota final

Los **doce** IE: IE1–IE4 al **5 %**; IE5–IE12 al **10 %**. Mismos entregables. Mínimo: dos supervisados, un no supervisado, optimización, validación, ética.

### Formativas 1, 2 y 3 (1 h cada una)

No hay zip. Son cuestionario de sala alineado al RA que cierra:

| Formativa | Después de | Foco |
|---|---|---|
| F1 | Act. 1.4 | Calidad + ética (sin Telco todavía) |
| F2 | Act. 2.4 | Supervisado vs no supervisado; métrica ≠ exactitud |
| F3 | Act. 3.3 | Hiperparámetros / ensamble / “¿ganó de verdad?” |

No sustituyen la rúbrica institucional.

## 5. Qué ya está en el repo (2026-09-09)

- `herramientas/preparar_casos_oficiales.py` + tests
- `src/casos.py` (`cargar_caso`, falla sin CSV)
- `15_alumno_evaluacion` / `15_docente_evaluacion`
- Formativas 1–3 + pautas; guion de sala EP1
- `calcular_nota.py --instrumento` y `--csv` con IE
- `.gitignore` → `datos/evaluaciones/`
- Spec y `docs/evaluaciones.md`

## 6. Cerrado (regla permanente)

No copiar el notebook institucional Telco al GitHub público. El resto del andamiaje
está en el repo: `matriz_xy` de los tres casos (Spotify recorta por álbum), guiones
EP1–EP3 + EFT, CSV de ejemplo por instrumento, plantilla `15_*_evaluacion`.

## 7. Fuera de alcance

- Subir el notebook institucional Telco al GitHub público.
- Reescribir las Act. 1.1–3.3 sobre Telco.
- `kedro run` sobre Housing/Spotify (el grafo de 34 nodos es Perception v2).
- Inventar filas si falta el zip.

## 8. Verificación

```bash
uv run pytest tests/test_nota_institucional.py tests/test_casos_oficiales.py
uv run python herramientas/preparar_casos_oficiales.py
# Los tres CSV en datos/evaluaciones/; git status no los lista.
```
