"""Fuente única de la plantilla de evaluación (parciales y EFT).

Genera:

- ``notebooks/15_alumno_evaluacion.ipynb``
- ``notebooks/15_docente_evaluacion.ipynb``

**No es el hilo Waymo.** Un equipo elige Telco, Housing o Spotify y lo mantiene
en EP1 → EP2 → EP3 → EFT. Los CSV salen de ``preparar_casos_oficiales.py`` y
están gitignored. Si faltan, ``casos.cargar_caso`` falla: no hay filas inventadas.

Regenerar tras editar:

    python herramientas/construir_notebooks.py
"""

from __future__ import annotations

from contenido_semana01 import URL_REPO, code, md, md_docente

CELDAS_EVALUACION: list[dict] = [
    md(
        """
# MLY1101 · Plantilla de evaluación
## Parciales 1–3 y EFT · casos oficiales

Este notebook **no** usa Perception v2. Las Act. 1.1–3.3 y `kedro run` siguen en
Waymo. Aquí el equipo trabaja **un** caso institucional y lo mantiene hasta el EFT.

| Caso | Archivo | Problema | Objetivo |
|---|---|---|---|
| `telco` | `Telco_Customer_Churn_Dataset.csv` | ¿Quién se va? | `Churn` (clasificación) |
| `housing` | `Ames_Iowa_Housing_Dataset.csv` | ¿Cuánto vale la casa? | `SalePrice` (regresión) |
| `spotify` | `Spotify_Tracks_Dataset.csv` | ¿Qué tan popular es la pista? | `popularity` (regresión; ojo con fuga) |

Los CSV **no están en GitHub**. Los entrega la coordinación (`EV PARCIALES MLY1101.zip`).
En tu máquina, una vez:

```bash
uv run python herramientas/preparar_casos_oficiales.py
```

En Colab: sube el CSV del caso a la carpeta
`mly1101-machine-learning/datos/evaluaciones/{telco|housing|spotify}/` después del clone.
Si el archivo no está, la celda **revienta** a propósito. No hay dataset de reemplazo.

---

### Qué pide cada instancia (no te adelantes)

| Instancia | Qué entra | Qué **no** entra |
|---|---|---|
| **EP1** | Fuentes, preparación, EDA, ética, pregunta de negocio, KPI | Modelos |
| **EP2** | CRISP-DM, **dos** supervisados, **un** no supervisado, interpretación | Aún no es el concurso de hiperparámetros |
| **EP3** | Hiperparámetros, ensamble, validación cruzada, justificación | Inventar un cuarto dataset |
| **EFT** | Los doce IE; defensa individual | Copiar el notebook institucional de Telco |

Entrega común: informe Markdown + este notebook ejecutable + datos + carpetas
`data/ notebooks/ models/ images/ README.md`. Presentación 10 min, preguntas cruzadas.
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — plantilla de evaluación
>
> **No proyectes el notebook institucional de Telco** (anexo EFT). Es pauta Duoc, no
> CC BY-NC-SA, y si lo muestran en el pizarrón la EP1 se vuelve un dictado.
>
> Cifras medidas 2026-09-09 sobre el zip oficial (no redondear de memoria):
>
> | Caso | Forma | Trampa de calidad |
> |---|---|---|
> | Telco | **7.043 × 21** | `Churn` Yes **1.869** (26,54 %). `TotalCharges` es `object` con **11** vacíos: `tenure = 0`. Nulos pandas = 0. |
> | Housing | **2.930 × 82** | Objetivo `SalePrice`. Identificadores `Order`, `PID`. |
> | Spotify | **114.000 × 21** | `Unnamed: 0` sobra. `popularity` no puede ser feature si es el target. |
>
> EP1: si aparece un `RandomForest` o un `KMeans`, la defensa individual pregunta por
> qué modelaron antes de formular el KPI. Eso es IE3/IE5 en rojo, no un extra.
>
> El alumno elige el caso en **una** variable (`CASO`). Si pone `"waymo"` debe fallar.
"""
    ),
    md(
        """
---
## 0 · Configuración
"""
    ),
    code(
        f"""
import sys
from pathlib import Path

EN_COLAB = "google.colab" in sys.modules

if EN_COLAB:
    REPO = Path("mly1101-machine-learning")
    if not REPO.exists():
        !git clone -q {URL_REPO}.git {{REPO}}
    RAIZ = REPO.resolve()
else:
    RAIZ = Path("..").resolve()

sys.path.insert(0, str(RAIZ / "src"))
import casos

print("Colab:", EN_COLAB)
print("Casos:", ", ".join(casos.CASOS))
print("Carpeta de evaluaciones:", RAIZ / "datos" / "evaluaciones")
"""
    ),
    md(
        """
---
## TODO 1 · Elegir el caso y cargarlo

Cambia **una** variable. El resto del semestre usa esa misma tabla.

Si falla con `FileNotFoundError`, el zip no está desempaquetado. No inventes un CSV.
"""
    ),
    code(
        """
import pandas as pd

import eda

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 140)

CASO = "telco"  # el equipo elige: "telco" | "housing" | "spotify"
tabla = casos.cargar_caso(CASO, raiz=RAIZ)
meta = casos.CASOS[CASO]
print(f"{CASO}: {tabla.shape[0]:,} filas × {tabla.shape[1]} columnas")
print("objetivo:", meta["objetivo"], "| tipo:", meta["tipo"])
tabla.head()
""",
        todo="""
import pandas as pd

import eda

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 140)

CASO = "____"  # "telco" | "housing" | "spotify"
tabla = casos.cargar_caso(CASO, raiz=RAIZ)
meta = casos.CASOS[CASO]
print(f"{CASO}: {tabla.shape[0]:,} filas × {tabla.shape[1]} columnas")
print("objetivo:", meta["objetivo"], "| tipo:", meta["tipo"])
tabla.head()
""",
    ),
    md(
        """
---
## TODO 2 · Radiografía de calidad (EP1)

Usa `eda.resumen_calidad`. En Telco, `TotalCharges` parece numérica y no lo es.
En Spotify, pregunta si alguna columna **es** el target disfrazado.
"""
    ),
    code(
        """
calidad = eda.resumen_calidad(tabla)
calidad.sort_values("pct_faltante_total", ascending=False).head(12)
""",
        todo="""
# calidad = eda.resumen_calidad(tabla)
# calidad.sort_values("pct_faltante_total", ascending=False).head(12)
""",
    ),
    md_docente(
        """
> ### 🎓 Pauta · TODO 2
>
> Telco: `TotalCharges` dtype `object`, 11 cadenas vacías, 0 NaN. Quien impute la
> mediana sin mirar `tenure == 0` se inventa cobros de clientes que aún no tienen
> factura. Housing: nulos reales en `Lot Frontage` y similares; no es 0 % como Waymo.
> Spotify: 114 mil filas — si el split es al azar por pista del mismo álbum, hay fuga.
"""
    ),
    md(
        """
---
## TODO 3 · Pregunta de negocio y KPI (todavía sin modelo)

Una frase que se pueda decir en la reunión, y un criterio de éxito **numérico**.
No escribas el nombre de un algoritmo.
"""
    ),
    code(
        """
pregunta = (
    "¿Qué contratos y cargos mensuales anticipan el abandono en 30 días "
    "mejor que el azar, sin usar el identificador del cliente?"
)
kpi = (
    "Recall de Churn=Yes ≥ 0,70 en un hold-out por cliente, "
    "con precisión de esa clase ≥ 0,50 (el falso positivo cuesta una oferta)."
)
print(pregunta)
print(kpi)
""",
        todo="""
pregunta = "____"  # una frase, sin el nombre de un algoritmo
kpi = "____"       # un número y sobre qué partición se mide
print(pregunta)
print(kpi)
""",
    ),
    md(
        """
---
## Qué sigue (no lo hagas hoy si es EP1)

| Cuando | Celda que todavía no tocas |
|---|---|
| EP2 | Dos supervisados (`sklearn`) + un no supervisado. RA2 = **los dos**. |
| EP3 | Grid/random search, un ensamble, validación cruzada. Eso es RA3, no “más k-medias”. |
| EFT | Los doce IE. Defensa individual: cualquiera responde por cualquier parte. |

Copia este notebook a `15_<equipo>_evaluacion.ipynb`. El original se sobrescribe
en el siguiente `git merge upstream/main`.
"""
    ),
]
