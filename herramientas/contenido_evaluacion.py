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
## EP2 · Modelos (después de la Formativa 2)

**No ejecutes este bloque en la EP1.** RA2 = dos supervisados **y** un no supervisado
(IE6 + IE7). El split no puede ser “al azar por fila” si hay una unidad de negocio
(cliente, casa, álbum).

Guion de sala: `docs/guion_ep2.md`.
"""
    ),
    code(
        """
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if meta["tipo"] != "clasificacion":
    raise SystemExit(
        "Este esqueleto es Telco (clasificación). Housing/Spotify: dos regresores "
        "+ un KMeans; MAE/RMSE, no exactitud. Ver docs/guion_ep2.md."
    )

trabajo = tabla.copy()
trabajo["TotalCharges"] = pd.to_numeric(trabajo["TotalCharges"], errors="coerce")
trabajo.loc[trabajo["tenure"] == 0, "TotalCharges"] = 0.0
y = (trabajo["Churn"] == "Yes").astype(int)
X = pd.get_dummies(trabajo.drop(columns=["Churn", "customerID"]), drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

logistica = LogisticRegression(max_iter=1000, class_weight="balanced")
bosque = RandomForestClassifier(
    n_estimators=80, max_depth=8, random_state=42, n_jobs=-1, class_weight="balanced"
)
logistica.fit(X_train, y_train)
bosque.fit(X_train, y_train)
print("— logística —")
print(classification_report(y_test, logistica.predict(X_test), digits=3))
print("— bosque —")
print(classification_report(y_test, bosque.predict(X_test), digits=3))

escala = StandardScaler()
grupos = KMeans(n_clusters=3, random_state=42, n_init=10)
etiquetas = grupos.fit_predict(escala.fit_transform(X_train))
print("tamaños k-medias:", pd.Series(etiquetas).value_counts().to_dict())
""",
        todo="""
# EP2: dos supervisados + un no supervisado sobre TU caso.
# Telco: no uses customerID; TotalCharges con to_numeric; stratify en Churn.
# Housing: MAE en pesos, no solo R²; no uses PID como feature.
# Spotify: popularity no va en X; no partas al azar pistas del mismo álbum.
#
# from sklearn...
# modelo_a = ____
# modelo_b = ____
# grupos = ____
""",
    ),
    md_docente(
        """
> ### 🎓 Pauta · EP2
>
> El esqueleto de Telco **no** es la nota máxima: es para no proyectar el notebook
> Duoc. IE8 se juega en la frase de negocio (recall de `Yes`, no la exactitud 78 %).
> IE7: el KMeans no predice `Churn`; tiene que aportar un segmento o un hallazgo.
> Housing/Spotify no corren esta celda a propósito (`SystemExit`).
"""
    ),
    md(
        """
---
## EP3 · ¿Ganó de verdad? (después de la Formativa 3)

Hiperparámetros, un ensamble, validación cruzada **solo en train**. El test se toca
al final. Si la ganancia no supera el ruido, el default es una respuesta válida (IE12).

Guion de sala: `docs/guion_ep3.md`.
"""
    ),
    code(
        """
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
busqueda = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1),
    {"n_estimators": [50, 80], "max_depth": [4, 8]},
    scoring="f1",
    cv=cv,
)
busqueda.fit(X_train, y_train)
print("mejor CV f1:", round(busqueda.best_score_, 4), busqueda.best_params_)

ensamble = VotingClassifier(
    [("log", logistica), ("rf", busqueda.best_estimator_)],
    voting="soft",
)
ensamble.fit(X_train, y_train)
f1_cv = cross_val_score(ensamble, X_train, y_train, cv=cv, scoring="f1")
print("ensamble CV f1: media", round(f1_cv.mean(), 4), "std", round(f1_cv.std(), 4))
print("test f1 bosque default", round(f1_score(y_test, bosque.predict(X_test)), 4))
print("test f1 ensamble", round(f1_score(y_test, ensamble.predict(X_test)), 4))
""",
        todo="""
# EP3: GridSearchCV o RandomizedSearchCV SOLO sobre X_train.
# Un VotingClassifier / bagging / boosting.
# cross_val_score en train; una sola vez el test.
# Si |ganancia| < ruido del CV, quédate con el default y justifícalo (IE12).
""",
    ),
    md_docente(
        """
> ### 🎓 Pauta · EP3
>
> Si el ensamble y el bosque caen dentro del `std` del CV, **no** hay victoria.
> Quien elige el modelo con el test (o busca hiperparámetros ahí) se lleva 0 en IE11.
> El EFT reutiliza este mismo notebook: los doce IE, defensa individual.
"""
    ),
    md(
        """
Copia este notebook a `15_<equipo>_evaluacion.ipynb`. El original se sobrescribe
en el siguiente `git merge upstream/main`.
"""
    ),
]
