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

Este notebook **no** es el hilo del curso. Las Act. 1.1–3.3 y `kedro run` son Waymo
(Perception v2). Telco / Housing / Spotify los entrega Duoc para las parciales y el
EFT: un equipo elige **un** caso institucional y lo mantiene hasta el EFT. No mezcles
detecciones LiDAR aquí.

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
No pide cuenta Waymo. Corre en **Chrome / Safari / Brave** (el IDE embebido no hace falta
aquí, pero si abres el notebook 14 sí: si no, `MessageError`).
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
    RAIZ = Path.cwd().resolve()
    if not (RAIZ / "src" / "waymo.py").exists():
        RAIZ = RAIZ.parent

sys.path.insert(0, str(RAIZ / "src"))
import casos

print("Colab:", EN_COLAB)
print("Casos:", ", ".join(casos.CASOS))
for nombre in casos.CASOS:
    (RAIZ / "datos" / "evaluaciones" / nombre).mkdir(parents=True, exist_ok=True)
print("Carpeta de evaluaciones:", RAIZ / "datos" / "evaluaciones")
print("Si cargar_caso falla: sube el CSV oficial a la carpeta de tu caso (no está en GitHub).")
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

es_clf = meta["tipo"] == "clasificacion"
X, y, grupo = casos.matriz_xy(
    tabla, CASO, muestra=8000 if CASO == "spotify" else None
)
print("X:", X.shape, "| y:", y.name, "| grupo:", None if grupo is None else grupo.name)

if grupo is not None:
    i_tr, i_te = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(X, y, grupo)
    )
    X_train, X_test = X.iloc[i_tr], X.iloc[i_te]
    y_train, y_test = y.iloc[i_tr], y.iloc[i_te]
    grupos_train = grupo.iloc[i_tr]
elif es_clf:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    grupos_train = None
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    grupos_train = None

if es_clf:
    lineal = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    bosque = RandomForestClassifier(
        n_estimators=80, max_depth=8, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    lineal.fit(X_train, y_train)
    bosque.fit(X_train, y_train)
    print("— logística (con escala) —")
    print(classification_report(y_test, lineal.predict(X_test), digits=3))
    print("— bosque —")
    print(classification_report(y_test, bosque.predict(X_test), digits=3))
else:
    lineal = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    bosque = RandomForestRegressor(
        n_estimators=80, max_depth=8, random_state=42, n_jobs=-1
    )
    lineal.fit(X_train, y_train)
    bosque.fit(X_train, y_train)
    print("MAE ridge", round(mean_absolute_error(y_test, lineal.predict(X_test)), 2))
    print("MAE bosque", round(mean_absolute_error(y_test, bosque.predict(X_test)), 2))

etiquetas = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(
    StandardScaler().fit_transform(X_train)
)
print("tamaños k-medias:", pd.Series(etiquetas).value_counts().to_dict())
""",
        todo="""
# EP2: casos.matriz_xy(tabla, CASO) arma X e y (sin IDs, sin el target en X).
# Spotify: muestra=8000 recorta álbumes enteros (no parte discos).
# Dos supervisados + un KMeans. Clasificación: recall de la clase cara.
# Regresión: MAE en unidades del objetivo, no solo R².
#
# X, y, grupo = casos.matriz_xy(tabla, CASO, muestra=____)
# modelo_a = ____
# modelo_b = ____
# grupos = ____
""",
    ),
    md_docente(
        """
> ### 🎓 Pauta · EP2
>
> El esqueleto **no** es la nota máxima: es para no proyectar el notebook Duoc.
> `matriz_xy` ya saca `customerID` / `PID` / `popularity` de X. IE8: Telco = recall
> de `Yes`; Housing/Spotify = error en pesos o en puntos de popularidad.
> IE7: el KMeans no predice el target; tiene que cambiar una decisión.
> Spotify se recorta por **álbumes enteros** (tope 8.000 filas) **solo en este
> esqueleto** (Colab free aguanta las 114 mil: el tope es por tiempo, no por RAM).
> Una fila del zip no tiene `album_name`: `matriz_xy` la descarta o el split revienta.
> La logística va en `make_pipeline(StandardScaler(), …)`; sin escala no converge.
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
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, StratifiedKFold, cross_val_score

if grupos_train is not None:
    cv = GroupKFold(n_splits=3)
    kw_cv = {"groups": grupos_train}
elif es_clf:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    kw_cv = {}
else:
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    kw_cv = {}

if es_clf:
    scoring = "f1"
    candidato = RandomForestClassifier(
        random_state=42, class_weight="balanced", n_jobs=-1
    )
else:
    scoring = "neg_mean_absolute_error"
    candidato = RandomForestRegressor(random_state=42, n_jobs=-1)

busqueda = GridSearchCV(
    candidato,
    {"n_estimators": [50, 80], "max_depth": [4, 8]},
    scoring=scoring,
    cv=cv,
)
busqueda.fit(X_train, y_train, **kw_cv)
print("mejor CV:", round(busqueda.best_score_, 4), busqueda.best_params_)

if es_clf:
    ensamble = VotingClassifier(
        [("lin", lineal), ("rf", busqueda.best_estimator_)],
        voting="soft",
    )
    ensamble.fit(X_train, y_train)
    cv_vals = cross_val_score(
        ensamble, X_train, y_train, cv=cv, scoring="f1", **kw_cv
    )
    print("ensamble CV f1: media", round(cv_vals.mean(), 4), "std", round(cv_vals.std(), 4))
    print("test f1 bosque", round(f1_score(y_test, bosque.predict(X_test)), 4))
    print("test f1 ensamble", round(f1_score(y_test, ensamble.predict(X_test)), 4))
else:
    ensamble = VotingRegressor(
        [("lin", lineal), ("rf", busqueda.best_estimator_)]
    )
    ensamble.fit(X_train, y_train)
    cv_vals = -cross_val_score(
        ensamble, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error", **kw_cv
    )
    print("ensamble CV MAE: media", round(cv_vals.mean(), 2), "std", round(cv_vals.std(), 2))
    print("test MAE bosque", round(mean_absolute_error(y_test, bosque.predict(X_test)), 2))
    print("test MAE ensamble", round(mean_absolute_error(y_test, ensamble.predict(X_test)), 2))
""",
        todo="""
# EP3: búsqueda SOLO sobre X_train. Un ensamble. CV en train; test una vez.
# Clasificación: f1. Regresión: MAE (neg_mean_absolute_error en CV).
# Si |ganancia| < ruido del CV, quédate con el default (IE12).
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
