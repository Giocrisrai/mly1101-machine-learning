"""Notebook opcional: un lote chico de Waymo real, una tabla, analítica.

Genera ``notebooks/14_opcional_waymo_buckets.ipynb``.

El alumno no abre la consola GCS ni copia terabytes. Baja segmentos livianos
(``lidar_box`` + ``stats``, ~1 MB cada uno), obtiene UNA tabla, parte por grupo
y ve que las imágenes no se cargan. Los otros buckets quedan en un apéndice.

Regenerar tras editar:

    python herramientas/construir_notebooks.py
"""

from __future__ import annotations

from contenido_semana01 import (
    URL_AWS_ACADEMY,
    URL_AWS_ACADEMY_LAB,
    URL_AWS_CONSOLA,
    URL_DATABRICKS_FREE,
    URL_REPO,
    code,
    md,
)

CELDAS_WAYMO_BUCKETS: list[dict] = [
    md(
        """
# Opcional · Un lote chico de datos reales

No vas a bajar el Waymo Open Dataset. Pesan **terabytes**. Hoy bajas **segmentos livianos**
(cajas LiDAR + clima, ~1 MB cada uno), los juntas en **una tabla** y haces analítica con
pandas. Esa tabla es la que usan el proyecto y el pipeline Kedro.

| Qué | Qué no |
|---|---|
| `lidar_box` + `stats` (tabla, ~1 MB por segmento) | `camera_image` (JPEG, GB) y nubes `lidar` |
| Varios `segment_id` → partir train/test **por grupo** | Un solo segmento (fuga: misma calle en los dos lados) |
| Un parquet: `datos/waymo_real/detecciones_reales.parquet` | `gsutil -m cp -r` de un bucket |
| Contar, agrupar, describir, un Random Forest pequeño | Entrenar una red sobre video |

Si ya hay una carpeta `muestra/` con varios segmentos (p. ej. los 40 del pipeline), **no
vuelve a bajar**: ensambla esa tabla. En clase, con 8 alcanza.

Licencia: [términos de Waymo](https://waymo.com/open/terms/) — uso no comercial, **no
redistribuir**. Cada uno baja lo suyo. La carpeta está en `.gitignore`.

### Guion de clase (paso a paso)

Hoy no corremos las Act. 1.1–3.3. Hoy armamos **la tabla real** y hacemos analítica. Las
actividades calificadas siguen el CSV del repo (mismas cifras que la pauta).

| Min | Qué |
|---|---|
| 0–5 | Cuenta en [waymo.com/open/download](https://waymo.com/open/download/) + Colab con la **misma** cuenta (sección 1) |
| 5–12 | Celda 2: lote de 8, o ensambla `muestra/` si ya está |
| 12–20 | Sección 3: tipos, dificultad, clima. Pregunta al curso: ¿qué clase va a costar? |
| 20–28 | Sección 4: partir por `segment_id`. La pregunta de oro: ¿algún segmento en los dos lados? |
| 28–38 | Sección 4b: EDA + RF chico. Cada equipo **extiende** una celda |
| 38–48 | Sección 4c: volúmenes, transformaciones (mismos nodos de Kedro) y estrategias |
| 48–55 | Cierre: el notebook 10 toma este parquet. Kedro `waymo_real` es el mismo grafo |
"""
    ),
    md(
        """
---
## 1 · Cuenta (una vez)

1. Entra a <https://waymo.com/open/download/> con **tu** Google y acepta el acuerdo.
   Esa página **es** el catálogo: Perception v2 / v1, Motion y E2E cámara. Cada uno abre
   un bucket de GCS (terabytes). **No pulses descargar el dataset.**
2. Colab: misma cuenta (avatar) y *Archivo → Guardar una copia en Drive* **antes** de
   autenticarte. No uses `!gsutil` en Colab.
3. Local: `gcloud auth login` con esa misma cuenta.

| En esa página | Bucket | En este curso |
|---|---|---|
| Perception **v2.0.1** (modular) | `waymo_open_dataset_v_2_0_1` | **sí**: `lidar_box` + `stats` |
| Perception v1.4.3 (con mapas) | `waymo_open_dataset_v_1_4_3` | no (tfrecord de GB) |
| Motion v1.3.1 | `waymo_open_dataset_motion_v_1_3_1` | no (~1 GB por archivo) |
| E2E cámara v1.0.0 | `waymo_open_dataset_end_to_end_camera_v_1_0_0` | no (tfrecord ~1,6 GB; hay un JSON de 36 KB) |
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
    RAIZ = REPO
    %pip install -q google-cloud-storage
    from google.colab import auth
    auth.authenticate_user()
    print("Autenticado en Colab.")
else:
    RAIZ = Path("..").resolve()
    print("Local. Si falla la descarga:  gcloud auth login")

sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "kedro_mly1101" / "src"))
DESTINO = RAIZ / "datos" / "waymo_real"
DESTINO.mkdir(parents=True, exist_ok=True)
print("Destino:", DESTINO)
"""
    ),
    md(
        """
## 2 · El lote (un comando, en batch)

`waymo.cargar_o_preparar` arma **una** tabla. Orden:

1. Si `muestra/` ya tiene ≥2 segmentos completos, los **ensambla** (es lo mismo que lee Kedro).
2. Si no, reutiliza `detecciones_reales.parquet`.
3. Si no hay nada, baja 8 × `lidar_box` + `stats` (~8 MB) y traduce.

En local:

```bash
uv run python herramientas/descargar_waymo.py --lote 8
# Si el docente ya dejó 40 segmentos en muestra/, esto no pide GCS: solo junta la tabla.
```
"""
    ),
    code(
        """
import pandas as pd

import waymo

tabla, informe, ruta = waymo.cargar_o_preparar(DESTINO, n=waymo.LOTE_CLASE)
print(ruta)
print(waymo.texto_informe(informe))
tabla.head()
"""
    ),
    md(
        """
## 3 · Analítica (esto es lo que importa)

Cuatro preguntas. Con las respuestas ya puedes seguir el curso. No hace falta otro archivo.
"""
    ),
    code(
        """
print("Filas:", len(tabla), "· segmentos:", tabla["segment_id"].nunique())
print("\\n¿Qué objetos hay?  (desbalance = la clase rara es la que cuesta modelar)")
print(tabla["object_type"].value_counts())
print("\\n¿Qué tan difíciles?  (el NaN de Waymo ya se tradujo a LEVEL_1: no hagas dropna)")
print(tabla["detection_difficulty"].value_counts())
"""
    ),
    code(
        """
print("Geometría y rapidez (unidades: metros, m/s, conteo de puntos láser)")
print(
    tabla[["box_length", "box_width", "box_height", "speed_mps", "num_lidar_points"]]
    .describe()
    .round(2)
)
sin_puntos = (tabla["num_lidar_points"] == 0).sum()
print(f"\\nCajas con 0 puntos LiDAR: {sin_puntos}  → decisión de limpieza, no un bug")
"""
    ),
    code(
        """
print("Contexto (clima / hora / ciudad). Un segmento suele tener UN valor.")
print(tabla.groupby(["weather", "time_of_day", "location"], dropna=False).size())
"""
    ),
    md(
        """
## 4 · Cargas por grupo (sin esto el modelo hace trampa)

Un `segment_id` son ~20 s de la **misma** calle, el mismo clima y los mismos objetos frame a
frame. Si partes al azar por fila, una detección cae en entrenamiento y la del fotograma
siguiente en prueba: la métrica mide memoria, no generalización.

`waymo.partir_por_grupo` es el mismo criterio que el pipeline (`GroupShuffleSplit`): un
segmento entero va a un solo lado.
"""
    ),
    code(
        """
print("Segmentos en la tabla:", tabla["segment_id"].nunique())
if tabla["segment_id"].nunique() < 2:
    raise SystemExit(
        "Con un solo segmento no se puede partir sin fuga. "
        "Baja más: uv run python herramientas/descargar_waymo.py --lote 8"
    )

marcada = waymo.partir_por_grupo(tabla, test_size=0.25, semilla=42)
train = set(marcada.loc[marcada["particion"] == "entrenamiento", "segment_id"])
test = set(marcada.loc[marcada["particion"] == "prueba", "segment_id"])
print(f"entrenamiento: {len(train)} segmentos · prueba: {len(test)} segmentos")
print("¿Algún segmento en los dos lados?", bool(train & test))
print(marcada["particion"].value_counts())
"""
    ),
    md(
        """
## 4b · Tu análisis (esto lo mejoran ustedes)

Misma tabla, las mismas funciones del curso (`eda.py`). El Random Forest de abajo es un
**demo**: pocos árboles, y si hay muchas filas toma una muestra. No mezcles `camera_box`
(píxeles) con estas columnas (metros). No uses `num_lidar_points` como predictora: la
etiqueta de dificultad se parece demasiado a esa columna (fuga).

Después de correrlo, **escriban tres hallazgos con cifra** en la celda Markdown que sigue.
Cambien una cosa y vuelvan a medir: otro `test_size`, otra variable, filtrar `cyclist`,
entrenar solo de noche. Eso es el proyecto, no copiar el output.
"""
    ),
    code(
        """
import eda
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

print(eda.resumen_calidad(tabla)[
    ["dtype", "n_nulos", "n_unicos", "pct_unicos"]
].to_string())

print("\\nDesbalance de la etiqueta (lo que el modelo va a esconder si miras solo exactitud):")
print(eda.resumen_desbalance(tabla["detection_difficulty"]))
"""
    ),
    code(
        """
PREDICTORAS = [
    "box_length", "box_width", "box_height",
    "box_center_x", "box_center_y", "box_center_z", "speed_mps",
]
objetivo = "detection_difficulty"

trabajo = marcada.dropna(subset=PREDICTORAS + [objetivo]).copy()
# Con 40 segmentos esto son cientos de miles de filas: el demo se queda en 30.000.
if len(trabajo) > 30_000:
    trabajo = trabajo.sample(n=30_000, random_state=42)

X_tr = trabajo.loc[trabajo["particion"] == "entrenamiento", PREDICTORAS]
y_tr = trabajo.loc[trabajo["particion"] == "entrenamiento", objetivo]
X_te = trabajo.loc[trabajo["particion"] == "prueba", PREDICTORAS]
y_te = trabajo.loc[trabajo["particion"] == "prueba", objetivo]

modelo = RandomForestClassifier(
    n_estimators=40, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
)
modelo.fit(X_tr, y_tr)
print(classification_report(y_te, modelo.predict(X_te), zero_division=0))
print("Importancia (geometría y rapidez; sin num_lidar_points a propósito):")
print(pd.Series(modelo.feature_importances_, index=PREDICTORAS).sort_values(ascending=False).round(3))
"""
    ),
    md(
        """
### Tres hallazgos de este lote (con cifra; no de memoria)

1. `____`
2. `____`
3. `____`

**Una cosa que cambiamos y qué pasó:** `____`

**Lo que no haríamos:** mezclar `camera_box` con esta tabla · partir al azar por fila ·
hacer `dropna` de la dificultad en `camera_box` (el NaN es LEVEL_1).
"""
    ),
    md(
        """
## 4c · Volúmenes, transformaciones y estrategias

Las Act. 1.2 y 1.3 practican esto sobre el CSV de la pauta (10 defectos). **Aquí** se aplica
al lote real, con los **mismos nodos** que Kedro (`parameters.yml` + `preprocesamiento`).
No se reescribe la limpieza: si cambia una regla, cambia en un solo sitio.

| Estrategia | Qué es | Qué no es |
|---|---|---|
| **Volumen de clase** | lote 8 (~8 MB) o `muestra/` ya bajada | `gsutil -m cp -r` del bucket (terabytes) |
| **Volumen de pipeline** | 40 segmentos, ~35 MB de `lidar_box` | JPEG / nubes LiDAR (~330 MB **por** segmento) |
| **Marcar, no borrar** | `np.nan` en lo imposible; el resto de la fila sigue | `dropna()` global o `pd.NA` (rompe scikit-learn) |
| **Un producto, un modelo** | RF solo ve v2 (metros) | pegar `camera_box` (píxeles) a estas columnas |
| **Partir por grupo** | un `segment_id` entero a un solo lado | `train_test_split` al azar por fila |
| **Explorar vs citar** | muestra de 30.000 para el demo | el número del informe sale del pipeline completo |
| **Guardar** | Parquet (conserva dtypes) | CSV de ida y vuelta (el tipo se pierde) |
| **RAM de Colab corta** | AWS Academy (curso de la asignatura) o [Databricks Free Edition](https://www.databricks.com/learn/free-edition) | abandonar el análisis o bajar el bucket |

En datos reales la tabla **ya viene limpia** (0 % nulos, categorías en inglés). Eso no anula
el pipeline: el informe de limpieza debe mostrar *casi ceros*. Si saliera un agujero grande,
habría un bug en la traducción, no "datos sucios".
"""
    ),
    code(
        """
import yaml

import eda
import formatos
from kedro_mly1101.pipelines.preprocesamiento import nodes as limpieza

PARAMETROS = yaml.safe_load(
    (RAIZ / "kedro_mly1101" / "conf" / "base" / "parameters.yml").read_text(encoding="utf-8")
)

# Las mismas 4 transformaciones, en el mismo orden que Kedro.
paso = limpieza.normalizar_categorias(tabla, PARAMETROS["mapas_categorias"])
paso = limpieza.descubrir_faltantes(paso, PARAMETROS["centinelas"])
paso = limpieza.marcar_imposibles(paso, PARAMETROS["reglas_dominio"])
limpia = limpieza.quitar_duplicados_y_constantes(paso, PARAMETROS["columnas_a_descartar"])
informe_limpieza = limpieza.resumir_limpieza(tabla, limpia)

print("Transformaciones (crudo → limpio). En Waymo real la diferencia debe ser chica:")
print(informe_limpieza.to_string(index=False))

print("\\nReglas de dominio (filas IMPOSIBLES, no atípicos legítimos):")
print(eda.valores_imposibles(
    tabla, {n: r["condicion"] for n, r in PARAMETROS["reglas_dominio"].items()}
).to_string(index=False))

print("\\nMemoria de esta tabla:")
print(f"  RAM: {tabla.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
print(f"  disco ({ruta.name}): {ruta.stat().st_size / 1024**2:.1f} MB")
print(f"  filas: {len(tabla):,}  ·  segmentos: {tabla['segment_id'].nunique()}")
"""
    ),
    code(
        """
# Volumen: el mismo recorte en CSV vs Parquet. Excel no entra (límite 1.048.576 filas
# y es lento). En 40 millones de detecciones esta decisión es la que aguanta.
muestra = limpia.head(min(5_000, len(limpia)))
carpeta = DESTINO / "_tmp_formatos"
comparativa = formatos.medir_formatos(muestra, carpeta, formatos=("csv", "parquet"))
print(comparativa.to_string(index=False))

csv_fila = comparativa.set_index("formato")
print("\\n¿El CSV conservó los dtypes?", bool(csv_fila.loc["csv", "conserva_dtypes"]))
print("¿El Parquet conservó los dtypes?", bool(csv_fila.loc["parquet", "conserva_dtypes"]))

# Categorías después de limpiar: cabe más en RAM y el modelo no ve 4 grafías de peatón.
mem_obj = limpia.memory_usage(deep=True).sum() / 1024**2
opt = limpia.copy()
for columna in opt.select_dtypes(include="object").columns:
    if opt[columna].nunique(dropna=True) < len(opt) * 0.5:
        opt[columna] = opt[columna].astype("category")
mem_cat = opt.memory_usage(deep=True).sum() / 1024**2
print(f"\\nobject → category: {mem_obj:.1f} MB → {mem_cat:.1f} MB "
      f"(ahorro {100 * (1 - mem_cat / mem_obj):.0f} %)")
print("Orden: limpiar PRIMERO, convertir a category DESPUÉS.")
"""
    ),
    md(
        """
**Qué deben poder explicar después de esta celda**

1. Por qué el informe de limpieza en datos reales da diferencia ~0, y en el CSV de la pauta no.
2. Por qué un `speed_mps` de 15 m/s no se marca (legítimo) y un `box_height <= 0` sí.
3. Qué estrategia usarían si el lote fueran 40 millones de filas: Parquet, tipos, partir por
   grupo, no bajar imágenes, muestrear para explorar y reservar el pipeline para el número
   que se cita.
"""
    ),
    md(
        f"""
### Si Colab o el portátil se quedan sin memoria

Eso no recorta el trabajo. Hay dos espacios de la asignatura, **gratis**, para seguir:

| Dónde | Para qué | Enlace |
|---|---|---|
| **AWS Academy** (curso Duoc) | Laboratorio con más RAM/disco; misma consola `us-east-1` | [curso]({URL_AWS_ACADEMY}) · [módulo de lab]({URL_AWS_ACADEMY_LAB}) · [consola]({URL_AWS_CONSOLA}) |
| **Databricks Free Edition** | Explorar Spark / escala con un clúster chico (no es Community Edition: esa se retiró) | [{URL_DATABRICKS_FREE}]({URL_DATABRICKS_FREE}) |

En ambos: clonas **este** repo. En Academy corres el notebook 14 o `kedro run --pipeline waymo_real`.
En Databricks subes el parquet a un Volume y pruebas Spark; el grafo Kedro se queda en
Colab/CloudShell (`docs/databricks_free.md`).
Sigue sin bajarse JPEG. El parquet de Waymo es de **tu** cuenta: no lo hagas público
([términos](https://waymo.com/open/terms/)).

Ninguno de los dos es evaluación. Colab alcanza para el lote de 8. Academy y Databricks son
para cuando el volumen o el pipeline completo piden más máquina, y para **probar** la
herramienta.

Paso a paso del lab (clone, sin EC2, S3 opcional, RAM medida): `docs/aws_academy_laboratorio.md`
y `docs/recorrido_waymo.md`. No EMR, no Bedrock, no GPU, no Docker.
"""
    ),
    md(
        """
## 5 · Imágenes: por qué no las cargas

En el bucket v2 cada segmento tiene más componentes. El curso **no** baja fotos ni nubes:

| Componente | Qué es | ¿En clase? |
|---|---|---|
| `lidar_box` | cajas 3D (tabla, ~1 MB) | sí |
| `stats` | clima / hora / ciudad (~23 KB) | sí |
| `camera_box` | cajas 2D sobre la foto (tabla, 50–360 KB) | opcional: sigue siendo pandas |
| `camera_image` | JPEG empaquetados | **no** (~330 MB **por segmento**) |
| `lidar` | nube de puntos | **no** (GB) |

`camera_box` no es una imagen: es otra tabla. Si aparece en tu `muestra/`, el inventario lo
muestra y el nodo `ensamblar_cajas_camara` de Kedro la junta. **El Random Forest no la usa:**
el modelo solo ve `lidar_box` + `stats` (Perception v2).
"""
    ),
    code(
        """
inventario = waymo.inventario_muestra(DESTINO / "muestra")
print("Componentes en disco:")
if inventario.empty:
    print("  (aún no hay muestra/; el lote de 8 la crea)")
else:
    curso = inventario[inventario["completo"]]
    print(curso.groupby("componente")["mb"].agg(["count", "sum"]).round(2))
    raros = set(curso["componente"]) - {"lidar_box", "stats"}
    print("Otros en segmentos del curso (no se usan):", raros or "ninguno — bien")
    sueltos = inventario.loc[~inventario["completo"], "segmento"].nunique()
    if sueltos:
        print(
            f"Carpetas incompletas (solo stats, no las usa el pipeline): {sueltos}. "
            "Puedes ignorarlas; el censo vive en datos/waymo_real/censo_stats/."
        )

print("\\nRegla:")
for nombre, meta in waymo.COMPONENTES_V2.items():
    print(f"  {nombre:13} [{meta['uso']:8}] {meta['peso']:18} {meta['que']}")

# Un camera_box del primer segmento: tabla 2D (~100–300 KB), no es la foto.
seg = tabla["segment_id"].iloc[0]
try:
    ruta_2d = waymo.descargar_camera_box(seg, DESTINO / "muestra" / seg)
    cajas_2d = pd.read_parquet(ruta_2d)
    print(f"\\ncamera_box de {seg[:24]}… → {cajas_2d.shape[0]} filas × {cajas_2d.shape[1]} cols")
    print("columnas:", list(cajas_2d.columns)[:8], "…")
except Exception as error:
    print("\\ncamera_box omitido (hace falta GCS):", error)
"""
    ),
    md(
        """
## 5b · Tipos y flujo de cada tabla que sí está en disco

El modelo **no** mezcla productos. Cada fuente se mira sola: dtypes, nulos, qué pregunta
responde. Cifras de una máquina con 40 segmentos (2026-09-08).

| Fuente | Forma | Dtypes que importan | Flujo | Análisis en este curso |
|---|---|---|---|---|
| CSV de la pauta | 40.680 × 16 | `object` sucio (`PEATON`, `soleado`) | Act. 1.1–3.3 + Kedro 30 nodos | EDA, RF, k-medias, RA3 |
| Perception v2 | 530.396 × 16 | `object` limpio; `speed_mps` float | `muestra/` → traducir → `waymo_real` | el mismo grafo, F1 LEVEL_2 = 0,0893 |
| `camera_box` | 407.267 × 11 | `type` int8; cajas en **píxeles**; dificultad con NaN | `ensamblar_cajas_camara` y se **detiene** | EDA de tabla 2D; no entra al RF |
| JSON E2E | 479 × 2 | dos `object` | `leer_metadatos_e2e` y se **detiene** | conteo de clusters; no es video |
| Censo `stats` | 798 parquet | clima / hora / ciudad | `analizar_sesgo_waymo.py` | sesgo (793/798 `sunny`) |
| v1 / Motion / JPEG | 0 archivos | — | no se bajan | no |
| Telco / House Prices / Spotify | no están en el repo | — | Parciales y EFT | no este hilo |

`camera_box` usa los mismos enteros de tipo que el LiDAR (1 vehículo, 2 peatón, 4 ciclista).
Las cajas son `(center.x, center.y, size.x, size.y)` en la imagen, no metros. El `NaN` de
dificultad es el **mismo reverso** de la Act. 1.3: vacío = fácil, no “falta el dato”.
"""
    ),
    code(
        """
cajas = waymo.ensamblar_camera_box(DESTINO / "muestra")
print("camera_box:", cajas.shape)
if cajas.empty:
    print("No hay camera_box en muestra/. El lote de clase no las exige.")
else:
    print("\\ndtypes:")
    print(cajas.dtypes.to_string())
    print("\\nnulos %:")
    print((cajas.isna().mean() * 100).round(2).to_string())
    tipo = cajas["[CameraBoxComponent].type"].map(waymo.TIPOS_DE_OBJETO)
    print("\\ntipo (entero Waymo → nombre):")
    print(tipo.value_counts().to_string())
    print("\\ncámaras (1–5, no son JPEG):")
    print(cajas["key.camera_name"].value_counts().sort_index().to_string())
    nan_dif = cajas["[CameraBoxComponent].difficulty_level.detection"].isna().mean()
    print(f"\\nNaN en dificultad de detección: {100 * nan_dif:.1f} %  → no hagas dropna")

e2e = waymo.leer_metadatos_e2e(DESTINO)
print("\\nE2E JSON:", e2e.shape, list(e2e.columns))
if not e2e.empty:
    print(e2e["cluster"].value_counts().to_string())
    print("(Interections es la grafía del archivo de Waymo, no un typo nuestro.)")
"""
    ),
    md(
        """
## 6 · Pipeline Kedro (el mismo grafo, datos reales)

Con ≥2 segmentos en `muestra/` el pipeline `ingesta` **ve** las fuentes locales
(Perception v2, `camera_box`, JSON E2E) y deja `detecciones_reales` para el modelo.
`waymo_real` es eso más calidad → preproceso → supervisado (por grupo) → no
supervisado → optimización. **No mezcla** productos: el Random Forest solo usa v2.

```bash
cd kedro_mly1101
uv run kedro run --pipeline ingesta       # 5 nodos, segundos
uv run kedro run --pipeline waymo_real    # 35 nodos: fuentes + EDA + ML
```

| Fuente | EDA | ML (RF / k-medias / RA3) |
|---|---|---|
| Esta tabla v2 (`lidar_box` + `stats`) | ✅ calidad + ML | ✅ partido por `segment_id` |
| `camera_box` | tipos, nulos, cámaras (celda 5b) | ❌ no entra al RF |
| JSON E2E | clusters (celda 5b) | ❌ no es video |
| v1 / Motion / fotos | no se bajan | ❌ |

No duplica código. Si solo hay un segmento, el nodo de partición **falla a propósito** y te
dice que bajes más. Eso no es un bug: es el desafío de la Act. 2.2 (fuga por agrupación).

En una máquina con los 40 segmentos del curso, el inventario Kedro (2026-09-08 10:55) midió:

| fuente | archivos | MB | ¿entra al RF? |
|---|---|---|---|
| percepcion_v2 | 40 | 34,943 | sí |
| camera_box | 40 | 7,181 | no (407.267 filas × 11) |
| e2e_camara | 1 | 0,035 | no (479 secuencias) |
| camera_image / v1 / motion | 0 | 0 | no |
"""
    ),
    md(
        """
## 7 · Qué sigue (para no perderse)

| Ya tienes | Lo usas en |
|---|---|
| Esta tabla (`detecciones_reales.parquet`) | **Proyecto de equipo** (notebook 10): lo carga solo |
| El método (calidad, modelos, métricas) | Actividades 1.1–3.3: CSV del repo, para que coincida la pauta |
| Hallazgos de la 4b | Mini-informe del proyecto: desbalance, grupos, clima |
| Varios `segment_id` + `partir_por_grupo` | Train/test **sin fuga**; Kedro hace lo mismo |

Cómo mejorar el análisis (eligen una y la miden):

- ¿El F1 de `LEVEL_2` sube si entrenan solo de día, o baja?
- ¿`cyclist` (la clase rara) se parece en geometría a `pedestrian`?
- `camera_box` es **otra** tabla: tipos y NaN de dificultad; no la peguen al RF de LiDAR.
- Kedro: `uv run kedro run --pipeline waymo_real` corre el grafo completo sobre este parquet.

En el proyecto, si este parquet existe, la plantilla lo toma sola. No hace falta copiar rutas.

Si quieres armar el lote de nuevo: borra el parquet **y** la carpeta `muestra/`, y reejecuta
la celda 2. Si solo borras el parquet y `muestra/` tiene varios segmentos, los vuelve a juntar.

---

### Apéndice · Los otros productos de la página de descarga

Son los mismos cuatro de <https://waymo.com/open/download/>. El curso trabaja Perception v2.
Los demás se **listan**; si el archivo más chico cabe, se baja. Nunca `cp -r` del bucket.
"""
    ),
    code(
        """
print("Página:", waymo.PAGINA_DESCARGA)
for clave, meta in waymo.CATALOGO_BUCKETS.items():
    print(f"\\n{clave}  [{meta['formato']}]  gs://{meta['bucket']}")
    print(" ", meta["para_que"])
    try:
        objetos = waymo.listar_objetos(meta["bucket"], meta["prefijo_muestra"], limite=3)
        for objeto in objetos[:2]:
            print(f"    {objeto['bytes'] / 1024**2:8.2f} MB  {objeto['nombre'][:72]}")
    except Exception as error:
        print("    (sin GCS ahora:", str(error).splitlines()[0], ")")
"""
    ),
]
