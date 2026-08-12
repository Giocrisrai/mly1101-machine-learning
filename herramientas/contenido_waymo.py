"""Contenido del notebook opcional con datos reales del Waymo Open Dataset.

Este notebook NO se ejecuta en clase: requiere aceptar los términos de Waymo,
autenticarse con Google Cloud y descargar cientos de MB. Está pensado para el
alumno que quiera repetir el análisis con datos reales, y para el docente que
quiera mostrar el puente entre el dataset de la clase y el mundo real.

Los nombres de columna usados aquí fueron verificados el 2026-08-12 contra el
código fuente oficial del dataset:

- ``src/waymo_open_dataset/v2/perception/box.py``     (LiDARBoxComponent)
- ``src/waymo_open_dataset/v2/perception/context.py`` (StatsComponent)
- ``tutorial/tutorial_v2.ipynb``                      (formato de las columnas)

en https://github.com/waymo-research/waymo-open-dataset
"""

from __future__ import annotations

from contenido_semana01 import code, md

CELDAS_WAYMO: list[dict] = [
    md(
        """
# Opcional · El mismo análisis con datos reales de Waymo

El notebook `01_alumno_exploracion.ipynb` usa un dataset **sintético** con el mismo esquema del
componente `lidar_box` del Waymo Open Dataset v2. Aquí bajamos un fragmento **real** y corremos
el mismo análisis, con las mismas funciones de `src/eda.py`.

> ### ⚠️ Estado de verificación
>
> Los **nombres de columna y las rutas** de este notebook fueron verificados contra el código
> fuente oficial del [waymo-open-dataset](https://github.com/waymo-research/waymo-open-dataset)
> (archivos `v2/perception/box.py`, `v2/perception/context.py` y `tutorial/tutorial_v2.ipynb`),
> revisados el **12 de agosto de 2026**.
>
> Lo que **no** está verificado es la ejecución de extremo a extremo: descargar los datos exige
> una cuenta de Google con los términos aceptados y varios cientos de MB de tráfico, así que no
> se ejecutó al construir este repositorio. Si el bucket o el esquema cambiaron después de esa
> fecha, la celda de traducción de columnas te avisará qué falta en lugar de fallar en silencio.
>
> **Puedes verificarlo tú:** una vez descargados los datos,
> `pytest tests/test_mapeo_waymo.py -v` comprueba que cada columna de este notebook existe en el
> archivo real, que la unión con `stats` encuentra correspondencia, y que las relaciones que
> reproduce el dataset sintético (distancia ↔ puntos láser, altura del peatón, minoría ciclista)
> también se cumplen en los datos reales. Sin datos descargados, esos tests se saltan.

### Por qué el dataset de la clase es sintético

1. **Tamaño:** el conjunto de percepción v2 pesa varios TB; un solo *shard* ronda los cientos de MB.
2. **Licencia:** el [Waymo Open Dataset License Agreement](https://waymo.com/open/terms/) es de
   uso **no comercial** y **no permite redistribuir** los datos. Por eso no hay ningún archivo de
   Waymo en este repositorio: cada persona debe aceptar los términos y descargarlos por su cuenta.
3. **Pedagogía:** en un dataset real y ya curado no está garantizado que aparezcan los problemas
   de calidad que queremos enseñar; en el sintético, sí (y hay tests que lo verifican).

Lo que **no** es sintético es el esquema. Por eso el código de abajo es casi idéntico al de clase.
"""
    ),
    md(
        """
## Paso 1 · Registrarse y aceptar los términos

1. Entra a <https://waymo.com/open/download/> con tu cuenta de Google.
2. Acepta el *License Agreement*. Basta una vez por cuenta.
3. Los datos viven en Google Cloud Storage, en el bucket `gs://waymo_open_dataset_v_2_0_1/`.

La organización del bucket es:

```
gs://waymo_open_dataset_v_2_0_1/
├── training/
│   ├── lidar_box/{context_name}.parquet     ← cajas 3D: lo que analizaremos
│   ├── stats/{context_name}.parquet         ← clima, hora del día, ubicación
│   ├── camera_image/…                       ← imágenes (muy pesado)
│   └── lidar/…                              ← nubes de puntos (muy pesado)
└── validation/…
```

Cada `context_name` es un segmento de conducción de unos 20 segundos. Nos bastan **dos archivos
pequeños** del mismo segmento: `lidar_box` y `stats`. No hace falta bajar imágenes ni nubes de
puntos, que son los componentes realmente grandes.

## Paso 2 · Autenticarse

**En Colab** (lo más simple, no requiere instalar nada):
"""
    ),
    code(
        """
# --- Solo en Google Colab ---
from google.colab import auth

auth.authenticate_user()   # usa la cuenta con la que aceptaste los términos de Waymo
print("Autenticado.")
"""
    ),
    md(
        """
**En local (macOS)**, hay que instalar el SDK de Google Cloud una sola vez:

```bash
brew install --cask google-cloud-sdk
gcloud auth login                        # abre el navegador
```

Si `gsutil` responde `401 Anonymous caller does not have storage.objects.list access`, es que no
hay sesión iniciada. Si responde `AccessDeniedException: 403`, la cuenta con la que te
autenticaste **no** es la misma con la que aceptaste los términos de Waymo.

> **Atajo:** el repositorio incluye `herramientas/descargar_waymo.py`, que hace los pasos 3 y 4
> por ti, comprueba los requisitos y da mensajes accionables si algo falta:
>
> ```bash
> python herramientas/descargar_waymo.py --listar      # ver segmentos disponibles
> python herramientas/descargar_waymo.py               # bajar el primero
> pytest tests/test_mapeo_waymo.py -v                  # verificar que el esquema calza
> ```
>
> Si prefieres entender qué hace por dentro, sigue las celdas de abajo: son lo mismo, paso a paso.

## Paso 3 · Elegir un segmento y descargar los dos componentes
"""
    ),
    code(
        """
# Listar los primeros segmentos disponibles del componente lidar_box.
!gsutil ls gs://waymo_open_dataset_v_2_0_1/training/lidar_box/ | head -5
"""
    ),
    code(
        """
from pathlib import Path

BUCKET = "gs://waymo_open_dataset_v_2_0_1/training"
DESTINO = Path("datos/waymo_real")     # ignorado por git: no se redistribuye
DESTINO.mkdir(parents=True, exist_ok=True)

# Reemplaza SEGMENTO por uno de los nombres listados arriba (sin la extensión .parquet).
SEGMENTO = "10023947602400723454_1120_000_1140_000"

!gsutil cp {BUCKET}/lidar_box/{SEGMENTO}.parquet {DESTINO}/lidar_box.parquet
!gsutil cp {BUCKET}/stats/{SEGMENTO}.parquet {DESTINO}/stats.parquet

for archivo in sorted(DESTINO.glob("*.parquet")):
    print(f"{archivo.name}: {archivo.stat().st_size / 1024**2:.1f} MB")
"""
    ),
    md(
        """
## Paso 4 · Cargar y mirar el esquema original

Waymo v2 usa nombres de columna jerárquicos: las llaves llevan el prefijo `key.` y los campos del
componente van entre corchetes, como `[LiDARBoxComponent].box.center.x`. Es feo de leer, pero
tiene una razón: permite unir componentes distintos (cajas, imágenes, estadísticas) por sus llaves
sin que los nombres choquen.
"""
    ),
    code(
        """
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")   # ajusta la ruta si ejecutas desde otra carpeta
import eda

cajas = pd.read_parquet(DESTINO / "lidar_box.parquet")
stats = pd.read_parquet(DESTINO / "stats.parquet")

print("lidar_box:", cajas.shape)
print("stats:    ", stats.shape)
print("\\nColumnas de lidar_box:")
for columna in cajas.columns:
    print("  ", columna)
"""
    ),
    md(
        """
## Paso 5 · Unir `lidar_box` con `stats` y traducir al esquema de la clase

`lidar_box` trae una fila por objeto detectado; `stats` trae una fila por *frame* con el clima y
el momento del día. Se unen por la llave común `(segment_context_name, frame_timestamp_micros)`.

Un par de detalles del esquema real que conviene conocer:

- El tipo de objeto es un **entero**: 1 = vehículo, 2 = peatón, 3 = señalética, 4 = ciclista.
- La dificultad de detección también es entera: 1 = `LEVEL_1`, 2 = `LEVEL_2`.
- **La velocidad es un vector**, no un escalar: hay `speed.x` y `speed.y` por separado. La
  rapidez que usamos en clase se obtiene con $v = \\sqrt{v_x^2 + v_y^2}$.
- `box.size.x/y/z` corresponden a largo, ancho y alto, en ese orden.
"""
    ),
    code(
        """
LB = "[LiDARBoxComponent]"     # prefijo del componente de cajas
ST = "[StatsComponent]"        # prefijo del componente de estadísticas

EQUIVALENCIAS_CAJAS = {
    "key.segment_context_name": "segment_id",
    "key.frame_timestamp_micros": "timestamp_micros",
    "key.laser_object_id": "id_interno",
    f"{LB}.type": "object_type",
    f"{LB}.box.center.x": "box_center_x",
    f"{LB}.box.center.y": "box_center_y",
    f"{LB}.box.center.z": "box_center_z",
    f"{LB}.box.size.x": "box_length",
    f"{LB}.box.size.y": "box_width",
    f"{LB}.box.size.z": "box_height",
    f"{LB}.num_lidar_points_in_box": "num_lidar_points",
    f"{LB}.difficulty_level.detection": "detection_difficulty",
}

EQUIVALENCIAS_STATS = {
    "key.segment_context_name": "segment_id",
    "key.frame_timestamp_micros": "timestamp_micros",
    f"{ST}.weather": "weather",
    f"{ST}.time_of_day": "time_of_day",
    f"{ST}.location": "location",
}


def traducir(datos: pd.DataFrame, equivalencias: dict) -> pd.DataFrame:
    \"\"\"Renombra al esquema de la clase y avisa si el esquema de Waymo cambió.\"\"\"
    presentes = {k: v for k, v in equivalencias.items() if k in datos.columns}
    faltan = set(equivalencias) - set(presentes)
    if faltan:
        print("⚠️ Columnas no encontradas (el esquema pudo cambiar desde 2026-08-12):")
        for columna in sorted(faltan):
            print("   ", columna)
    return datos[list(presentes)].rename(columns=presentes)


cajas_tr = traducir(cajas, EQUIVALENCIAS_CAJAS)
stats_tr = traducir(stats, EQUIVALENCIAS_STATS)

df_waymo = cajas_tr.merge(stats_tr, on=["segment_id", "timestamp_micros"], how="left")
print("Tras unir lidar_box + stats:", df_waymo.shape)
"""
    ),
    code(
        """
# Decodificación de los campos enteros y cálculo de la rapidez.
TIPOS = {0: "unknown", 1: "vehicle", 2: "pedestrian", 3: "sign", 4: "cyclist"}
DIFICULTAD = {1: "LEVEL_1", 2: "LEVEL_2"}

df_waymo["object_type"] = df_waymo["object_type"].map(TIPOS).fillna("desconocido")
df_waymo["detection_difficulty"] = df_waymo["detection_difficulty"].map(DIFICULTAD)

# La velocidad viene como vector; la clase usa la rapidez en el plano horizontal.
vx, vy = f"{LB}.speed.x", f"{LB}.speed.y"
if vx in cajas.columns:
    df_waymo["speed_mps"] = np.sqrt(cajas[vx].astype(float) ** 2 + cajas[vy].astype(float) ** 2)
else:
    print("Este shard no trae velocidad: speed es un campo opcional en el esquema v2.")

df_waymo.head()
"""
    ),
    md(
        """
## Paso 6 · El mismo diagnóstico de la clase

A partir de aquí el código es idéntico al del notebook `01_alumno_exploracion.ipynb`, porque el
esquema ya es el mismo.
"""
    ),
    code(
        """
eda.resumen_calidad(df_waymo)
"""
    ),
    code(
        """
numericas = [c for c in ["box_center_x", "box_center_y", "box_center_z",
                         "box_length", "box_width", "box_height", "speed_mps"]
             if c in df_waymo.columns]
eda.perfil_numerico(df_waymo, numericas)
"""
    ),
    code(
        """
display(eda.resumen_desbalance(df_waymo["object_type"]))
display(eda.reporte_duplicados(df_waymo, ["segment_id", "timestamp_micros", "id_interno"]))
"""
    ),
    code(
        """
# ¿Existe aquí la relación distancia / puntos láser que vimos en clase?
df_waymo["distancia"] = np.sqrt(df_waymo["box_center_x"] ** 2 + df_waymo["box_center_y"] ** 2)
correlacion = df_waymo["distancia"].corr(df_waymo["num_lidar_points"], method="spearman")
print(f"Correlación de Spearman distancia vs. puntos láser: {correlacion:.3f}")

if "speed_mps" in df_waymo.columns:
    print(eda.matriz_nulos_por_grupo(df_waymo, "speed_mps", ["detection_difficulty"]))
"""
    ),
    md(
        """
## Para discutir

Compara lo que ves aquí con el dataset de la clase:

1. **Desbalance:** ¿los ciclistas siguen siendo la minoría? ¿En qué proporción?
2. **Atípicos:** ¿aparecen valores extremos en las dimensiones? ¿Son errores o vehículos grandes?
3. **La relación distancia ↔ puntos láser** que descubrimos en el dataset sintético, ¿se cumple
   también aquí? (debería: no la inventamos, la copiamos de la física del sensor).
4. ¿Qué problemas del dataset de clase **no** existen aquí? Los datos publicados por Waymo ya
   pasaron por un proceso de curación, validación y anonimización.
5. ¿Qué problemas existen aquí que **no** anticipamos en clase?

La pregunta 4 tiene una moraleja incómoda: un dataset publicado y curado es un lujo. En un
proyecto real los datos llegan como el CSV de la clase, no como el Parquet de Waymo. Buena parte
del trabajo de un equipo de datos consiste, justamente, en convertir lo primero en lo segundo.

Y una diferencia que conviene notar: el clima real en Waymo v2 solo toma los valores **Sunny** y
**Rain**. En el dataset de clase agregamos `fog` y variantes en español para que la limpieza de
categorías tuviera algo que hacer.

---

### Recordatorio de licencia

Los datos de Waymo son de **uso no comercial** y **no se pueden redistribuir**. No subas los
Parquet descargados a este repositorio ni se los pases a otra persona: cada quien debe aceptar
los términos y descargarlos. La carpeta `datos/waymo_real/` está en `.gitignore` por esa razón.
"""
    ),
]
