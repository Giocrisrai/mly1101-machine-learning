"""Contenido del notebook opcional con datos reales del Waymo Open Dataset.

Este notebook NO se ejecuta en clase: requiere aceptar los términos de Waymo,
autenticarse con Google Cloud y descargar cientos de MB. Está pensado para el
alumno que quiera repetir el análisis con datos reales, y para el docente que
quiera mostrar el puente entre el dataset de la clase y el mundo real.
"""

from __future__ import annotations

from contenido_semana01 import code, md

CELDAS_WAYMO: list[dict] = [
    md(
        """
# Opcional · El mismo análisis con datos reales de Waymo

> ⚠️ **Este notebook no está verificado de extremo a extremo.** Requiere una cuenta de Google,
> aceptar los términos del Waymo Open Dataset y descargar cientos de MB. Los comandos son los
> documentados por Waymo, pero no se ejecutaron al construir este repositorio. Si algo cambió
> en el bucket, revisa la [documentación oficial](https://waymo.com/open/data/perception/).

El notebook `01_alumno_exploracion.ipynb` usa un dataset **sintético** con el mismo esquema del
componente `lidar_box` del Waymo Open Dataset v2. Aquí bajamos un fragmento **real** y corremos
el mismo análisis.

### Por qué el dataset de la clase es sintético

1. **Tamaño:** el conjunto de percepción v2 pesa varios TB; un solo *shard* ronda los cientos de MB.
2. **Licencia:** el [Waymo Open Dataset License Agreement](https://waymo.com/open/terms/) es de
   uso **no comercial** y **no permite redistribuir** los datos. Por eso no hay ningún archivo de
   Waymo en este repositorio: cada persona debe aceptar los términos y descargarlos por su cuenta.
3. **Pedagogía:** en un dataset real no está garantizado que aparezcan todos los problemas de
   calidad que queremos enseñar; en el sintético, sí (y hay tests que lo verifican).

Lo que **no** es sintético es el esquema. Por eso el código de abajo es casi idéntico.
"""
    ),
    md(
        """
## Paso 1 · Registrarse y aceptar los términos

1. Entra a <https://waymo.com/open/download/> con tu cuenta de Google.
2. Acepta el *License Agreement*. Basta una vez por cuenta.
3. Los datos viven en Google Cloud Storage, en el bucket público
   `gs://waymo_open_dataset_v_2_0_1/`.

## Paso 2 · Autenticarse

**En Colab** (lo más simple, no requiere instalar nada):
"""
    ),
    code(
        """
# --- Solo en Google Colab ---
from google.colab import auth
auth.authenticate_user()   # usa la cuenta con la que aceptaste los términos
print("Autenticado.")
"""
    ),
    md(
        """
**En local (macOS)**, hay que instalar el SDK de Google Cloud una vez:

```bash
brew install --cask google-cloud-sdk
gcloud auth login          # abre el navegador
gcloud auth application-default login
```

## Paso 3 · Explorar el bucket y elegir un shard

El componente `lidar_box` está en formato **Parquet**, que pandas lee directamente. Cada archivo
corresponde a un segmento de conducción de unos 20 segundos.
"""
    ),
    code(
        """
# Listar los primeros archivos disponibles del componente lidar_box.
!gsutil ls gs://waymo_open_dataset_v_2_0_1/training/lidar_box/ | head -5
"""
    ),
    code(
        """
from pathlib import Path

DESTINO = Path("datos/waymo_real")     # ignorado por git: no se redistribuye
DESTINO.mkdir(parents=True, exist_ok=True)

# Copia un único shard (cientos de MB). Reemplaza el nombre por uno de los listados arriba.
SHARD = "10023947602400723454_1120_000_1140_000.parquet"
!gsutil -m cp gs://waymo_open_dataset_v_2_0_1/training/lidar_box/{SHARD} {DESTINO}/

print("Descargado en:", DESTINO / SHARD)
"""
    ),
    md(
        """
## Paso 4 · Cargar y traducir al esquema de la clase

Las columnas de Waymo v2 usan nombres jerárquicos del tipo
`[LiDARBoxComponent].box.center.x`. Las renombramos al esquema que ya conocemos para poder
reutilizar `src/eda.py` tal cual.
"""
    ),
    code(
        """
import sys
import pandas as pd

sys.path.insert(0, "src")
import eda

crudo = pd.read_parquet(DESTINO / SHARD)
print("Forma:", crudo.shape)
print("\\nPrimeras columnas del esquema original de Waymo:")
for columna in list(crudo.columns)[:12]:
    print("  ", columna)
"""
    ),
    code(
        """
# Traducción del esquema de Waymo v2 al de la clase.
# Los nombres exactos pueden variar entre versiones: ajusta según lo impreso arriba.
EQUIVALENCIAS = {
    "key.segment_context_name": "segment_id",
    "key.frame_timestamp_micros": "timestamp_micros",
    "key.laser_object_id": "id_interno",
    "[LiDARBoxComponent].type": "object_type",
    "[LiDARBoxComponent].box.center.x": "box_center_x",
    "[LiDARBoxComponent].box.center.y": "box_center_y",
    "[LiDARBoxComponent].box.center.z": "box_center_z",
    "[LiDARBoxComponent].box.size.x": "box_length",
    "[LiDARBoxComponent].box.size.y": "box_width",
    "[LiDARBoxComponent].box.size.z": "box_height",
    "[LiDARBoxComponent].num_lidar_points_in_box": "num_lidar_points",
    "[LiDARBoxComponent].difficulty_level.detection": "detection_difficulty",
}

presentes = {k: v for k, v in EQUIVALENCIAS.items() if k in crudo.columns}
faltan = set(EQUIVALENCIAS) - set(presentes)
if faltan:
    print("⚠️ No se encontraron estas columnas (el esquema pudo cambiar):")
    for columna in sorted(faltan):
        print("  ", columna)

df_waymo = crudo[list(presentes)].rename(columns=presentes)

# En Waymo el tipo de objeto es un entero: 1=vehículo, 2=peatón, 3=señalética, 4=ciclista.
TIPOS = {1: "vehicle", 2: "pedestrian", 3: "sign", 4: "cyclist"}
if "object_type" in df_waymo.columns:
    df_waymo["object_type"] = df_waymo["object_type"].map(TIPOS).fillna("desconocido")

df_waymo.head()
"""
    ),
    md(
        """
## Paso 5 · El mismo diagnóstico

A partir de aquí, el código es exactamente el del notebook de clase.
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
                         "box_length", "box_width", "box_height"] if c in df_waymo.columns]
eda.perfil_numerico(df_waymo, numericas)
"""
    ),
    code(
        """
if "object_type" in df_waymo.columns:
    display(eda.resumen_desbalance(df_waymo["object_type"]))
"""
    ),
    code(
        """
LLAVE = [c for c in ["segment_id", "timestamp_micros", "id_interno"] if c in df_waymo.columns]
eda.reporte_duplicados(df_waymo, LLAVE)
"""
    ),
    md(
        """
## Para discutir

Compara lo que ves aquí con el dataset de la clase:

1. ¿El desbalance entre tipos de objeto es parecido? ¿Los ciclistas siguen siendo la minoría?
2. ¿Aparecen valores atípicos en las dimensiones? ¿Son errores o vehículos grandes?
3. ¿Qué problemas del dataset de clase **no** existen aquí? (los datos publicados por Waymo ya
   pasaron por un proceso de limpieza y validación).
4. ¿Qué problemas existen aquí que **no** anticipamos en clase?

La pregunta 3 tiene una moraleja: un dataset publicado y curado es un lujo. En un proyecto real,
los datos llegan como el CSV de la clase, no como el Parquet de Waymo.

---

### Recordatorio de licencia

Los datos de Waymo son de **uso no comercial** y **no se pueden redistribuir**. No subas el
Parquet descargado a este repositorio ni se lo pases a otra persona: cada quien debe aceptar los
términos y descargarlo. La carpeta `datos/waymo_real/` está en `.gitignore` por esa razón.
"""
    ),
]
