"""Contenido del notebook opcional con datos reales del Waymo Open Dataset.

Verificado de extremo a extremo el 2026-08-13 contra el segmento
``10023947602400723454_1120_000_1140_000`` (San Francisco, soleado, de día):
18.633 detecciones en 1,0 MB de Parquet.

Los nombres de columna corresponden al esquema documentado en
https://github.com/waymo-research/waymo-open-dataset
(``v2/perception/box.py``, ``v2/perception/context.py``) y fueron confirmados
contra el archivo real. ``tests/test_mapeo_waymo.py`` los revalida.
"""

from __future__ import annotations

from contenido_semana01 import code, md

SEGMENTO_VERIFICADO = "10023947602400723454_1120_000_1140_000"

CELDAS_WAYMO: list[dict] = [
    md(
        f"""
# Opcional · El mismo análisis con datos reales de Waymo

El notebook `01_alumno_exploracion.ipynb` usa un dataset **sintético** con el esquema del
componente `lidar_box` del Waymo Open Dataset v2. Aquí bajamos datos **reales** y corremos el
mismo análisis, con las mismas funciones de `src/eda.py`.

> ### ✅ Estado de verificación
>
> Este notebook fue **ejecutado de extremo a extremo el 13 de agosto de 2026** sobre el segmento
> `{SEGMENTO_VERIFICADO}` (San Francisco, soleado, de día): 18.633 detecciones.
> `tests/test_mapeo_waymo.py` revalida el esquema y las relaciones cada vez que hay datos
> descargados.

### Buenas noticias sobre el tamaño

El Waymo Open Dataset completo pesa varios TB, pero eso son las **imágenes y las nubes de
puntos**. Los dos componentes que necesita este análisis son livianos:

| Componente | Contenido | Tamaño real medido |
|---|---|---|
| `lidar_box` | Cajas 3D: posición, tamaño, tipo, velocidad | **~1,0 MB** por segmento |
| `stats` | Clima, hora del día, ubicación | **~23 KB** por segmento |

O sea: esto se puede hacer en clase. Lo único que cuesta es el registro y la autenticación.

### Entonces, ¿por qué el dataset de la clase es sintético?

1. **Licencia:** el [Waymo Open Dataset License Agreement](https://waymo.com/open/terms/) es de
   uso no comercial y **no permite redistribuir** los datos. No podemos dejar un archivo de Waymo
   en el repositorio: cada persona debe registrarse y descargarlo.
2. **Pedagogía:** los datos publicados por Waymo ya pasaron por curación y validación. Como
   verás al final de este notebook, **casi no tienen los defectos que queremos enseñar**: cero
   duplicados, categorías consistentes, nada de valores imposibles. Aprender a limpiar con datos
   ya limpios no funciona.

Lo que **no** es sintético es el esquema ni la física. Este notebook lo demuestra.
"""
    ),
    md(
        """
---
## Paso 1 · Registrarse y aceptar los términos

1. Entra a <https://waymo.com/open/download/> con tu cuenta de Google.
2. Acepta el *License Agreement*. Basta una vez por cuenta.
3. Los datos viven en Google Cloud Storage, en `gs://waymo_open_dataset_v_2_0_1/`:

```
gs://waymo_open_dataset_v_2_0_1/
├── training/
│   ├── lidar_box/{context_name}.parquet     ← cajas 3D: ~1 MB      ✔ lo usamos
│   ├── stats/{context_name}.parquet         ← clima y hora: ~23 KB ✔ lo usamos
│   ├── camera_image/…                       ← imágenes: GB         ✘
│   └── lidar/…                              ← nubes de puntos: GB  ✘
└── validation/…
```

Cada `context_name` es un segmento de conducción de unos 20 segundos.

## Paso 2 · Preparar el entorno y autenticarse
"""
    ),
    code(
        """
import sys
from pathlib import Path

EN_COLAB = "google.colab" in sys.modules

if EN_COLAB:
    REPO = Path("mly1101-machine-learning")
    if not REPO.exists():
        !git clone -q https://github.com/Giocrisrai/mly1101-machine-learning.git {REPO}
    RAIZ = REPO
    from google.colab import auth
    auth.authenticate_user()          # usa la cuenta que aceptó los términos de Waymo
    print("Autenticado en Colab.")
else:
    RAIZ = Path("..").resolve()       # el notebook vive en notebooks/
    print("Entorno local. Si no has iniciado sesión, ejecuta en el terminal:")
    print("    brew install --cask google-cloud-sdk")
    print("    gcloud auth login")

sys.path.insert(0, str(RAIZ / "src"))
DESTINO = RAIZ / "datos" / "waymo_real"     # en .gitignore: no se redistribuye
DESTINO.mkdir(parents=True, exist_ok=True)
print("Raíz:", RAIZ)
"""
    ),
    md(
        """
Si `gsutil` responde `401 Anonymous caller…`, no hay sesión iniciada. Si responde
`AccessDeniedException: 403`, la cuenta autenticada **no** es la misma con la que aceptaste los
términos de Waymo.

## Paso 3 · Elegir un segmento y descargar

> **Atajo:** desde el terminal, `python herramientas/descargar_waymo.py` hace este paso completo,
> comprueba los requisitos y da mensajes accionables si algo falta. Las celdas de abajo son lo
> mismo, paso a paso.
"""
    ),
    code(
        """
BUCKET = "gs://waymo_open_dataset_v_2_0_1/training"

# Primeros segmentos disponibles.
!gsutil ls {BUCKET}/lidar_box/ | head -5
"""
    ),
    code(
        f"""
# Segmento verificado para este notebook (San Francisco, soleado, de día).
SEGMENTO = "{SEGMENTO_VERIFICADO}"

for componente in ["lidar_box", "stats"]:
    destino = DESTINO / f"{{componente}}.parquet"
    if destino.exists():
        print(f"{{componente}}: ya descargado ({{destino.stat().st_size / 1024**2:.2f}} MB)")
    else:
        !gsutil cp {{BUCKET}}/{{componente}}/{{SEGMENTO}}.parquet {{destino}}
        print(f"{{componente}}: descargado ({{destino.stat().st_size / 1024**2:.2f}} MB)")
"""
    ),
    md(
        """
## Paso 4 · Cargar y mirar el esquema original

Waymo v2 usa nombres de columna jerárquicos: las llaves llevan el prefijo `key.` y los campos del
componente van entre corchetes, como `[LiDARBoxComponent].box.center.x`. Es incómodo de leer,
pero tiene una razón: permite unir componentes distintos (cajas, imágenes, estadísticas) por sus
llaves sin que los nombres choquen.
"""
    ),
    code(
        """
import numpy as np
import pandas as pd

import eda

cajas = pd.read_parquet(DESTINO / "lidar_box.parquet")
stats = pd.read_parquet(DESTINO / "stats.parquet")

print("lidar_box:", cajas.shape, " (una fila = una detección)")
print("stats:    ", stats.shape, " (una fila = un frame)")
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

Cuatro detalles del esquema real que conviene conocer:

- El **tipo de objeto es un entero**: 1 = vehículo, 2 = peatón, 3 = señalética, 4 = ciclista.
- La **dificultad también es entera**: 1 = `LEVEL_1`, 2 = `LEVEL_2`.
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
    f"{LB}.speed.x": "speed_x",
    f"{LB}.speed.y": "speed_y",
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
        print("⚠️ Columnas no encontradas (el esquema pudo cambiar):")
        for columna in sorted(faltan):
            print("   ", columna)
    return datos[list(presentes)].rename(columns=presentes)


df = traducir(cajas, EQUIVALENCIAS_CAJAS).merge(
    traducir(stats, EQUIVALENCIAS_STATS),
    on=["segment_id", "timestamp_micros"],
    how="left",
)
print("Tras unir lidar_box + stats:", df.shape)
"""
    ),
    code(
        """
# Decodificación de los campos enteros y cálculo de la rapidez.
TIPOS = {0: "unknown", 1: "vehicle", 2: "pedestrian", 3: "sign", 4: "cyclist"}
DIFICULTAD = {1: "LEVEL_1", 2: "LEVEL_2"}

df["object_type"] = df["object_type"].map(TIPOS).fillna("desconocido")
df["detection_difficulty"] = df["detection_difficulty"].map(DIFICULTAD)
df["speed_mps"] = np.sqrt(df["speed_x"] ** 2 + df["speed_y"] ** 2)

df[["segment_id", "object_type", "box_length", "speed_mps",
    "num_lidar_points", "weather", "time_of_day"]].head()
"""
    ),
    md(
        """
## Paso 6 · El mismo diagnóstico de la clase

A partir de aquí el código es el del notebook `01_alumno_exploracion.ipynb`: el esquema ya es el
mismo.
"""
    ),
    code(
        """
eda.resumen_calidad(df)
"""
    ),
    md(
        """
### 🔎 Primer hallazgo real

Mira la fila de `detection_difficulty`. En este segmento, **el 82 % de los valores son nulos**, y
los que existen son todos `LEVEL_2`.

Eso no es un error del archivo: es una convención de codificación. Waymo escribe el campo solo
cuando la detección es difícil; el nulo significa *"dificultad estándar"*, es decir `LEVEL_1`.

Es exactamente el mismo tipo de problema que el `-1` de `num_lidar_points` en el dataset de
clase, pero al revés: allá un valor válido escondía un nulo, acá un nulo esconde un valor válido.
**Un nulo sin diccionario de datos es indescifrable.** Si lo imputáramos con la moda o
elimináramos esas filas, borraríamos el 82 % del dataset por no haber leído la documentación.
"""
    ),
    code(
        """
numericas = ["box_center_x", "box_center_y", "box_center_z",
             "box_length", "box_width", "box_height", "speed_mps"]
eda.perfil_numerico(df, numericas)
"""
    ),
    code(
        """
display(eda.resumen_desbalance(df["object_type"]))
display(eda.reporte_duplicados(df, ["segment_id", "timestamp_micros", "id_interno"]))
"""
    ),
    code(
        """
# ¿Existe aquí la relación distancia / puntos láser que vimos en clase?
distancia = np.sqrt(df["box_center_x"] ** 2 + df["box_center_y"] ** 2)
correlacion = distancia.corr(df["num_lidar_points"], method="spearman")
print(f"Correlación de Spearman distancia vs. puntos láser: {correlacion:.3f}")

# ¿Y las dimensiones típicas por tipo de objeto?
print("\\nAltura mediana por tipo de objeto (m):")
print(df.groupby("object_type")["box_height"].median().round(2).to_string())
"""
    ),
    code(
        """
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))

muestra = df.sample(min(3000, len(df)), random_state=42)
axes[0].scatter(np.sqrt(muestra["box_center_x"] ** 2 + muestra["box_center_y"] ** 2),
                muestra["num_lidar_points"], s=4, alpha=0.3)
axes[0].set_yscale("log")
axes[0].set_xlabel("distancia al vehículo (m)")
axes[0].set_ylabel("puntos láser (escala log)")
axes[0].set_title("Waymo real: menos puntos a mayor distancia")

conteo = df["object_type"].value_counts(normalize=True).mul(100).sort_values()
conteo.plot.barh(ax=axes[1], color="#4C72B0")
axes[1].set_xlabel("% de detecciones")
axes[1].set_title("Waymo real: composición por tipo")
for i, valor in enumerate(conteo):
    axes[1].text(valor + 0.5, i, f"{valor:.1f}%", va="center")

plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
---
## Comparación: sintético vs. real

Esta tabla se generó con las cifras medidas en el segmento
`10023947602400723454_1120_000_1140_000`. Las tuyas pueden variar si eliges otro segmento.

| | Dataset de clase (sintético) | Waymo real (este segmento) |
|---|---|---|
| Filas | 40.680 | 18.633 |
| Altura mediana del peatón | 1,72 m | **1,74 m** |
| Largo mediano del vehículo | 4,61 m | **4,42 m** |
| Correlación distancia ↔ puntos láser | −0,93 | **−0,64** |
| Ciclistas | 1,9 % | **0,8 %** |
| Duplicados | 480 exactos + 200 lógicos | **0** |
| Categorías inconsistentes | 11 variantes de clima | **0** |
| Valores imposibles | 157 velocidades absurdas, altos 0, largos < 0 | **0** |
| Nulos | 5 tipos distintos, incluidos 2 ocultos | 1 tipo, **con significado semántico** |

### Para discutir

1. **La física coincide, pero el sintético es demasiado prolijo.** La altura del peatón (1,72 vs.
   1,74 m) y el largo del vehículo (4,61 vs. 4,42 m) son casi idénticos. En cambio la correlación
   distancia ↔ puntos láser es **−0,93 en el sintético y −0,64 en el real**: la relación existe en
   ambos, pero en el mundo real es mucho más ruidosa. ¿Por qué? Porque ahí intervienen cosas que
   el generador no modela: oclusiones (un auto tapa a otro), el tamaño del objeto, el ángulo de
   incidencia del láser y la reflectancia del material. **Un dato simulado casi siempre es más
   limpio que la realidad**, y conviene desconfiar de un modelo que se ve perfecto en simulación.
2. **El desbalance es peor en la realidad.** 0,8 % de ciclistas, no 1,9 %. El problema que
   discutimos en clase es más grave, no menos.
3. **La suciedad no está.** Cero duplicados, cero categorías inconsistentes, cero valores
   imposibles. Waymo publicó un dataset curado.
4. **Pero apareció un problema que no anticipamos:** el 82 % de nulos en `detection_difficulty`,
   que resultó ser una convención de codificación y no un dato faltante.

La moraleja incómoda: **un dataset publicado y curado es un lujo.** En un proyecto real los datos
llegan como el CSV de la clase, no como el Parquet de Waymo. Buena parte del trabajo de un equipo
de datos consiste, precisamente, en convertir lo primero en lo segundo.

Y la moraleja técnica: los problemas de calidad **no se repiten** entre proyectos. Este segmento
no tenía ninguno de los diez defectos de la clase, y tenía uno que la clase no cubría. Lo que se
transfiere no es la lista de defectos: es el hábito de mirar antes de modelar.

### Una limitación de esta comparación

Un solo segmento son 20 segundos de conducción, en San Francisco, soleado y de día. Por eso aquí
`weather` y `time_of_day` no varían: son propiedades del segmento completo. Para estudiar el
sesgo de muestreo hay que comparar **entre** segmentos, no dentro de uno. Eso es lo que sigue.

---
## Paso 7 · Sesgo de muestreo, medido sobre 250 segmentos

Como `stats` pesa solo ~23 KB, se puede caracterizar la composición del dataset a bajo costo:

```bash
python herramientas/descargar_waymo.py --muestra 250 --solo-stats   # ~6 MB
python herramientas/descargar_waymo.py --muestra 40                 # ~45 MB, con detecciones
python herramientas/analizar_sesgo_waymo.py
```

Estos son los resultados medidos el 16 de agosto de 2026 (250 segmentos para las condiciones,
40 con detecciones completas, 530.396 detecciones):

### 1. El clima: 1 de cada 250

| Clima | Segmentos | % |
|---|---|---|
| `sunny` | 249 | **99,6 %** |
| `rain` | 1 | **0,4 %** |

La documentación de Waymo dice que el clima es *"Sunny o Rain"*. En la práctica, la lluvia
prácticamente no existe en esta muestra. Un modelo entrenado con esto **nunca vio llover**.

### 2. La hora del día

| Momento | Segmentos | % |
|---|---|---|
| `Day` | 200 | 80,0 % |
| `Dawn/Dusk` | 26 | 10,4 % |
| `Night` | 24 | **9,6 %** |

### 3. Los usuarios vulnerables desaparecen de noche

| Momento | Peatones + ciclistas | Ciclistas |
|---|---|---|
| `Day` | **27,05 %** | 0,47 % |
| `Night` | **14,11 %** | 0,38 % |
| `Dawn/Dusk` | 6,81 % | 0,00 % |

De noche, la proporción de peatones y ciclistas cae a **casi la mitad**. No es que dejen de
existir: es que hay menos ejemplos para aprender a detectarlos, justo cuando son más difíciles de
ver y cuando un error cuesta más caro.

### 4. El hallazgo que se desmintió solo

Al mirar la tasa de detecciones marcadas como difíciles, apareció algo raro:

| Momento | Agregando TODAS las detecciones | Mediana **por segmento** | Rango entre segmentos |
|---|---|---|---|
| `Day` | 13,19 % | **4,81 %** | 0,00 % – 53,81 % |
| `Night` | 7,04 % | **4,25 %** | 0,00 % – 13,17 % |

La primera columna sugiere que **de día es más difícil detectar que de noche**, lo que no tiene
sentido. La segunda muestra que la diferencia casi desaparece: 4,81 % contra 4,25 %.

¿Qué pasó? El promedio global juntó 530.396 detecciones como si fueran observaciones
independientes, cuando en realidad vienen de **40 grabaciones**. Un único segmento diurno con
53,81 % de detecciones difíciles y muchísimas filas arrastró el promedio de todo el grupo.

> **La unidad de análisis es el segmento, no la detección.**
>
> Es el mismo error que arruina un modelo cuando se separa entrenamiento y prueba **por fila**:
> detecciones del mismo segmento terminan a ambos lados del split, el modelo reconoce la escena
> en vez de aprender el objeto, y la evaluación miente. En EA2 el split tendrá que ser
> **por `segment_id`**, y esta tabla es la razón.

### Para discutir

1. Si este dataset se usara tal cual para entrenar, ¿en qué condición meteorológica esperarías el
   peor desempeño? ¿Podrías siquiera medirlo con estos datos?
2. Con 24 segmentos nocturnos, ¿cuánta confianza tienes en cualquier conclusión sobre la noche?
3. ¿Qué recolectarías tú antes de desplegar?

### Una advertencia sobre esta muestra

Los 250 segmentos son **los primeros del listado del bucket**, no una muestra aleatoria. Los
nombres son identificadores, así que el orden es arbitrario en la práctica, pero no es lo mismo
que un muestreo aleatorio: las cifras de arriba son indicativas, no un censo del dataset
completo. Decirlo es parte del trabajo.

---

### Recordatorio de licencia

Los datos de Waymo son de **uso no comercial** y **no se pueden redistribuir**. No subas los
Parquet descargados al repositorio ni se los pases a otra persona: cada quien debe aceptar los
términos y descargarlos. La carpeta `datos/waymo_real/` está en `.gitignore` por esa razón.
"""
    ),
]
