# Los cuatro productos Waymo · qué queda resuelto (y qué no, ni en AWS)

**Para el docente y el alumno.** Respuesta corta a “¿podemos tener Motion / Perception v1 /
Driving E2E resueltos, aunque sea en AWS?”.

**No.** AWS Academy da más RAM y disco para **el mismo parquet v2**. No convierte un
`tfrecord` de 1 GB en una tabla de pandas, ni mete esos productos en `kedro run`.

Cifras de GCS **en vivo el 2026-09-08** (cuenta que aceptó
[waymo.com/open/download](https://waymo.com/open/download/)). No redondear de memoria.

El lote de clase y las Act. 1.1–3.3 siguen siendo Perception **v2** (`lidar_box` + `stats`).
Telco / House Prices / Spotify son **evaluaciones**, otro hilo.

---

## Qué sí queda resuelto para el alumno

No hace falta el 100 % de Waymo. Cada etapa del curso usa **lo que cabe**:

| Etapa | Qué hay en disco | Qué hace el alumno |
|---|---|---|
| **A** Listar | catálogo + `muestra/` | notebook 14 sección 5 |
| **B** Bajar lo chico | `camera_box` + pose + calibración (KB) | `descargar_waymo.py --tablas-chicas --lote 8` |
| **C** Traducir | nombres de clase, cajas en píxeles | `traducir_camera_box` (Kedro `ingesta` también) |
| **D** Comparar | conteos 3D vs 2D | `comparar_conteos_por_tipo` — **sin** merge de filas |
| **E** Pose / E2E | trayectoria x/y · JSON 479 clusters | explorar; no es video |
| **F** Modelar | parquet v2 (`lidar_box`+`stats`) | Act. 1.1–3.3 y `kedro run` (34 nodos) |

| Pregunta del alumno | Dónde está la pauta |
|---|---|
| ¿Qué hay en la página de descarga? | Esta guía + apéndice del notebook **14** |
| ¿Bajo el bucket? | **No.** `TAMANO_MAXIMO_CLASE_MB = 250`. Un objeto de v1/Motion/E2E video **no pasa**. |
| ¿Cómo armo la tabla del curso? | Notebook 14 · `descargar_waymo.py --lote 8` · `kedro run --pipeline ingesta` |
| ¿Y `camera_box` / pose / el JSON E2E? | Etapas A–E del notebook 14. **No** entran al Random Forest. |
| ¿Y si Colab se queda sin RAM? | CloudShell o SageMaker `large`/`xlarge` sobre **el mismo parquet v2** |
| ¿Motion en SageMaker? | Se **lista**. No se baja. No hay pipeline de trayectorias en este repo. |

El “solucionario” de los otros productos **es este mapa + listar + las tablas chicas**.
No hay (ni va a haber) un `kedro run --pipeline motion`.

---

## Los cuatro productos (medido 2026-09-08)

| Producto | Bucket | Objeto más chico que vimos | ¿En clase? |
|---|---|---|---|
| **Perception v2.0.1** | `waymo_open_dataset_v_2_0_1` | `lidar_box` 0,25–0,95 MB · `stats` 0,02 MB | **Sí.** Única tabla del RF / k-medias / RA3 |
| **Perception v1.4.3** | `waymo_open_dataset_v_1_4_3` | tfrecord **894–1.062 MB** por segmento | Listar. Un Frame protobuf con fotos y LiDAR pegados |
| **Motion v1.3.1** | `waymo_open_dataset_motion_v_1_3_1` | `tf_example` **1.168–1.325 MB** (hay 1000 shards) · `scenario` **434–480 MB** | Listar. Predicción de trayectorias, no clasificación `LEVEL_2` |
| **E2E Driving v1.0.0** | `waymo_open_dataset_end_to_end_camera_v_1_0_0` | JSON **0,03 MB** (479 clusters) · tfrecord **1.590–1.679 MB** (266 shards de *test* solo en esa lista) | El JSON sí. El video **no** |

El tope de clase (`TAMANO_MAXIMO_CLASE_MB = 250`) está **por debajo** del objeto más chico de
v1, Motion y el video E2E. El código **omite** esos archivos a propósito; no es un bug de
credenciales.

---

## Por qué AWS no lo resuelve

El lab de Academy ([guía](aws_academy_laboratorio.md)) tiene **USD 50**, CloudShell con
**~1 GB** en `$HOME`, y SageMaker con disco razonable si alguien prende `large`/`xlarge`.

| Idea | Qué pasa de verdad |
|---|---|
| “Lo bajo en CloudShell” | Un shard de Motion (**~1,2 GB**) **no cabe** en el `$HOME` de 1 GB. |
| “SageMaker tiene más disco” | Cabe **un** tfrecord. Sigue siendo protobuf + TensorFlow + `waymo-open-dataset`, no pandas. El RA de MLY1101 no es predicción de trayectorias ni detección en video. |
| “Lo copio a S3 y se lo paso al curso” | La licencia Waymo **prohíbe redistribuir**. Cada alumno baja lo suyo. S3 privado solo para *tu* parquet v2. |
| “Más RAM arregla el F1 de Motion” | No hay modelo de Motion en el grafo. El F1 `LEVEL_2` = 0,0893 es del RF sobre v2. |
| “Databricks / Spark” | Lee el **mismo** parquet v2. No abre tfrecord de 1 GB como tabla del curso. |

AWS **sí** sirve para: clonar el repo, correr `kedro run` sobre 530 k filas, y no perder el
parquet (S3 privado). Eso ya está en [`recorrido_waymo.md`](recorrido_waymo.md).

---

## Perception v2: 16 componentes, no solo LiDAR

Un segmento `training/` no es “el dataset”. Son parquet **modulares**. Tamaños de **un**
segmento (`10017090168044687777_6380_000_6400_000`, GCS 2026-09-08):

| Componente | MB | Uso en MLY1101 |
|---|---|---|
| `lidar_box` | 0,46 | **Curso.** Cajas 3D → tabla del RF |
| `stats` | 0,02 | **Curso.** Clima / hora / ciudad |
| `camera_box` | 0,09 | Opcional. Cajas 2D en píxeles; EDA, no RF |
| `camera_calibration` | 0,01 | Tabla chica; no se exige |
| `lidar_calibration` | < 0,01 | Tabla chica; no se exige |
| `vehicle_pose` | 0,04 | Tabla chica; no se exige |
| `camera_hkp` / `lidar_hkp` | 0,01 | Puntos clave; no se exige |
| `camera_to_lidar_box_association` | < 0,01 | Cruce 2D–3D; no se exige |
| `projected_lidar_box` | 0,21 | Proyección; no se exige |
| `lidar_camera_synced_box` | 0,20 | Cajas sincronizadas; no se exige |
| `camera_segmentation` | 1,75 | Máscaras; no se baja en el lote |
| `lidar_segmentation` | 0,70 | Máscaras LiDAR; no se baja en el lote |
| `lidar_pose` | 28,08 | Pose densa; no se baja en el lote |
| `lidar_camera_projection` | 71,14 | Pasa el tope de 50 MB/archivo del lote |
| `lidar` (nube) | 165,49 | **No.** Puntos 3D |
| `camera_image` (JPEG) | 319,59 | **No.** Fotos |

El lote (`COMPONENTES_LIVIANOS`) sigue siendo **solo** `lidar_box` + `stats`. El resto se
**nombra** para que nadie crea que “falta bajar el dataset”.

---

## Cómo ayudar en clase (guion corto)

1. El alumno abre [waymo.com/open/download](https://waymo.com/open/download/) y ve cuatro
   tarjetas. “Las cuatro existen; el curso usa la v2 liviana.”
2. Notebook 14, apéndice: `CATALOGO_BUCKETS` lista los cuatro. Si hay GCS, imprime MB reales.
3. Si pide Motion: “un shard pesa más que CloudShell; el problema no es AWS, es el formato y
   el RA.” Mostrar esta tabla.
4. Si Colab revienta con 40 segmentos: SageMaker `large` sobre **v2**, no un tfrecord.
5. Nunca `gsutil -m cp -r`. Nunca pegar `camera_box` al RF. Nunca un CSV de pauta: si falta el
   parquet, el código **falla** y dice cómo bajarlo.

Código que lista (no descarga el bucket):

```bash
uv run python herramientas/validar_buckets_waymo.py        # GCS + un objeto que quepa
uv run python herramientas/validar_buckets_waymo.py --local  # parquet ya en disco
```
