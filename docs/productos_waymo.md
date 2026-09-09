# Los cuatro productos Waymo · qué queda resuelto (y qué no, ni en AWS)

**Respuesta corta a “¿y los videos / Motion / Perception v1?”**

**No se ven ni se entrenan.** Lo comprobamos en GCS el 2026-09-08: el archivo de
video E2E más chico mide **1,56 GB**; un shard de Motion **1,17 GB**; un segmento v1
**825 MB**. El tope de clase es **250 MB**. CloudShell tiene ~1 GB. No hay un
tfrecord “de muestra” más chico: buscamos JSON, tutorial y metadata en esos
buckets y **no hay** nada bajo 250 MB salvo el JSON E2E (0,03 MB), que ya usamos.

Lo que **sí** se hace con esa información, sin el video:

| Lo que el alumno pide | Lo que cabe y ya está resuelto |
|---|---|
| “Ver las cámaras / el video E2E” | `camera_box`: qué hay **en el cuadro** (píxeles, sin JPEG). JSON E2E: **de qué tipo** es la escena (479 clusters). |
| “Motion, las trayectorias” | `vehicle_pose` (x/y del auto, ~40 KB) + `speed_mps` en v2. No hay predicción a 9 s. |
| “Perception v1, con fotos pegadas” | Perception **v2** `lidar_box` (~1 MB): las mismas cajas 3D, en parquet. |
| “Bajar un JPEG para el informe” | No. Un `camera_image` pesa **~320 MB por segmento**. La ficha de fuentes cita `camera_box`. |

Act. 1.1–3.3 y `kedro run` siguen siendo **solo** v2. Telco / House Prices / Spotify
son el instrumento Duoc de las parciales y el EFT, no un segundo hilo de clase.

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
| **F** Un fotograma | `recorte_de_un_frame` + lienzo de calibración | cajas 2D **sin** JPEG (mismo gesto que el Colab oficial) |
| **G** Modelar | parquet v2 (`lidar_box`+`stats`) | Act. 1.1–3.3 y `kedro run` (34 nodos) |

| Pregunta del alumno | Dónde está la pauta |
|---|---|
| ¿Qué hay en la página de descarga? | Esta guía + apéndice del notebook **14** |
| ¿Bajo el bucket? | **No.** `TAMANO_MAXIMO_CLASE_MB = 250`. Un objeto de v1/Motion/E2E video **no pasa**. |
| ¿Cómo armo la tabla del curso? | Notebook 14 · `descargar_waymo.py --lote 8` · `kedro run --pipeline ingesta` |
| ¿Y `camera_box` / pose / el JSON E2E? | Etapas A–F del notebook 14. Kedro `ingesta` traduce las cajas. **No** entran al Random Forest. |
| ¿Y si Colab se queda sin RAM? | CloudShell o SageMaker `large`/`xlarge` sobre **el mismo parquet v2** |
| ¿Motion en SageMaker / Databricks? | Se **lista**. No se baja. Spark lee v2. No hay pipeline de trayectorias en este repo. |

### Dónde se ve cada cosa (mismo dato, distinta superficie)

| Superficie | Tabla v2 (RF) | Cajas 2D | Un fotograma | Motion / v1 / video |
|---|---|---|---|---|
| Notebook **14** | arma `detecciones_reales` | traduce y compara | **etapa F** (lienzo, sin JPEG) | lista + Colab oficial |
| Kedro (`ingesta` + `waymo_real`) | 30 nodos de ML | `cajas_camara_2d.parquet` | no pinta (34 nodos, sin matplotlib) | 0 datasets en el catálogo |
| Notebook **04** | lee salidas de `kedro run` | recorta un frame si el parquet existe | el dibujo está en el 14 | — |
| Databricks Volume | **este** parquet v2 + `count()` | no hace falta subirla | no | no `spark.read` de un tfrecord |
| AWS CloudShell | `kedro run` sobre v2 | las mismas tablas chicas | no | un shard **no cabe** en `$HOME` |

El “solucionario” de los otros productos **es este mapa + listar + las tablas chicas**.
No hay (ni va a haber) un `kedro run --pipeline motion`.

---

## Qué hacen otros (y nosotros, sin redistribuir)

Waymo **tampoco** baja el bucket para enseñar el formato. Lo dice la
[FAQ](https://waymo.com/open/faq/): *“the tutorial currently uses some sample
frames — it does not access the actual dataset files.”* En su Colab hay **2
fotogramas** embebidos (`tutorial/frames` en el repo oficial). No los copiamos:
la licencia **prohíbe redistribuir**. El alumno abre el Colab de ellos.

| Recurso | Para qué | En MLY1101 |
|---|---|---|
| [Colab percepción · 2 frames](https://colab.research.google.com/github/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial.ipynb) | Ver JPEG + cajas de **dos** instantes (TensorFlow + protobuf) | Enlace. No entra al RF. |
| [Colab Perception v2](https://colab.research.google.com/github/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_v2.ipynb) | Parquet modular (Dask), el mismo formato del lote | Enlace. El curso usa pandas. |
| [Colab Motion](https://colab.research.google.com/github/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_motion.ipynb) | Un ejemplo del tutorial; un shard real sigue siendo ~1 GB | Enlace. En clase: `vehicle_pose`. |
| [Colab E2E](https://colab.research.google.com/github/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_vision_based_e2e_driving.ipynb) | Un frame si ya tienes el tfrecord (~1,6 GB) | Enlace. En clase: JSON 479. |
| [Muestras en GitHub](https://github.com/waymo-research/waymo-open-dataset/tree/master/tutorial) | `frames`, `frames_with_maps.tfrecord`, `frame_with_keypoints.tfrecord` | Enlace. **No** al repo del curso. |
| [EgoLens](https://egolens.org) ([código](https://github.com/egolens/egolens)) | Viewer OMSCS: arrastras parquet v2 **local** | Útil si alguien ya bajó `camera_image`. No resuelve los 320 MB. |

El repo oficial ([waymo-research/waymo-open-dataset](https://github.com/waymo-research/waymo-open-dataset), carpeta `tutorial/`, leído 2026-09-09) tiene **16** notebooks. MLY1101 **no los copia** (licencia + TensorFlow + challenges). De cada uno, qué se aprovecha:

| Tutorial oficial | Qué enseña Waymo | En este curso |
|---|---|---|
| `tutorial.ipynb` | 2 frames protobuf, JPEG, cajas, nube 3D, métricas C++/TF | **Enlace.** El gesto (un cuadro) está en el notebook 14 etapa F, **sin** JPEG. |
| `tutorial_v2.ipynb` | Parquet modular, joins objeto/frame/escena, API `v2`, Dask. Muestra `lidar_box`, `camera_box`, `camera_image`, `lidar`. **No usa `stats`.** | **Enlace.** El lote de clase es ese formato. Nosotros: pandas + `lidar_box`+`stats` (el RF). No unimos filas 2D↔3D. No bajamos `camera_image` ni `lidar`. |
| `tutorial_local.ipynb` | Mismo que el de 2 frames, kernel local + `compute_detection_metrics` (mAP) | No. MLY1101 no es un challenge de detección. |
| `tutorial_motion.ipynb` | Decodificar Motion + entrenar un modelo TF | **Enlace.** Un shard real ~1 GB. En clase: `vehicle_pose` + `speed_mps`. |
| `tutorial_vision_based_e2e_driving.ipynb` | Cargar/visualizar/submit E2E (challenge 2025) | **Enlace.** En clase: JSON 479 clusters, no el tfrecord 1,6 GB. |
| `tutorial_camera_only.ipynb` | Labels 3D sincronizados a cámara (challenge 2022) | No. Pide imágenes. |
| `tutorial_keypoints.ipynb` | Keypoints humanos | No. |
| `tutorial_maps.ipynb` | Alinear nube a mapa (`frames_with_maps.tfrecord`) | No. Es Perception v1. |
| `tutorial_2d_pvps.ipynb` | Panóptica 2D de video | No. |
| `tutorial_3d_semseg.ipynb` | Semántica 3D de la nube | No. Pide `lidar`. |
| `tutorial_object_asset.ipynb` | Assets 3D (parches de vehículo/peatón, v2.0.0) | No. Otro recorte del bucket. |
| `tutorial_occupancy_flow.ipynb` | Occupancy/flow sobre Motion + submit | No. Challenge, TensorFlow. |
| `tutorial_sim_agents.ipynb` | Sim Agents challenge 2025 | No. |
| `tutorial_scenario_gen.ipynb` | Scenario Gen challenge 2025 | No. |
| `tutorial_womd_camera.ipynb` | Tokens de cámara en WOMD + codebook `.npy` | No. |
| `tutorial_womd_lidar.ipynb` | Nubes comprimidas en WOMD | No. |

Lo que **sí** tomamos de `tutorial_v2` (sin copiar celdas): el dataset es **tablas parquet por componente**; se baja solo lo que cabe. `stats` (clima, hora, ciudad) no aparece en ese Colab y **sí** entra a nuestra tabla de 530.396 filas: es un componente real de v2, no un invento.

Lo que **aplicamos aquí** (notebook 14, etapa F): el mismo gesto —ver **un**
cuadro— con las tablas que sí caben. `recorte_de_un_frame` +
`rectangulos_del_frame` dibujan las cajas 2D sobre un lienzo `ancho×alto` de
calibración. No hay foto; hay el mapa de objetos en píxeles. Eso evita la
frustración de “no puedo ver nada de las cámaras” sin pedir 1,5 GB.

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
   el RA.” Mostrar esta tabla. Si quiere **ver** el formato: Colab oficial de Motion (un
   ejemplo), no el bucket.
4. Si pide video: notebook 14 etapa F (cajas en el lienzo) + Colab oficial de 2 frames.
   No hay clip chico en GCS.
5. Si Colab revienta con 40 segmentos: SageMaker `large` sobre **v2**, no un tfrecord.
6. Nunca `gsutil -m cp -r`. Nunca pegar `camera_box` al RF. Nunca un CSV de pauta: si falta el
   parquet, el código **falla** y dice cómo bajarlo.

Código que lista (no descarga el bucket):

```bash
uv run python herramientas/validar_buckets_waymo.py        # GCS + un objeto que quepa
uv run python herramientas/validar_buckets_waymo.py --local  # parquet ya en disco
```
