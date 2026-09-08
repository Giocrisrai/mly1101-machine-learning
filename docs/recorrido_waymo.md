# Recorrido Waymo · del clone al modelo comparado

**Para el alumno.** Un mapa de **este** repositorio: qué clonas, qué bajas, dónde corre, y
qué hace cada fase (analítica, EDA, modelamiento, pipelines, reentrenamiento y “productivo”).

No es evaluación. Las Act. 1.1–3.3 y `kedro run` usan **Perception v2 real**. Las parciales
y el EFT van sobre *Telco Churn*, *House Prices* o *Spotify Tracks*.

El mapa de **servicios AWS** (sí/no, USD 50) está en
[`aws_academy_laboratorio.md`](aws_academy_laboratorio.md). Aquí está el **ciclo de ML**.

Verificado en el lab de Academy el **2026-09-08**: consola `us-east-1`, usuario federado
`voclabs/…`, CloudShell abre a `~ $` (git y AWS CLI ya vienen). **Actions → Upload file** es
cómo se enlaza un parquet que armaste en Colab.

---

## El parquet real es el mismo archivo en todos lados

No hay un dataset “de Colab”, otro “de AWS” y otro “de Databricks”. Hay **una** tabla
`datos/waymo_real/detecciones_reales.parquet` (y, para Kedro, `muestra/` con ≥ 2 segmentos).
Esa tabla es Perception v2 (`lidar_box` + `stats`). No hay un CSV de pauta.

```
GCS Waymo (lidar_box + stats, ~1 MB/segmento)
        │  notebook 14  /  descargar_waymo.py --lote 8  o  --muestra 40
        ▼
datos/waymo_real/detecciones_reales.parquet     ← gitignore, licencia Waymo
datos/waymo_real/muestra/<segmento>/*.parquet
        │
        ├─ Colab / local     notebooks 14 · 10 · 00     waymo.cargar_tabla_curso
        ├─ Kedro             waymo_real (35 nodos)      catalog.yml remapea la entrada
        │                    salidas → kedro_mly1101/data/waymo/
        ├─ CloudShell        Upload file  o  aws s3 cp  al mismo path
        ├─ S3 privado        s3://…/detecciones_reales.parquet   (opcional, no público)
        └─ Databricks Volume /Volumes/…/mly1101/detecciones_reales.parquet
                             pandas en el driver; Spark solo para un count()
```

Si el parquet no está, el material **cae al CSV** para no romperse. Eso no es el hilo de
clase: el hilo es el real. `camera_box` y el JSON E2E se inventarian; **no** entran al RF.

---

## Respuestas cortas

| Pregunta | Respuesta en MLY1101 |
|---|---|
| ¿Tengo que crear una **EC2**? | **No.** CloudShell es suficiente. SageMaker notebook `medium` solo si el Jupyter no cabe en Colab. |
| ¿Tengo que crear un **bucket S3**? | **No** para trabajar. Es **opcional**: guardar *tu* parquet entre sesiones. Privado, `us-east-1`, Block public access ON. |
| ¿El `git clone` trae Waymo? | **No.** Trae código, notebooks y el CSV de la pauta. `datos/waymo_real/` está en `.gitignore` (licencia: no redistribuir). |
| ¿Qué problema resuelve el hilo Waymo? | **Clasificación** binaria: `detection_difficulty` (`LEVEL_1` / `LEVEL_2`). |
| ¿Y la **regresión** del IL2.2? | En las evaluaciones (*House Prices*), no en este parquet. |
| ¿Supervisado vs no supervisado? | Los **dos** están en el **RA2**. El RA3 es ajuste, ensamble y validación cruzada. |
| ¿“Puesta en productivo” es un endpoint? | **No** en este lab (se come los USD 50). Productivo = el mismo grafo Kedro, decisiones en YAML, un comando. |

---

## 1 · Dónde corre (sin inventar infra)

| Máquina | Cuándo | Qué haces |
|---|---|---|
| **Colab** | Actividades y lote de 8 (~8 MB) | Badges del [README](../README.md). *Archivo → Guardar copia en Drive* antes de autenticar. |
| **CloudShell** (Academy) | Pipeline, más disco, misma cuenta AWS | `git clone` + `pip` + Kedro. **No** es una EC2. |
| **SageMaker notebook** `ml.t3.medium` + rol `LabRole` | Jupyter con más RAM | Clonas el mismo repo. **Stop** al terminar. Sin GPU. |
| **Databricks Free Edition** | Probar Spark sobre **el mismo** parquet | Git Folder + Volume. No es EMR. |
| **EC2** | No hace falta | Permitida en Academy (nano–large). Si la prendes: AMI Linux, `LabInstanceProfile`, **Stop**. Máx. 9 instancias. |
| **EMR / Bedrock / endpoints SageMaker** | No | La UI abre; el lab de USD 50 no es para eso. |

El curso **no** se reescribe para AWS. Es el mismo GitHub.

---

## 2 · Clonar el repo

En CloudShell (icono de terminal arriba en la consola, región **N. Virginia**):

```bash
git clone https://github.com/Giocrisrai/mly1101-machine-learning.git
cd mly1101-machine-learning
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Si hay `uv`: `uv sync`. Kedro (notebook 04 y pipelines): `pip install kedro kedro-datasets`
o `uv sync --extra kedro`.

**Qué llega con el clone**

| Hay | No hay |
|---|---|
| notebooks, `src/`, `kedro_mly1101/` (sin `data/`) | `datos/waymo_real/detecciones_reales.parquet` |
| | segmentos en `muestra/`, JPEG, nubes LiDAR |

Compruébalo:

```bash
ls datos/waymo_real 2>/dev/null || echo "vacío: hay que descargar el lote (licencia Waymo)"
```

---

## 3 · Enlazar los datos (elige una vía)

`waymo.cargar_o_preparar` busca, en este orden:

1. `datos/waymo_real/muestra/` con **≥ 2** segmentos (`lidar_box` + `stats`) — es lo que usa Kedro.
2. `datos/waymo_real/detecciones_reales.parquet` ya armado.
3. Un par suelto `lidar_box.parquet` + `stats.parquet` (un segmento: **no** alcanza para partir sin fuga).
4. Si no hay nada: baja `n` segmentos livianos (**pide GCS**).

### Vía A — Colab ya armó el parquet (la más simple en clase)

1. Notebook **14** en Colab → queda `detecciones_reales.parquet` en el runtime o en Drive.
2. Bájalo a tu disco (no lo subas a un GitHub **público**).
3. En CloudShell: **Actions → Upload file** → déjalo en
   `mly1101-machine-learning/datos/waymo_real/detecciones_reales.parquet`.

Para el pipeline `waymo_real` hace falta **`muestra/` con varios segmentos**, no solo el parquet
suelto. Un segmento único hace fallar el split a propósito.

### Vía B — Descargar el lote en CloudShell (GCS)

Misma cuenta Google que aceptó [waymo.com/open/download](https://waymo.com/open/download/):

```bash
# gcloud / gsutil tienen que estar autenticados en ESA sesión
python herramientas/descargar_waymo.py --lote 8      # clase: ~8 MB, una tabla
python herramientas/descargar_waymo.py --muestra 40  # Kedro real: varios segmentos
```

`--lote 8` **no** pide GCS otra vez si `muestra/` ya tiene segmentos: los junta.

Nunca `gsutil -m cp -r` del bucket Waymo (terabytes, JPEG, nubes).

### Vía C — Bucket S3 **privado** (opcional)

Solo si quieres **persistir** el parquet cuando CloudShell se reinicia o para pasarlo a
SageMaker. No sustituye a GCS: Waymo sigue viviendo en Google; S3 guarda **tu** copia de trabajo.

```bash
# un nombre globalmente único; región del lab
aws s3 mb s3://mly1101-waymo-<tu-alias>-$(date +%s) --region us-east-1
aws s3api put-public-access-block --bucket mly1101-waymo-... \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

aws s3 cp datos/waymo_real/detecciones_reales.parquet \
  s3://mly1101-waymo-.../detecciones_reales.parquet
aws s3 cp s3://mly1101-waymo-.../detecciones_reales.parquet \
  datos/waymo_real/detecciones_reales.parquet
```

En la consola: **S3 → Create bucket** → `us-east-1` → **Block all public access** ON →
sin website, sin ACL pública. Licencia Waymo: **no redistribuir**.

**Persistencia (no confundir):**

| Sitio | Qué dura |
|---|---|
| Disco de CloudShell (`$HOME`, ~1 GB) | Suele sobrevivir cierres; no es backup |
| Bucket S3 **de tu lab** | Suele sobrevivir **End Lab** (apaga cómputo). **Reset** o presupuesto 50/50 lo borra |
| Drive / fork **privado** | Lo que quieras conservar al cierre del curso |
| GitHub público | Código sí. Parquet Waymo **no** |

---

## 4 · Qué se descarga (y qué no)

| Comando / fuente | Qué es | Entra al modelo |
|---|---|---|
| `--lote 8` | Tabla liviana `lidar_box` + `stats` | Notebook 14, proyecto |
| `--muestra 40` | Varios segmentos en `muestra/` | `kedro run` (34 nodos) |
| `camera_box` | Cajas 2D (píxeles), otra tabla | Inventario / Kedro `ingesta`. **No** al RF |
| `camera_image` / `lidar` | JPEG y nubes | **No** |
| Perception v1, Motion | Otros buckets GCS | Se **listan** en el notebook 14; no se bajan |

Medido 2026-09-08 con 40 segmentos v2: 530.396 filas, 0 % nulos, F1 `LEVEL_2` = 0,0893.

---

## 5 · El ciclo completo (CRISP-DM en este repo)

Supervisado y no supervisado son **RA2**. Ajuste / ensamble / CV son **RA3**.

| Fase | Pregunta | Notebook | Pipeline Kedro | Salida que importa |
|---|---|---|---|---|
| **Fuentes** | ¿De dónde sale y con qué licencia? | 02 · Act 1.1 · **14** | `ingesta` (5 nodos) | `inventario_fuentes_waymo.csv` |
| **Estructuras** | ¿CSV o Parquet? ¿Qué dtype? | 03 · Act 1.2 | — | Parquet en `03_primary` |
| **EDA / calidad** | ¿Nulos, imposibles, desbalance, sesgo? | 01 · Act 1.3 · 07 ética | `calidad` | `resumen_calidad`, `valores_imposibles` |
| **Limpieza** | ¿Qué se marca y qué se tira? | 14 (nodos en proceso) | `preprocesamiento` | `detecciones_limpias.parquet` |
| **CRISP-DM** | ¿Cómo se cuenta el proyecto? | 12 · Act 2.1 | — | Informe de fases |
| **Supervisado · clasificación** | ¿Esta caja será `LEVEL_2`? | 05 · Act 2.2 | `supervisado` | `clasificador.pickle`, `metricas_por_clase.csv` |
| **No supervisado** | ¿Hay grupos sin etiqueta? | 06 · Act 2.3 | `no_supervisado` | `busqueda_de_k.csv`, `perfil_de_grupos.csv` |
| **Interpretación** | ¿El F1 se traduce a negocio? | 13 · Act 2.4 | (lee métricas) | Matriz, por clase, no solo exactitud |
| **Hiperparámetros** | ¿Ajustar gana más que el ruido? | 08 · Act 3.1 | `optimizacion` | `ganancia_del_ajuste.csv` |
| **Ensamble** | ¿Voting / boosting justifican el costo? | 09 · Act 3.2 | `optimizacion` | `comparacion_modelos.csv` |
| **Robustez / selección** | ¿La diferencia es real? | 11 · Act 3.3 | `optimizacion` | `robustez_modelos.csv`, `seleccion_de_modelo.csv` |
| **Proyecto** | Todo sobre *tu* tabla | **10** | `waymo_real` (35 nodos) | `data/waymo/` |

Comandos:

```bash
cd kedro_mly1101
kedro run                              # 30 nodos · CSV de la pauta
kedro run --pipeline ingesta           # 5 nodos · inventario de lo que hay en disco
kedro run --pipeline waymo_real        # 35 nodos · mismos análisis, entrada real
kedro run --pipeline supervisado       # solo clasificación (pide datos limpios)
kedro run --pipeline no_supervisado    # solo k-medias / PCA
kedro run --pipeline optimizacion      # RA3 (pide la partición del supervisado)
```

`waymo_real` **no duplica nodos**: remapea `detecciones_crudas` → `detecciones_reales`.
Las salidas reales van a `kedro_mly1101/data/waymo/` (no se versionan).

---

## 6 · Analítica y EDA

Sobre **datos reales** el informe de limpieza suele dar diferencia ~0 (Waymo entrega cajas
limpias). El trabajo no es “encontrar nulos”: es desbalance (`cyclist`, `LEVEL_2`), clima
casi todo `sunny`, y **no partir al azar por fila**.

Funciones puras en `src/eda.py` (las mismas que Kedro). Notebook 01 = pauta sobre el CSV;
notebook 14 = la misma lógica sobre el parquet.

Partición: `segment_id` es el grupo. `waymo.partir_por_grupo` / nodo `particionar`. Un solo
segmento → el pipeline **falla a propósito** (todas las filas comparten contexto).

---

## 7 · Modelamiento

### Clasificación (hilo Waymo y Act. 2.2)

| | |
|---|---|
| Objetivo | `detection_difficulty` |
| Variables | geometría de la caja + `speed_mps` |
| **No** entra | `num_lidar_points` (fuga: la etiqueta se deriva de ahí) · `segment_id` (no es feature) · `camera_box` |
| Split | 75/25 por **grupo** (`segment_id`), semilla 42 |
| Métrica que manda | **F1 macro** (desbalance ~88/12). La exactitud miente |
| Modelo de referencia | Random Forest (`parameters.yml` → `modelo`) |

No se clasifica `object_type` en la pauta: en el CSV sintético se resuelve al 99,98 % por
cómo se sortean las dimensiones. No enseña a evaluar.

### Regresión (IL2.2, otro caso)

El programa pide regresión **y** clasificación. En **esta** tabla el objetivo es binario.
La regresión se trabaja en la Evaluación Parcial 2 con *House Prices*. No hace falta un
segundo pipeline Waymo de `speed_mps` para cumplir el IL.

### No supervisado (Act. 2.3 — sigue siendo RA2)

k-medias sobre geometría + velocidad + puntos láser. `object_type` **no** agrupa: se usa
después para contrastar. Se prueba k ∈ {2,3,4,5,6,8}; k por defecto = 4. PCA a 2D para ver,
no para entrenar el RF.

En v2 medido: silueta 0,5228 (k=2) → 0,6103 (k=8), sin codo claro. Dos componentes PCA
explican 0,7447. Tres grupos casi 100 % `vehicle`; el otro mezcla peatón y señalética.

---

## 8 · Comparativa validada (RA3)

Todo con **GroupKFold** sobre `segment_id` (5 pliegues). Métrica `f1_macro`.

Candidatos en `kedro_mly1101/.../optimizacion/nodes.py`:

1. `DummyClassifier` (mayoría) — piso
2. Árbol
3. Regresión logística (con imputación: no traga `NaN`)
4. Bosque aleatorio
5. Gradient boosting
6. Ensamble (voting) en la comparación

La tabla `seleccion_de_modelo` no elige “el número más alto”. Si la diferencia entre dos
modelos es menor que la variabilidad entre particiones (`robustez_modelos`), **no hay
evidencia** de que uno sea mejor.

Medido en v2 (2026-09-08): default 0,5104 → búsqueda 0,5893, ganancia 0,0789 (supera ruido).
Boosting 0,594 vs ensamble 0,5938: no distinguibles (ruido 0,0282).

Ajustar hiperparámetros **mirando el test** se mide a propósito (`fuga_por_ajuste.csv`): es
el anti-ejemplo.

---

## 9 · Reentrenamiento

No hay un botón “retrain” en SageMaker. Reentrenar es **volver a correr el grafo** cuando
cambia el dato o una decisión:

```bash
# 1) Más segmentos o un parquet nuevo en datos/waymo_real/
python herramientas/descargar_waymo.py --muestra 40   # o Upload file

# 2) Si cambia un umbral, un mapa de categorías o k: edita
#    kedro_mly1101/conf/base/parameters.yml
#    (no el Python de los nodos)

# 3) Mismo comando
cd kedro_mly1101 && kedro run --pipeline waymo_real
```

El modelo queda en `data/06_models/clasificador.pickle` (recorrido sintético). El real
escribe métricas en `data/waymo/07_model_output/`. Compara **antes/después** con las
tablas CSV, no con un pantallazo.

---

## 10 · “Puesta en productivo” en esta asignatura

En CRISP-DM, *Deployment* es dejar el proceso **repetible** sin reescribir el análisis.
Aquí eso ya está:

| Pieza | Dónde | Para qué |
|---|---|---|
| Rutas y formatos | `kedro_mly1101/conf/base/catalog.yml` | Único sitio con paths. CSV local vs Parquet vs (más adelante) un URI |
| Decisiones | `parameters.yml` | Umbrales, mapas, k, hiperparámetros |
| Grafo | `pipeline_registry.py` | `__default__` = 30 nodos; `waymo_real` = 35 |
| Artefacto | `clasificador.pickle` | El modelo entrenado, no un servicio HTTP |
| Contrato de tests | `tests/test_pipeline_supervisado.py` | El RF **no** ve `camera_box` |

**No** formes parte del lab:

- Endpoint SageMaker, Canvas, JumpStart, HyperPod
- Cluster EMR “porque es productivo”
- API Flask en una EC2
- Bucket **público** o CloudFront con el parquet

Si más adelante un equipo quiere servir predicciones, eso es extra-curricular y **fuera**
de los USD 50. El entregable del curso es el pipeline + las tablas de selección + el
informe que traduce F1 a “nos perdemos las cajas difíciles”.

---

## 11 · Checklist de un día en Academy

1. Start Lab · punto AWS **verde** · región `us-east-1`.
2. CloudShell → clone → `pip` (sección 2).
3. Enlazar parquet o `--lote 8` (sección 3). **No** EC2.
4. `kedro run` (CSV) para ver que el grafo vive. Luego `waymo_real` solo si hay **≥ 2**
   segmentos en `muestra/`.
5. Leer `metricas_por_clase` y `seleccion_de_modelo` (o las `*_real`).
6. Stop de SageMaker si lo usaste. **End Lab**. Parquet: Drive o S3 privado, no GitHub público.

Guía de botones del lab: [`aws_academy_laboratorio.md`](aws_academy_laboratorio.md).
Databricks Free Edition (Spark, Volumes, Kedro): [`databricks_free.md`](databricks_free.md).
Plantilla de equipo: notebook **10**. Inventario de buckets GCS: notebook **14**.

---

## 12 · Cómo se almacenan y cuánta máquina hace falta

Cifras **medidas** el 2026-09-08 en esta máquina (pandas 2.x). No las redondees de memoria.

| Tabla | Disco | RAM pandas | Filas |
|---|---|---|---|
| Parquet real (40 segmentos v2) | **19,85 MB** | **257 MB** | 530.396 |
| Carpeta `datos/waymo_real/` (muestra + extras) | **83 MB** | — | — |
| `muestra/` (`lidar_box`+`stats`+algo de `camera_box`) | 42 MB | — | — |
| Salidas Kedro `data/` | 61 MB | — | se regeneran |
| JPEG `camera_image` × 40 | ~13 GB | no entra | **no se baja** |

Regla práctica: pandas suele ocupar **4–13×** el archivo (aquí el real es ~13× por columnas
`object`). El Random Forest + `GroupKFold` del RA3 suma RAM **encima** de la tabla. Por eso
530 k filas “pesan poco” en disco y aún así CloudShell (≈ 2 GB) se puede quedar corto en el
ajuste de hiperparámetros.

### Dónde guardar cada cosa

| Qué | Dónde | Formato |
|---|---|---|
| Código y CSV de la pauta | Git clone | CSV versionado |
| Lote Waymo (`detecciones_reales.parquet`, `muestra/`) | Disco local gitignore · Drive · **S3 privado** · Volume Databricks | **Parquet** |
| Salidas Kedro (`data/`) | Se regeneran con `kedro run`. No se commitean | Parquet + CSV de métricas |
| Modelo | `data/06_models/clasificador.pickle` | pickle, no endpoint |

Capas Kedro (`01_raw` no se toca → `03_primary` limpio → `07_model_output` métricas):
`kedro_mly1101/conf/base/catalog.yml`.

### Qué máquina

| Trabajo | RAM de verdad | Dónde |
|---|---|---|
| Act. 1.1–3.3 (CSV) | 20 MB + sklearn chico → **< 1 GB** | Colab o CloudShell |
| Notebook 14, lote 8 | decenas de MB | Colab |
| `kedro run` (30 nodos, CSV) | < 2 GB | CloudShell o Colab |
| `waymo_real` 40 segmentos (RF + RA3) | tabla 257 MB + bosques → **pide 8–16 GB** | Colab (≈ 12 GB) o SageMaker **`ml.t3.large` / `xlarge`**, `LabRole`, **sin GPU**. Stop al terminar |
| Spark “a ver” | el driver del Free Edition | [Databricks Free](databricks_free.md), no EMR |

**EC2: no.** Si alguien la prende igual: `t3.medium` o `large`, `LabInstanceProfile`, disco
≤ 100 GB, **Stop**. Máx. 9 instancias en el lab.

**Contenedores (Docker / ECS / EKS): no.** El entorno reproducible es `uv.lock` + Kedro.
CloudShell y Databricks ya son máquinas gestionadas. Un `Dockerfile` no suma al F1 y ECS/EKS
no están en el allowlist (o se llevan los USD 50).

### Bucket S3 (opcional, barato)

No hace falta para clonar ni para el lote de 8. Sirve para **no perder** el parquet cuando
Reset o se llena CloudShell (1 GB en `$HOME`).

- Un bucket, `us-east-1`, **Block all public access** ON, sin website.
- Contenido esperado: **< 200 MB** (tabla + `muestra/` liviana). A precio de S3 eso es
  **fracciones de centavo** del cupo de USD 50. Lo que funde el lab es compute prendido, no
  el parquet.
- No copies JPEG ni el bucket GCS de Waymo a S3.

```bash
aws s3 mb s3://mly1101-waymo-<alias>-$(date +%s) --region us-east-1
aws s3 cp datos/waymo_real/detecciones_reales.parquet s3://…/detecciones_reales.parquet
```

### Otros servicios

No agregues Glue, RDS, Lambda, ECR, NAT, Bedrock ni endpoints. El preproceso ya está en
Kedro. Si la consola muestra el botón, mira la tabla 4.1 de
[`aws_academy_laboratorio.md`](aws_academy_laboratorio.md).
