# Laboratorio AWS Academy · paso a paso (MLY1101)

**Para el alumno.** Cuando Colab o el portátil se quedan sin RAM, este es el espacio de la
asignatura. **No es evaluación.** El lote de clase (~8 MB) sigue cabiendo en Colab; aquí corres
el mismo repo con más máquina.

Verificado en vivo el **2026-09-08** sobre el curso
[AWS Academy Learner Lab 183052](https://awsacademy.instructure.com/courses/183052)
(código Canvas `ALLv2ES-LA-LTI13-183052`), con el laboratorio **encendido** (punto verde AWS) y
la consola abierta en **N. Virginia (`us-east-1`)**.

Entras con **tu** cuenta de estudiante de Academy. No uses la del docente.

---

## Qué vas a tener en la mano

| Herramienta | Para qué en este curso | Enlace |
|---|---|---|
| **Google Colab** | Actividades 1.1–3.3 y el lote de 8 segmentos | Badges de cada notebook en el [README](../README.md) |
| **AWS Academy Learner Lab** | RAM/disco cuando Colab no alcanza | [Curso](https://awsacademy.instructure.com/courses/183052) · [Módulos](https://awsacademy.instructure.com/courses/183052/modules) · [Iniciar lab](https://awsacademy.instructure.com/courses/183052/modules/items/18057525) |
| **Consola AWS** | CloudShell primero; S3 privado opcional; SageMaker `medium`/`large` si falta RAM | [us-east-1](https://us-east-1.console.aws.amazon.com/console/home?region=us-east-1) |
| **Databricks Free Edition** | Spark / Volumes; **no** sustituye a Kedro | [Guía](databricks_free.md) · [alta](https://www.databricks.com/learn/free-edition) |
| **Ciclo ML (EDA → modelos → reentrenar)** | Mismo repo, mismos nodos | [recorrido_waymo.md](recorrido_waymo.md) |
| **Este repositorio** | Notebooks, Kedro, lote Waymo liviano | [github.com/Giocrisrai/mly1101-machine-learning](https://github.com/Giocrisrai/mly1101-machine-learning) |

Sigue **sin** bajarse el bucket de Waymo (terabytes). Solo `lidar_box` + `stats` (~1 MB por
segmento). El parquet es de **tu** cuenta: no lo publiques
([términos Waymo](https://waymo.com/open/terms/)). Motion, Perception v1 y el video E2E **no
caben** en este lab: [por qué, con MB medidos](productos_waymo.md).

---

## 1 · Entrar a AWS Academy

1. Abre [awsacademy.com](https://www.awsacademy.com/) → **Student Login**
   *(no Educator Login)*.
2. Entra con el correo que te inscribió el docente (institucional Duoc, no la cuenta
   personal de Google de Waymo — pueden ser distintas).
3. En **Cursos** abre **AWS Academy Learner Lab [183052]**.
   El código interno es `ALLv2ES-LA-LTI13-183052`.
4. En el menú del curso: **Módulos**.

Si no ves el curso, avisa al docente: hay que estar **inscrito** en esa sección.

---

## 2 · Recorrer los módulos (en este orden)

Así están en Canvas (2026-09-08). El laboratorio no se “adivina”: hay una guía, un quiz de
seguridad y **después** el Start Lab.

### Bienvenida e información general

1. **Encuesta previa al curso** (si aparece).
2. **Guía del estudiante del Laboratorio de aprendizaje de AWS Academy.** Léela: explica
   presupuesto, duración de la sesión y cómo no dejar recursos prendidos.

### Cumplimiento y seguridad *(hay que completar los ítems)*

3. **Cómo usar de manera eficaz el Laboratorio de aprendizaje de Academy.**
4. **Evaluación de conocimientos del módulo** — 100 puntos; **mínimo 70 %** para marcar el
   ítem como completo. Sin esto, Canvas puede bloquearte el laboratorio.

### Laboratorio (el que usamos en MLY1101)

5. **Iniciar el Laboratorio de aprendizaje de AWS Academy**
   → [modules/items/18057525](https://awsacademy.instructure.com/courses/183052/modules/items/18057525)

### Recursos si algo falla

6. **Demostración: cómo acceder al Laboratorio de aprendizaje**
7. **Demostración: sugerencias para la solución de problemas generales**
8. **Demostración: cómo iniciar servicios a través de la Consola de AWS**
9. **Actividad: desarrollador de Amazon Q** (opcional; no es de MLY1101)

Al final del semestre hay encuesta de cierre y el alta de **AWS Skill Builder** (gratis con
Academy). No hace falta para las actividades calificadas.

---

## 3 · Encender el laboratorio

En **Iniciar el Laboratorio…** verás una barra encima del iframe:

| Control | Qué hace |
|---|---|
| **Start Lab** | Prende el entorno. La primera vez tarda uno o dos minutos. |
| Punto **AWS** verde | Ya está listo. Si está gris, espera o pulsa Start otra vez. |
| **AWS Details** | Abre usuario, clave y el enlace a la **Consola**. |
| **Readme** | Instrucciones del lab (presupuesto, región). |
| **End Lab** | Apaga todo. Hazlo al terminar: si no, comes el presupuesto. |
| **Reset** | Solo si el lab quedó roto. Borra lo que hayas creado en esa sesión. |

En la sesión verificada:

- Presupuesto del lab: **USD 50** (`Used $0 of $50` al partir).
- Reloj de sesión: del orden de **4 horas** (en pantalla, `03:54` restantes).
- A la izquierda hay una **terminal** Linux del propio lab; a la derecha, el panel *Learner Lab*
  (idioma EN-US: Environment Overview, Environment Navigation, Access the AWS Console).

**No dejes el lab encendido de un día para otro.** End Lab apaga el cómputo (EC2, notebooks).
Un bucket S3 **privado** suele sobrevivir hasta un **Reset** o hasta que el presupuesto llega
a 50/50. CloudShell guarda ~1 GB en `$HOME`, no es backup. Drive o fork **privado** para lo
que quieras al cierre del curso. El parquet de Waymo **no** va a un bucket público ni a GitHub.

---

## 4 · Abrir la consola AWS (`us-east-1`)

1. En la barra del lab: **AWS Details** → enlace a la consola, o entra directo a
   [console us-east-1](https://us-east-1.console.aws.amazon.com/console/home?region=us-east-1).
2. Arriba a la derecha debe decir **United States (N. Virginia)** / `us-east-1`.
   **No cambies de región:** el Learner Lab trabaja ahí. Otras regiones aparecen “not enabled”.
3. El usuario es federado **voclabs/…** (Academy), no tu IAM personal. En la prueba se veía
   como `voclabs/user…` y un Account ID del lab. El tuyo será otro número: da igual.

---

## 4.1 · Servicios: cuáles sí y cuáles no (USD 50)

La consola **lista casi todos** los productos de AWS (Bedrock, EMR, Canvas, HyperPod…). Eso **no**
significa que el lab te deje usarlos. El Learner Lab es un sandbox con IAM recortado: región
`us-east-1` (y a veces `us-west-2`); presupuesto **USD 50**; rol listo **`LabRole`** /
**`LabInstanceProfile`**. Si un botón existe y al crear sale `AccessDenied` o te come el
presupuesto, no es un bug del curso: es el lab.

Verificado el 2026-09-08 en esta cuenta voclabs: Home con **CloudShell**, **EC2**, **SageMaker**
y **S3** en *Recently visited*; SageMaker AI abre (Studio / Notebooks / Canvas en el menú);
EMR abre la lista de clusters (0) **y el botón Create cluster está ahí**; Bedrock también abre
la UI. Abrir la página ≠ estar autorizado a entrenar o a gastar.

La lista oficial de Academy (*Foundation Services*) admite un subconjunto. Abajo está **filtrada
para MLY1101 y el lote real de Waymo** (tablas parquet, no video).

### Usar en este curso (con datos reales)

| Servicio | Cómo | Límite del lab | Con Waymo |
|---|---|---|---|
| **CloudShell** | Icono de terminal arriba en la consola | Casi no gasta presupuesto | **Por defecto.** `git clone` del repo, lote `--lote 8`, pandas, Kedro |
| **S3** | Bucket **privado**, bloqueo de acceso público ON | Permitido | Guardar `detecciones_reales.parquet` de **tu** sesión. **Nunca** ACL pública ni website (licencia Waymo: no redistribuir) |
| **SageMaker Notebook** | Notebook instance, rol **`LabRole`** | Solo tamaños **medium / large / xlarge**. **Sin GPU** | Jupyter con más RAM que Colab para el notebook **14** o **10**. **Stop** al terminar |
| **SageMaker Studio** | Launch Studio, rol **`LabRole`**. Ignora 1–2 avisos `iam:CreateRole` | Studio parcial; JumpStart a menudo falla | Alternativa al notebook clásico. No abras Canvas ni HyperPod |
| **CloudWatch** | Logs | Permitido | Ver por qué falló un notebook |
| **IAM** | Solo el rol que ya existe | **No** puedes crear usuarios ni grupos | Adjuntar **`LabRole`** / **`LabInstanceProfile`**. Nada más |

Región: **N. Virginia (`us-east-1`)**. En otra, la consola carga y las API responden `AccessDenied`.

### Permitidos por Academy, pero **no** para MLY1101

Queman los USD 50 o no aportan al parquet de detecciones.

| Servicio | Por qué no |
|---|---|
| **EC2** (aunque esté permitido: nano–large, máx. 9 instancias / 32 vCPU, disco ≤ 100 GB) | El lote Waymo no necesita un servidor. Si lo prendes, **Stop** al salir |
| **Cloud9** | Duplica CloudShell. Solo nano–medium |
| **RDS / Aurora** | No hay base SQL en este hilo. Si lo dejas parado 7 días, AWS **lo enciende solo** y cobra |
| **Lambda, SQS, SNS, Step Functions, DynamoDB** | Otra asignatura. No hace falta |
| **Glue / DataBrew** | El preproceso ya está en `kedro_mly1101` + `src/eda.py` |
| **Rekognition, Textract, Polly, Translate, Comprehend, Lex, Forecast** | El curso **no baja JPEG**. Rekognition sobre fotos Waymo viola el alcance y la licencia |
| **Lightsail, Elastic Beanstalk, ELB, EFS, ECR** | Infra que no usamos |
| **RoboMaker, DeepRacer** | Fuera de MLY1101 |

### No usar (UI engaña / no están en el allowlist / se llevan el presupuesto)

| Servicio | Qué vimos / qué pasa |
|---|---|
| **EMR** (Hadoop/Spark en EC2) | La consola abre *Clusters (0)* y **Create cluster** está activo. Un cluster se come el cupo de EC2 y los USD 50. Spark de exploración: **Databricks Free Edition**, no EMR |
| **EMR on EKS / EKS / ECS** | No están en el lab. Kubernetes no es este curso |
| **SageMaker Canvas, JumpStart, HyperPod, endpoints, training GPU** | El menú se ve. GPU **no** está soportada. JumpStart pide más IAM del que hay. Un endpoint 24 h vacía el presupuesto |
| **Amazon Bedrock** | La UI de Overview **abre** en `us-east-1`. Bedrock **no** está en la lista de servicios del Learner Lab. No playground, no fine-tune, no Marketplace de modelos |
| **NAT Gateway** | Sigue cobrando **después** de End Lab |
| **Marketplace de pago** | Solo lectura limitada |
| **Crear usuarios IAM, access keys permanentes** | Bloqueado. Credenciales = las temporales del lab |
| **Otra región** (Santiago, Ohio, Irlanda…) | Error de acceso |
| **S3 público / CloudFront con el parquet** | Licencia Waymo |

### Cómo no fundir los USD 50

1. Empieza por **CloudShell**. Si corre el notebook 14, no abras SageMaker.
2. SageMaker notebook: **ml.t3.medium** (o el medium que ofrezca). **Stop** cuando dejes de teclear.
3. Nada que quede “running” de noche: EC2, RDS, NAT, EMR, endpoints.
4. End Lab al terminar el día. RDS parado 7 días se reenciende solo: **bórralo**.
5. El presupuesto en la barra del lab puede ir **hasta 8 h atrasado**. Si llega a 50/50, el lab se
   apaga y **pierdes** lo que no hayas copiado a Drive/GitHub.

---

## 5 · Correr el material de MLY1101 dentro del lab

El curso **no** se reescribe para AWS. Clonas el mismo GitHub. El mapa de datos, RAM, S3,
EC2 y “productivo” está en [`recorrido_waymo.md`](recorrido_waymo.md).

**No necesitas EC2.** **No necesitas un bucket** para empezar. Verificado el 2026-09-08:
CloudShell en `us-east-1` abre a `~ $` (git y AWS CLI ya vienen). **Actions → Upload file**
enlaza el parquet que armaste en Colab.

### 5.1 CloudShell (por defecto)

1. Consola → **CloudShell** (barra superior). Primera vez: ~30 s + modal *Welcome*.
2. Pega:

```bash
git clone https://github.com/Giocrisrai/mly1101-machine-learning.git
cd mly1101-machine-learning
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

El clone trae código y notebooks. **No** trae `datos/waymo_real/` (gitignore). **No** hay CSV
de pauta: si falta el parquet, el material falla y te dice cómo bajarlo.

3. Enlaza datos (una vía):

```bash
# A) Sube detecciones_reales.parquet vía Actions → Upload file a datos/waymo_real/
# B) Si gcloud está autenticado con la Google de Waymo:
python herramientas/descargar_waymo.py --lote 8
# C) Si creaste un bucket privado:
# aws s3 cp s3://mly1101-waymo-…/detecciones_reales.parquet datos/waymo_real/
```

4. Pipeline de la pauta (cabe en CloudShell):

```bash
cd kedro_mly1101 && pip install kedro kedro-datasets && kedro run && cd ..
```

`kedro run --pipeline waymo_real` pide **≥ 2** segmentos en `muestra/` y más RAM (RA3).
Un segmento suelto **falla a propósito**. Con 40 segmentos usa SageMaker `large`/`xlarge`.

### 5.2 SageMaker (solo si CloudShell o Colab se quedan cortos)

Notebook instance, tamaño **ml.t3.medium** (4 GB) para el 14; **large / xlarge** para
`waymo_real` + ajuste. Rol **`LabRole`**. Sin GPU. Clona el mismo repo. **Stop** al terminar.

### 5.3 Bucket S3 (opcional)

Create bucket → `us-east-1` → Block all public access ON. Sube solo **tu** parquet
(< 200 MB). No website, no ACL pública. Detalle y costos: sección 12 de
[`recorrido_waymo.md`](recorrido_waymo.md).

### 5.4 Qué no hacer

- EC2, EMR, Docker/ECS, endpoints SageMaker, Bedrock.
- `gsutil -m cp -r` del bucket Waymo.
- Pegar `camera_box` al Random Forest de LiDAR.
- Partir train/test al azar por fila.
- Publicar el parquet.
- Dejar SageMaker/EC2 prendidos.
- Bajar un tfrecord de Motion / v1 / E2E “porque SageMaker tiene disco”. Un shard mide
  **1,2–1,7 GB**, CloudShell tiene **1 GB**, y el grafo Kedro no los usa. Detalle:
  [`productos_waymo.md`](productos_waymo.md).

### 5.5 Motion, v1 y video E2E (AWS no los vuelve pandas)

La consola de Google muestra cuatro productos. El lab de USD 50 **no** es un atajo para
tenerlos “resueltos”:

| Producto | Objeto típico (GCS 2026-09-08) | En Academy |
|---|---|---|
| Perception v2 (`lidar_box`+`stats`) | 0,25–0,95 MB | **Sí.** Es el parquet del curso |
| Perception v1 | tfrecord 894–1.062 MB | Listar. No cabe en CloudShell; no es pandas |
| Motion `tf_example` | shard 1,17–1,32 GB | Listar. Un shard **ya** supera el `$HOME` de CloudShell |
| E2E JSON | 0,03 MB | **Sí**, inventario (479 clusters). El video ~1,6 GB **no** |

SageMaker `xlarge` puede *almacenar* un tfrecord. Sigue haciendo falta TensorFlow y una
pregunta de ML que **no** está en los RA (trayectorias / video). No copies esos shards a S3
para repartirlos: la licencia Waymo lo prohíbe.

---

## 6 · Databricks Free Edition (explorar escala)

Paso a paso (cuenta, Git Folder, Volume, qué hace Kedro y qué no):
[`databricks_free.md`](databricks_free.md).

No reemplaza Academy ni Colab. `kedro run` sigue en CloudShell/local. Aquí subes el **mismo**
parquet a un Volume privado y pruebas `spark.read.parquet(…).count()`. No EMR.

---

## 7 · Colab sigue siendo el camino de clase

| Situación | Dónde |
|---|---|
| Actividades 1.1–3.3 (parquet v2) | Colab, notebook alumno |
| Lote real de 8 segmentos | Colab, notebook **14** |
| “¿Y Motion / v1 / el video?” | No se bajan. [Mapa de productos](productos_waymo.md) |
| “Se me acabó la RAM / el disco” | AWS Academy, esta guía |
| Quiero ver Spark | Databricks Free Edition |
| Proyecto de equipo | 14 → parquet → notebook **10**; Kedro `waymo_real` si hay ≥2 segmentos |

En Colab: *Archivo → Guardar una copia en Drive* **antes** de autenticar. La cuenta de Colab
tiene que ser la misma que aceptó Waymo.

---

## 8 · Si algo no abre

1. ¿El punto AWS está **verde**? Si no, Start Lab y espera.
2. ¿La región es **N. Virginia**? Si no, cámbiala; no uses Santiago ni Ohio.
3. ¿Hiciste el quiz de seguridad (≥ 70 %)? Canvas puede ocultar el lab.
4. Mira las tres **demostraciones** del módulo *Recursos* (acceso, problemas generales, consola).
5. Presupuesto en 50/50: End Lab, avisa al docente. No crees otra cuenta por tu cuenta.
6. Error 403 al bajar Waymo: la Google de GCS no es la que aceptó los términos.

---

## 9 · Cerrar (obligatorio)

1. En SageMaker/EC2: Stop / Terminate.
2. En el iframe del lab: **End Lab**.
3. El parquet y los notebooks que quieras entregar: Drive o el fork de GitHub, **no** el lab.

El docente ve el mismo curso en *vista del estudiante*. Tú no tienes esos botones
(“Abandonar la vista del estudiante”); ignóralos si algún pantallazo los muestra.
