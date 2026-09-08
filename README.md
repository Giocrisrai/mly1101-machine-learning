# MLY1101 · Machine Learning — Duoc UC

Material de trabajo de la asignatura **Machine Learning (MLY1101)**, Escuela de Informática y
Telecomunicaciones, Duoc UC. Segundo semestre 2026.

Docente: Giocrisrai Godoy Bonillo · `gi.godoy@profesor.duoc.cl`

---

## Empieza aquí · el flujo (sin terabytes)

No vas a bajar el Waymo Open Dataset. Vas a bajar **unos pocos archivos de ~1 MB**, armar **una
tabla** y partir train/test **por segmento**. Las fotos (JPEG) y las nubes LiDAR no entran.

| Paso | Qué haces | Dónde |
|---|---|---|
| **1** | Acepta los términos con tu cuenta Google en [waymo.com/open/download](https://waymo.com/open/download/) | Ahí ves Perception v2/v1, Motion y E2E. **No** descargues el bucket: el curso solo usa v2 liviano. |
| **2** | Abre el notebook, *guardar copia en Drive*, misma cuenta | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/14_opcional_waymo_buckets.ipynb) `14_opcional_waymo_buckets.ipynb` |
| **3** | Una celda arma el lote (o en local: `uv run python herramientas/descargar_waymo.py --lote 8`) | Sale `datos/waymo_real/detecciones_reales.parquet`. Si ya hay `muestra/` con varios segmentos, **no pide GCS**: los junta. |
| **4** | Analítica, **partir por grupo**, transformaciones (mismos nodos de Kedro), volúmenes | Mismas celdas del notebook 14 |
| **5** | Actividades 1.1–3.3 del aula | El mismo parquet real |
| **6** | Proyecto de equipo · pipeline | `cd kedro_mly1101 && uv run kedro run` (34 nodos) |
| **7** | RAM / disco / S3 / Databricks | [`docs/recorrido_waymo.md`](docs/recorrido_waymo.md) · [lab AWS](docs/aws_academy_laboratorio.md) · [Databricks Free](docs/databricks_free.md) |

En local, si ya bajaste segmentos, el paso 3 **no pide GCS de nuevo**.

No abras la consola de Google “a ver qué hay”: ahí hay terabytes de video y LiDAR. El curso
usa solo `lidar_box` + `stats`. `camera_box` es otra tabla (cajas 2D), no una foto.

### Dónde computar (la RAM no es una excusa)

El lote de clase (~8 MB) cabe en Colab. **Es el mismo parquet** el que leen el notebook 10,
Kedro `waymo_real`, CloudShell y un Volume de Databricks. Si el pipeline de 40 segmentos, un
RF más grande o el proyecto se quedan sin memoria, hay espacio de la asignatura:

| Dónde | Para qué | Enlace |
|---|---|---|
| **Google Colab** | Actividades y lote de 8 | Los badges de cada notebook |
| **AWS Academy** (curso Duoc) | Laboratorio con más RAM/disco (**USD 50**). Guía: qué servicios sí/no | [Paso a paso](docs/aws_academy_laboratorio.md) · [Curso](https://awsacademy.instructure.com/courses/183052) · [Iniciar lab](https://awsacademy.instructure.com/courses/183052/modules/items/18057525) |
| **Databricks Free Edition** | Explorar Spark / escala (gratis; no es Community Edition) | [Guía](docs/databricks_free.md) · [alta](https://www.databricks.com/learn/free-edition) |

En Academy o Databricks: clona este repo y corre el mismo notebook. El parquet de Waymo es de
**tu** cuenta — no lo publiques. Ninguno de los dos es evaluación.

En AWS usa **CloudShell** (no EC2). SageMaker `medium`/`large` solo si el RA3 sobre 530 k
filas pide más RAM. Bucket S3 **privado** solo para no perder el parquet. Contenedores: no.
Databricks Free Edition: Spark sobre el **mismo** parquet, Kedro no se mueve ahí. En
2026-09-08 el Git Folder del repo público y el Volume managed
`/Volumes/workspace/default/mly1101` se crearon en vivo. Detalle en
[`docs/aws_academy_laboratorio.md`](docs/aws_academy_laboratorio.md) y
[`docs/databricks_free.md`](docs/databricks_free.md).

---

## Ruta de aprendizaje

```
Problema → Datos → Exploración → Preprocesamiento → Modelamiento → Evaluación → Interpretación
```

| RA | Experiencia de aprendizaje | Actividades (horas) | Estado |
|---|---|---|---|
| **RA1** | Ingeniería de Datos y Análisis Exploratorio | 1.1 Fuentes (6) · 1.2 Estructuras (6) · 1.3 EDA (6) · 1.4 Ética (5) | ✅ las cuatro |
| **RA2** | Implementación y Análisis de Modelos de ML | 2.1 CRISP-DM (6) · 2.2 Supervisado (6) · 2.3 No supervisado (12) · 2.4 Interpretación (5) | ✅ las cuatro |
| **RA3** | Optimización y Ensamble de Modelos Avanzados | 3.1 Hiperparámetros (6) · 3.2 Ensamble (6) · 3.3 Robustez (11) | ✅ las tres |
| — | **Evaluación Final Transversal** | 12 h · 40 % de la nota final | ⏳ |

**108 horas · 4 SCT.** Las evaluaciones parciales ponderan 30 / 40 / 30 y suman el **60 %** de la
nota final; el EFT, el **40 %** restante.

> **Los notebooks de actividad usan un hilo de detecciones LiDAR (Perception v2).** El lote
> se arma en el paso 2–3 de arriba. Las Act. 1.1–3.3 y el **proyecto** leen el mismo parquet.
> Las **parciales y el EFT** se rinden sobre los casos oficiales: *Telco Customer Churn*,
> *House Prices* o *Spotify Tracks*.

---

## RA1 · Ingeniería de datos y análisis exploratorio

**Contexto:** trabajas en el equipo de percepción de una empresa de conducción autónoma. Antes de
entrenar cualquier modelo, hay que responder si se puede confiar en las detecciones del sensor
LiDAR.

Son **cuatro actividades** que comparten el mismo dataset y se encadenan: de dónde vienen los
datos → cómo se almacenan y manipulan → qué tan sucios están → a quién perjudican.

| Act. | Indicador · horas | Notebook del alumno | Solucionario docente |
|---|---|---|---|
| **1.1** Fuentes de datos y trabajo colaborativo | IL 1.1 · 6 h | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/02_alumno_fuentes.ipynb) `02_alumno_fuentes.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/02_docente_fuentes.ipynb) |
| **1.2** Estructuras de datos y almacenamiento | IL 1.2 · 6 h | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/03_alumno_estructuras.ipynb) `03_alumno_estructuras.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/03_docente_estructuras.ipynb) |
| **1.3** Análisis exploratorio de datos (EDA) | IL 1.3 · 6 h | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/01_alumno_exploracion.ipynb) `01_alumno_exploracion.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/01_docente_solucionario.ipynb) |
| **1.4** Impacto ético, sesgos y privacidad | IL 1.4 · 5 h | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/07_alumno_etica.ipynb) `07_alumno_etica.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/07_docente_etica.ipynb) |

| Notebook transversal | Para quién | Abrir |
|---|---|---|
| `14_opcional_waymo_buckets.ipynb` | **Empieza por aquí** si vas a usar datos reales: lote liviano, grupos, sin imágenes | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/14_opcional_waymo_buckets.ipynb) |
| `00_opcional_waymo_real.ipynb` | EDA profundo sobre un segmento real (un segmento no alcanza para train/test) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/00_opcional_waymo_real.ipynb) |
| `10_proyecto_equipo_plantilla.ipynb` | El equipo la copia; si ya corriste el 14, usa el parquet real | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/10_proyecto_equipo_plantilla.ipynb) |
| `04_opcional_kedro_databricks.ipynb` | Quien quiera ver el análisis como pipeline ([`kedro_mly1101/`](kedro_mly1101/)) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/04_opcional_kedro_databricks.ipynb) |

> **El número del archivo no coincide con el de la actividad.** El notebook de EDA se publicó
> primero como `01` y sus enlaces ya circulan, así que se mantuvo. El número de actividad está
> declarado en la primera celda de cada notebook.

> Los enlaces de Colab apuntan a `github.com/Giocrisrai/mly1101-machine-learning`. Si publicas el
> repositorio con otro nombre, actualiza `URL_REPO` en `herramientas/contenido_semana01.py` y en
> este README, y vuelve a ejecutar `python herramientas/construir_notebooks.py`.

### Estructura de la sesión de EDA · Actividad 1.3 (4 h)

| Bloque | Min | Foco |
|---|---|---|
| 0 · El problema antes del algoritmo | 15 | Por qué no se empieza eligiendo un algoritmo |
| 1 · Carga e inspección | 45 | `.info()`, `.dtypes`, memoria, diagnóstico general |
| 2 · Tipos de variables | 45 | Taxonomía estadística y categorías inconsistentes |
| 3 · Nulos y duplicados | 45 | Nulos ocultos, patrón MNAR, duplicado lógico |
| 4 · Valores atípicos | 45 | IQR vs z-score; imposible vs legítimo |
| 5 · Decisiones | 30 | Tabla de decisiones y fuga de información |
| 6 · Datos responsables | 20 | Sesgo de muestreo y datos personales |
| Cierre | 15 | Mini-informe de calidad de datos |

### Material docente

| Documento | Para qué |
|---|---|
| [`docs/programa_oficial.md`](docs/programa_oficial.md) | **Resumen operativo del Programa de Asignatura**: los 3 RA con sus 12 indicadores, horas por actividad, ponderaciones y qué exige el EFT. Verificar contra esto antes de escribir material nuevo |
| [`docs/guion_clase_actividades_11_12.md`](docs/guion_clase_actividades_11_12.md) | Guion de las actividades 1.1 y 1.2: coreografía de sala, qué preguntar antes de mostrar la cifra, qué recortar y en qué orden |
| [`docs/guion_clase_semana01.md`](docs/guion_clase_semana01.md) | Guion minuto a minuto de la Actividad 1.3 (EDA): preguntas para el curso, momentos críticos, qué recortar si falta tiempo |
| [`docs/guion_clase_actividad_21.md`](docs/guion_clase_actividad_21.md) | Guion de la Actividad 2.1 (CRISP-DM): coreografía, qué preguntar antes de mostrar, qué recortar |
| [`docs/guion_clase_actividad_24.md`](docs/guion_clase_actividad_24.md) | Guion de la Actividad 2.4 (interpretación): de la matriz a la frase de negocio |
| [`docs/rubrica_ra1.md`](docs/rubrica_ra1.md) | Pauta de las cuatro actividades del RA1, con las cifras exactas para corregir |
| [`docs/rubrica_act_2_1.md`](docs/rubrica_act_2_1.md) · [`docs/rubrica_act_2_2.md`](docs/rubrica_act_2_2.md) · [`docs/rubrica_act_2_3.md`](docs/rubrica_act_2_3.md) · [`docs/rubrica_act_2_4.md`](docs/rubrica_act_2_4.md) · [`docs/rubrica_ra3.md`](docs/rubrica_ra3.md) | Pautas de las actividades del RA2 y del RA3 |
| [`docs/superpowers/specs/2026-08-12-mly1101-semana01-eda-design.md`](docs/superpowers/specs/2026-08-12-mly1101-semana01-eda-design.md) | Especificación completa: decisiones de diseño, catálogo de defectos, protocolo de verificación |

### Evaluar las entregas

La rúbrica se convierte a nota con la escala chilena de exigencia 60 %:

```bash
python herramientas/calcular_nota.py 3 4 2 3 3          # IL1 IL2 IL3 IL4 IL5 → 5,2
python herramientas/calcular_nota.py --csv docs/ejemplo_notas.csv   # el curso completo
```

El modo CSV entrega también el promedio del curso y el porcentaje de aprobación. Si tu sede usa
otra exigencia, `--exigencia 0.5`.

---

## RA2 · Modelamiento

La Actividad **2.1** nombra el mapa (CRISP-DM) y cierra la comprensión del negocio que el RA1
se saltó. Las actividades **2.2** y **2.3** parten del **mismo dataset limpio** y hacen
preguntas opuestas. La **2.4** traduce esas métricas a una frase que la organización puede
usar. Las cuatro pertenecen al RA2; 2.2, 2.3 y 2.4 se evalúan juntas en la **Parcial 2**.

| Act. | Sesión | Notebook del alumno | Solucionario | Pauta |
|---|---|---|---|---|
| **2.1** | CRISP-DM · IL2.1 · 6 h | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/12_alumno_crispdm.ipynb) `12_alumno_crispdm.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/12_docente_crispdm.ipynb) | [`rubrica_act_2_1.md`](docs/rubrica_act_2_1.md) |
| **2.2** | Supervisado · IL2.2 · 6 h · 17 TODO | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/05_alumno_supervisado.ipynb) `05_alumno_supervisado.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/05_docente_supervisado.ipynb) | [`rubrica_act_2_2.md`](docs/rubrica_act_2_2.md) |
| **2.3** | No supervisado · IL2.3 · 12 h · 11 TODO | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/06_alumno_no_supervisado.ipynb) `06_alumno_no_supervisado.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/06_docente_no_supervisado.ipynb) | [`rubrica_act_2_3.md`](docs/rubrica_act_2_3.md) |
| **2.4** | Interpretación · IL2.4 · 5 h | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/13_alumno_interpretacion.ipynb) `13_alumno_interpretacion.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/13_docente_interpretacion.ipynb) | [`rubrica_act_2_4.md`](docs/rubrica_act_2_4.md) |

**Los notebooks de 2.2–2.4 reutilizan los nodos del pipeline**, no una copia. Esos módulos solo importan
pandas, así que funcionan en Colab **sin instalar Kedro**.

### Los dos momentos que cargan cada sesión

**Act. 2.2 — el baseline.** Los alumnos entrenan un bosque aleatorio que alcanza un **89,65 %** de
exactitud. Después descubren que un modelo que **responde siempre lo mismo, sin mirar los datos**,
saca **88,96 %**. Siete décimas de diferencia. Y sin embargo el F1-macro pasa de 0,47 a 0,70.

> Dos métricas sobre el mismo modelo, con conclusiones opuestas. Esa es la sesión.

**Act. 2.3 — los buses.** El agrupamiento encuentra cuatro grupos; el más pequeño (1,5 % de las filas,
`box_length` a casi **cinco desviaciones típicas**) son los **buses**: los mismos atípicos
legítimos que en la Actividad 1.3 se aprendió a *no* eliminar. Aparecen solos, sin que nadie se lo pidiera.

> Si en la Actividad 1.3 hubieran hecho caso al criterio IQR y eliminado los atípicos, este
> grupo no existiría.

**Act. 2.4 — de cada 100.** El F1 de 0,46 no entra a una reunión. Traducido: **de cada 100
detecciones difíciles, el modelo pierde 60.** Esa es la frase que la organización puede usar.

---

## RA3 · Optimización y selección

Tres actividades que producen el **mismo resultado incómodo**, y esa repetición es deliberada.

| Act. | Sesión | Notebook del alumno | Solucionario |
|---|---|---|---|
| **3.1** | Hiperparámetros · IL3.1 · 6 h | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/08_alumno_hiperparametros.ipynb) `08_alumno_hiperparametros.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/08_docente_hiperparametros.ipynb) |
| **3.2** | Ensamble · IL3.2 · 6 h | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/09_alumno_ensamble.ipynb) `09_alumno_ensamble.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/09_docente_ensamble.ipynb) |
| **3.3** | Robustez y selección · IL3.3/3.4 · 11 h | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/11_alumno_seleccion.ipynb) `11_alumno_seleccion.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/11_docente_seleccion.ipynb) |

Pauta común: [`docs/rubrica_ra3.md`](docs/rubrica_ra3.md).

| Actividad | Lo que se intenta | Lo que se mide | Conclusión |
|---|---|---|---|
| 3.1 | Ajustar 12 configuraciones | **−0,0006** de F1-macro | El ajuste no mejora nada |
| 3.2 | Combinar tres modelos | **−0,0040**, y más lento | El ensamble tampoco |
| 3.3 | Distinguir cuál es mejor | Diferencias **< ruido (0,0079)** | No se puede distinguir |

**La conclusión no es que estas técnicas no sirvan.** Es que atacan la **varianza**, y aquí el
cuello de botella es el **sesgo**: la información necesaria no está en las variables. Saberlo con
evidencia vale más que sospecharlo — y es lo que permite dejar de gastar tiempo por la vía
equivocada.

> Un alumno que reporte *"mejoré el modelo ajustando hiperparámetros"* no ha entendido la
> experiencia, por bien ejecutado que esté el código.

---

## El pipeline: `kedro_mly1101/`

El análisis del RA1 también existe como **pipeline reproducible de [Kedro](https://kedro.org)**,
versionado en [`kedro_mly1101/`](kedro_mly1101/) y cubierto por los tests de `tests/test_pipeline_*.py`. No es una demostración:
es la columna de ingeniería sobre la que crecen las experiencias siguientes.

```bash
uv sync --extra kedro
cd kedro_mly1101 && uv run kedro run
```

**34 nodos** (`__default__` = `waymo_real`): ingesta de Perception v2 + EDA + ML. Del parquet
real a la selección sustentada del modelo:

| Experiencia | Pipeline | Nodos | Consume | Estado |
|---|---|---|---|---|
| **RA1** · Datos | `calidad` · `preprocesamiento` | 4 + 5 | `detecciones_reales` | ✅ |
| **RA2** · Supervisado (Act. 2.2) | `supervisado` | 8 | `detecciones_limpias` | ✅ |
| **RA2** · No supervisado (Act. 2.3) | `no_supervisado` | 7 | `detecciones_limpias` | ✅ |
| **RA3** · Optimización (Act. 3.1–3.3) | `optimizacion` | 6 | Salidas de `supervisado` | ✅ |
| — · Fuentes | `ingesta` | 4 | `datos/waymo_real/muestra/` | ✅ inventario + v2 + camera_box + E2E |
| — · Todo | `__default__` = `waymo_real` | 34 | Perception v2 al modelo | ✅ |

El pipeline `supervisado` responde una pregunta con sustancia: **¿se puede anticipar qué
detecciones van a ser difíciles?** Sobre v2 alcanza exactitud 0,7805 con un F1 de **0,0893 en
`LEVEL_2`** — se pierde el 94 % de las detecciones difíciles. Ese contraste es el material de
clase, no un defecto que haya que tapar.

```bash
python herramientas/descargar_waymo.py --muestra 40      # ~40 MB, tras aceptar los términos
cd kedro_mly1101 && uv run kedro run
```

Cifras **medidas** el 2026-09-08 sobre Perception v2 (40 segmentos):

| | Waymo v2 |
|---|---|
| Filas · segmentos | **530.396 · 40** |
| vehicle / sign / peatón / ciclista | **256.855 / 140.319 / 130.836 / 2.386** |
| `weather` | **530.396 `sunny`** |
| Celdas faltantes / valores imposibles | **0 / 0** |
| F1 `LEVEL_2` (prueba) | **0,0893** (recall 0,0588 · 1.572 aciertos de 26.713) |
| Exactitud | **0,7805** |
| Silueta | **0,5228 → 0,6103 (k=2…8), sin codo** |

**Qué hay en disco.** v2 entra al modelo. `camera_box` = 407.267 filas (40 segmentos).
E2E = JSON de 479 secuencias. `camera_image`, v1 y Motion = 0 archivos.
`kedro run` cierra **34/34**; F1 `LEVEL_2` = 0,0893.

Los detalles y el informe por clase están en [`kedro_mly1101/README.md`](kedro_mly1101/README.md).

Cada experiencia **añade nodos, no reescribe el análisis anterior**. Los detalles, las decisiones
de diseño y qué cambiaría en Databricks están en
[`kedro_mly1101/README.md`](kedro_mly1101/README.md) y en el notebook 04.

---

## Cómo empezar

### Opción A — Google Colab (recomendada para los estudiantes)

Clic en el badge de Colab del notebook del alumno. La primera celda clona el repositorio y deja
los datos disponibles. No hay que instalar nada.

> **Avísales de esto antes de la clase.** Al ejecutar la primera celda, Colab muestra
> *"Advertencia: Este cuaderno no lo ha creado Google"*. Es el aviso estándar para cualquier
> notebook abierto desde GitHub, no una señal de problema. Hay que pulsar **"Ejecutar de todos
> modos"**. Si no se les advierte, la mitad del curso se detiene ahí.

### Opción B — Haz un fork y trabaja sobre esta base (recomendada para el proyecto)

El proyecto de equipo se construye **encima** de este repositorio, no al lado. Un integrante hace
el fork y el resto colabora sobre él.

```bash
# 1. Fork desde la web: botón "Fork" en github.com/Giocrisrai/mly1101-machine-learning
# 2. Clonar TU fork (cambia TU-USUARIO)
git clone https://github.com/TU-USUARIO/mly1101-machine-learning.git
cd mly1101-machine-learning

# 3. Dejar el original como "upstream" para poder traer material nuevo después
git remote add upstream https://github.com/Giocrisrai/mly1101-machine-learning.git
git fetch upstream

# 4. Trabajar siempre en una rama, nunca en main
git checkout -b eda-nuestro-dataset
```

Cuando se publique material de una semana nueva, se trae así sin perder su trabajo:

```bash
git fetch upstream && git merge upstream/main
```

**Importante:** los enlaces de Colab de este README apuntan al repositorio original. Para abrir
los notebooks de *tu* fork, cambia `Giocrisrai` por tu usuario en la URL de Colab, o actualiza
`URL_REPO` en `herramientas/contenido_semana01.py` y vuelve a generar los notebooks.

### Opción C — Entorno local con `uv`

[`uv`](https://docs.astral.sh/uv/) es el gestor de entornos y dependencias de Astral. Crea el
entorno, resuelve las versiones y las deja fijadas en `uv.lock`, de modo que **todos los
integrantes del equipo y el docente ejecutan exactamente las mismas versiones**.

```bash
# Instalar uv (una sola vez)
curl -LsSf https://astral.sh/uv/install.sh | sh        # macOS y Linux
# En Windows:  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Crear el entorno e instalar todo, incluidas las herramientas de desarrollo
uv sync

# Ejecutar cualquier cosa dentro del entorno, sin activarlo
uv run pytest
uv run python herramientas/construir_notebooks.py
uv run jupyter lab notebooks/03_alumno_estructuras.ipynb
```

`uv sync` instala las dependencias de la asignatura más el grupo `dev` (pytest, JupyterLab,
nbconvert, ruff). Los extras se piden aparte porque no todo el mundo los necesita:

```bash
uv sync --extra kedro     # notebook 04: pipeline reproducible
uv sync --extra waymo     # notebook 00: descarga de datos reales de Waymo
```

| Archivo | Para qué |
|---|---|
| `pyproject.toml` | Dependencias declaradas, extras y configuración de pytest y ruff |
| `uv.lock` | Versiones exactas resueltas. **Va versionado**: es lo que hace reproducible el entorno |
| `.python-version` | Versión de Python que `uv` usa por defecto (3.13) |

> **`pandas` está fijado por debajo de 3.0 a propósito.** En pandas 3 una columna de texto deja de
> tener `dtype == object` y pasa a `str`, y varias celdas del material enseñan justamente a leer
> ese `object`. Google Colab sigue en la serie 2.x, así que el tope mantiene alineado lo local con
> lo que ocurre en clase.

### Opción D — Local con `pip`

Si prefieres no instalar `uv`, `requirements.txt` sigue funcionando:

```bash
git clone https://github.com/Giocrisrai/mly1101-machine-learning.git
cd mly1101-machine-learning
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/01_alumno_exploracion.ipynb
```

---

## Los datos

**No hay CSV en el repositorio.** La tabla de las Act. 1.1–3.3 es Perception v2 real:

`datos/waymo_real/detecciones_reales.parquet` — **530.396** detecciones, **40** segmentos
(medido 2026-09-08). Está en `.gitignore`: la licencia de Waymo es de uso no comercial y
**prohíbe redistribuir** los datos.

```bash
# Tras aceptar https://waymo.com/open/terms/
uv run python herramientas/descargar_waymo.py --muestra 40
```

El diccionario y los conteos están en [`datos/README.md`](datos/README.md). El notebook
`14_opcional_waymo_buckets.ipynb` arma el lote; `00_opcional_waymo_real.ipynb` hace EDA de un
segmento. `kedro run` (34 nodos) es el ciclo completo sobre esa tabla. `camera_box` y E2E se
inventarian; **no** entran al modelo.

Las evaluaciones parciales y el EFT usan los casos oficiales (*Telco Churn*, *House Prices*,
*Spotify Tracks*), no este hilo.

---

## Estructura del repositorio

```
pyproject.toml       dependencias, extras y configuración de pytest/ruff (entorno con uv)
uv.lock              versiones exactas: es lo que hace reproducible el entorno
requirements.txt     el mismo entorno para quien prefiera pip

datos/waymo_real/     Perception v2 (gitignored). Tabla de trabajo: detecciones_reales.parquet

src/
  eda.py                utilidades de diagnóstico de calidad de datos      (Act. 1.3)
  fuentes.py            lectura desde SQL, JSON anidado y texto libre      (Act. 1.1)
  formatos.py           benchmark de formatos y pérdida de tipos           (Act. 1.2)
  waymo.py              descarga GCS de Waymo (v2 + catálogo Motion/E2E/v1)

notebooks/           los notebooks generados. NO se editan a mano

herramientas/
  contenido_actividad11.py  fuente única de la Actividad 1.1
  contenido_actividad12.py  fuente única de la Actividad 1.2
  contenido_semana01.py     fuente única de la Actividad 1.3
  contenido_actividad14.py  fuente única de la Actividad 1.4 (ética)
  contenido_actividad22.py  fuente única de la Actividad 2.2 (supervisado)
  contenido_actividad23.py  fuente única de la Actividad 2.3 (no supervisado)
  contenido_actividad31.py  fuente única de la Actividad 3.1 (hiperparámetros)
  contenido_actividad32.py  fuente única de la Actividad 3.2 (ensamble)
  contenido_actividad33.py  fuente única de la Actividad 3.3 (selección)
  contenido_waymo.py        fuente del notebook de datos reales (Perception v2)
  contenido_waymo_buckets.py fuente del tutorial de los cuatro buckets GCS
  construir_notebooks.py    genera todos los .ipynb
  calcular_nota.py          rúbrica → nota de 1,0 a 7,0
  descargar_waymo.py        descarga segmentos reales de Waymo
  analizar_sesgo_waymo.py   análisis de sesgo sobre varios segmentos

kedro_mly1101/       el pipeline reproducible (Kedro). Versionado; sus salidas no
  conf/base/            catalog.yml (dónde vive el dato) y parameters.yml (las decisiones)
  src/.../pipelines/    calidad, preprocesamiento (RA1), supervisado (Act. 2.2),
                        no_supervisado (Act. 2.3), optimizacion (RA3) e ingesta

tests/               pytest de todo lo anterior
docs/                guiones de clase, rúbrica y documentos de diseño
```

### Los notebooks se generan, no se editan a mano

El notebook del alumno y el solucionario salen de **una sola fuente** para que no se
desincronicen. Para modificar un ejercicio:

```bash
# 1. Editar el contenido
$EDITOR herramientas/contenido_semana01.py

# 2. Regenerar los tres notebooks
python herramientas/construir_notebooks.py

# 3. Verificar que el solucionario sigue ejecutando completo
cd notebooks && python -m jupyter nbconvert --to notebook --execute --stdout 01_docente_solucionario.ipynb > /dev/null
```

Editar los `.ipynb` directamente funciona hasta el siguiente build, que los sobrescribe.

---

## Tests

```bash
uv run pytest        # o simplemente `pytest` si ya activaste el entorno
```

**228 tests en total** (contado con `pytest --collect-only` el 2026-09-08), repartidos así:

| Archivo | Tests | Qué verifica |
|---|---|---|
| `tests/test_pipeline_kedro.py` | 19 | Los nodos de calidad y limpieza, y que el grafo se construya sin ciclos |
| `tests/test_pipeline_supervisado.py` | 17 | La partición sin fuga, el entrenamiento, las fugas y que `__default__` = `waymo_real` = 34 nodos |
| `tests/test_pipeline_no_supervisado.py` | 14 | El escalado antes de agrupar, la elección de k y la interpretación |
| `tests/test_ingesta_waymo.py` | 17 | Traducción v2, `camera_box`/E2E a la vista y que no entran al RF |
| `tests/test_pipeline_optimizacion.py` | 14 | Que la búsqueda nunca toque la prueba, y la selección sustentada |
| `tests/test_formatos.py` | 17 | El benchmark de formatos de la Act. 1.2: peso, tiempos y pérdida de tipos |
| `tests/test_fuentes.py` | 15 | La lectura desde SQL, JSON anidado y texto libre de la Act. 1.1 |
| `tests/test_eda.py` | 14 | Las utilidades de diagnóstico de `src/eda.py` |
| `tests/test_calcular_nota.py` | 14 | La conversión de rúbrica a nota (escala 1,0–7,0, exigencia 60 %) |
| `tests/test_analisis_sesgo.py` | 11 | La lógica del análisis de sesgo, incluida la unidad de análisis |
| `tests/test_crispdm.py` | 11 | Las seis fases, el mapa RA2/RA3 y que la carta del proyecto sea usable |
| `tests/test_interpretacion.py` | 9 | Traducir la matriz a frecuencia, costo y error en unidades |
| `tests/test_waymo_descarga.py` | 46 | Descarga, catálogo de buckets, inventario de fuentes y partir por grupo |
| `tests/test_mapeo_waymo.py` | 10 | El mapeo al esquema real de Waymo (se saltan sin datos descargados) |

Dos de esos tests existen para dejar por escrito matices que el material afirma y que serían
fáciles de aceptar sin comprobar:

- `test_el_csv_pierde_los_tipos_y_el_parquet_no` — el hallazgo central de la Actividad 1.2.
- `test_con_pocas_filas_el_parquet_puede_pesar_mas` — el matiz honesto: con unos cientos de filas
  el encabezado de Parquet pesa más de lo que ahorra, y el CSV gana.

Si un test del generador falla, el solucionario dejó de coincidir con lo que reciben los
alumnos.

En esta máquina, con extras `kedro`+`waymo` y la muestra en disco, el 2026-09-08
fue **247 passed**. Sin el extra `waymo` se salta el que importa `google.cloud.storage`.
Sin la muestra se saltan `tests/test_mapeo_waymo.py` y los de `test_ingesta_waymo.py`
que piden 40 segmentos. Los 10 de mapeo pasaron 10/10 contra el segmento
`10023947602400723454_1120_000_1140_000` el 2026-08-13. Para reproducirlos:

```bash
brew install --cask google-cloud-sdk
gcloud auth login                              # cuenta con los términos de Waymo aceptados
python herramientas/descargar_waymo.py         # baja lidar_box + stats de un segmento
pytest tests/test_mapeo_waymo.py -v
```

Comprueban que los nombres de columna del notebook opcional siguen siendo los del dataset real.

### Estado de verificación

| Qué | Cómo | Estado |
|---|---|---|
| Utilidades de `src/eda.py` | `pytest`, 14 tests | ✅ |
| El solucionario ejecuta completo | `jupyter nbconvert --execute` | ✅ |
| El notebook del alumno no filtra la pauta | `grep` sobre el `.ipynb` | ✅ |
| Cifras de la pauta y la rúbrica | Comprobadas contra el parquet real (530.396) | ✅ |
| Esquema del notebook de Waymo | Contrastado con el código fuente oficial (2026-08-12) | ✅ |
| Ejecución del notebook de Waymo | `jupyter nbconvert --execute` sobre datos reales descargados | ✅ ejecutado de extremo a extremo el 2026-08-13 |
| Análisis de sesgo de muestreo | **Censo** de los 798 segmentos de training de Waymo | ✅ medido el 2026-08-16 · [informe](docs/sesgo_waymo.md) |
| **Ejecución en Google Colab** (notebooks 01) | Abiertos desde el badge y ejecutados | ✅ 2026-08-16 · 29 celdas, 0 errores, 0 warnings |
| Ejecución en Colab del notebook de Waymo | Ejecutado en Colab el 2026-08-16 | ⚠️ **depende de tu cuenta**: el código llega a Google Cloud, pero la descarga exige que la cuenta de Colab sea la que aceptó los términos de Waymo. Tres trampas documentadas en el Paso 2 del notebook |
| Mapeo del esquema de Waymo | `pytest tests/test_mapeo_waymo.py` contra un Parquet real | ✅ 10/10 |
| Kedro `__default__` = `waymo_real` | `kedro run` | ✅ **34 nodos** · F1 `LEVEL_2` = **0,0893** |
| Kedro `ingesta` | `kedro run --pipeline ingesta` 2026-09-08 | ✅ **4/4** · v2 40×34,943 MB · camera_box 40×7,181 MB · E2E 1×0,035 MB |
| Notebooks docente 1.1–3.3 + 04 + 10 | `nbconvert --execute` | ✅ |
| Perception v1 / Motion / `camera_image` | 0 archivos en disco | no se bajan |

---

## Licencia

Copyright © 2026 Giocrisrai Godoy Bonillo.

Este material está bajo licencia
[Creative Commons Atribución-NoComercial-CompartirIgual 4.0 Internacional](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.es)
(CC BY-NC-SA 4.0). El texto completo está en [`LICENSE`](LICENSE).

En términos prácticos, puedes:

- **usarlo y adaptarlo** para tus propias clases, citando la fuente;
- **compartir** tus adaptaciones, siempre bajo la misma licencia;
- pero **no** usarlo con fines comerciales.

Si lo reutilizas, una atribución razonable sería:
*"Basado en material de MLY1101 · Giocrisrai Godoy Bonillo, Duoc UC (github.com/Giocrisrai/mly1101-machine-learning), CC BY-NC-SA 4.0"*.

El dataset de Waymo **no está** en este repositorio (licencia no comercial, sin redistribución).
El código y el material docente quedan cubiertos por CC BY-NC-SA 4.0.
El **Waymo Open Dataset no se distribuye aquí**: tiene su propia licencia de uso no comercial
([términos](https://waymo.com/open/terms/)) y cada persona debe aceptarla y descargar los datos
por su cuenta.
