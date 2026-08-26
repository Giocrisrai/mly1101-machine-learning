# MLY1101 · Machine Learning — Duoc UC

Material de trabajo de la asignatura **Machine Learning (MLY1101)**, Escuela de Informática y
Telecomunicaciones, Duoc UC. Segundo semestre 2026.

Docente: Giocrisrai Godoy Bonillo · `gi.godoy@profesor.duoc.cl`

---

## Ruta de aprendizaje

```
Problema → Datos → Exploración → Preprocesamiento → Modelamiento → Evaluación → Interpretación
```

| Experiencia | Contenido | Estado |
|---|---|---|
| **EA1** · Análisis y preprocesamiento de datos | Fuentes, estructuras, exploración y calidad de datos | ✅ Semana 1 completa (Act. 1.1, 1.2 y 1.3) |
| **EA2** · Aprendizaje supervisado | Regresión y clasificación | 🔧 pipeline listo; material docente ⏳ |
| **EA3** · Aprendizaje no supervisado | Segmentación, reducción de dimensionalidad | ⏳ |
| **EFT** · Evaluación final transversal | Integra los tres RA (40 % de la nota) | ⏳ |

---

## Semana 1 · EA1 — Análisis exploratorio y calidad de datos

**Contexto:** trabajas en el equipo de percepción de una empresa de conducción autónoma. Antes de
entrenar cualquier modelo, hay que responder si se puede confiar en las detecciones del sensor
LiDAR.

La Semana 1 son **tres actividades** que comparten el mismo dataset y se encadenan: de dónde
vienen los datos → cómo se almacenan y manipulan → qué tan sucios están.

| Act. | Indicador | Notebook del alumno | Solucionario docente |
|---|---|---|---|
| **1.1** Fuentes de datos y trabajo colaborativo | IL 1.1 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/02_alumno_fuentes.ipynb) `02_alumno_fuentes.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/02_docente_fuentes.ipynb) |
| **1.2** Estructuras de datos y almacenamiento | IL 1.2 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/03_alumno_estructuras.ipynb) `03_alumno_estructuras.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/03_docente_estructuras.ipynb) |
| **1.3** Análisis exploratorio de datos (EDA) | IL 1.3 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/01_alumno_exploracion.ipynb) `01_alumno_exploracion.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/01_docente_solucionario.ipynb) |

| Notebook transversal | Para quién | Abrir |
|---|---|---|
| `10_proyecto_equipo_plantilla.ipynb` | El equipo la copia y la rellena con **su** dataset | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/10_proyecto_equipo_plantilla.ipynb) |
| `04_opcional_kedro_databricks.ipynb` | Quien quiera ver el análisis como pipeline reproducible ([`kedro_mly1101/`](kedro_mly1101/)) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/04_opcional_kedro_databricks.ipynb) |
| `00_opcional_waymo_real.ipynb` | Quien quiera repetirlo con datos reales de Waymo | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/00_opcional_waymo_real.ipynb) |

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
| [`docs/guion_clase_actividades_11_12.md`](docs/guion_clase_actividades_11_12.md) | Guion de las actividades 1.1 y 1.2: coreografía de sala, qué preguntar antes de mostrar la cifra, qué recortar y en qué orden |
| [`docs/guion_clase_semana01.md`](docs/guion_clase_semana01.md) | Guion minuto a minuto de la Actividad 1.3 (EDA): preguntas para el curso, momentos críticos, qué recortar si falta tiempo |
| [`docs/rubrica_ea1.md`](docs/rubrica_ea1.md) | Rúbrica por indicador de logro, mapeada al RA1, con las cifras exactas para corregir |
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

## El pipeline: `kedro_mly1101/`

El análisis de la EA1 también existe como **pipeline reproducible de [Kedro](https://kedro.org)**,
versionado en [`kedro_mly1101/`](kedro_mly1101/) y cubierto por 16 tests. No es una demostración:
es la columna de ingeniería sobre la que crecen las experiencias siguientes.

```bash
uv sync --extra kedro
cd kedro_mly1101 && uv run kedro run
```

**17 nodos** en tres pipelines, del CSV crudo al modelo evaluado:

| Experiencia | Pipeline | Nodos | Consume | Estado |
|---|---|---|---|---|
| **EA1** · Datos | `calidad` · `preprocesamiento` | 4 + 5 | El CSV crudo | ✅ |
| **EA2** · Supervisado | `supervisado` | 8 | `detecciones_limpias` | ✅ |
| **EA3** · No supervisado | `no_supervisado` | — | `detecciones_limpias` | ⏳ |

El pipeline `supervisado` responde una pregunta con sustancia: **¿se puede anticipar qué
detecciones van a ser difíciles?** Alcanza un 90 % de exactitud con un F1 de **0,46 en la clase
minoritaria** — se pierde el 60 % de las detecciones difíciles, que eran justo las que
interesaban. Ese contraste es el material de clase, no un defecto que haya que tapar.

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

`datos/crudos/detecciones_waymo_like.csv` — 40.680 detecciones de objetos, 16 columnas.

Es un dataset **sintético** que usa el mismo esquema del componente `lidar_box` del
[Waymo Open Dataset v2](https://waymo.com/open/). Contiene **10 defectos de calidad inyectados a
propósito**, documentados en `src/generar_dataset.py::CATALOGO_DEFECTOS` y verificados por
`pytest`. El diccionario de datos está en [`datos/README.md`](datos/README.md).

No hay ningún dato real de Waymo en este repositorio: su licencia es de uso no comercial y no
permite redistribución. El notebook `00_opcional_waymo_real.ipynb` explica cómo obtenerlos
directamente, aceptando los términos, y compara el sintético con el real.

**Los datos reales son más livianos de lo que parece:** el análisis solo necesita los componentes
`lidar_box` (~1 MB por segmento) y `stats` (~23 KB). Los terabytes del Waymo Open Dataset son las
imágenes y las nubes de puntos, que aquí no se usan. Un alumno con cuenta de Google puede hacerlo
en clase.

Para regenerarlo o cambiar su tamaño:

```bash
python src/generar_dataset.py --filas 40000 --semilla 42
```

Misma semilla ⇒ archivo idéntico byte a byte.

---

## Estructura del repositorio

```
pyproject.toml       dependencias, extras y configuración de pytest/ruff (entorno con uv)
uv.lock              versiones exactas: es lo que hace reproducible el entorno
requirements.txt     el mismo entorno para quien prefiera pip

datos/crudos/        dataset de trabajo (versionado, para que Colab funcione con un clic)

src/
  generar_dataset.py    generador determinista del dataset
  eda.py                utilidades de diagnóstico de calidad de datos      (Act. 1.3)
  fuentes.py            lectura desde SQL, JSON anidado y texto libre      (Act. 1.1)
  formatos.py           benchmark de formatos y pérdida de tipos           (Act. 1.2)
  waymo.py              descarga de segmentos reales

notebooks/           los notebooks generados. NO se editan a mano

herramientas/
  contenido_actividad11.py  fuente única de la Actividad 1.1
  contenido_actividad12.py  fuente única de la Actividad 1.2
  contenido_semana01.py     fuente única de la Actividad 1.3
  contenido_waymo.py        fuente del notebook de datos reales
  construir_notebooks.py    genera todos los .ipynb
  calcular_nota.py          rúbrica → nota de 1,0 a 7,0
  descargar_waymo.py        descarga segmentos reales de Waymo
  analizar_sesgo_waymo.py   análisis de sesgo sobre varios segmentos

kedro_mly1101/       el pipeline reproducible (Kedro). Versionado; sus salidas no
  conf/base/            catalog.yml (dónde vive el dato) y parameters.yml (las decisiones)
  src/.../pipelines/    calidad, preprocesamiento (EA1) y supervisado (EA2)

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

**142 tests en total**, repartidos así:

| Archivo | Tests | Qué verifica |
|---|---|---|
| `tests/test_generar_dataset.py` | 19 | Reproducibilidad byte a byte y presencia de **cada uno de los 10 defectos** |
| `tests/test_pipeline_kedro.py` | 16 | Los nodos de calidad y limpieza, y que el grafo se construya sin ciclos |
| `tests/test_pipeline_supervisado.py` | 15 | La partición sin fuga, el entrenamiento y las dos mediciones de fuga |
| `tests/test_formatos.py` | 17 | El benchmark de formatos de la Act. 1.2: peso, tiempos y pérdida de tipos |
| `tests/test_fuentes.py` | 15 | La lectura desde SQL, JSON anidado y texto libre de la Act. 1.1 |
| `tests/test_eda.py` | 14 | Las utilidades de diagnóstico de `src/eda.py` |
| `tests/test_calcular_nota.py` | 14 | La conversión de rúbrica a nota (escala 1,0–7,0, exigencia 60 %) |
| `tests/test_analisis_sesgo.py` | 11 | La lógica del análisis de sesgo, incluida la unidad de análisis |
| `tests/test_waymo_descarga.py` | 11 | La descarga de Waymo y la traducción de sus errores (1 se salta sin el extra `waymo`) |
| `tests/test_mapeo_waymo.py` | 10 | El mapeo al esquema real de Waymo (se saltan sin datos descargados) |

Dos de esos tests existen para dejar por escrito matices que el material afirma y que serían
fáciles de aceptar sin comprobar:

- `test_el_csv_pierde_los_tipos_y_el_parquet_no` — el hallazgo central de la Actividad 1.2.
- `test_con_pocas_filas_el_parquet_puede_pesar_mas` — el matiz honesto: con unos cientos de filas
  el encabezado de Parquet pesa más de lo que ahorra, y el CSV gana.

Si un test del generador falla, el solucionario dejó de coincidir con lo que reciben los
alumnos.

Sin datos de Waymo descargados el resultado es `58 passed, 10 skipped`. Los 10 de Waymo pasaron
10/10 contra el segmento
`10023947602400723454_1120_000_1140_000` el 2026-08-13. Para reproducirlos:

```bash
brew install --cask google-cloud-sdk
gcloud auth login                              # cuenta con los términos de Waymo aceptados
python herramientas/descargar_waymo.py         # baja lidar_box + stats de un segmento
pytest tests/test_mapeo_waymo.py -v
```

Comprueban que los nombres de columna del notebook opcional siguen siendo los del dataset real, y
que las relaciones que reproduce el dataset sintético existen también en los datos reales.

### Estado de verificación

| Qué | Cómo | Estado |
|---|---|---|
| Reproducibilidad del dataset | `pytest` (SHA-256 de dos generaciones) | ✅ |
| Presencia de los 10 defectos | `pytest`, un test por defecto | ✅ |
| Utilidades de `src/eda.py` | `pytest`, 14 tests | ✅ |
| El solucionario ejecuta completo | `jupyter nbconvert --execute` | ✅ |
| El notebook del alumno no filtra la pauta | `grep` sobre el `.ipynb` | ✅ |
| Cifras de la pauta y la rúbrica | Comprobadas contra el CSV publicado | ✅ |
| Esquema del notebook de Waymo | Contrastado con el código fuente oficial (2026-08-12) | ✅ |
| Ejecución del notebook de Waymo | `jupyter nbconvert --execute` sobre datos reales descargados | ✅ ejecutado de extremo a extremo el 2026-08-13 |
| Análisis de sesgo de muestreo | **Censo** de los 798 segmentos de training de Waymo | ✅ medido el 2026-08-16 · [informe](docs/sesgo_waymo.md) |
| **Ejecución en Google Colab** (notebooks 01) | Abiertos desde el badge y ejecutados | ✅ 2026-08-16 · 29 celdas, 0 errores, 0 warnings |
| Ejecución en Colab del notebook de Waymo | Ejecutado en Colab el 2026-08-16 | ⚠️ **depende de tu cuenta**: el código llega a Google Cloud, pero la descarga exige que la cuenta de Colab sea la que aceptó los términos de Waymo. Tres trampas documentadas en el Paso 2 del notebook |
| Mapeo del esquema de Waymo | `pytest tests/test_mapeo_waymo.py` contra un Parquet real | ✅ 10/10 |

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

El dataset `detecciones_waymo_like.csv` es sintético y queda cubierto por esta misma licencia.
El **Waymo Open Dataset no se distribuye aquí**: tiene su propia licencia de uso no comercial
([términos](https://waymo.com/open/terms/)) y cada persona debe aceptarla y descargar los datos
por su cuenta.
