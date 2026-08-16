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
| **EA1** · Análisis y preprocesamiento de datos | Exploración, calidad de datos, preparación | ✅ Semana 1 disponible |
| **EA2** · Aprendizaje supervisado | Regresión y clasificación | ⏳ |
| **EA3** · Aprendizaje no supervisado | Segmentación, reducción de dimensionalidad | ⏳ |
| **EFT** · Evaluación final transversal | Integra los tres RA (40 % de la nota) | ⏳ |

---

## Semana 1 · EA1 — Análisis exploratorio y calidad de datos

**Contexto:** trabajas en el equipo de percepción de una empresa de conducción autónoma. Antes de
entrenar cualquier modelo, hay que responder si se puede confiar en las detecciones del sensor
LiDAR.

| Notebook | Para quién | Abrir |
|---|---|---|
| `notebooks/01_alumno_exploracion.ipynb` | Estudiantes (18 ejercicios con TODO y autochequeo) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/01_alumno_exploracion.ipynb) |
| `notebooks/01_docente_solucionario.ipynb` | Docente (código resuelto + pauta + criterios de logro) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/01_docente_solucionario.ipynb) |
| `notebooks/00_opcional_waymo_real.ipynb` | Quien quiera repetirlo con datos reales de Waymo | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Giocrisrai/mly1101-machine-learning/blob/main/notebooks/00_opcional_waymo_real.ipynb) |

> Los enlaces de Colab apuntan a `github.com/Giocrisrai/mly1101-machine-learning`. Si publicas el
> repositorio con otro nombre, actualiza `URL_REPO` en `herramientas/contenido_semana01.py` y en
> este README, y vuelve a ejecutar `python herramientas/construir_notebooks.py`.

### Estructura de la sesión (4 h)

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
| [`docs/guion_clase_semana01.md`](docs/guion_clase_semana01.md) | Guion minuto a minuto: preguntas para el curso, momentos críticos, qué recortar si falta tiempo |
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

## Cómo empezar

### Opción A — Google Colab (recomendada para los estudiantes)

Clic en el badge de Colab del notebook del alumno. La primera celda clona el repositorio y deja
los datos disponibles. No hay que instalar nada.

> **Avísales de esto antes de la clase.** Al ejecutar la primera celda, Colab muestra
> *"Advertencia: Este cuaderno no lo ha creado Google"*. Es el aviso estándar para cualquier
> notebook abierto desde GitHub, no una señal de problema. Hay que pulsar **"Ejecutar de todos
> modos"**. Si no se les advierte, la mitad del curso se detiene ahí.

### Opción B — Local

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
datos/crudos/        dataset de trabajo (versionado, para que Colab funcione con un clic)
src/generar_dataset.py   generador determinista del dataset
src/eda.py               utilidades de diagnóstico de calidad de datos
notebooks/           los tres notebooks de la Semana 1
herramientas/
  contenido_semana01.py     fuente única del contenido de la clase
  contenido_waymo.py        fuente del notebook de datos reales
  construir_notebooks.py    genera los tres .ipynb
  calcular_nota.py          rúbrica → nota de 1,0 a 7,0
  descargar_waymo.py        descarga segmentos reales de Waymo
  analizar_sesgo_waymo.py   análisis de sesgo sobre varios segmentos
tests/               pytest del generador, de las utilidades y del mapeo de Waymo
docs/                guion de clase y documento de diseño
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
pytest
```

**68 tests en total**, repartidos así:

| Archivo | Tests | Qué verifica |
|---|---|---|
| `tests/test_generar_dataset.py` | 19 | Reproducibilidad byte a byte y presencia de **cada uno de los 10 defectos** |
| `tests/test_eda.py` | 14 | Las utilidades de diagnóstico de `src/eda.py` |
| `tests/test_calcular_nota.py` | 14 | La conversión de rúbrica a nota (escala 1,0–7,0, exigencia 60 %) |
| `tests/test_analisis_sesgo.py` | 11 | La lógica del análisis de sesgo, incluida la unidad de análisis |
| `tests/test_mapeo_waymo.py` | 10 | El mapeo al esquema real de Waymo (se saltan sin datos descargados) |

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
| Análisis de sesgo de muestreo | 250 segmentos reales de Waymo | ✅ medido el 2026-08-16 · [informe](docs/sesgo_waymo.md) |
| **Ejecución en Google Colab** | Notebook abierto desde el badge y ejecutado completo | ✅ 2026-08-16 · 29 celdas, 0 errores, 0 warnings |
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
