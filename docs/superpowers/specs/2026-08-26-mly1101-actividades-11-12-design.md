# Especificación — Semana 1 completa: actividades 1.1 y 1.2, proyecto de equipo y entorno reproducible

**Fecha:** 2026-08-26
**Estado:** implementado y verificado
**Extiende:** [`2026-08-12-mly1101-semana01-eda-design.md`](2026-08-12-mly1101-semana01-eda-design.md)

---

## 1. Problema

El repositorio cubría solo la **Actividad 1.3** (EDA) de la Semana 1. Los PPT de la asignatura
definen tres actividades con tres indicadores de logro distintos:

| Actividad | Indicador | Cobertura antes de este trabajo |
|---|---|---|
| 1.1 Fuentes de datos y trabajo colaborativo | IL 1.1 | ninguna |
| 1.2 Estructuras de datos y almacenamiento | IL 1.2 | ninguna |
| 1.3 Análisis exploratorio de datos | IL 1.3 | completa |

Además, las actividades prácticas de los PPT 1.1 y 1.2 piden que cada equipo trabaje con **su
propio caso de negocio y su propio dataset**, y el repositorio solo sabía operar sobre
`detecciones_waymo_like.csv`.

### 1.1 Restricciones de contexto

- El docente tiene clases inmediatas: el material debe ser utilizable, no solo estar diseñado.
- El notebook `01_alumno_exploracion.ipynb` ya está verificado en Colab y sus badges circulan:
  **no se puede renumerar**.
- Los alumnos harán **fork** del repositorio y trabajarán encima; hay que soportar el flujo de
  traer material nuevo sin perder el suyo (`upstream`).
- El entorno local debe ser reproducible y coincidir con Colab.

---

## 2. Decisiones y sus alternativas descartadas

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| Un notebook por actividad, más una plantilla de proyecto aparte | Que los notebooks de actividad trabajen directamente sobre el dataset del equipo | Sin dataset fijo no hay `assert` deterministas ni verificación con `nbconvert`; se pierde el autochequeo, que es lo que permite al docente recorrer la sala mirando los ✅ |
| Numerar los archivos `02` y `03` aunque sean las actividades 1.1 y 1.2 | Renumerar todo a `11`, `12`, `13` | Rompería los enlaces de Colab ya publicados. El número de actividad va en la primera celda |
| La plantilla de proyecto **cae al dataset de la asignatura** si el equipo no configuró el suyo | Dejar las celdas rotas hasta que alguien complete `RUTA_MI_DATASET` | Una plantilla con celdas rotas impide distinguir "error mío" de "error de la plantilla", y no se puede verificar con `nbconvert` |
| Kedro ejecutable, Databricks conceptual | Ambos ejecutables | Databricks exige cuenta y clúster; no se puede pedir en clase. Lo que sí aporta es el código equivalente lado a lado |
| **El proyecto Kedro vive en el repositorio** (`kedro_mly1101/`), versionado y testeado | Crearlo dentro de una celda con `%%writefile` y descartarlo | *(Corregido el mismo día, tras revisión.)* Un proyecto que se crea y se borra en una celda no es base para las Act. 2.2 y 2.3. El notebook ahora **lee y ejecuta el proyecto real** en vez de fabricar uno de juguete |
| Las decisiones de limpieza en `conf/base/parameters.yml` | Escritas dentro de los nodos | Cambiar un umbral o una variante de escritura no debería exigir tocar Python ni saber programar. Cada bloque del YAML es una fila de la tabla de decisiones de la EA1 |
| `marcar antes que eliminar` en todos los nodos de limpieza | `dropna()` sobre las filas con valores imposibles | El resto de la fila era válido. Solo se eliminan los duplicados exactos |
| Cada regla de dominio declara su `columna` además de su `condicion` | Deducir la columna del texto con `expresion.split()[0]` | **Defecto encontrado en revisión.** Una regla que menciona dos columnas (`num_lidar_points < 0 or speed_mps > 100`) limpiaba solo la primera, y en silencio |
| Los nodos de Kedro **reutilizan `src/eda.py`** | Reescribir el análisis dentro del pipeline | El punto pedagógico es que industrializar no exige reescribir: es el premio de haber escrito funciones puras |
| `pandas` fijado en `<3` | Aceptar pandas 3 | En pandas 3 una columna de texto deja de tener `dtype == object`. El material enseña a leer ese `object` y Colab sigue en 2.x |
| E501 fuera de las reglas de ruff | Ignorarlo solo en `contenido_*.py` | También choca con los f-strings de salida alineada de `calcular_nota.py` y con tablas en docstrings. Partirlos empeora el código |

---

## 3. Arquitectura

### 3.1 Módulos nuevos en `src/`

Puros y sin efectos secundarios visibles, como `src/eda.py`: sin `print`, sin gráficos, sin
estado.

| Módulo | Funciones | Sirve a |
|---|---|---|
| `fuentes.py` | `a_sqlite`, `consultar`, `contexto_por_segmento`, `aplanar_contexto`, `aplanar_objetos`, `generar_partes_incidente`, `extraer_segmentos`, `segmentos_comprometidos` | Act. 1.1 |
| `formatos.py` | `medir_formatos`, `dtypes_conservados`, `columnas_con_dtype_cambiado`, `memoria_lista_python` | Act. 1.2 |

`memoria_lista_python` existe porque `sys.getsizeof` sobre una lista mide solo el arreglo de
punteros y no los objetos apuntados: la comparación honesta contra `ndarray.nbytes` requiere
sumar ambos.

### 3.2 El pipeline `kedro_mly1101/`

Proyecto Kedro versionado, con **tres pipelines y 17 nodos**, del CSV crudo a las métricas:

| Pipeline | Nodos | Qué hace |
|---|---|---|
| `calidad` | 4, **independientes entre sí** | Diagnóstico: radiografía por columna, desbalance, reglas de dominio, patrón de faltantes |
| `preprocesamiento` | 5, **encadenados** | Normaliza categorías → descubre nulos ocultos → marca imposibles → quita duplicados y constantes → resume |
| `supervisado` (RA2 · Act. 2.2) | 8 | Prepara variables → parte por segmento → entrena → evalúa por clase → mide dos tipos de fuga |

El orden entre pipelines no está escrito en ninguna parte: `supervisado` consume
`detecciones_limpias`, que `preprocesamiento` produce, y Kedro deduce el resto. El de la Act. 2.3 se
registró igual, sin tocar lo existente.

#### Dos decisiones del pipeline supervisado que se tomaron midiendo

**El objetivo no es `object_type`.** Era el candidato natural, pero se resuelve al **99,98 %** con
cualquier partición: el generador sortea las dimensiones por tipo de objeto y basta el largo de la
caja. Un ejercicio donde todo sale perfecto no enseña a evaluar. Se cambió a
`detection_difficulty`, que está desbalanceado 88,9 / 11,1 y da 90 % de exactitud con **F1 de 0,46
en la clase minoritaria**: el caso de manual de que el promedio oculta a la minoría.

**La demostración de fuga por partición se midió antes de escribirla, y salió en cero.** Se
mantiene el nodo, pero el material lo declara: en este dataset sintético cada detección se sortea
de forma independiente dentro del segmento, así que la dependencia que la fuga explotaría no
existe (−0,005, con 153 segmentos compartidos contra 0). Se añadió en su lugar una fuga que **sí**
se manifiesta: incluir `num_lidar_points`, del que se deriva la etiqueta, infla el F1-macro de
0,7025 a 0,7543. Presentar una demostración de un efecto inexistente habría sido engañoso.

Los nodos reutilizan `src/eda.py` sin reimplementar nada; `src/kedro_mly1101/__init__.py` añade
la carpeta `src/` del repositorio al `sys.path` porque es lo primero que Kedro importa.

### 3.3 Fuentes de contenido

`herramientas/contenido_actividad11.py`, `contenido_actividad12.py`, `contenido_proyecto.py` y
`contenido_kedro.py`, todas reutilizando `md` / `md_docente` / `code` de `contenido_semana01.py`.
`construir_notebooks.py` genera nueve notebooks.

### 3.4 Entorno

`pyproject.toml` + `uv.lock` + `.python-version`. Extras `kedro` y `waymo` fuera del conjunto por
defecto. `requirements.txt` se mantiene alineado a mano para quien prefiera pip.

---

## 4. Diseño instruccional

### 4.1 Actividad 1.1 (2 h, 11 TODO)

El mismo dataset traído por cuatro naturalezas de fuente: CSV local, CSV por URL, SQLite en
memoria, JSON anidado y texto libre. El bloque de SQL destapa que hay **7 categorías de
`object_type` en vez de 4**, lo que permite decir en voz alta que *cambiar de herramienta no
arregla los datos*.

El bloque de ética usa el censo real de Waymo: **793 segmentos soleados de 798**. Es la cifra que
debe quedar si solo queda una.

### 4.2 Actividad 1.2 (2 h, 15 TODO)

El bloque 4 (`.loc` contra `.iloc`) es el que justifica la sesión. Diseño del ejercicio: tras
filtrar los 789 ciclistas y reiniciar el índice, asignar la Series de km/h deja **771 `NaN` y 18
valores cruzados**, sin lanzar ningún error.

Esos 18 no son un detalle: son las etiquetas originales de ciclistas menores que 789, que existen
también en el índice nuevo. El resultado **parcialmente lleno** es el caso peligroso, y por eso el
ejercicio se diseñó para producirlo en vez de un fallo limpio.

El bloque 6 encadena con el 2: el 57 % de memoria ahorrado al optimizar tipos se pierde entero al
guardar en CSV (11 de 16 columnas cambian de tipo). Parquet lo conserva y pesa 0,40×.

### 4.3 Plantilla de proyecto

Una sola versión, sin solucionario. Ejecuta de extremo a extremo sobre el dataset de la
asignatura y se aplica al del equipo cambiando una variable. Incluye la lista de chequeo de
privacidad, la tabla de decisiones con columna **"qué se pierde"**, y una autoevaluación que
exige señalar dónde está la evidencia de cada criterio.

---

## 5. Cifras verificadas contra el CSV publicado

Medidas el 2026-08-26 sobre `detecciones_waymo_like.csv` (semilla 42, 40.680 filas).

| Magnitud | Valor |
|---|---|
| Lista de Python vs `ndarray` (39.893 flotantes) | 1,22 MB vs 0,30 MB (**4,0×**) |
| Ciclo vs vectorizado | del orden de **20×** |
| Memoria del DataFrame, antes → después de optimizar tipos | 20,1 → 8,7 MB (**57 %**) |
| Detecciones `CYCLIST` | 789 (primeras etiquetas: 76, 194, 199, 305, 312) |
| `NaN` de la asignación desalineada | **771 de 789**, más 18 valores cruzados |
| Parquet vs CSV (5.000 filas optimizadas) | 0,23 MB vs 0,57 MB (**0,40×**) |
| Columnas que el CSV pierde de tipo | **11 de 16** |
| Nulos de `speed_mps` por momento del día | Noche 4,08 % · Día 1,47 % · Amanecer 0,91 % |
| Censo de Waymo (798 segmentos de training) | 793 soleados / 5 con lluvia |

Y lo que produce el pipeline sobre el dataset completo:

| Magnitud | Crudo | Limpio |
|---|---|---|
| Filas | 40.680 | 40.200 (−480 duplicados exactos) |
| Columnas | 16 | 15 (−`sensor_version`, constante) |
| Categorías de `object_type` | 7 | 4 |
| Categorías de `weather` | 11 | 3 |
| Celdas faltantes | 2.862 | **4.420 (+1.558)** |

El aumento de faltantes es correcto y contraintuitivo: no aparecen faltantes nuevos, se cuentan
los que estaban disfrazados de `-1` o `"N/D"`. Si esa cifra bajara sería la señal de que se
eliminaron filas en vez de marcarlas, y hay un test que lo fija.

El patrón MNAR que la rúbrica documenta queda medido por el nodo `medir_sesgo_de_faltantes`:
**33,8 % de velocidad faltante en LEVEL_2 nocturno contra 0,36 % en LEVEL_1 nocturno**.

---

## 6. Verificación

```bash
uv sync
uv run pytest                                        # 110 passed, 1 skipped (111 con el extra waymo)
uv run ruff check .
uv run python herramientas/construir_notebooks.py    # nueve notebooks

for nb in 02_docente_fuentes 03_docente_estructuras 01_docente_solucionario \
          10_proyecto_equipo_plantilla 04_opcional_kedro_databricks; do
  uv run python -m jupyter nbconvert --to notebook --execute --stdout \
      --output-dir=/tmp notebooks/$nb.ipynb > /dev/null && echo "$nb OK"
done

grep -c "Pauta docente" notebooks/0[123]_alumno*.ipynb notebooks/10_proyecto*.ipynb   # todos 0
rm -rf notebooks/kedro_mly1101 notebooks/salidas_act12 notebooks/salidas_proyecto
```

### 6.1 Resultado (2026-08-26)

| Qué | Estado |
|---|---|
| Tests | ✅ 142 (63 nuevos: 15 de `fuentes`, 17 de `formatos`, 16 del pipeline de datos, 15 del supervisado) |
| `ruff check` | ✅ limpio |
| Los cinco notebooks resueltos ejecutan completos | ✅ |
| Ningún notebook de alumno filtra la pauta | ✅ |
| El pipeline de Kedro corre 17/17 nodos sobre el proyecto versionado | ✅ |

### 6.2 Tres hallazgos al armar el entorno con uv

Los tres corregidos, y los tres vale la pena tener escritos porque volverán a aparecer:

1. **uv resolvió `pandas` a 3.0.5**, donde una columna de texto ya no tiene `dtype == object`
   sino `str`. Rompía `test_defecto_1_timestamp_corrupto` y desalineaba el material respecto de
   Colab. → `pandas>=2.2,<3`, con el motivo escrito en `pyproject.toml`.
2. **Faltaba `scipy`.** pandas lo necesita por debajo para `corr(method="spearman")`, que usa el
   test de distancia contra puntos láser. El fallo aparece como un `ModuleNotFoundError` dentro
   de pandas, no como una dependencia declarada.
3. **Kedro arrastra `google-api-core`.** Un `pytest.importorskip("google.api_core")` deja de
   saltarse en cuanto se instala el extra `kedro`, aunque el extra `waymo` no esté. El guard
   correcto es sobre `google.cloud.storage`.

### 6.3 Limitaciones que persisten

| Limitación | Estado |
|---|---|
| Ningún notebook nuevo se ha ejecutado **en Colab** | ⏳ pendiente; el 01 sí, el 2026-08-16 |
| El bloque de Databricks no está verificado contra una cuenta real | Es conceptual por diseño |
| La lectura por URL del TODO 2 de la Act. 1.1 depende de que el repositorio sea público | Mitigado: la celda cae al CSV local si falla |
| Los tiempos medidos varían por máquina | Los `assert` comprueban el orden de magnitud, no el valor |

---

## 7. Trabajo futuro

| Cuándo | Qué |
|---|---|
| Hecho (2026-08) | Material docente de Act. 2.2, 2.3 y RA3; rúbrica del RA1 extendida a 1.1–1.4 |
| Hecho (2026-09) | Act. 2.1 y 2.4 |
| Evaluaciones | Formativas, parciales y EFT sobre los casos oficiales |
| Cuando Colab migre a pandas 3 | Migrar el material: `dtype == object` deja de valer y hay que reescribir esas celdas y sus tests |
