# Especificación — MLY1101 Semana 01 / EA1: Análisis y Preprocesamiento de Datos

- **Fecha:** 2026-08-12
- **Autor:** Giocrisrai Godoy Bonillo (docente)
- **Estado:** aprobado e implementado
- **Repositorio:** <https://github.com/Giocrisrai/mly1101-machine-learning>

---

## 1. Problema

La Semana 1 de MLY1101 inicia la EA1 (20 h). El objetivo pedagógico es que el estudiante entienda
que un proyecto de Machine Learning no empieza eligiendo un algoritmo, sino comprendiendo el
problema y los datos:

```
Problema → Datos → Exploración → Preprocesamiento → Modelamiento → Evaluación → Interpretación
           └────────── alcance de esta especificación ──────────┘
```

Se necesita material para una sesión de taller de ~4 h, con alumnos que ya manejan pandas,
ejecutable en Google Colab (recomendación del PIA) y con respaldo local en Jupyter.

### 1.1 Restricciones de contexto

| Restricción | Origen | Consecuencia de diseño |
|---|---|---|
| Colab como entorno primario | PIA de la asignatura | Los datos deben viajar con el repositorio; nada de instalaciones |
| Datos "reales o suficientemente contextualizados" | Coordinación de la línea | Dataset sintético admisible si el contexto y el esquema son reales |
| 90 h de laboratorio/taller | Programa | El material es práctico; la teoría entra como pregunta, no como diapositiva |
| Progresión hacia la EFT (40 %) | Programa | Un repositorio por asignatura, no uno por semana |
| Licencia no comercial de Waymo | [Términos de Waymo](https://waymo.com/open/terms/) | Prohibido incluir datos de Waymo en el repositorio |

---

## 2. Decisiones y sus alternativas descartadas

| Decisión | Elección | Alternativa descartada | Motivo |
|---|---|---|---|
| Dataset principal | Sintético con esquema Waymo v2 y defectos inyectados | Waymo real; dataset público chileno (CONASET) | Control exacto de lo que el alumno debe descubrir; sin login, sin descarga, sin problema de licencia; la pauta puede afirmar cifras exactas |
| Waymo real | Notebook opcional, verificado de extremo a extremo | Incluir un subconjunto en el repo | La licencia prohíbe la redistribución |
| Entrega | Repo público en GitHub con badges de Colab | ZIP en la plataforma | Un clic para el alumno, sin cuenta de GitHub |
| Versiones del notebook | Alumno (TODO) + docente (solucionario) | Un solo notebook resuelto | El docente necesita pauta; el alumno necesita trabajo por hacer |
| Generación de los notebooks | Fuente única en Python → dos `.ipynb` | Editar dos `.ipynb` a mano | Evita que se desincronicen al corregir un ejercicio |
| Alcance | ~4 h, alumnos que ya manejan pandas | 2 h nivel básico | Permite llegar a outliers y a las decisiones de preprocesamiento |
| Estructura | Un repo por asignatura, carpetas por experiencia | Un repo por semana | La EFT exige lógica de proyecto progresiva |

**Fuera de alcance en esta iteración:** algoritmos predictivos, material de EA2/EA3, evaluación
automática de las entregas de los alumnos.

---

## 3. Arquitectura

Cinco unidades, cada una con una responsabilidad y un contrato explícito.

### 3.1 `src/generar_dataset.py` — generación del dataset

- **Entrada:** `--filas` (int), `--semilla` (int), `--salida` (ruta).
- **Salida:** un CSV.
- **Invariante:** misma semilla ⇒ archivo idéntico byte a byte. Verificado por
  `test_es_reproducible_byte_a_byte`.
- **Estructura:** cada defecto se inyecta en una función nombrada
  (`_inyectar_nulos_ocultos`, `_inyectar_duplicados`, …) para que el catálogo se lea sin entrar
  en la implementación. `CATALOGO_DEFECTOS` es la lista canónica de los 10 defectos.
- **Orden de inyección** (importa): base limpia → outliers legítimos → outliers imposibles →
  nulos MNAR → nulos ocultos → categorías inconsistentes → timestamps corruptos → duplicados.
  Los duplicados van al final para que arrastren los defectos ya inyectados, y la permutación
  final impide que queden contiguos.

### 3.2 `src/eda.py` — diagnóstico reutilizable

Funciones puras, sin estado, sin `print` ni gráficos, para que sirvan en los notebooks y en los
tests por igual.

| Función | Contrato |
|---|---|
| `resumen_calidad(df, centinelas, max_ejemplos)` | Una fila por columna: dtype, cardinalidad, nulos, centinelas, `pct_faltante_total` |
| `detectar_outliers_iqr(serie, k)` | Serie booleana; los `NaN` son `False` |
| `limites_iqr(serie, k)` | `(inferior, superior)` coherente con la función anterior |
| `detectar_outliers_zscore(serie, umbral)` | Serie booleana; devuelve todo `False` si σ = 0 |
| `normalizar_categoria(serie, mapa)` | `strip` → `lower` → `mapa`; conserva los `NaN` |
| `reporte_duplicados(df, llave)` | `dup_exactos`, `dup_por_llave`, `dup_logicos = por_llave − exactos` |
| `matriz_nulos_por_grupo(df, columna, grupos)` | % de nulos cruzado por 1 o 2 variables |
| `valores_imposibles(df, reglas)` | Cuenta violaciones de reglas de dominio expresadas como `query` |
| `a_numerico(serie)` | `to_numeric(errors="coerce")` con nombre explícito |
| `resumen_desbalance(serie)` | `n`, `pct`, `ratio_vs_mayoritaria` |
| `perfil_numerico(df, columnas)` | `describe()` + asimetría, curtosis y outliers IQR |

### 3.3 `herramientas/` — fuente única de contenido

- `contenido_semana01.py`: lista `CELDAS`. Cada celda de código declara su versión resuelta y,
  opcionalmente, su versión con `TODO`. Las celdas `solo_docente` contienen la pauta.
- `contenido_waymo.py`: celdas del notebook opcional.
- `construir_notebooks.py`: emite los tres `.ipynb` (nbformat 4.4) y antepone el badge de Colab.

**Regla:** los `.ipynb` son artefactos generados. Editarlos a mano funciona hasta el siguiente
build, que los sobrescribe.

### 3.4 `tests/` — el contrato con la pauta

63 tests (19 del generador, 14 de `eda.py`, 14 del cálculo de notas, 6 del análisis de sesgo y
10 del mapeo de Waymo, que se saltan sin datos descargados). Su función no es solo detectar
regresiones de código: **si un defecto desaparece del
generador, la pauta del docente pasa a mentir**. Los tests son lo que mantiene alineados el
dataset, el solucionario y la rúbrica.

### 3.5 `docs/` — material docente

`guion_clase_semana01.md` (minuto a minuto), `rubrica_ea1.md` (evaluación por indicador) y esta
especificación.

### 3.6 Flujo de datos

```
src/generar_dataset.py --semilla 42
        ↓
datos/crudos/detecciones_waymo_like.csv   (versionado en el repositorio)
        ↓                    ↓                       ↓
notebook alumno      notebook docente         tests (verifican los defectos)
        ↓                    ↓
   mini-informe        pauta + rúbrica
```

En Colab, la primera celda detecta el entorno y clona el repositorio. En local usa rutas
relativas (`RAIZ = Path("..")` desde `notebooks/`).

---

## 4. El dataset

### 4.1 Esquema y correspondencia con Waymo v2

Los nombres de la columna de la derecha fueron verificados el 2026-08-12 contra
`v2/perception/box.py`, `v2/perception/context.py` y `tutorial/tutorial_v2.ipynb` del repositorio
[waymo-research/waymo-open-dataset](https://github.com/waymo-research/waymo-open-dataset).

| Columna de clase | Tipo estadístico | Columna real en Waymo Open Dataset v2 |
|---|---|---|
| `segment_id` | nominal | `key.segment_context_name` |
| `timestamp_micros` | temporal | `key.frame_timestamp_micros` |
| `id_interno` | identificador | `key.laser_object_id` |
| `object_type` | nominal | `[LiDARBoxComponent].type` (entero 1–4) |
| `box_center_x/y/z` | continua | `[LiDARBoxComponent].box.center.x/y/z` |
| `box_length/width/height` | continua | `[LiDARBoxComponent].box.size.x/y/z` |
| `speed_mps` | continua | √(`speed.x`² + `speed.y`²) — en Waymo es un **vector** |
| `num_lidar_points` | discreta | `[LiDARBoxComponent].num_lidar_points_in_box` |
| `detection_difficulty` | ordinal | `[LiDARBoxComponent].difficulty_level.detection` (entero) |
| `weather` | nominal | `[StatsComponent].weather` (solo `Sunny` / `Rain`) |
| `time_of_day` | nominal | `[StatsComponent].time_of_day` (`Day` / `Night` / `Dawn/Dusk`) |
| `sensor_version` | constante | — (columna inventada para el ejercicio) |

Diferencias deliberadas respecto del original: `fog` y las variantes en español de `weather`
(el Waymo real solo distingue `Sunny` y `Rain`) y la columna `sensor_version`.

### 4.2 Estructura generativa (lo que *no* es defecto)

El dataset tiene señal real que el análisis exploratorio puede descubrir:

- `num_lidar_points ~ Poisson(900 / (1 + d/12)²)` con `d = √(x² + y²)`. Correlación de Spearman
  esperada con la distancia: **< −0,5** (verificado por test).
- Las detecciones con pocos puntos tienden a `LEVEL_2`.
- Las dimensiones son coherentes por tipo de objeto (peatón ≈ 1,72 m de alto).
- La señalética no se mueve (< 2 % supera 1 m/s, y solo por los outliers inyectados).

Sin esta estructura, el ejercicio sería solo limpieza; con ella, hay algo que aprender de los
datos.

### 4.3 Catálogo de defectos — cifras exactas

Con `--filas 40000 --semilla 42` ⇒ 40.680 filas × 16 columnas (4,6 MB).

| # | Defecto | Cifra verificada | Enseña |
|---|---|---|---|
| 1 | `timestamp_micros` con `"N/D"` | 60 filas (0,15 %) | Un valor de texto rompe el dtype de toda la columna |
| 2 | `num_lidar_points = -1` | 1.198 (2,9 %) | Un nulo puede estar disfrazado de valor válido |
| 3 | `weather`: 11 variantes → 3 categorías | 2.075 nulos (5,1 %) | Normalización; espacios invisibles |
| 4 | `object_type`: 7 variantes → 4 categorías | 2.343 filas | Impacto directo sobre la variable objetivo |
| 5 | Duplicados exactos y lógicos | 480 + 200 | `drop_duplicates()` no basta |
| 6 | Outliers imposibles | 157 > 60 m/s · 122 con alto 0 · 80 con largo < 0 | Reglas de dominio |
| 7 | Outliers legítimos (buses) | 608 con largo > 12 m | No todo atípico es error |
| 8 | Nulos MNAR en `speed_mps` | 787; 33,8 % en LEVEL_2 nocturno vs. 0,4 % en LEVEL_1 | El patrón importa más que el conteo |
| 9 | Desbalance | `CYCLIST` 1,9 % | Anticipa las métricas de EA2 |
| 10 | `sensor_version` constante · `id_interno` 98,3 % único | — | Qué no es una feature |

Cada uno tiene su test en `tests/test_generar_dataset.py`.

### 4.4 Composición del dataset (para el bloque de sesgo)

~20,0 % nocturnas · ~21,3 % con lluvia · 1,9 % ciclistas. La intersección *nocturna + difícil* es
pequeña y además es donde más faltan las velocidades: doble carencia, y es exactamente el punto
del bloque 6.

---

## 5. Diseño instruccional

### 5.1 Estructura de la sesión

| Bloque | Min | TODO | Objetivo |
|---|---|---|---|
| 0 · El problema | 15 | — | Que nadie abra scikit-learn hoy |
| 1 · Carga e inspección | 45 | 1–3 | Que `.info()` deje de ser un trámite |
| 2 · Tipos de variables | 45 | 4–6 | Tipo de pandas ≠ tipo estadístico |
| 3 · Nulos y duplicados | 45 | 7–11 | Nulos ocultos y patrón MNAR ⭐ |
| 4 · Outliers | 45 | 12–15 | Imposible vs. legítimo |
| 5 · Decisiones | 30 | 16–17 | Del diagnóstico a la decisión documentada |
| 6 · Datos responsables | 20 | 18 | Sesgo de muestreo con consecuencia concreta |
| Cierre | 15 | — | Mini-informe |

**Bloque 3 es el imprescindible.** Si el tiempo se acorta, se recorta el 2 y el 6 (ver
`docs/guion_clase_semana01.md`).

### 5.2 Tres ideas que deben quedar

1. *Eliminar filas nunca es gratis: siempre estás eligiendo qué parte de la realidad borrar.*
2. *El umbral estadístico propone; el conocimiento del dominio dispone.*
3. *Un promedio global oculta a las minorías.*

### 5.3 Autochequeo

Cada bloque de ejercicios termina en una celda `assert`. El alumno sabe si acertó sin esperar al
docente, y el docente puede recorrer la sala mirando los ✅ en vez de revisar código.

Diseño de los `assert`: mensaje en español que **orienta sin dar la respuesta**
(`"revisa: ¿estás mirando la columna correcta?"`).

### 5.4 Evaluación

`docs/rubrica_ea1.md`, cinco indicadores mapeados al RA1, con el peso mayor (30 %) en la
identificación y cuantificación de problemas de calidad.

---

## 6. Verificación

| Qué | Cómo | Resultado |
|---|---|---|
| Reproducibilidad del dataset | `pytest` (hash SHA-256 de dos generaciones) | ✅ |
| Presencia de los 10 defectos | `pytest`, un test por defecto | ✅ |
| Utilidades de `eda.py` | `pytest`, 14 tests | ✅ |
| El solucionario ejecuta completo | `jupyter nbconvert --execute` | ✅ exit 0 |
| El notebook del alumno no filtra la pauta | `grep "Pauta docente"` sobre el `.ipynb` | ✅ 0 coincidencias |
| Cifras citadas en la pauta | Comprobadas contra el CSV publicado | ✅ |
| Accesibilidad pública del repo | `curl` sobre `raw.githubusercontent.com` | ✅ HTTP 200 |
| Esquema del notebook de Waymo | Contrastado con el código fuente oficial (2026-08-12) | ✅ |
| Ejecución del notebook de Waymo | `jupyter nbconvert --execute` sobre datos reales | ✅ 2026-08-13, sin errores |
| Mapeo del esquema de Waymo | `pytest tests/test_mapeo_waymo.py` contra un Parquet real | ✅ 10/10 |

### 6.1 Comando de verificación completa

```bash
pytest -q
python herramientas/construir_notebooks.py
cd notebooks && python -m jupyter nbconvert --to notebook --execute --stdout 01_docente_solucionario.ipynb > /dev/null
```

### 6.2 Verificación del notebook de Waymo

El repositorio trae la herramienta para cerrarla cuando haya credenciales:

```bash
brew install --cask google-cloud-sdk     # ya instalado en la máquina del docente (SDK 580.0.0)
gcloud auth login                        # interactivo: abre el navegador
python herramientas/descargar_waymo.py   # baja lidar_box + stats de UN segmento (no imágenes)
pytest tests/test_mapeo_waymo.py -v
```

`tests/test_mapeo_waymo.py` contiene 10 tests que se **saltan** si no hay datos descargados.
Verifican dos cosas distintas:

1. **Que el esquema no cambió:** cada columna que usa el notebook existe en el Parquet real, la
   velocidad sigue siendo un vector, `type` y `difficulty_level.detection` siguen siendo enteros,
   `time_of_day` y `weather` mantienen sus categorías, y la unión `lidar_box` × `stats` por
   `(segment_context_name, frame_timestamp_micros)` encuentra correspondencia para >95 % de las
   filas.
2. **Que el dataset sintético no enseña algo falso:** la correlación negativa entre distancia y
   puntos láser, la altura mediana del peatón (1,4–2,1 m) y la minoría ciclista deben existir
   también en los datos reales.

El segundo grupo es el que importa pedagógicamente: si fallara, el dataset de clase estaría
enseñando una física que no existe.

### 6.3 Resultado de la validación contra datos reales (2026-08-13)

Segmento `10023947602400723454_1120_000_1140_000` (San Francisco, soleado, de día): 18.633
detecciones en 0,95 MB de Parquet más 23 KB de `stats`. Los 10 tests pasaron y el notebook se
ejecutó completo sin errores ni avisos de columnas faltantes.

**Lo que la validación confirmó del dataset sintético:**

| Magnitud | Sintético | Waymo real | Veredicto |
|---|---|---|---|
| Altura mediana del peatón | 1,72 m | 1,74 m | ✅ |
| Largo mediano del vehículo | 4,61 m | 4,42 m | ✅ |
| Correlación distancia ↔ puntos láser | −0,93 | −0,64 | ⚠️ ver abajo |
| Proporción de ciclistas | 1,9 % | 0,8 % | ✅ (la realidad es peor) |

**Lo que la validación reveló y no estaba previsto:**

1. **El sintético es demasiado prolijo.** La relación distancia ↔ puntos láser es casi
   determinista en el generador (−0,93) y claramente más ruidosa en la realidad (−0,64), porque
   el modelo de Poisson no incorpora oclusiones, tamaño del objeto, ángulo de incidencia ni
   reflectancia. **No se corrigió el generador**: la limpieza de la señal es lo que hace que la
   relación sea descubrible en una clase de 4 h. Pero el notebook lo declara explícitamente y lo
   usa como advertencia metodológica (*un dato simulado casi siempre es más limpio que la
   realidad*).
2. **Los datos reales son mucho más livianos de lo estimado.** El diseño asumía "cientos de MB";
   los dos componentes necesarios pesan ~1 MB. Los terabytes del dataset son imágenes y nubes de
   puntos. Consecuencia: la actividad con datos reales **es viable en clase**, no solo como tarea
   avanzada. La documentación se corrigió.
3. **El dataset real trae un problema de calidad que la clase no cubría:** 82 % de nulos en
   `difficulty_level.detection`, que no son datos faltantes sino una convención de codificación
   (el campo se escribe solo cuando la detección es difícil; el nulo significa `LEVEL_1`). Es el
   espejo exacto del `-1` sintético: allá un valor válido escondía un nulo, acá un nulo esconde
   un valor válido. Se incorporó al notebook como hallazgo.
4. **Cero duplicados, cero categorías inconsistentes, cero valores imposibles.** Confirma la
   justificación del dataset sintético: con datos ya curados no se puede enseñar a limpiar.

### 6.4 Limitaciones que persisten

- La comparación de la §6.3 se hizo sobre **un** segmento. El análisis de sesgo posterior
  (§6.5) usa 250 segmentos para las condiciones y 40 para las detecciones, pero **no es una
  muestra aleatoria**: son los primeros del listado del bucket. Los nombres son identificadores,
  así que el orden es arbitrario en la práctica, pero las cifras son indicativas y no un censo.
- `tests/test_mapeo_waymo.py` se salta sin datos descargados, de modo que en una máquina sin
  credenciales el `pytest` pasa en limpio pero **no** revalida el esquema de Waymo.
- La celda `traducir()` avisa qué columnas faltan en lugar de fallar en silencio, para que un
  cambio futuro de esquema se manifieste como mensaje legible y no como `KeyError`.

### 6.5 Análisis de sesgo de muestreo (2026-08-16)

Medido con `herramientas/analizar_sesgo_waymo.py` sobre 250 segmentos (condiciones) y 40
segmentos con detecciones completas (530.396 detecciones):

| Hallazgo | Cifra |
|---|---|
| Segmentos con lluvia | **1 de 250** (0,4 %) |
| Segmentos nocturnos | 24 de 250 (9,6 %) |
| Ubicaciones | 116 SF · 109 Phoenix · 25 otras |
| Peatones + ciclistas de día | 27,05 % |
| Peatones + ciclistas de noche | **14,11 %** |
| Ciclistas | 0,47 % (día) · 0,38 % (noche) · 0,00 % (amanecer/atardecer) |

**El hallazgo metodológico.** Agregando todas las detecciones, la tasa de detecciones difíciles
parecía mayor de día (13,19 %) que de noche (7,04 %), lo que es absurdo. Calculada **por
segmento**, la diferencia se evapora: mediana 4,81 % contra 4,25 %. Un único segmento diurno con
53,81 % de difíciles y muchas filas dominaba el promedio.

Consecuencias sobre el diseño:

1. `analizar_sesgo_waymo.py` reporta **ambas** tablas y advierte explícitamente sobre la unidad
   de análisis. La herramienta no debe cometer el error que enseña a evitar.
2. Se añadió `calidad_por_segmento()`, con tests que usan un caso construido a propósito (dos
   segmentos limpios y uno atípico enorme) para fijar el contraste entre ambas agregaciones.
3. El material conecta esto con EA2: **el split de entrenamiento/prueba debe ser por
   `segment_id`**, no por fila. Es el mismo error con otra cara.

Las cifras se incorporaron al Paso 7 de `00_opcional_waymo_real.ipynb` y al bloque 6 del guion de
clase, donde reemplazan una discusión hipotética por uno medido.

---

## 7. Riesgos asumidos

| Riesgo | Mitigación |
|---|---|
| El dataset es sintético y podría leerse como "de juguete" | Se declara como tal en el README, en `datos/README.md` y en la primera celda; el esquema es real y el notebook opcional permite repetir todo con datos reales |
| Los alumnos podrían leer el solucionario | Está en el mismo repo público. Se asume: quien quiera copiar puede, y el mini-informe con cifras propias es lo que se evalúa |
| El CSV de 4,6 MB versionado | Aceptable: es lo que permite que Colab funcione con un clic |
| El esquema de Waymo puede cambiar | La fecha de verificación está declarada y la traducción avisa qué falta |
| Los badges dependen del nombre del repositorio | `URL_REPO` está en un solo lugar (`contenido_semana01.py`) y el README documenta cómo cambiarlo |

---

## 8. Trabajo futuro

| Cuándo | Qué |
|---|---|
| Semana 2 | Preprocesamiento aplicado: imputación por grupo, codificación, escalado, y `Pipeline` de scikit-learn para evitar la fuga de información |
| EA2 | Aprendizaje supervisado. El desbalance de `CYCLIST` (1,9 %) y las métricas por clase quedaron sembrados en esta sesión |
| EA3 | Aprendizaje no supervisado sobre el mismo dominio: segmentación de detecciones, reducción de dimensionalidad |
| EFT | El mismo dataset o su versión real de Waymo, integrando los tres RA |
| Cuando se amplíe la EA1 | Extender `docs/rubrica_ea1.md`: hoy cubre solo la Semana 1 de las 20 h de la experiencia |
