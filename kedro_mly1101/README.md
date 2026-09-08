# `kedro_mly1101` — el pipeline de datos de la asignatura

Proyecto [Kedro](https://kedro.org) que convierte el análisis del RA1 en un proceso
reproducible: un comando, sin intervención humana, con las decisiones en configuración y cada
paso cubierto por tests.

**No es una demostración.** Es la columna de ingeniería de la asignatura: los pipelines de las
Act. 2.2, 2.3 y 3.1–3.3 se enchufan aquí, sobre el mismo dataset de detecciones.

El notebook [`notebooks/04_opcional_kedro_databricks.ipynb`](../notebooks/04_opcional_kedro_databricks.ipynb)
lo explica pieza por pieza y lo ejecuta.

---

## Ejecutar

```bash
uv sync --extra kedro          # una sola vez
cd kedro_mly1101
uv run kedro run               # todo: datos y modelo
uv run kedro run --pipeline calidad            # solo el diagnóstico
uv run kedro run --pipeline preprocesamiento   # solo la limpieza
uv run kedro run --pipeline supervisado        # solo el modelamiento
uv run kedro run --pipeline no_supervisado     # solo el agrupamiento
uv run kedro run --pipeline ingesta            # 4 nodos: inventario + v2 + camera_box + E2E
uv run kedro run --pipeline waymo_real         # alias de `kedro run`: 34 nodos, Perception v2
```

Las salidas van a `data/` (no se versionan). Sobre 530 k filas tarda alrededor de 15 minutos.

Para ver el grafo en el navegador:

```bash
uv run pip install kedro-viz && uv run kedro viz run
```

---

## Estructura

```
conf/base/
  catalog.yml       dónde vive cada dato y en qué formato — el único sitio con rutas
  parameters.yml    las DECISIONES de limpieza — el único sitio con umbrales y mapas
src/kedro_mly1101/
  __init__.py             pone src/ del repositorio en el sys.path (ver abajo)
  pipeline_registry.py    qué pipelines existen
  pipelines/
    calidad/              diagnóstico: 4 nodos independientes  (RA1 · Act. 1.3)
    preprocesamiento/     limpieza: 5 nodos encadenados        (RA1 · tabla de decisiones)
    supervisado/          modelamiento: 8 nodos                (RA2 · Act. 2.2)
    no_supervisado/       agrupamiento y PCA: 7 nodos          (RA2 · Act. 2.3)
    optimizacion/         ajuste, ensamble y selección: 6 nodos (RA3)
    ingesta/              fuentes: 4 nodos (inventario, v2, camera_box, E2E)
data/                     salidas. No se versiona
```

### Las capas del catálogo

| Capa | Qué contiene | Regla |
|---|---|---|
| `01_raw` | Lo que llegó (`detecciones_reales.parquet`) | **No se toca nunca.** Es la evidencia de origen |
| `02_intermediate` | Diagnósticos y tablas de trabajo | Se puede borrar y regenerar |
| `03_primary` | `detecciones_limpias.parquet`, listo para modelar | Lo consume el pipeline `supervisado` |
| `04_feature` … `07_model_output` | Tabla de modelamiento, partición, modelo y métricas | Salidas de las Act. 2.2, 2.3 y 3.1–3.3 |

`detecciones_limpias` se guarda en **Parquet y no en CSV** por lo medido en la Actividad 1.2: el
CSV pierde el tipo de 11 de las 16 columnas, así que guardar ahí desharía el preprocesamiento en
el mismo momento de escribirlo.

---

## Tres decisiones de diseño

**1. Los nodos reutilizan `src/eda.py`; no reimplementan nada.**
Son las mismas funciones que los alumnos usan en los notebooks. Si el diagnóstico del notebook y
el del pipeline se separaran, habría dos verdades sobre los mismos datos. Para que ese
`import eda` funcione, `src/kedro_mly1101/__init__.py` añade la carpeta `src/` del repositorio al
`sys.path`. Se hace ahí porque es lo primero que Kedro importa.

**2. Las decisiones viven en `parameters.yml`, no en el código.**
Cada bloque de ese archivo es una fila de la tabla de decisiones del RA1. Agregar una variante
de escritura no debería exigir tocar Python, ni volver a probar nada, ni que quien la agrega sepa
programar.

**3. Marcar antes que eliminar.**
Un valor imposible se convierte en faltante; no se borra la fila entera, porque el resto de esa
fila sí era válido. Lo único que se elimina son los duplicados exactos, que por definición no
aportan nada.

> **En Perception v2 las celdas faltantes siguen en 0.** El lote llega curado: no hay `-1`
> ni `"N/D"`. El código igual convierte centinelas (los tests los inyectan) para que una
> corrida sucia no ensucie un promedio. Marcar antes que borrar sigue siendo la regla.

---

## El pipeline `supervisado` (RA2 · Act. 2.2)

**La pregunta:** ¿se puede anticipar qué detecciones van a ser difíciles
(`detection_difficulty`) a partir de la geometría del objeto y de dónde está?

Se descartó clasificar `object_type`: las dimensiones de la caja ya separan tipos casi a la
perfección y no enseña a evaluar un modelo. El objetivo es `detection_difficulty`.

### Lo que produce, medido (Perception v2, 2026-09-08)

| Clase | Precisión | Recall | F1 | Soporte |
|---|---|---|---|---|
| `LEVEL_1` (87,67 %) | 0,8173 | 0,9419 | 0,8752 | 119.403 |
| `LEVEL_2` (12,33 %) | **0,1847** | **0,0588** | **0,0893** | 26.713 |
| **exactitud** | | | **0,7805** | |
| **macro avg** | 0,5010 | 0,5004 | **0,4822** | |

**Ese contraste es el material de clase.** El 78 % de exactitud suena mejor que el F1 de
`LEVEL_2` (0,0893): el modelo se pierde el 94 % de las detecciones difíciles. Un promedio
global oculta a la minoría.

### Dos fugas de información

| Tipo de fuga | Qué es | En este lote |
|---|---|---|
| **Por variable derivada** — incluir `num_lidar_points`, de donde sale la etiqueta | Contarle la respuesta al modelo | Se mide en la Act. 2.2 |
| **Por agrupación** — partir al azar por fila en vez de por segmento | Fotogramas del mismo `segment_id` en train y test | Por grupo: **30 / 10** segmentos, **0 compartidos**. Al azar: **40 compartidos** |

**Partir por grupo sigue siendo lo correcto**: en Waymo los fotogramas consecutivos siguen al
mismo objeto. Un riesgo que no se manifieste en la métrica de prueba sigue siendo un riesgo.

> La conclusión que se busca no es *"partir por grupo da igual"*, sino una más incómoda y más
> útil: **un riesgo que no se manifiesta en tus datos de prueba sigue siendo un riesgo.**

---

## El pipeline `no_supervisado` (RA2 · Act. 2.3)

**La pregunta:** sin decirle a nadie qué es cada objeto, ¿aparecen grupos naturales? ¿Y coinciden
con los tipos que el sensor etiquetó?

Es la contracara de la Act. 2.2: allí había etiqueta y se medía el acierto; aquí no la hay y hay que
**justificar** que la estructura encontrada significa algo. `object_type` viaja en la tabla pero
**no entra en el agrupamiento**: se usa solo para contrastar después.

### Lo que salió, y por qué es mejor que un resultado limpio

| | Perception v2 (40 segmentos) |
|---|---|
| Silueta | **sin codo**: 0,5228 (k=2) … **0,6103** (k=8) |
| ¿Los grupos recuperan el tipo de objeto? | **No** (tres ~100 % vehicle; uno 47,24 % peatón / 50,59 % sign) |
| PCA: 2 componentes explican | **74,47 %** |

En los datos reales, tres grupos son 99,9 / 99,57 / 100 % `vehicle` y el otro mezcla peatones
(47,24 %) y señalética (50,59 %). El agrupamiento **no descubrió los tipos de objeto**: descubrió
estructura de tamaño y densidad de puntos, que es otra cosa.

Y la silueta no tiene máximo, así que el criterio automático para elegir `k` **falla**. Ambas
cosas son el material de clase:

> No existe "el k correcto". La inercia siempre baja al añadir grupos; la silueta a veces tampoco
> decide. La decisión final es de dominio: **cuántos grupos son útiles para quien va a usar el
> resultado.**

---

## Los datos: `kedro run` (Perception v2)

**Un solo grafo.** `__default__` y `waymo_real` son el mismo: ingesta + calidad +
preprocesamiento + supervisado + no supervisado + optimización (34 nodos).

```bash
# 1. Aceptar los términos en https://waymo.com/open/terms/ con tu cuenta de Google
brew install --cask google-cloud-sdk
gcloud auth login

# 2. Descargar VARIOS segmentos livianos (lidar_box + stats; no imágenes)
python herramientas/descargar_waymo.py --muestra 40     # ~40 MB

# 3. Ver las fuentes o el análisis completo
cd kedro_mly1101
uv run kedro run --pipeline ingesta        # 4 nodos: inventario + v2 + camera_box + E2E
uv run kedro run                           # 34 nodos: eso + EDA + ML
```

Los datos **no están en el repositorio**: la licencia de Waymo es de uso no comercial y prohíbe
redistribuirlos. Sin ellos, `kedro run` **no arranca**.

### Qué corre EDA + ML y qué solo se ve

| Fuente | Nodos | EDA | Supervisado / k-medias / RA3 |
|---|---|---|---|
| Perception v2 (`lidar_box` + `stats`) | 30 de análisis | 530.396 filas, 0 nulos | F1 LEVEL_2 = 0,0893; k sin codo |
| `camera_box` | 1 (`ensamblar_cajas_camara`) | **407.267** × 11 | no entra al RF |
| JSON E2E | 1 (`leer_metadatos_e2e`) | **479** clusters | no |
| v1 / Motion / `camera_image` | 0 | 0 archivos en disco | no |

`ingesta` = 4 nodos. El modelo no mezcla productos: una fila de `camera_box` no se concatena
con `lidar_box`.

### Por qué varios segmentos y no uno

Con un solo segmento **no se puede partir en entrenamiento y prueba sin fuga**: las ~18.000
detecciones comparten clima, hora y ubicación, así que cualquier corte deja las dos mitades
contaminadas. El pipeline no hace un apaño cayendo a una partición al azar —sería justo la mala
práctica que el material enseña a evitar—: **falla, y el error dice qué descargar.**

### Tres traducciones que no son un cambio de nombre

Están en `src/waymo.py::traducir_esquema`, con tests:

1. **La velocidad es un vector.** Waymo da `speed.x` y `speed.y`; la rapidez es su módulo.
   Quedarse con `speed.x` da valores plausibles y equivocados.
2. **El tipo de objeto es un entero**, no una cadena. Y existe el `0` (*unknown*).
3. **El `NaN` de la dificultad NO es un dato faltante.** Waymo solo rellena
   `difficulty_level.detection` cuando la detección es difícil; vacío significa `LEVEL_1`. En el
   segmento verificado son **15.356 `NaN` de 18.633**: tratarlos como faltantes borraría el 82 %
   de los datos y dejaría una sola clase.

### Cifras medidas el 2026-09-08 (Perception v2, 40 segmentos)

Inventario: `data/02_intermediate/inventario_fuentes_waymo.csv`.
Clasificador: `data/07_model_output/metricas_por_clase.csv`.

| | Waymo v2 |
|---|---|
| Filas · segmentos | **530.396 · 40** |
| vehicle / sign / peatón / ciclista | **48,43 / 26,46 / 24,67 / 0,45 %** |
| LEVEL_2 | 12,33 % (65.394) |
| Mediana `speed_mps` | **0,0133** |
| Clima | **530.396 `sunny`** |
| Ubicación | SF 398.065 · Phoenix 132.331 |
| Nulos / imposibles | **0 / 0** |
| F1 LEVEL_2 | **0,0893** (prec. 0,1847 · rec. 0,0588) |
| Exactitud | **0,7805** |
| F1-macro | **0,4822** |
| Silueta | **0,5228 (k=2) … 0,6103 (k=8), sin codo** |
| PCA, 2 componentes | **74,47 %** |
| Ajuste de hiperparámetros | **+0,0789** (0,5104 → 0,5893) |

Otras fuentes en el mismo `datos/waymo_real/` (no entran al RF):

| | Medido |
|---|---|
| `camera_box` | 407.267 × 11 · 40 segmentos · 7,181 MB · tipos 1/2/4 = 297.902 / 107.507 / 1.858 |
| JSON E2E | 479 secuencias · 36.235 bytes · cluster más frecuente: `Interections` (116, grafía de Waymo) |
| `camera_image`, v1, Motion | **0 archivos** |

**Las filas de F1 y de clima son las que hay que discutir en clase.** Waymo publicado está
curado (0 imposibles). El modelo pierde casi todas las detecciones difíciles (recall 5,88 %).
El 100 % `sunny` es el sesgo del censo (793 de 798 soleados), ahora en el lote con el que se
entrena.

---

## Tests

```bash
uv run pytest tests/test_pipeline_kedro.py tests/test_pipeline_supervisado.py \
              tests/test_pipeline_no_supervisado.py tests/test_pipeline_optimizacion.py \
              tests/test_ingesta_waymo.py -v
```

81 tests de pipeline, desde la raíz del repositorio. Los que necesitan datos reales de Waymo **se saltan** si
no están descargados, así que `pytest` pasa en limpio sin credenciales. Los nodos son funciones normales de Python, así que se
prueban sin levantar catálogo, ni runner, ni sesión — que es justamente una de las ventajas del
pipeline sobre el notebook.

Son el contrato entre el pipeline y la tabla de decisiones del RA1: si alguien cambia una
decisión de limpieza, los tests dicen qué pauta quedó desalineada.

---

## Qué cubre cada experiencia

| Experiencia | Pipeline | Consume | Estado |
|---|---|---|---|
| **RA1** · Datos | `calidad` · `preprocesamiento` | `detecciones_reales` | ✅ |
| **RA2** · Supervisado (Act. 2.2) | `supervisado` — partición sin fuga, entrenamiento, evaluación por clase | `detecciones_limpias` | ✅ |
| **RA2** · No supervisado (Act. 2.3) | `no_supervisado` — agrupamiento y reducción de dimensionalidad | `detecciones_limpias` | ✅ |
| **RA3** · Optimización (Act. 3.1–3.3) | `optimizacion` — ajuste, ensamble y selección sustentada | Salidas de `supervisado` | ✅ |
| — | `ingesta` (4) + `waymo_real` (34) — EDA + ML sobre v2; camera_box y E2E a la vista | Parquet de Waymo | ✅ |

Se registran en `pipeline_registry.py` sin tocar lo que ya existe. Cada experiencia **añade
nodos, no reescribe el análisis anterior**. Que `supervisado` corra después de
`preprocesamiento` no está escrito en ninguna parte: se deduce de que consume
`detecciones_limpias`, que el otro produce.

El material docente de las Act. 2.1–2.4 y 3.1–3.3 (notebooks de alumno, solucionario y rúbrica)
ya existe. Lo que sigue pendiente son las evaluaciones sobre los casos oficiales.

Sobre llevar esto a Databricks Free Edition (Git Folder, Volume, `catalog.yml` y por qué
`kedro run` no se muda al workspace), ver [`docs/databricks_free.md`](../docs/databricks_free.md)
y el bloque 7 del notebook 04. Tamaños medidos y qué máquina usar:
[`docs/recorrido_waymo.md`](../docs/recorrido_waymo.md).
