# `kedro_mly1101` — el pipeline de datos de la asignatura

Proyecto [Kedro](https://kedro.org) que convierte el análisis de la EA1 en un proceso
reproducible: un comando, sin intervención humana, con las decisiones en configuración y cada
paso cubierto por tests.

**No es una demostración.** Es la columna de ingeniería sobre la que crecen las experiencias
siguientes: las pipelines de EA2 y EA3 se enchufan aquí, sobre el mismo dataset de detecciones.

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
```

Las salidas van a `data/`, que **no se versiona**: se regenera en un par de segundos.

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
    calidad/              diagnóstico: 4 nodos independientes  (EA1 · Act. 1.3)
    preprocesamiento/     limpieza: 5 nodos encadenados        (EA1 · tabla de decisiones)
    supervisado/          modelamiento: 8 nodos                (EA2)
data/                     salidas. No se versiona
```

### Las capas del catálogo

| Capa | Qué contiene | Regla |
|---|---|---|
| `01_raw` | Lo que llegó (el CSV, fuera de este proyecto) | **No se toca nunca.** Es la evidencia de origen |
| `02_intermediate` | Diagnósticos y tablas de trabajo | Se puede borrar y regenerar |
| `03_primary` | `detecciones_limpias.parquet`, listo para modelar | Lo consume el pipeline `supervisado` |
| `04_feature` … `07_model_output` | Tabla de modelamiento, partición, modelo y métricas | Salidas de la EA2 |

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
Cada bloque de ese archivo es una fila de la tabla de decisiones de la EA1. Agregar una variante
de escritura no debería exigir tocar Python, ni volver a probar nada, ni que quien la agrega sepa
programar.

**3. Marcar antes que eliminar.**
Un valor imposible se convierte en faltante; no se borra la fila entera, porque el resto de esa
fila sí era válido. Lo único que se elimina son los duplicados exactos, que por definición no
aportan nada.

> **Consecuencia contraintuitiva:** después de limpiar hay **más** celdas faltantes que antes
> (2.862 → 4.420). No aparecieron faltantes nuevos: los que estaban disfrazados de `-1` o de
> `"N/D"` pasaron a contarse. Si esa cifra bajara, sería la señal de que se eliminaron filas en
> vez de marcarlas. Hay un test que lo deja por escrito.

---

## El pipeline `supervisado` (EA2)

**La pregunta:** ¿se puede anticipar qué detecciones van a ser difíciles
(`detection_difficulty`) a partir de la geometría del objeto y de dónde está?

Se descartó clasificar `object_type`, que parecía lo natural: sobre este dataset se resuelve al
**99,98 %** con cualquier partición, porque el generador sortea las dimensiones por tipo de
objeto y basta el largo de la caja. Un ejercicio donde todo sale perfecto no enseña a evaluar.

### Lo que produce, medido

| Clase | Precisión | Recall | F1 | Soporte |
|---|---|---|---|---|
| `LEVEL_1` (88,9 %) | 0,93 | 0,96 | 0,94 | 9.122 |
| `LEVEL_2` (11,1 %) | **0,54** | **0,40** | **0,46** | 1.132 |
| **exactitud** | | | **0,90** | |
| **macro avg** | 0,74 | 0,68 | **0,70** | |

**Ese contraste es el material de clase.** El 90 % de exactitud suena bien hasta que se mira la
fila de `LEVEL_2`: el modelo se pierde el **60 % de las detecciones difíciles**, que son justo las
que interesaban. Un promedio global oculta a la minoría.

### Dos fugas de información, y solo una se manifiesta

| Tipo de fuga | Qué se midió | Resultado |
|---|---|---|
| **Por variable derivada** — incluir `num_lidar_points`, de donde sale la etiqueta | F1-macro con y sin ella | **0,7025 → 0,7543 (+0,052)**. Real y medible |
| **Por agrupación** — partir al azar por fila en vez de por segmento | F1-macro con las dos particiones | **−0,005**, es decir nada. 153 segmentos compartidos contra 0 |

La segunda merece una explicación honesta, porque un resultado nulo es fácil de malinterpretar:
en este dataset sintético cada detección se sortea de forma independiente dentro del segmento,
así que la dependencia que la fuga explotaría no existe. **Partir por grupo sigue siendo lo
correcto** —en datos reales de Waymo los fotogramas consecutivos siguen al mismo objeto—, pero
aquí se justifica por cómo se generaron los datos, no por la diferencia que se mide.

> La conclusión que se busca no es *"partir por grupo da igual"*, sino una más incómoda y más
> útil: **un riesgo que no se manifiesta en tus datos de prueba sigue siendo un riesgo.**

---

## Tests

```bash
uv run pytest tests/test_pipeline_kedro.py tests/test_pipeline_supervisado.py -v
```

31 tests, desde la raíz del repositorio. Los nodos son funciones normales de Python, así que se
prueban sin levantar catálogo, ni runner, ni sesión — que es justamente una de las ventajas del
pipeline sobre el notebook.

Son el contrato entre el pipeline y la tabla de decisiones de la EA1: si alguien cambia una
decisión de limpieza, los tests dicen qué pauta quedó desalineada.

---

## Qué se suma en las experiencias siguientes

| Experiencia | Pipeline | Consume | Estado |
|---|---|---|---|
| **EA1** · Datos | `calidad` · `preprocesamiento` | El CSV crudo | ✅ |
| **EA2** · Supervisado | `supervisado` — partición sin fuga, entrenamiento, evaluación por clase | `detecciones_limpias` | ✅ |
| **EA3** · No supervisado | `no_supervisado` — segmentación, reducción de dimensionalidad | `detecciones_limpias` | ⏳ |

Se registran en `pipeline_registry.py` sin tocar lo que ya existe. Cada experiencia **añade
nodos, no reescribe el análisis anterior**. Que `supervisado` corra después de
`preprocesamiento` no está escrito en ninguna parte: se deduce de que consume
`detecciones_limpias`, que el otro produce.

Sobre llevar esto a Databricks (cambiar `pandas.CSVDataset` por `spark.SparkDataset` y qué
implica de verdad), ver el bloque 7 del notebook 04.
