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
uv run kedro run               # todo el pipeline
uv run kedro run --pipeline calidad            # solo el diagnóstico
uv run kedro run --pipeline preprocesamiento   # solo la limpieza
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
data/                     salidas. No se versiona
```

### Las capas del catálogo

| Capa | Qué contiene | Regla |
|---|---|---|
| `01_raw` | Lo que llegó (el CSV, fuera de este proyecto) | **No se toca nunca.** Es la evidencia de origen |
| `02_intermediate` | Diagnósticos y tablas de trabajo | Se puede borrar y regenerar |
| `03_primary` | `detecciones_limpias.parquet`, listo para modelar | Es lo que consumirá la EA2 |

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

## Tests

```bash
uv run pytest tests/test_pipeline_kedro.py -v
```

16 tests, desde la raíz del repositorio. Los nodos son funciones normales de Python, así que se
prueban sin levantar catálogo, ni runner, ni sesión — que es justamente una de las ventajas del
pipeline sobre el notebook.

Son el contrato entre el pipeline y la tabla de decisiones de la EA1: si alguien cambia una
decisión de limpieza, los tests dicen qué pauta quedó desalineada.

---

## Qué se suma en las experiencias siguientes

| Experiencia | Pipeline | Consume | Estado |
|---|---|---|---|
| **EA1** · Datos | `calidad` · `preprocesamiento` | El CSV crudo | ✅ |
| **EA2** · Supervisado | `supervisado` — partición sin fuga, entrenamiento, evaluación por clase | `detecciones_limpias` | ⏳ |
| **EA3** · No supervisado | `no_supervisado` — segmentación, reducción de dimensionalidad | `detecciones_limpias` | ⏳ |

Se registran en `pipeline_registry.py` sin tocar lo que ya existe. Cada experiencia **añade
nodos, no reescribe el análisis anterior**.

Sobre llevar esto a Databricks (cambiar `pandas.CSVDataset` por `spark.SparkDataset` y qué
implica de verdad), ver el bloque 7 del notebook 04.
