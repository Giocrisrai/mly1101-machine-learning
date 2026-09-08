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
uv run kedro run --pipeline ingesta            # 5 nodos: inventario + v2 + camera_box + E2E
uv run kedro run --pipeline waymo_real         # 35 nodos: eso + EDA + ML sobre Perception v2
```

Las salidas van a `data/` (sintético) y `data/waymo/` (real), que **no se versionan**. El
recorrido sintético tarda un par de minutos; `waymo_real` sobre 530 k filas, alrededor de 15.

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
    ingesta/              fuentes reales: 5 nodos (inventario, v2, camera_box, E2E, comparar)
data/                     salidas del recorrido sintetico. No se versiona
data/waymo/               salidas del recorrido real.        No se versiona
```

### Las capas del catálogo

| Capa | Qué contiene | Regla |
|---|---|---|
| `01_raw` | Lo que llegó (el CSV, fuera de este proyecto) | **No se toca nunca.** Es la evidencia de origen |
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

> **Consecuencia contraintuitiva:** después de limpiar hay **más** celdas faltantes que antes
> (2.862 → 4.420). No aparecieron faltantes nuevos: los que estaban disfrazados de `-1` o de
> `"N/D"` pasaron a contarse. Si esa cifra bajara, sería la señal de que se eliminaron filas en
> vez de marcarlas. Hay un test que lo deja por escrito.

---

## El pipeline `supervisado` (RA2 · Act. 2.2)

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

## El pipeline `no_supervisado` (RA2 · Act. 2.3)

**La pregunta:** sin decirle a nadie qué es cada objeto, ¿aparecen grupos naturales? ¿Y coinciden
con los tipos que el sensor etiquetó?

Es la contracara de la Act. 2.2: allí había etiqueta y se medía el acierto; aquí no la hay y hay que
**justificar** que la estructura encontrada significa algo. `object_type` viaja en la tabla pero
**no entra en el agrupamiento**: se usa solo para contrastar después.

### Lo que salió, y por qué es mejor que un resultado limpio

| | CSV del repo | Waymo v2 (40 segmentos) |
|---|---|---|
| Silueta | **k = 3** (0,473), luego cae | **sin codo**: 0,5228 (k=2) … **0,6103** (k=8) |
| ¿Los grupos recuperan el tipo de objeto? | Parcialmente | **No** (tres ~100 % vehicle; uno 47,24 % peatón / 50,59 % sign) |
| PCA: 2 componentes explican | 70,2 % | **74,47 %** |

En los datos reales, tres grupos son 99,9 / 99,57 / 100 % `vehicle` y el otro mezcla peatones
(47,24 %) y señalética (50,59 %). El agrupamiento **no descubrió los tipos de objeto**: descubrió
estructura de tamaño y densidad de puntos, que es otra cosa.

Y la silueta no tiene máximo, así que el criterio automático para elegir `k` **falla**. Ambas
cosas son el material de clase:

> No existe "el k correcto". La inercia siempre baja al añadir grupos; la silueta a veces tampoco
> decide. La decisión final es de dominio: **cuántos grupos son útiles para quien va a usar el
> resultado.**

---

## Los datos REALES de Waymo: `kedro run --pipeline waymo_real`

**El mismo análisis, sobre datos reales, sin duplicar un solo nodo.** El pipeline `waymo_real`
reutiliza `calidad`, `preprocesamiento`, `supervisado`, `no_supervisado` y `optimizacion` remapeando su entrada:
donde leían el CSV sintético, leen la salida de la ingesta de Waymo. Esa es, en una línea, la
razón de haber separado el catálogo del análisis.

```bash
# 1. Aceptar los términos en https://waymo.com/open/terms/ con tu cuenta de Google
brew install --cask google-cloud-sdk
gcloud auth login

# 2. Descargar VARIOS segmentos livianos (lidar_box + stats; no imágenes)
python herramientas/descargar_waymo.py --muestra 40     # ~40 MB
# Si ya están en disco: --lote 8 ensambla la tabla sin pedir GCS

# 3. Ver las fuentes (rápido) o el análisis completo
cd kedro_mly1101
uv run kedro run --pipeline ingesta        # inventario + v2 + camera_box + E2E
uv run kedro run --pipeline waymo_real     # eso + el mismo grafo de siempre
```

Los datos **no están en el repositorio**: la licencia de Waymo es de uso no comercial y prohíbe
redistribuirlos. Sin ellos, `kedro run` funciona igual; solo `ingesta` y `waymo_real` los
necesitan.

### Qué corre EDA + ML y qué solo se ve

| Fuente | Nodos | EDA | Supervisado / k-medias / RA3 |
|---|---|---|---|
| CSV sintético (pauta) | 30 (`kedro run`) | ✅ | ✅ |
| Perception v2 (`lidar_box` + `stats`) | 30 remapeados | 530.396 filas, 0 nulos | F1 LEVEL_2 = 0,0893; k sin codo |
| `camera_box` | 1 (`ensamblar_cajas_camara`) | **407.267** × 11 | no entra al RF |
| JSON E2E | 1 (`leer_metadatos_e2e`) | **479** clusters | no |
| v1 / Motion / `camera_image` | 0 | 0 archivos en disco | no |

`ingesta` = 5 nodos. `waymo_real` = esas 5 + el grafo de 30 **sin duplicar código**. El modelo
no mezcla productos: una fila de `camera_box` no se concatena con `lidar_box`.

### Por qué varios segmentos y no uno

Con un solo segmento **no se puede partir en entrenamiento y prueba sin fuga**: las ~18.000
detecciones comparten clima, hora y ubicación, así que cualquier corte deja las dos mitades
contaminadas. El pipeline no hace un apaño cayendo a una partición al azar —sería justo la mala
práctica que el material enseña a evitar—: **falla, y el error dice qué descargar.**

### Tres traducciones que no son un cambio de nombre

Están en `src/waymo.py::traducir_esquema`, con tests:

1. **La velocidad es un vector.** Waymo da `speed.x` y `speed.y`; la rapidez es su módulo.
   Quedarse con `speed.x` da valores plausibles y equivocados.
2. **El tipo de objeto es un entero**, no una cadena. Y existe el `0` (*unknown*), que el
   sintético no tiene.
3. **El `NaN` de la dificultad NO es un dato faltante.** Waymo solo rellena
   `difficulty_level.detection` cuando la detección es difícil; vacío significa `LEVEL_1`. En el
   segmento verificado son **15.356 `NaN` de 18.633**: tratarlos como faltantes borraría el 82 %
   de los datos y dejaría una sola clase.

   Es el **reverso exacto** del defecto que se estudia en la Actividad 1.3, donde un `-1`
   disfraza un faltante. Aquí un faltante disfraza un valor.

### Lo que cambia al pasar del CSV de pauta a Waymo v2 (medido 2026-09-08)

Inventario: `data/02_intermediate/inventario_fuentes_waymo.csv`. Clasificador:
`data/waymo/07_model_output/metricas_por_clase.csv` (09:09, mismos 530.396). k/PCA: 10:55.

| | CSV del repo | Waymo v2 (40 segmentos) |
|---|---|---|
| Filas | 40.680 | **530.396** |
| Segmentos | 153 | 40 |
| vehicle / sign / peatón / ciclista | 61,73 / 8,12 / 26,22 / 1,94 % | **48,43 / 26,46 / 24,67 / 0,45 %** (256.855 / 140.319 / 130.836 / 2.386) |
| LEVEL_2 | 11,1 % | 12,33 % (65.394) |
| Mediana `speed_mps` | 5,35 | **0,0133** |
| Mediana `num_lidar_points` | 96 | 36 |
| Mediana `box_length` | 3,99 m | 1,21 m |
| Clima | 3 categorías sucias | **530.396 `sunny`** |
| Ubicación | — | SF 398.065 · Phoenix 132.331 |
| Valores imposibles | hay (inyectados) | **0 filas** |
| Celdas faltantes tras limpiar | suben (defectos destapados) | **0 → 0** |
| F1 LEVEL_2 | 0,46 | **0,0893** (prec. 0,1847 · rec. 0,0588 · 1.572/26.713) |
| Exactitud | 0,90 | **0,7805** |
| F1-macro | 0,70 | **0,4822** |
| Silueta | máximo k=3 (0,473) | **0,5228 (k=2) … 0,6103 (k=8), sin codo** |
| PCA, 2 componentes | 70,2 % | **74,47 %** |
| Ajuste de hiperparámetros | no mejora (pauta Act. 3.1) | **+0,0789** (0,5104 → 0,5893; sí supera ruido) |

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
| **RA1** · Datos | `calidad` · `preprocesamiento` | El CSV crudo | ✅ |
| **RA2** · Supervisado (Act. 2.2) | `supervisado` — partición sin fuga, entrenamiento, evaluación por clase | `detecciones_limpias` | ✅ |
| **RA2** · No supervisado (Act. 2.3) | `no_supervisado` — agrupamiento y reducción de dimensionalidad | `detecciones_limpias` | ✅ |
| **RA3** · Optimización (Act. 3.1–3.3) | `optimizacion` — ajuste, ensamble y selección sustentada | Salidas de `supervisado` | ✅ |
| — | `ingesta` (5) + `waymo_real` (35) — EDA + ML sobre v2; camera_box y E2E a la vista | Parquet de Waymo | ✅ |

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
