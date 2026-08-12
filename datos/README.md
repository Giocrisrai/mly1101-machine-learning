# Datos

## `crudos/detecciones_waymo_like.csv`

**Dataset sintético.** No contiene datos reales de Waymo, ni de personas, ni de ningún sensor
físico. Fue generado por `src/generar_dataset.py` con la semilla 42.

- 40.680 filas × 16 columnas (~4,6 MB)
- **Una fila = una detección de un objeto en un instante determinado**

Usa el mismo esquema del componente `lidar_box` del
[Waymo Open Dataset v2](https://waymo.com/open/data/perception/), de modo que el análisis hecho
sobre este archivo se traslada tal cual a los datos reales
(ver `notebooks/00_opcional_waymo_real.ipynb`).

### Diccionario de datos

| Columna | Tipo estadístico | Unidad | Descripción |
|---|---|---|---|
| `segment_id` | nominal | — | Segmento de conducción (~20 s de grabación) |
| `timestamp_micros` | temporal | µs | Instante de la detección |
| `id_interno` | identificador | — | Identificador único de la detección |
| `object_type` | nominal | — | Tipo de objeto: vehículo, peatón, ciclista, señalética |
| `box_center_x` | continua | m | Posición longitudinal (positivo = adelante del vehículo) |
| `box_center_y` | continua | m | Posición lateral |
| `box_center_z` | continua | m | Altura del centro de la caja |
| `box_length` | continua | m | Largo de la caja delimitadora |
| `box_width` | continua | m | Ancho de la caja |
| `box_height` | continua | m | Alto de la caja |
| `speed_mps` | continua | m/s | Velocidad estimada del objeto |
| `num_lidar_points` | discreta | conteo | Puntos láser que cayeron sobre el objeto |
| `weather` | nominal | — | Condición climática del segmento |
| `time_of_day` | nominal | — | Momento del día |
| `detection_difficulty` | ordinal | — | `LEVEL_1` (fácil) < `LEVEL_2` (difícil) |
| `sensor_version` | constante | — | Versión del firmware |

### Correspondencia con el esquema real de Waymo v2

Verificada el 2026-08-12 contra `v2/perception/box.py`, `v2/perception/context.py` y
`tutorial/tutorial_v2.ipynb` del repositorio
[waymo-research/waymo-open-dataset](https://github.com/waymo-research/waymo-open-dataset).

| Columna aquí | Columna real en Waymo Open Dataset v2 |
|---|---|
| `segment_id` | `key.segment_context_name` |
| `timestamp_micros` | `key.frame_timestamp_micros` |
| `id_interno` | `key.laser_object_id` |
| `object_type` | `[LiDARBoxComponent].type` (entero: 1 vehículo, 2 peatón, 3 señalética, 4 ciclista) |
| `box_center_x/y/z` | `[LiDARBoxComponent].box.center.x/y/z` |
| `box_length/width/height` | `[LiDARBoxComponent].box.size.x/y/z` |
| `speed_mps` | √(`speed.x`² + `speed.y`²) — en Waymo la velocidad es un **vector** |
| `num_lidar_points` | `[LiDARBoxComponent].num_lidar_points_in_box` |
| `detection_difficulty` | `[LiDARBoxComponent].difficulty_level.detection` (entero 1 o 2) |
| `weather` | `[StatsComponent].weather` |
| `time_of_day` | `[StatsComponent].time_of_day` |
| `sensor_version` | — (columna inventada para el ejercicio) |

**Diferencias deliberadas respecto del original:** el `weather` real de Waymo v2 solo toma los
valores `Sunny` y `Rain`; aquí se agregó `fog` y variantes en español para que la limpieza de
categorías tuviera algo que hacer. La columna `sensor_version` no existe en Waymo: está para
ilustrar una variable de varianza cero.

### Relaciones que el dataset respeta (no son defectos)

- A mayor distancia del objeto, menos puntos láser: aproximadamente
  `num_lidar_points ~ Poisson(900 / (1 + d/12)²)` con `d = √(x² + y²)`.
- Las detecciones con pocos puntos tienden a clasificarse como `LEVEL_2`.
- Las dimensiones son coherentes con el tipo de objeto (un peatón mide ~1,7 m).
- La señalética no se mueve.

Estas relaciones existen para que el análisis exploratorio tenga algo real que descubrir, no solo
suciedad que limpiar.

### ⚠️ Defectos inyectados a propósito

El archivo contiene 10 problemas de calidad deliberados. **No los corrijas en el CSV**: son el
objeto de estudio de la EA1. Están listados en `src/generar_dataset.py::CATALOGO_DEFECTOS` y cada
uno tiene un test en `tests/test_generar_dataset.py`.

Si necesitas la lista completa con las respuestas, está al final de
`notebooks/01_docente_solucionario.ipynb`.

---

## `waymo_real/`

Carpeta para los datos reales del Waymo Open Dataset. **Está en `.gitignore`**: la licencia de
Waymo es de uso no comercial y no permite redistribuir los datos. Cada persona debe aceptar los
términos en <https://waymo.com/open/terms/> y descargarlos por su cuenta.
