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
