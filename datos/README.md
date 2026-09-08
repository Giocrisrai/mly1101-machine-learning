# Datos

La única tabla de las actividades 1.1–3.3 es **Perception v2 real**. No hay CSV de pauta.

## `waymo_real/` (gitignored)

La licencia de Waymo es de uso no comercial y **prohíbe redistribuir** los datos. Esta carpeta
no viaja con el clone: hay que aceptarla en [waymo.com/open/terms](https://waymo.com/open/terms/)
y bajar el lote:

```bash
uv run python herramientas/descargar_waymo.py --muestra 40
```

| Archivo | Qué es |
|---|---|
| `detecciones_reales.parquet` | **530.396** filas, **40** `segment_id`. Ensamblado desde `muestra/` (`lidar_box` + `stats`). **Esta es la tabla de trabajo.** |
| `muestra/` | 40 `lidar_box` + 40 `stats` (entran al modelo) y 40 `camera_box` (407.267 filas, **no** entran). |
| `val_sequence_name_to_scenario_cluster.json` | 479 secuencias E2E. Se inventarian; no se modelan. |
| `lidar_box.parquet` + `stats.parquet` | Un segmento suelto (notebook 00). Un solo segmento **no** alcanza para train/test. |

### Diccionario (tabla de trabajo)

| Columna | Tipo | Unidad | Descripción |
|---|---|---|---|
| `segment_id` | nominal | — | Segmento de conducción (~20 s) |
| `timestamp_micros` | temporal | µs | Instante de la detección (`int64` en el parquet) |
| `id_interno` | identificador | — | `laser_object_id` |
| `object_type` | nominal | — | vehicle, pedestrian, sign, cyclist |
| `box_center_x/y/z` | continua | m | Centro de la caja respecto del vehículo |
| `box_length/width/height` | continua | m | Dimensiones |
| `speed_mps` | continua | m/s | Módulo de `speed.x`, `speed.y` |
| `num_lidar_points` | discreta | conteo | Puntos láser sobre el objeto |
| `weather` | nominal | — | En este lote: solo `sunny` |
| `time_of_day` | nominal | — | Day / Night / Dawn/Dusk |
| `detection_difficulty` | ordinal | — | `LEVEL_1` < `LEVEL_2` |
| `location` | nominal | — | `location_sf` / `location_phx` |

### Conteos medidos 2026-09-08

| object_type | n | detection_difficulty | n |
|---|---|---|---|
| vehicle | 256.855 | LEVEL_1 | 465.002 |
| sign | 140.319 | LEVEL_2 | 65.394 |
| pedestrian | 130.836 | | |
| cyclist | 2.386 | | |

`weather`: 530.396 `sunny` (100 %). Nulos: 0 %. Valores imposibles: 0. Mediana `speed_mps`: 0,0133.
`location`: SF 398.065 · PHX 132.331. `time_of_day`: Day 461.090 · Night 51.867 · Dawn/Dusk 17.439.

| Fuente local | En disco | Modelo |
|---|---|---|
| Perception v2 | 40 segmentos, 530.396 filas | **sí** |
| `camera_box` | 407.267 filas | no |
| JSON E2E | 479 filas | no |
| v1 / Motion / JPEG | 0 archivos | no |

Pipelines: `cd kedro_mly1101 && uv run kedro run` (34 nodos; `__default__` = `waymo_real`).
