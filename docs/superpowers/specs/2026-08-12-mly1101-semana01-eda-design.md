# Diseño — MLY1101 Semana 01 / EA1: Análisis y Preprocesamiento de Datos

Fecha: 2026-08-12
Autor: Giocrisrai Godoy (docente)
Estado: aprobado por el docente el 2026-08-12

## Problema

La Semana 1 de MLY1101 inicia la EA1 (20 h). El objetivo pedagógico es que el
estudiante entienda que un proyecto de Machine Learning no empieza eligiendo un
algoritmo, sino comprendiendo el problema y los datos:

```
Problema → Datos → Exploración → Preprocesamiento → Modelamiento → Evaluación → Interpretación
```

Se necesita material listo para una sesión de taller de ~4 h, con alumnos que ya
manejan pandas, ejecutable en Google Colab (recomendación del PIA) y con
respaldo local en Jupyter.

## Decisiones tomadas

| Decisión | Elección | Motivo |
| --- | --- | --- |
| Dataset principal | Sintético con esquema Waymo v2 y defectos inyectados | Control total de lo que el alumno debe descubrir; pauta exacta; sin login ni descarga; sin problema de licencia |
| Waymo real | Notebook opcional `00_opcional_waymo_real.ipynb` | La licencia de Waymo prohíbe redistribuir los datos; cada usuario acepta los términos por su cuenta |
| Entrega | Repo git local, publicable en GitHub con badges "Abrir en Colab" | Un clic para el alumno, sin instalación |
| Versiones | Notebook de alumno (con TODO) + solucionario docente | El docente necesita pauta, no ayuda para aprender |
| Alcance | ~4 h, alumnos que ya manejan pandas | Permite llegar a outliers y decisiones de preprocesamiento |
| Alcance del repo | Un repo por asignatura, con carpetas por experiencia | La EFT exige lógica de proyecto progresiva EA1 → EA2 → EA3 |

Explícitamente **fuera de alcance** en esta iteración: rúbrica de evaluación de
la EA1, algoritmos predictivos, material de EA2/EA3.

## Arquitectura

Cuatro unidades con una responsabilidad cada una:

1. **`src/generar_dataset.py`** — genera el CSV de trabajo.
   - Entrada: `--filas`, `--semilla`, `--salida`. Salida: CSV.
   - Determinista: misma semilla ⇒ mismo archivo byte a byte.
   - Cada defecto se inyecta en una función propia y nombrada
     (`_inyectar_nulos_ocultos`, `_inyectar_duplicados`, …), de modo que el
     catálogo de defectos sea legible sin leer el detalle de la implementación.
2. **`src/eda.py`** — utilidades de diagnóstico reutilizables por los tres
   notebooks (`resumen_calidad`, `detectar_outliers_iqr`,
   `normalizar_categoria`). Sin estado, sin efectos secundarios, sin `print`.
3. **`herramientas/contenido_semana01.py` + `herramientas/construir_notebooks.py`** —
   fuente única del contenido de la clase. Cada celda de código declara su
   versión resuelta y, opcionalmente, su versión con `TODO`. El constructor
   emite los dos notebooks (`.ipynb`, nbformat 4) desde esa fuente.
   - Razón de diseño: evita que el notebook del alumno y el solucionario se
     desincronicen al corregir un ejercicio.
4. **`tests/`** — pytest sobre el generador (reproducibilidad + presencia de
   cada defecto) y sobre `src/eda.py`. Si un defecto desaparece del generador,
   la pauta deja de mentir: falla un test.

### Flujo de datos

```
generar_dataset.py --semilla 42
        ↓
datos/crudos/detecciones_waymo_like.csv   (committeado al repo)
        ↓                          ↓
notebook alumno            notebook docente        (ambos usan src/eda.py)
        ↓
mini-informe en Markdown del estudiante
```

En Colab, la primera celda detecta el entorno y hace `git clone` del repo para
disponer del CSV y de `src/`. En local, usa las rutas relativas.

## Esquema del dataset

Mismo esquema del componente `lidar_box` (+ campos de `stats`) de Waymo Open
Dataset v2, para que el notebook opcional con datos reales sea un reemplazo
directo:

| Columna | Tipo esperado | Notas |
| --- | --- | --- |
| `segment_id` | object | `seg_0147` |
| `timestamp_micros` | int64 | contaminado ⇒ se lee como object |
| `object_type` | object | VEHICLE / PEDESTRIAN / CYCLIST / SIGN |
| `box_center_x`, `box_center_y`, `box_center_z` | float64 | metros, sistema del vehículo |
| `box_length`, `box_width`, `box_height` | float64 | metros |
| `speed_mps` | float64 | nulos MNAR + outliers imposibles |
| `num_lidar_points` | int64 | `-1` como nulo oculto |
| `weather` | object | categorías inconsistentes + NaN |
| `time_of_day` | object | Day / Night / Dawn/Dusk |
| `detection_difficulty` | object | LEVEL_1 / LEVEL_2 |
| `sensor_version` | object | constante: no es feature |
| `id_interno` | object | casi único: no es feature |

## Catálogo de defectos inyectados

Cada uno tiene un test que lo verifica y una respuesta esperada en el
solucionario.

1. `timestamp_micros` contiene `"N/D"` ⇒ dtype `object`.
2. `num_lidar_points == -1` como nulo oculto (~3 %).
3. `weather` inconsistente: `sunny`, `Sunny`, `"RAIN "`, `rain`, `lluvia`, NaN (~5 %).
4. `object_type` inconsistente: `PEDESTRIAN`, `Pedestrian`, `PEATON`, `Ped`.
5. Duplicados exactos (~1,2 %) y duplicados lógicos (~0,5 %: misma llave
   `segment_id`+`timestamp_micros`+`id_interno`, valores distintos).
6. Outliers imposibles: `speed_mps` hasta ~340, `box_height == 0`,
   `box_length` negativo.
7. Outliers legítimos: buses con `box_length` de 12 a 18 m (no se eliminan).
8. Nulos MNAR en `speed_mps`, concentrados en `detection_difficulty == LEVEL_2`
   y `time_of_day == Night`.
9. Desbalance de clases: `CYCLIST` ≈ 2 %.
10. `sensor_version` constante y `id_interno` de cardinalidad casi única.

## Estructura de la sesión (4 h)

| Bloque | Min | Contenido |
| --- | --- | --- |
| 0 | 15 | El problema antes del algoritmo: ¿podemos confiar en las detecciones del sensor? |
| 1 | 45 | Carga e inspección: `shape`, `head`, `info`, `dtypes`, memoria |
| 2 | 45 | Tipos de variables: nominal / ordinal / discreta / continua; `value_counts`, `nunique` |
| 3 | 45 | Calidad: nulos (patrón, no solo conteo) y duplicados exactos vs. lógicos |
| 4 | 45 | Outliers: IQR y z-score; imposibles vs. legítimos |
| 5 | 30 | Tabla de decisiones de preprocesamiento; mención de fuga de información |
| 6 | 20 | Tratamiento responsable: peatones, geolocalización, sesgo de muestreo |
| Cierre | 15 | Mini-informe en Markdown: 5 hallazgos y 3 decisiones |

## Manejo de errores y verificación

- El notebook del alumno incluye celdas `assert` de autochequeo tras los
  ejercicios clave: el alumno sabe si acertó sin esperar al docente.
- La primera celda verifica versiones de pandas/numpy y la existencia del CSV;
  si falta, ofrece regenerarlo con `src/generar_dataset.py`.
- Los tests se ejecutan con `pytest` desde la raíz del repo.
- El notebook de Waymo real **no puede verificarse** en esta sesión (requiere
  `gcloud`, aceptación de términos y descarga de GB). Queda marcado como *no
  verificado* dentro del propio notebook.

## Riesgos asumidos

- **Dataset sintético**: se declara como tal en el README, en `datos/README.md`
  y en la primera celda del notebook. Mitigación pedagógica: el esquema es real
  y el notebook opcional permite repetir el análisis con datos reales de Waymo.
- **Licencia de Waymo**: no se redistribuye ningún dato de Waymo en el repo.
- **CSV committeado** (~4 MB): aceptable para que Colab funcione con un clic.
