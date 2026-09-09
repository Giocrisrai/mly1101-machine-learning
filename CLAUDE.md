# CLAUDE.md — MLY1101 Machine Learning (Duoc UC)

Contexto del repositorio para asistentes de código. Complementa, no reemplaza, al `CLAUDE.md`
global del usuario.

## ⚠️ La estructura oficial, y por qué importa

El **Programa de Asignatura** (`docs/programa_oficial.md`) define 108 h, 3 RA con **cuatro
indicadores cada uno**, y no coincide con la intuición:

| RA | Qué es realmente | Actividades (horas) |
|---|---|---|
| **RA1** | Recopila datos de calidad | 1.1 Fuentes (6) · 1.2 Estructuras (6) · 1.3 EDA (6) · 1.4 Ética (5) |
| **RA2** | Aplica modelos estadísticos — **supervisado Y no supervisado** | 2.1 CRISP-DM (6) · 2.2 Supervisado (6) · 2.3 No supervisado (12) · 2.4 Interpretación (5) |
| **RA3** | **Hiperparámetros, ensamble y validación cruzada** | 3.1 Ajuste (6) · 3.2 Ensamble (6) · 3.3 Robustez (11) |

**El error que ya se cometió una vez:** asumir que "RA2 = supervisado" y "RA3 = no supervisado".
Los dos están en el **RA2**; el RA3 es optimización. Antes de citar un RA o un IL, mirar el
programa.

**Los datasets también se dividen:** las actividades usan el hilo de Waymo; las **evaluaciones
parciales y el EFT** se rinden sobre los casos oficiales (*Telco Churn*, *House Prices*,
*Spotify Tracks*). No mezclar. Dónde corre cada nube: `docs/integraciones.md`.

**El instrumento sumativo no es esta rúbrica.** Las pautas de `docs/` son formativas. Las
evaluaciones calificadas son las tres Parciales (30/40/30 → 60 % final) y el EFT (40 %), con la
rúbrica institucional de indicadores ponderados por % de logro (100/80/60/30/0).

## Qué es esto

Material docente de la asignatura **Machine Learning (MLY1101)**, Duoc UC, semestre 2026-2. El
usuario es el **docente**: cuando pide resolver ejercicios, necesita **pautas y solucionarios
para sus alumnos**, no explicaciones para aprender.

La especificación completa está en
`docs/superpowers/specs/2026-08-12-mly1101-semana01-eda-design.md`. Léela antes de cambiar algo
estructural.

## Reglas del repositorio

1. **Los `.ipynb` son artefactos generados. No los edites a mano.** El contenido vive en los
   módulos `herramientas/contenido_*.py`; los notebooks salen de
   `python herramientas/construir_notebooks.py`. Editar el `.ipynb` funciona hasta el siguiente
   build, que lo sobrescribe.

   | Fuente | Genera | Actividad |
   |---|---|---|
   | `contenido_actividad11.py` | `02_alumno_fuentes` + `02_docente_fuentes` | 1.1 · IL1.1 |
   | `contenido_actividad12.py` | `03_alumno_estructuras` + `03_docente_estructuras` | 1.2 · IL1.2 |
   | `contenido_semana01.py` | `01_alumno_exploracion` + `01_docente_solucionario` | 1.3 · IL1.3 |
   | `contenido_actividad14.py` | `07_alumno_etica` + `07_docente_etica` | 1.4 · IL1.4 |
   | `contenido_actividad21.py` | `12_alumno_crispdm` + `12_docente_crispdm` | 2.1 · IL2.1 |
   | `contenido_actividad22.py` | `05_alumno_supervisado` + `05_docente_supervisado` | 2.2 · IL2.2 |
   | `contenido_actividad23.py` | `06_alumno_no_supervisado` + `06_docente_no_supervisado` | 2.3 · IL2.3 |
   | `contenido_actividad24.py` | `13_alumno_interpretacion` + `13_docente_interpretacion` | 2.4 · IL2.4 |
   | `contenido_actividad31.py` | `08_alumno_hiperparametros` + `08_docente_*` | 3.1 · IL3.1 |
   | `contenido_actividad32.py` | `09_alumno_ensamble` + `09_docente_ensamble` | 3.2 · IL3.2 |
   | `contenido_actividad33.py` | `11_alumno_seleccion` + `11_docente_seleccion` | 3.3 · IL3.3/3.4 |
   | `contenido_proyecto.py` | `10_proyecto_equipo_plantilla` | transversal |
   | `contenido_kedro.py` | `04_opcional_kedro_databricks` | opcional |
   | `contenido_waymo.py` | `00_opcional_waymo_real` | opcional |
   | `contenido_waymo_buckets.py` | `14_opcional_waymo_buckets` | opcional |
   | `contenido_evaluacion.py` | `15_alumno_evaluacion` + `15_docente_evaluacion` | EP/EFT (casos oficiales) |

   El número del archivo **no** coincide con el de la actividad: el notebook de EDA se publicó
   primero como `01` y sus enlaces de Colab ya circulan.

2. **El notebook del alumno y el solucionario salen de la misma fuente.** Una celda de código
   declara su versión resuelta y, opcionalmente, su versión con `TODO`. Nunca crees dos versiones
   separadas: se desincronizan.

3. **No hay dataset sintético.** La única tabla de las Act. 1.1–3.3 es Perception v2
   (`datos/waymo_real/detecciones_reales.parquet`, gitignored). Si faltan los datos, el código
   **falla** y dice cómo bajarlos. No generes CSV, no inventes filas, no pongas un fallback.

4. **El material está bajo CC BY-NC-SA 4.0** (ver `LICENSE`). No agregues contenido de terceros
   con licencia incompatible.

5. **Nunca agregues datos reales de Waymo al repositorio.** Su licencia es de uso no comercial y
   prohíbe la redistribución. `datos/waymo_real/` está en `.gitignore`. Sin la muestra,
   `kedro run` no arranca; los tests de Waymo se **saltan** y el resto de `pytest` sigue verde.

   Para el recorrido supervisado hacen falta **varios segmentos**
   (`descargar_waymo.py --muestra 40`): con uno solo no se puede partir en entrenamiento y prueba
   sin fuga, y el pipeline falla a propósito con un mensaje que lo explica en vez de apañarlo.

6. **Las cifras de la pauta y de la rúbrica se verifican contra el parquet real**, no de
   memoria. El lote medido el 2026-09-08: 530.396 filas, 40 segmentos.

7. **`pandas` está fijado por debajo de 3.0 a propósito.** En pandas 3 una columna de texto deja
   de tener `dtype == object` y pasa a `str`. Varias celdas del material —y sus tests— enseñan
   justamente a leer ese `object`, y Colab sigue en la serie 2.x. No subas el tope sin migrar el
   material completo.

8. **Los nodos de limpieza deben dejar las columnas numéricas como `float`, no `object`.**
   Usa `np.nan`, nunca `pd.NA`: en una columna numérica, `pd.NA` la degrada a `object` y
   scikit-learn revienta mucho más tarde con un `TypeError` sobre `NAType`. El pipeline lo
   ocultaba porque escribe a Parquet y al releer vuelve a `float`; los notebooks de las
   Act. 2.2 y 2.3 llaman los nodos **en proceso** y ahí sí falla. Hay dos tests que lo fijan.

9. **El pipeline de `kedro_mly1101/` SÍ se versiona; sus salidas (`data/`) no.**
   `__default__` y `waymo_real` son **el mismo grafo** (34 nodos): ingesta de la muestra +
   calidad + preprocesamiento + supervisado + no supervisado + optimización. Nunca dupliques
   nodos. Reutilizan `src/eda.py`. Las decisiones de limpieza viven en
   `conf/base/parameters.yml`, no en el código.

10. **La rúbrica usa dos numeraciones y ambas son necesarias.** Los PPT definen IL 1.1, 1.2 y 1.3
   (uno por actividad); la corrección usa cinco dimensiones D1–D5, que en el código y en
   `calcular_nota.py` se siguen llamando `IL1`…`IL5`. La tabla de correspondencia está al
   principio de `docs/rubrica_ra1.md`.

## Verificación obligatoria antes de dar algo por terminado

```bash
uv sync                                        # entorno reproducible (pyproject.toml + uv.lock)
uv run pytest                                  # suite local (extras + muestra + casos oficiales si están)
cd kedro_mly1101 && uv run kedro run && cd ..  # 34 nodos, Perception v2 real
# equivalente: uv run kedro run --pipeline waymo_real
uv run python herramientas/construir_notebooks.py   # regenera todos los notebooks

# Los notebooks con código resuelto deben ejecutar completos:
for nb in 02_docente_fuentes 03_docente_estructuras 01_docente_solucionario \
          07_docente_etica 12_docente_crispdm 05_docente_supervisado 06_docente_no_supervisado \
          13_docente_interpretacion \
          08_docente_hiperparametros 09_docente_ensamble 11_docente_seleccion \
          10_proyecto_equipo_plantilla 04_opcional_kedro_databricks \
          15_docente_evaluacion; do
  uv run python -m jupyter nbconvert --to notebook --execute --stdout \
      --output-dir=/tmp notebooks/$nb.ipynb > /dev/null && echo "$nb OK"
done

# 00_opcional_waymo_real y 14_opcional_waymo_buckets necesitan cuenta Waymo: no van en este bucle.
# 15_docente_evaluacion necesita el zip institucional en datos/evaluaciones/ (gitignored).

# Ningún notebook de alumno puede filtrar la pauta (todos deben dar 0):
grep -c "Pauta docente" notebooks/*alumno*.ipynb notebooks/10_proyecto*.ipynb

# Limpiar los artefactos que dejan los notebooks al ejecutarse:
rm -rf notebooks/kedro_mly1101 notebooks/salidas_act12 notebooks/salidas_proyecto
```

El notebook 04 necesita el extra de Kedro: `uv sync --extra kedro`. Ojo con un efecto lateral
que ya mordió una vez: **Kedro arrastra `google-api-core`**, así que un `importorskip` sobre
`google.api_core` deja de saltarse aunque el extra `waymo` no esté instalado. El guard correcto
es sobre `google.cloud.storage`.

## Convenciones

- Español en código, comentarios, docstrings y material.
- numpy/scipy antes que implementaciones manuales.
- pytest para los tests.
- Notación matemática clara en los docstrings (ver `src/eda.py::detectar_outliers_iqr`).
- Las funciones de `src/eda.py` son puras: sin `print`, sin gráficos, sin estado.

## Entorno

- **`uv` es el gestor del entorno** (`pyproject.toml` + `uv.lock` + `.python-version`). Usa
  `uv run <comando>` en vez de activar el entorno a mano. `requirements.txt` se mantiene
  alineado para quien prefiera pip.
- pyenv por debajo, no el Python de Homebrew. Global 3.13.1.
- Los notebooks deben funcionar **también** en Google Colab: la primera celda detecta el entorno
  y clona el repositorio.
- Extras opcionales: `uv sync --extra kedro` (notebook 04) y `uv sync --extra waymo`
  (notebook 00).

## Estado actual

| Experiencia | Estado |
|---|---|
| RA1 · Act. 1.1 Fuentes y colaboración | ✅ completa y verificada |
| RA1 · Act. 1.2 Estructuras y almacenamiento | ✅ completa y verificada |
| RA1 · Act. 1.3 EDA | ✅ completa y verificada |
| RA1 · Act. 1.4 Ética, sesgos y privacidad | ✅ completa y verificada |
| RA2 · Act. 2.1 CRISP-DM | ✅ completa (notebooks, pauta, `src/crispdm.py`) |
| RA2 · Act. 2.4 Interpretación y métricas | ✅ completa (notebooks, pauta, `src/interpretacion.py`) |
| RA3 · Act. 3.1, 3.2 y 3.3 | ✅ completas y verificadas |
| Evaluaciones formativas, parciales y EFT | ✅ formativas 1–3 + plantilla `15_*_evaluacion` + calculadora IE + CSV de ejemplo ep1–ep3/eft. Spotify recorta por álbum. Hub: `docs/integraciones.md`. CSV oficiales **no** en git |
| Plantilla de proyecto de equipo | ✅ ejecuta de extremo a extremo |
| Pipeline Kedro (`kedro_mly1101/`) | ✅ `__default__` = `waymo_real` = 34 nodos (ingesta 4 + análisis 30) |
| RA2 · Act. 2.2 y 2.3 (notebooks, pautas, pipeline) | ✅ completas y verificadas |
| Datos reales de Waymo (`waymo_real`) | medido 2026-09-08: 530.396 detecciones v2 · 407.267 `camera_box` · 479 secuencias E2E. El RF **solo** ve v2. `kedro run --pipeline waymo_real` **34/34**. F1 `LEVEL_2` = 0,0893 (mismos números que 09:09) |
| Notebook opcional de Kedro y Databricks | ✅ `04_opcional_kedro_databricks` ejecutó local (nbconvert, 191,9 s). Free Edition en vivo 2026-09-08: Git Folder del repo público (rama `main`, sin PAT) + Volume managed `workspace.default.mly1101` (`/Volumes/workspace/default/mly1101`, **LIST 0 filas** en 5,8 s; parquet no va en git). SQL warehouse *Serverless Starter Warehouse*. `kedro run` **no** se muda al workspace |
| Notebook opcional de buckets Waymo (Motion / E2E / v1) | ✅ `14_opcional_waymo_buckets`: lote + tablas chicas + **fotograma sin JPEG** (`recorte_de_un_frame`); v1/Motion/video se listan; Colabs oficiales de Waymo enlazados. Guía: `docs/productos_waymo.md`. Kedro `ingesta` traduce `camera_box`; Databricks Spark = parquet v2. **Auth Colab:** Copy to Drive + Chrome/Safari/Brave; el navegador embebido del IDE suelta `MessageError` (medido 2026-09-08) |
| EFT | ✅ mismos 12 IE; plantilla `15_*_evaluacion`; CSV gitignored; no copiar el notebook Duoc de Telco |

### Cifras medidas (esta máquina, 2026-09-08) — no redondear de memoria

Fuente: `kedro_mly1101/data/02_intermediate/inventario_fuentes_waymo.csv`,
parquet de `datos/waymo_real/`, `data/07_model_output/metricas_por_clase.csv`.

| Fuente (disco) | Archivos | MB | EDA | ML (RF / k-medias / RA3) |
|---|---|---|---|---|
| Perception v2 `lidar_box`+`stats` | 40 segmentos | 34,943 | sí | **sí** (único que entra al modelo) |
| `camera_box` | 40 · **407.267** filas × 11 cols | 7,181 | tabla 2D | no |
| JSON E2E | 1 · **479** secuencias | 0,035 | clusters | no (no es video) |
| `camera_image` | 0 | 0 | no se baja | no |
| Perception v1 | 0 | 0 | no se baja | no |
| Motion | 0 | 0 | no se baja | no |

**Perception v2** (530.396 filas, 40 segmentos, 0 % nulos):

| | n | % |
|---|---|---|
| vehicle | 256.855 | 48,43 |
| sign | 140.319 | 26,46 |
| pedestrian | 130.836 | 24,67 |
| cyclist | 2.386 | 0,45 |
| LEVEL_1 / LEVEL_2 | 465.002 / 65.394 | 87,67 / 12,33 |
| weather | 530.396 `sunny` | 100 |
| location | SF 398.065 · PHX 132.331 | |
| time_of_day | Day 461.090 · Night 51.867 · Dawn/Dusk 17.439 | |
| mediana `speed_mps` | 0,0133 | |

**Supervisado v2** (`metricas_por_clase.csv`, soporte de prueba 146.116):

| Clase | Precisión | Recall | F1 | Soporte |
|---|---|---|---|---|
| LEVEL_1 | 0,8173 | 0,9419 | 0,8752 | 119.403 |
| LEVEL_2 | 0,1847 | 0,0588 | **0,0893** | 26.713 |
| exactitud | | | 0,7805 | |
| macro avg | 0,5010 | 0,5004 | 0,4822 | |

Matriz: TN 112.466 · FP 6.937 · FN 25.141 · TP 1.572.

**No supervisado v2** (`busqueda_de_k.csv`): silueta 0,5228 (k=2) → 0,6103 (k=8), sin codo.
PCA: 2 componentes explican **0,7447**. Grupos vs tipo: tres ~100 % `vehicle`; el otro 47,24 % peatón / 50,59 % señalética.

**RA3 v2** (`ganancia_del_ajuste.csv`, 09:09): default 0,5104 → búsqueda 0,5893, ganancia **0,0789** (supera ruido). Mejor: gradient boosting 0,594; ensamble 0,5938 (no distinguible, ruido 0,0282).

**E2E JSON** (479, grafía de Waymo): Interections 116 · Foreign Object Debris 78 · Cyclist 71 · Pedestrian 52 · Multi-Lane Maneuvers 42 · Single-Lane Maneuvers 38 · Special Vehicles 25 · Others 22 · Cut_ins 20 · Construction 15.

**`camera_box` tipos** (enteros Waymo): type 1 = 297.902 · type 2 = 107.507 · type 4 = 1.858.

Telco / House Prices / Spotify: evaluaciones, no este hilo.

`notebooks/00_opcional_waymo_real.ipynb`: EDA de un segmento, 2026-08-13,
`10023947602400723454_1120_000_1140_000` (18.633 detecciones). `test_mapeo_waymo.py` 10/10.

Para reproducirlo en otra máquina:

```bash
brew install --cask google-cloud-sdk
gcloud auth login                        # interactivo: NO lo ejecutes tú, pídeselo al usuario
python herramientas/descargar_waymo.py --lote 8   # 8 segmentos livianos → una tabla
pytest tests/test_mapeo_waymo.py -v
```

Los tests de Waymo se **saltan** si no hay datos descargados, así que `pytest` sigue pasando en
limpio sin credenciales.

Para validar los cuatro buckets GCS y el tratamiento de un fragmento:

```bash
gcloud auth login                                          # interactivo: el usuario
uv run python herramientas/validar_buckets_waymo.py        # lista + baja + abre
uv run python herramientas/validar_buckets_waymo.py --local  # solo parquet ya en disco
```
