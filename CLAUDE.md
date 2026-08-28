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
*Spotify Tracks*). No mezclar.

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
   | `contenido_actividad22.py` | `05_alumno_supervisado` + `05_docente_supervisado` | 2.2 · IL2.2 |
   | `contenido_actividad23.py` | `06_alumno_no_supervisado` + `06_docente_no_supervisado` | 2.3 · IL2.3 |
   | `contenido_actividad31.py` | `08_alumno_hiperparametros` + `08_docente_*` | 3.1 · IL3.1 |
   | `contenido_actividad32.py` | `09_alumno_ensamble` + `09_docente_ensamble` | 3.2 · IL3.2 |
   | `contenido_actividad33.py` | `11_alumno_seleccion` + `11_docente_seleccion` | 3.3 · IL3.3/3.4 |
   | `contenido_proyecto.py` | `10_proyecto_equipo_plantilla` | transversal |
   | `contenido_kedro.py` | `04_opcional_kedro_databricks` | opcional |
   | `contenido_waymo.py` | `00_opcional_waymo_real` | opcional |

   El número del archivo **no** coincide con el de la actividad: el notebook de EDA se publicó
   primero como `01` y sus enlaces de Colab ya circulan.

2. **El notebook del alumno y el solucionario salen de la misma fuente.** Una celda de código
   declara su versión resuelta y, opcionalmente, su versión con `TODO`. Nunca crees dos versiones
   separadas: se desincronizan.

3. **Los 10 defectos del dataset son intencionales.** Están en
   `src/generar_dataset.py::CATALOGO_DEFECTOS` y cada uno tiene un test. No los "arregles". Si
   cambias el generador, los tests te dirán qué pauta quedó desalineada.

4. **El material está bajo CC BY-NC-SA 4.0** (ver `LICENSE`). No agregues contenido de terceros
   con licencia incompatible.

5. **Nunca agregues datos reales de Waymo al repositorio.** Su licencia es de uso no comercial y
   prohíbe la redistribución. `datos/waymo_real/` está en `.gitignore`. El pipeline `waymo_real`
   los lee de ahí y **se salta limpio** si no están: `pytest` y `kedro run` funcionan sin ellos.

   Para el recorrido supervisado hacen falta **varios segmentos**
   (`descargar_waymo.py --muestra 40`): con uno solo no se puede partir en entrenamiento y prueba
   sin fuga, y el pipeline falla a propósito con un mensaje que lo explica en vez de apañarlo.

6. **Las cifras de la pauta y de la rúbrica deben verificarse contra el CSV**, no citarse de
   memoria. Si cambias `--filas` o la semilla, todas las cifras del solucionario, del guion de
   clase y de la rúbrica quedan obsoletas.

7. **`pandas` está fijado por debajo de 3.0 a propósito.** En pandas 3 una columna de texto deja
   de tener `dtype == object` y pasa a `str`. Varias celdas del material —y sus tests— enseñan
   justamente a leer ese `object`, y Colab sigue en la serie 2.x. No subas el tope sin migrar el
   material completo.

8. **Los nodos de limpieza deben dejar las columnas numéricas como `float`, no `object`.**
   Usa `np.nan`, nunca `pd.NA`: en una columna numérica, `pd.NA` la degrada a `object` y
   scikit-learn revienta mucho más tarde con un `TypeError` sobre `NAType`. El pipeline lo
   ocultaba porque escribe a Parquet y al releer vuelve a `float`; los notebooks de EA2 y EA3
   llaman los nodos **en proceso** y ahí sí falla. Hay dos tests que lo fijan.

9. **El pipeline de `kedro_mly1101/` SÍ se versiona; sus salidas (`data/`) no.** Es la columna
   de ingeniería del curso: `calidad`, `preprocesamiento`, `supervisado` (EA2),
   `no_supervisado` (EA3) e `ingesta`. El pipeline `waymo_real` corre **el mismo grafo sobre
   datos reales remapeando su entrada**: nunca dupliques nodos para datos reales. Sus nodos **reutilizan `src/eda.py`**, nunca reimplementan el análisis, y las decisiones
   de limpieza viven en `conf/base/parameters.yml`, no en el código.

10. **La rúbrica usa dos numeraciones y ambas son necesarias.** Los PPT definen IL 1.1, 1.2 y 1.3
   (uno por actividad); la corrección usa cinco dimensiones D1–D5, que en el código y en
   `calcular_nota.py` se siguen llamando `IL1`…`IL5`. La tabla de correspondencia está al
   principio de `docs/rubrica_ra1.md`.

## Verificación obligatoria antes de dar algo por terminado

```bash
uv sync                                        # entorno reproducible (pyproject.toml + uv.lock)
uv run pytest                                  # 187 tests; algunos se saltan sin datos/extras
cd kedro_mly1101 && uv run kedro run && cd ..  # sintético: 30/30 nodos
# Con datos reales descargados:  uv run kedro run --pipeline waymo_real   # 32/32 nodos
uv run python herramientas/construir_notebooks.py   # regenera los nueve notebooks

# Los cinco notebooks con código resuelto deben ejecutar completos:
for nb in 02_docente_fuentes 03_docente_estructuras 01_docente_solucionario \
          07_docente_etica 05_docente_supervisado 06_docente_no_supervisado \
          08_docente_hiperparametros 09_docente_ensamble 11_docente_seleccion \
          10_proyecto_equipo_plantilla 04_opcional_kedro_databricks; do
  uv run python -m jupyter nbconvert --to notebook --execute --stdout \
      --output-dir=/tmp notebooks/$nb.ipynb > /dev/null && echo "$nb OK"
done

# Ningún notebook de alumno puede filtrar la pauta (todos deben dar 0):
grep -c "Pauta docente" notebooks/0[123]_alumno*.ipynb notebooks/10_proyecto*.ipynb

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
| RA2 · Act. 2.1 CRISP-DM | ⏳ pendiente |
| RA2 · Act. 2.4 Interpretación y métricas | ⏳ pendiente |
| RA3 · Act. 3.1, 3.2 y 3.3 | ✅ completas y verificadas |
| Evaluaciones formativas, parciales y EFT | ⏳ pendientes, sobre los casos oficiales |
| Plantilla de proyecto de equipo | ✅ ejecuta de extremo a extremo |
| Pipeline Kedro (`kedro_mly1101/`) | ✅ 30 nodos en 5 pipelines, versionado, 74 tests |
| RA2 · Act. 2.2 y 2.3 (notebooks, pautas, pipeline) | ✅ completas y verificadas |
| Datos reales de Waymo (`waymo_real`) | ✅ 26/26 nodos sobre 530.396 detecciones reales |
| Notebook opcional de Kedro y Databricks | ✅ usa el pipeline real; Databricks queda conceptual |
| EA1 semanas siguientes | ⏳ pendiente (la EA1 son 20 h en total) |
| EFT | ⏳ pendiente |

Todo está verificado, incluido `notebooks/00_opcional_waymo_real.ipynb`: se ejecutó de extremo a
extremo el 2026-08-13 contra el segmento real `10023947602400723454_1120_000_1140_000`
(18.633 detecciones) y `tests/test_mapeo_waymo.py` pasó 10/10.

Para reproducirlo en otra máquina:

```bash
brew install --cask google-cloud-sdk
gcloud auth login                        # interactivo: NO lo ejecutes tú, pídeselo al usuario
python herramientas/descargar_waymo.py   # baja lidar_box (~1 MB) + stats (~23 KB)
pytest tests/test_mapeo_waymo.py -v
```

Los tests de Waymo se **saltan** si no hay datos descargados, así que `pytest` sigue pasando en
limpio sin credenciales.
