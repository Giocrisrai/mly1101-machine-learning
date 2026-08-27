# CLAUDE.md — MLY1101 Machine Learning (Duoc UC)

Contexto del repositorio para asistentes de código. Complementa, no reemplaza, al `CLAUDE.md`
global del usuario.

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
   | `contenido_ea2.py` | `05_alumno_supervisado` + `05_docente_supervisado` | EA2 · RA2 |
   | `contenido_ea3.py` | `06_alumno_no_supervisado` + `06_docente_no_supervisado` | EA3 · RA3 |
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
   principio de `docs/rubrica_ea1.md`.

## Verificación obligatoria antes de dar algo por terminado

```bash
uv sync                                        # entorno reproducible (pyproject.toml + uv.lock)
uv run pytest                                  # 173 tests; algunos se saltan sin datos/extras
cd kedro_mly1101 && uv run kedro run && cd ..  # sintético: 24/24 nodos
# Con datos reales descargados:  uv run kedro run --pipeline waymo_real   # 26/26 nodos
uv run python herramientas/construir_notebooks.py   # regenera los nueve notebooks

# Los cinco notebooks con código resuelto deben ejecutar completos:
for nb in 02_docente_fuentes 03_docente_estructuras 01_docente_solucionario \
          05_docente_supervisado 06_docente_no_supervisado \
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
| EA1 Semana 1 · Act. 1.1 Fuentes y colaboración | ✅ completa y verificada |
| EA1 Semana 1 · Act. 1.2 Estructuras y almacenamiento | ✅ completa y verificada |
| EA1 Semana 1 · Act. 1.3 EDA | ✅ completa y verificada |
| Plantilla de proyecto de equipo | ✅ ejecuta de extremo a extremo |
| Pipeline Kedro (`kedro_mly1101/`) | ✅ 24 nodos en 4 pipelines, versionado, 60 tests |
| EA2 · notebooks, rúbrica y pipeline | ✅ completo y verificado |
| EA3 · notebooks, rúbrica y pipeline | ✅ completo y verificado |
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
