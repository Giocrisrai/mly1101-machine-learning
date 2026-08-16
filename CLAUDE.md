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

1. **Los `.ipynb` son artefactos generados. No los edites a mano.** El contenido vive en
   `herramientas/contenido_semana01.py` y `herramientas/contenido_waymo.py`; los notebooks salen
   de `python herramientas/construir_notebooks.py`. Editar el `.ipynb` funciona hasta el
   siguiente build, que lo sobrescribe.

2. **El notebook del alumno y el solucionario salen de la misma fuente.** Una celda de código
   declara su versión resuelta y, opcionalmente, su versión con `TODO`. Nunca crees dos versiones
   separadas: se desincronizan.

3. **Los 10 defectos del dataset son intencionales.** Están en
   `src/generar_dataset.py::CATALOGO_DEFECTOS` y cada uno tiene un test. No los "arregles". Si
   cambias el generador, los tests te dirán qué pauta quedó desalineada.

4. **El material está bajo CC BY-NC-SA 4.0** (ver `LICENSE`). No agregues contenido de terceros
   con licencia incompatible.

5. **Nunca agregues datos reales de Waymo al repositorio.** Su licencia es de uso no comercial y
   prohíbe la redistribución. `datos/waymo_real/` está en `.gitignore`.

6. **Las cifras de la pauta y de la rúbrica deben verificarse contra el CSV**, no citarse de
   memoria. Si cambias `--filas` o la semilla, todas las cifras del solucionario, del guion de
   clase y de la rúbrica quedan obsoletas.

## Verificación obligatoria antes de dar algo por terminado

```bash
pytest -q                                    # 63 tests (53 + 10 de Waymo, que se saltan sin datos)
python herramientas/construir_notebooks.py   # regenera los tres notebooks
cd notebooks && python -m jupyter nbconvert --to notebook --execute --stdout \
    01_docente_solucionario.ipynb > /dev/null   # el solucionario debe ejecutar completo
grep -c "Pauta docente" notebooks/01_alumno_exploracion.ipynb   # debe dar 0
```

## Convenciones

- Español en código, comentarios, docstrings y material.
- numpy/scipy antes que implementaciones manuales.
- pytest para los tests.
- Notación matemática clara en los docstrings (ver `src/eda.py::detectar_outliers_iqr`).
- Las funciones de `src/eda.py` son puras: sin `print`, sin gráficos, sin estado.

## Entorno

- pyenv, no el Python de Homebrew. Global 3.13.1.
- Los notebooks deben funcionar **también** en Google Colab: la primera celda detecta el entorno
  y clona el repositorio.

## Estado actual

| Experiencia | Estado |
|---|---|
| EA1 Semana 1 | ✅ completa y verificada |
| EA1 semanas siguientes | ⏳ pendiente (la EA1 son 20 h en total) |
| EA2 / EA3 / EFT | ⏳ pendiente |

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
