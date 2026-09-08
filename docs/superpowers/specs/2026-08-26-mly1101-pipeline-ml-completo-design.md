# Especificación — El proceso completo de ML sobre datos reales de Waymo

**Fecha:** 2026-08-26
**Estado:** implementado y verificado. Actualizado 2026-09-02: el material docente de las
Act. 2.2, 2.3 y 3.1–3.3 ya existe; el grafo sintético es 30 nodos.
**Actualizado 2026-09-08:** `ingesta` pasó de 2 a **5** nodos (inventario, v2, `camera_box`,
E2E, comparar). `waymo_real` = **35** (5 + 30 remapeados). El RF sigue usando **solo**
Perception v2; camera_box y E2E se ven, no se modelan.
**Extiende:** [`2026-08-26-mly1101-actividades-11-12-design.md`](2026-08-26-mly1101-actividades-11-12-design.md)

> **Nomenclatura:** no usar EA2 = supervisado ni EA3 = no supervisado. Ambos están en el **RA2**;
> el **RA3** es optimización. Esta spec se escribió antes de esa corrección.

---

## 1. Problema

El pipeline llegaba hasta el modelo supervisado, y **solo sobre el dataset sintético**. Faltaban
dos cosas para que la asignatura pudiera recorrer el proceso completo:

1. El tramo **no supervisado** (RA2 · Act. 2.3).
2. Poder aplicar todo eso a los **datos reales** del Waymo Open Dataset, que es lo que el docente
   pidió explícitamente: *"no solo esto mock sino con los datos reales para que los chicos sepan
   cómo usarlos"*.

---

## 2. Decisiones y sus alternativas descartadas

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| **`waymo_real` reutiliza el grafo remapeando su entrada** (`pipeline(..., inputs={...}, namespace="real")`) | Duplicar los nodos con nombres `_real` | Dos copias se desincronizan. Y es la demostración práctica de por qué se separó el catálogo del análisis |
| La traducción del esquema vive en `src/waymo.py`, pura y testeada | Dejarla como texto de celda del notebook 00, donde estaba | La usan por igual el notebook, el pipeline y los tests. Es la regla del repositorio |
| La ingesta lee **varios segmentos** con `PartitionedDataset` | Un solo segmento, como hacía el notebook 00 | Con un segmento no se puede partir sin fuga. Ver §4 |
| Con menos de dos grupos, `particionar` **falla con un mensaje que dice qué descargar** | Caer a una partición al azar | Sería exactamente la mala práctica que el material enseña a evitar |
| `object_type` viaja en la matriz de agrupamiento pero **no se usa para agrupar** | Excluirla del todo | Permite contrastar después si la estructura descubierta tiene lectura de dominio |
| Escalar con `StandardScaler` antes de K-medias | Agrupar sobre las variables crudas | K-medias mide distancias euclídeas: sin escalar, `num_lidar_points` (miles) aplasta a `box_height` (~1,7) |
| La varianza explicada de la proyección va como **columna**, no en `df.attrs` | `df.attrs["varianza_explicada"] = array` | **Defecto encontrado al ejecutar:** Parquet no serializa un ndarray en `attrs` y la escritura reventaba |

---

## 3. Arquitectura

```
                    ┌── detecciones_crudas (CSV sintético) ──┐
                    │                                         │
waymo_muestra ──► ingesta ──► detecciones_reales ─────────────┤
(40 segmentos                                                 │
 particionados)                                               ▼
                                          calidad · preprocesamiento
                                                      │
                                          detecciones_limpias.parquet
                                                   ┌──┴──┐
                                          supervisado   no_supervisado
                                          (RA2 · 2.2)     (RA2 · 2.3)
```

| Pipeline | Nodos | Qué hace |
|---|---|---|
| `calidad` | 4 | Diagnóstico de calidad |
| `preprocesamiento` | 5 | Limpieza según la tabla de decisiones |
| `supervisado` (RA2 · Act. 2.2) | 8 | Partición sin fuga, entrenamiento, evaluación por clase, dos mediciones de fuga |
| `no_supervisado` (RA2 · Act. 2.3) | 7 | Escalado, búsqueda de *k*, K-medias, perfilado, contraste con la etiqueta, PCA |
| `optimizacion` (RA3) | 6 | Ajuste, ensamble y selección sustentada |
| `ingesta` | 5 | Inventario de fuentes, traduce v2, ensambla `camera_box`, lee JSON E2E, compara con el sintético |
| `waymo_real` | 35 | `ingesta` + los 30 anteriores **remapeados**, sin duplicar nodos. El modelo solo ve v2 |

---

## 4. Las tres traducciones del esquema real

Verificadas contra Parquet reales el 2026-08-26. Están en `src/waymo.py::traducir_esquema`.

1. **La velocidad es un vector.** `speed.x` y `speed.y` por separado; la rapidez es su módulo.
   Quedarse con `speed.x` da valores plausibles y equivocados.
2. **El tipo de objeto es un entero** (0–4), y existe el `0` (*unknown*) que el sintético no tiene.
3. **El `NaN` de la dificultad no es un dato faltante.** Waymo solo rellena
   `difficulty_level.detection` cuando la detección es difícil: **15.356 `NaN` de 18.633** en el
   segmento verificado. Tratarlos como faltantes borraría el 82 % de los datos y dejaría una sola
   clase.

   Es el **reverso** del defecto nº 2 del dataset sintético, donde un `-1` disfraza un faltante.
   Aquí un faltante disfraza un valor.

### Por qué varios segmentos

Con uno solo, `GroupShuffleSplit` no puede partir: hay un único grupo. Y no es un tecnicismo — es
la definición del problema: las ~18.000 detecciones de un segmento comparten clima, hora y
ubicación, así que cualquier corte contamina las dos mitades. El pipeline falla con un mensaje
que dice `descargar_waymo.py --muestra 40`, en vez de apañarlo con una partición al azar.

---

## 5. Resultados medidos (2026-09-08, leídos de `data/waymo/` y del parquet)

Sobre 40 segmentos reales, 530.396 detecciones. El CSV de la pauta (40.680) va en la columna
de al lado **solo para contrastar**; no es Waymo.

| | CSV del repo | Waymo v2 |
|---|---|---|
| Filas · segmentos | 40.680 · 153 | 530.396 · 40 |
| vehicle / sign / peatón / ciclista (%) | 61,73 / 8,12 / 26,22 / 1,94 | 48,43 / 26,46 / 24,67 / 0,45 |
| LEVEL_2 | 11,1 % | 12,33 % (65.394) |
| Mediana `speed_mps` | 5,35 | 0,0133 |
| Clima | 3 categorías sucias | 530.396 `sunny` |
| Valores imposibles | 10 defectos inyectados | 0 |
| Act. 2.2 · exactitud | 0,897 | 0,7805 |
| Act. 2.2 · F1 LEVEL_2 | 0,462 | **0,0893** (recall 0,0588 · TP 1.572 / 26.713) |
| Act. 2.3 · silueta | máximo k=3 (0,473) | 0,5228 (k=2) … 0,6103 (k=8), sin codo |
| Act. 2.3 · ¿los grupos recuperan el tipo? | Parcialmente | No (tres ~100 % vehicle; uno 47,24/50,59 peatón/sign) |
| Act. 3.1 · ganancia | ~0 (pauta) | +0,0789 |

### Tres conclusiones que el material declara en vez de esconder

1. **La limpieza no encuentra nada en los datos reales.** El Waymo Open Dataset está curado; los
   10 defectos son sintéticos y existen para que haya algo que descubrir. Lo que se aprende a
   detectar existe en el mundo real, pero no en *este* dataset publicado.
2. **El modelo cae de 0,462 a 0,0893 de F1 en LEVEL_2.** Recall 0,0588 (1.572 de 26.713 en prueba).
   El CSV de pauta no predice el Open Dataset.
3. **El agrupamiento no descubre los tipos de objeto.** Descubre estructura de tamaño y densidad
   de puntos: tres grupos de `vehicle` y uno que mezcla peatones con señalética al 47/51. Y la
   silueta no tiene máximo, así que el criterio automático para elegir `k` falla.

---

## 6. Verificación

```bash
uv sync --extra kedro
uv run pytest        # 247 tests; los de datos reales se saltan si no están descargados
uv run ruff check .

cd kedro_mly1101
uv run kedro run                          # 30/30 nodos, dataset sintético
uv run kedro run --pipeline ingesta       # 5/5, fuentes locales
uv run kedro run --pipeline waymo_real    # 35 nodos, 530.396 detecciones v2
```

### Resultado (leído de disco el 2026-09-08)

| Qué | Evidencia | Estado |
|---|---|---|
| Tests | `uv run pytest` | 247 recolectados |
| `kedro run` CSV pauta | 30 nodos | 30/30 |
| `kedro run --pipeline ingesta` | `inventario_fuentes_waymo.csv` 10:55 | 5/5 · v2 40×34,943 MB · camera_box 40×7,181 MB · E2E 479×0,035 MB · v1/Motion/JPEG = 0 |
| Clasificador v2 | `metricas_por_clase.csv` 09:09 | F1 LEVEL_2 = **0,0893** · exactitud 0,7805 · 530.396 filas |
| k-medias / PCA | `busqueda_de_k.csv` / `varianza_pca.csv` 10:55 | silueta 0,5228→0,6103 · PCA 0,7447 |
| RA3 v2 | `ganancia_del_ajuste.csv` 09:09 | +0,0789 (0,5104→0,5893) |
| Corrida `waymo_real` 35 nodos | log 10:55–11:21 | **35/35** en 1568,4 s |
| `pytest` sin Waymo | skips | los tests que piden muestra se saltan |

---

## 7. Limitaciones que persisten

| Limitación | Estado |
|---|---|
| **Ninguna** | El material docente de 2.1–2.4 y del RA3 ya existe |
| El recorrido real tarda ~15 min | 530.396 filas con RandomForest y K-medias. Aceptable fuera de clase, no para ejecutar en vivo |
| Los 40 segmentos son todos `sunny` | No es un defecto del código: es el sesgo del propio dataset (793 de 798 soleados) |
| v1 / Motion / JPEG no tienen EDA ni ML en el curso | Deliberado: tfrecord de GB. El notebook 14 los lista |

---

## 8. Trabajo futuro

| Cuándo | Qué |
|---|---|
| Hecho | Act. 2.1 (CRISP-DM) y Act. 2.4 (interpretación) |
| Evaluaciones | Formativas, parciales y EFT sobre los casos oficiales (Telco, House Prices, Spotify) |
