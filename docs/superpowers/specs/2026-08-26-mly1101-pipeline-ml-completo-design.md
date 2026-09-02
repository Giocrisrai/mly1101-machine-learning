# Especificación — El proceso completo de ML sobre datos reales de Waymo

**Fecha:** 2026-08-26
**Estado:** implementado y verificado. Actualizado 2026-09-02: el material docente de las
Act. 2.2, 2.3 y 3.1–3.3 ya existe; el grafo sintético es 30 nodos y `waymo_real` 32.
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
| `ingesta` | 2 | Traduce Waymo real y lo compara con el sintético |
| `waymo_real` | 32 | `ingesta` + los 30 anteriores **remapeados**, sin duplicar nodos |

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

## 5. Resultados medidos (2026-08-26)

Sobre 40 segmentos reales, 530.396 detecciones, frente al CSV sintético de 40.680.

| | Sintético | Real |
|---|---|---|
| Filas · segmentos | 40.680 · 153 | 530.396 · 40 |
| % `cyclist` | 1,94 % | **0,45 %** |
| % `LEVEL_2` | 11,1 % | 12,3 % |
| Mediana `speed_mps` | 5,35 | **0,01** |
| Clima | 3 categorías sucias | **100 % `sunny`** |
| Defectos de calidad encontrados | 10 | **0** |
| Act. 2.2 · exactitud | 0,897 | 0,781 |
| Act. 2.2 · **F1 de la clase minoritaria** | **0,462** | **0,089** |
| Act. 2.3 · silueta | máximo en k = 3 (0,473) | **sin codo**: sube a 0,610 en k = 8 |
| Act. 2.3 · ¿los grupos recuperan el tipo? | Parcialmente | **No** |

### Tres conclusiones que el material declara en vez de esconder

1. **La limpieza no encuentra nada en los datos reales.** El Waymo Open Dataset está curado; los
   10 defectos son sintéticos y existen para que haya algo que descubrir. Lo que se aprende a
   detectar existe en el mundo real, pero no en *este* dataset publicado.
2. **El modelo cae de 0,46 a 0,089 de F1 en la clase minoritaria.** Acierta el 5,9 % de las
   detecciones difíciles. *Un buen resultado sobre datos de juguete no predice nada.*
3. **El agrupamiento no descubre los tipos de objeto.** Descubre estructura de tamaño y densidad
   de puntos: tres grupos de `vehicle` y uno que mezcla peatones con señalética al 47/51. Y la
   silueta no tiene máximo, así que el criterio automático para elegir `k` falla.

---

## 6. Verificación

```bash
uv sync --extra kedro
uv run pytest        # 187 tests; los de datos reales se saltan si no están descargados
uv run ruff check .

cd kedro_mly1101
uv run kedro run                          # 30/30 nodos, dataset sintético
uv run kedro run --pipeline waymo_real    # 32/32 nodos, 530.396 detecciones reales
```

### Resultado (cifras actuales a 2026-09-02)

| Qué | Estado |
|---|---|
| Tests | ✅ 187 recolectados |
| `ruff check` | ✅ limpio |
| `kedro run` sobre el sintético | ✅ 30/30 |
| `kedro run --pipeline waymo_real` | ✅ 32/32 sobre datos reales |
| `pytest` sin datos de Waymo | ✅ los tests que los necesitan se saltan |

---

## 7. Limitaciones que persisten

| Limitación | Estado |
|---|---|
| **Act. 2.1 y 2.4 aún no tienen notebook** | 2.2, 2.3 y todo el RA3 ya tienen notebooks, solucionario y rúbrica |
| El recorrido real tarda varios minutos | 530.396 filas con RandomForest y K-medias. Aceptable fuera de clase, no para ejecutar en vivo |
| Los 40 segmentos son todos `sunny` | No es un defecto del código: es el sesgo del propio dataset (793 de 798 soleados) |
| Ningún notebook nuevo se ha ejecutado en Colab | El runtime no arranca por automatización; verificado con `nbconvert` en local |

---

## 8. Trabajo futuro

| Cuándo | Qué |
|---|---|
| Act. 2.1 | Gestión de proyectos con CRISP-DM (6 h) |
| Act. 2.4 | Interpretación y métricas de desempeño (5 h) |
| Evaluaciones | Formativas, parciales y EFT sobre los casos oficiales (Telco, House Prices, Spotify) |
