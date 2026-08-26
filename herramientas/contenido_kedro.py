"""Fuente única del notebook de Kedro y Databricks.

Genera ``notebooks/04_opcional_kedro_databricks.ipynb``.

**Cambio de enfoque respecto de la primera versión (2026-08-26):** el notebook ya no
construye un proyecto Kedro de juguete con ``%%writefile``. Ahora **usa el proyecto real
del repositorio**, ``kedro_mly1101/``, que está versionado y cubierto por
``tests/test_pipeline_kedro.py``.

El motivo es que ese proyecto no es una demostración: es la columna de ingeniería sobre la
que crecen las experiencias siguientes. Un proyecto que se crea y se borra dentro de una
celda no sirve como base para la EA2 ni para la EA3.

Alcance de Databricks: **conceptual por diseño**. Exige cuenta y clúster que no se pueden
pedir en clase. Lo que sí se muestra es el cambio exacto en ``catalog.yml``, que es el
argumento de por qué se separó el catálogo del análisis.

Regenerar tras editar:

    python herramientas/construir_notebooks.py
"""

from __future__ import annotations

from contenido_semana01 import URL_REPO, code, md

CELDAS_KEDRO: list[dict] = [
    md(
        """
# MLY1101 · Notebook opcional
## De notebooks a pipeline: Kedro, y qué cambiaría en Databricks

**Este notebook es opcional y no entra en la evaluación de la EA1.** Está para responder una
pregunta que aparece en cuanto un análisis deja de ser un ejercicio:

> El notebook funciona. ¿Por qué no basta con eso?

---

### El problema del notebook

Un notebook tiene un defecto que no se nota mientras trabajas solo: **el estado depende del
orden en que ejecutaste las celdas, no del orden en que están escritas.**

Puedes borrar una celda y su resultado sigue en memoria. Puedes ejecutar la 12 antes que la 7 y
todo funciona. Al día siguiente lo abres, ejecutas de arriba abajo y falla, y nadie sabe por qué.

En una entrega de clase eso es una molestia. En un sistema que decide algo, es un incidente.

| | Notebook | Pipeline |
|---|---|---|
| Orden de ejecución | El que hiciste tú | Declarado y resuelto automáticamente |
| Dónde están las rutas de archivo | Repartidas por el código | En un solo catálogo |
| Dónde están las decisiones | Mezcladas con el código | En un archivo de configuración |
| Reejecutar solo lo que cambió | No | Sí |
| Probar una pieza con `pytest` | Difícil | Es una función normal |
| Correr sin que nadie mire | No | Sí |

**Kedro** aplica esa estructura a un proyecto de Python. **Databricks** resuelve otro problema
distinto: qué hacer cuando los datos ya no caben en un computador.

---

### Este no es un ejemplo de juguete

El proyecto que vamos a ejecutar **está en el repositorio**, en `kedro_mly1101/`. No se crea
aquí ni se borra al cerrar el notebook: está versionado en Git y tiene 16 tests propios en
`tests/test_pipeline_kedro.py`.

Es la columna de ingeniería de la asignatura. Las pipelines de aprendizaje supervisado (EA2) y
no supervisado (EA3) se enchufarán en este mismo proyecto, sobre el mismo dataset.
"""
    ),
    md(
        """
---
## 0 · Preparación

En Colab hay que instalar Kedro, que no viene preinstalado. Tarda cerca de un minuto.
"""
    ),
    code(
        f"""
import sys
from pathlib import Path

EN_COLAB = "google.colab" in sys.modules

if EN_COLAB:
    REPO = Path("mly1101-machine-learning")
    if not REPO.exists():
        !git clone -q {URL_REPO}.git {{REPO}}
    RAIZ = REPO.resolve()
else:
    RAIZ = Path("..").resolve()

PROYECTO = RAIZ / "kedro_mly1101"

print("Colab:", EN_COLAB)
print("Raíz del repositorio:", RAIZ)
print("¿Existe el proyecto Kedro?:", PROYECTO.exists())
"""
    ),
    code(
        """
try:
    import kedro
    print("Kedro ya instalado:", kedro.__version__)
except ImportError:
    print("Instalando Kedro...")
    %pip install -q kedro kedro-datasets
    import kedro
    print("Kedro instalado:", kedro.__version__)
"""
    ),
    # ------------------------------------------------------------------
    md(
        """
---
# Bloque 1 · Anatomía del proyecto

Cuatro piezas, y cada una responde a una pregunta distinta:

| Pieza | Archivo | Pregunta que responde |
|---|---|---|
| **Catálogo** | `conf/base/catalog.yml` | ¿Dónde vive cada dato y en qué formato? |
| **Parámetros** | `conf/base/parameters.yml` | ¿Cuáles son las decisiones? |
| **Nodos** | `pipelines/*/nodes.py` | ¿Qué transformación hace cada paso? |
| **Pipeline** | `pipelines/*/pipeline.py` | ¿Qué depende de qué? |

Lo importante es lo que **no** hay: ninguna ruta de archivo dentro del código de análisis,
ninguna decisión escrita a mano en medio de una transformación, y ningún orden de ejecución.
"""
    ),
    code(
        """
# El proyecto real, tal como está en el repositorio.
for ruta in sorted(PROYECTO.rglob("*")):
    if any(p in ruta.parts for p in ("data", "__pycache__", ".ipynb_checkpoints")):
        continue
    if ruta.is_file():
        relativa = ruta.relative_to(PROYECTO)
        print(f"  kedro_mly1101/{relativa}")
"""
    ),
    # ------------------------------------------------------------------
    md(
        """
---
# Bloque 2 · El catálogo: dónde vive el dato

Esta es la pieza que más cambia la forma de trabajar. En un notebook, la ruta está escrita en
medio del análisis:

```python
df = pd.read_csv("../datos/crudos/detecciones_waymo_like.csv")   # ¿y en producción?
```

En Kedro el dato tiene un **nombre**, y el nodo solo conoce ese nombre:

```python
def diagnosticar(detecciones_crudas):    # de dónde sale, no es asunto suyo
    ...
```

Dónde está y en qué formato se declara aparte. Consecuencia práctica: cambiar de CSV a Parquet,
o de disco local a un bucket en la nube, es editar tres líneas de YAML. **Ni una línea del
análisis se toca.** Esto vuelve en el bloque 6, cuando hablemos de Databricks.
"""
    ),
    code(
        """
print((PROYECTO / "conf" / "base" / "catalog.yml").read_text(encoding="utf-8"))
"""
    ),
    md(
        """
### Las capas del catálogo

Los nombres `01_raw`, `02_intermediate`, `03_primary` son una convención de Kedro, y valen la
pena aunque no uses Kedro:

| Capa | Qué contiene | Regla |
|---|---|---|
| **01_raw** | Lo que llegó | **No se toca nunca.** Es la evidencia de origen |
| **02_intermediate** | Diagnósticos y tablas de trabajo | Se puede borrar y regenerar |
| **03_primary** | El dataset limpio, listo para modelar | Es el que consume la EA2 |

Fíjate en que `detecciones_limpias` se guarda en **Parquet y no en CSV**. No es capricho: es lo
que se midió en la Actividad 1.2. El CSV pierde el tipo de 11 de las 16 columnas, así que
guardar ahí desharía el trabajo de preprocesamiento en el mismo momento de escribirlo.
"""
    ),
    # ------------------------------------------------------------------
    md(
        """
---
# Bloque 3 · Los parámetros: dónde viven las decisiones

Esta es la parte que más cuesta ver al empezar, y la que más se agradece después.

Las decisiones de limpieza —qué variantes de escritura unificar, qué valor es un centinela, qué
regla define lo imposible— **no están en el código**. Están en `parameters.yml`.

Agregar una variante de escritura no debería exigir tocar Python, ni volver a probar nada, ni
que la persona que la agrega sepa programar.
"""
    ),
    code(
        """
print((PROYECTO / "conf" / "base" / "parameters.yml").read_text(encoding="utf-8"))
"""
    ),
    md(
        """
**Cada bloque de ese archivo es una fila de la tabla de decisiones de la EA1.** Lo que el alumno
escribe en Markdown en su informe, aquí está en un formato que además se ejecuta.

Esa es la diferencia de fondo: una tabla de decisiones en un informe se interpreta distinto cada
vez que alguien la lee. Un `parameters.yml` se ejecuta igual siempre.
"""
    ),
    # ------------------------------------------------------------------
    md(
        """
---
# Bloque 4 · Los nodos: funciones puras y nada más

Un nodo de Kedro es **una función normal de Python**. Recibe DataFrames y parámetros, devuelve
DataFrames, y no sabe de dónde vinieron ni a dónde van.

Mira lo que hacen estos nodos: **reutilizan `src/eda.py`**, exactamente las mismas funciones que
usaste en la Actividad 1.3. No hubo que reescribir el análisis para industrializarlo.

Ese es el premio de haber escrito funciones puras desde el principio: sin `print`, sin gráficos,
sin estado.
"""
    ),
    code(
        """
nodos_calidad = PROYECTO / "src" / "kedro_mly1101" / "pipelines" / "calidad" / "nodes.py"
print(nodos_calidad.read_text(encoding="utf-8"))
"""
    ),
    md(
        """
### El pipeline de limpieza

El de calidad solo diagnostica. El de preprocesamiento **aplica** las decisiones, y tiene una
regla que gobierna todo el módulo:

> **Marcar antes que eliminar.** Un valor imposible se convierte en faltante; no se borra la
> fila entera, porque el resto de esa fila sí era válido.

Lo único que se elimina son los duplicados exactos, que por definición no aportan nada.
"""
    ),
    code(
        """
import ast

nodos_prep = PROYECTO / "src" / "kedro_mly1101" / "pipelines" / "preprocesamiento" / "nodes.py"
arbol = ast.parse(nodos_prep.read_text(encoding="utf-8"))

for definicion in [n for n in arbol.body if isinstance(n, ast.FunctionDef)]:
    argumentos = ", ".join(a.arg for a in definicion.args.args)
    resumen = (ast.get_docstring(definicion) or "").split("\\n")[0]
    print(f"{definicion.name}({argumentos})")
    print(f"    → {resumen}\\n")
"""
    ),
    # ------------------------------------------------------------------
    md(
        """
---
# Bloque 5 · El grafo: declarar dependencias, no orden

Aquí está la idea central de Kedro, y conviene detenerse.

**En ningún archivo se escribe en qué orden ejecutar los nodos.** Se declara qué necesita cada
uno (`inputs`) y qué produce (`outputs`). Kedro construye el grafo de dependencias y deduce el
orden solo.

Lo que se obtiene gratis:

- Los nodos independientes **pueden correr en paralelo** (`kedro run --runner=ParallelRunner`).
- Si cambia un nodo, se puede reejecutar **solo lo que depende de él**.
- El grafo se dibuja con `kedro viz` y sirve para explicarle el proceso a quien no lee código.
- Un ciclo o una dependencia que falta es un **error al construir**, no un fallo a mitad de
  ejecución.
"""
    ),
    code(
        """
sys.path.insert(0, str(PROYECTO / "src"))

from kedro_mly1101.pipeline_registry import register_pipelines

pipelines = register_pipelines()
print("Pipelines registradas:", sorted(pipelines), "\\n")

completo = pipelines["__default__"]
for i, nodo in enumerate(completo.nodes, start=1):
    entradas = [e for e in sorted(nodo.inputs) if not e.startswith("params:")]
    parametros = [e.replace("params:", "") for e in sorted(nodo.inputs) if e.startswith("params:")]
    print(f"{i}. {nodo.name}")
    print(f"     datos     : {entradas}")
    if parametros:
        print(f"     parámetros: {parametros}")
    print(f"     produce   : {sorted(nodo.outputs)}\\n")
"""
    ),
    code(
        """
# Lo único que el pipeline espera de fuera, y lo que entrega al final.
entradas_externas = {e for e in completo.inputs() if not e.startswith("params:")}
print("Entradas externas :", sorted(entradas_externas))
print("Salidas finales   :", sorted(completo.outputs()))
print()
print("Los nombres que empiezan por '_' no están en el catálogo: son datasets de")
print("memoria, existen solo durante la ejecución y no tocan el disco.")
"""
    ),
    md(
        """
Fíjate en dos cosas del listado:

1. Los cuatro nodos de **calidad** dependen solo de `detecciones_crudas`: son independientes
   entre sí y podrían correr en paralelo.
2. Los cinco de **preprocesamiento** forman una cadena, porque cada uno consume la salida del
   anterior. Kedro lo dedujo de los nombres, no de una lista de pasos.
"""
    ),
    # ------------------------------------------------------------------
    md(
        """
---
# Bloque 6 · Ejecutar

Un comando. Sin argumentos, sin rutas, sin orden.

```bash
cd kedro_mly1101 && kedro run
```
"""
    ),
    code(
        """
import os
import subprocess

entorno = dict(
    os.environ,
    PYTHONPATH=str(PROYECTO / "src"),
    KEDRO_DISABLE_TELEMETRY="1",
    NO_COLOR="1",   # sin esto el log llega lleno de códigos de color, ilegible aquí
    TERM="dumb",
)

resultado = subprocess.run(
    [sys.executable, "-m", "kedro", "run"],
    cwd=PROYECTO,
    env=entorno,
    capture_output=True,
    text=True,
)

salida = resultado.stdout + resultado.stderr
ejecutados = salida.count("Completed node")

print("Código de salida :", resultado.returncode)
print(f"Nodos ejecutados : {ejecutados} de {len(completo.nodes)}")
print("Pipeline completo:", "Pipeline execution completed" in salida)

assert resultado.returncode == 0, f"el pipeline falló:\\n{salida[-2000:]}"
assert ejecutados == len(completo.nodes), "no se ejecutaron todos los nodos"
print(f"\\n✅ Los {len(completo.nodes)} nodos corrieron: del CSV crudo a las métricas del modelo.")
print("   Las salidas quedaron en kedro_mly1101/data/")
"""
    ),
    code(
        """
import pandas as pd

print("Archivos producidos:\\n")
for archivo in sorted((PROYECTO / "data").rglob("*")):
    if archivo.is_file():
        print(f"  {str(archivo.relative_to(PROYECTO)):48s} {archivo.stat().st_size/1024:8.1f} KB")
"""
    ),
    code(
        """
print("=== Qué cambió al limpiar ===")
print(pd.read_csv(PROYECTO / "data" / "03_primary" / "informe_limpieza.csv").to_string(index=False))
"""
    ),
    md(
        """
### Una cifra que parece un error y no lo es

Mira la fila **`celdas faltantes`**: después de limpiar hay **más** faltantes que antes.

No aparecieron faltantes nuevos. Los que estaban **disfrazados** —el `-1` de `num_lidar_points`,
el `"N/D"` de `timestamp_micros`, las dimensiones imposibles— pasaron a contarse como lo que
siempre fueron.

Es exactamente lo que tiene que pasar. Si esa cifra **bajara**, sería la señal de que se
eliminaron filas en vez de marcarlas. Hay un test que lo deja por escrito:
`test_resumir_limpieza_reporta_el_aumento_de_faltantes`.

> *Limpiar datos no es hacer que los problemas desaparezcan de la vista. Es hacerlos visibles.*
"""
    ),
    code(
        """
print("=== Reglas de dominio violadas ===")
print(pd.read_csv(PROYECTO / "data" / "02_intermediate" / "valores_imposibles.csv").to_string(index=False))

print("\\n=== Desbalance de clases (tras unificar variantes) ===")
print(pd.read_csv(PROYECTO / "data" / "02_intermediate" / "desbalance_clases.csv", index_col=0))
"""
    ),
    code(
        """
print("=== % de velocidad faltante, por momento del día y dificultad ===")
print(pd.read_csv(PROYECTO / "data" / "02_intermediate" / "nulos_por_grupo.csv", index_col=0))
print()
print("El faltante NO es aleatorio: se concentra en las detecciones difíciles nocturnas.")
print("Un dropna() global dejaría al modelo aún más ciego de noche de lo que ya estaba,")
print("y como el conjunto de prueba tiene el mismo sesgo, la métrica no lo mostraría.")
"""
    ),
    code(
        """
# El dataset limpio, en Parquet: conserva los tipos.
limpio = pd.read_parquet(PROYECTO / "data" / "03_primary" / "detecciones_limpias.parquet")
crudo = pd.read_csv(RAIZ / "datos" / "crudos" / "detecciones_waymo_like.csv")

print(f"Crudo  : {len(crudo):,} filas × {crudo.shape[1]} columnas")
print(f"Limpio : {len(limpio):,} filas × {limpio.shape[1]} columnas")
print()
print("Este Parquet es lo que consumirá la EA2 para entrenar el primer modelo.")
limpio.head(3)
"""
    ),
    md(
        """
### Lo que acaba de pasar

El mismo análisis de la Actividad 1.3, pero ahora:

- **Se ejecuta con un comando**, sin intervención humana. Se puede programar cada noche.
- **Las rutas están en un solo archivo**, y las decisiones en otro.
- **Cada nodo es una función normal**, así que `pytest` la prueba sin levantar nada. Son los 16
  tests de `tests/test_pipeline_kedro.py`.
- **El orden lo deduce Kedro** del grafo.
- **La limpieza quedó escrita como código**, no como una lista de decisiones que alguien tendrá
  que volver a implementar.

> **`kedro viz`** dibuja el grafo en el navegador: `pip install kedro-viz` y `kedro viz run`
> dentro de la carpeta del proyecto. Es la mejor forma de explicarle un pipeline a alguien que
> no lee código.

> **Cuándo *no* usar Kedro:** para una exploración de media hora. Todo este andamiaje se paga
> cuando el proceso se va a repetir, cuando lo va a mantener alguien más, o cuando tiene que
> correr sin que nadie mire. Para responder una pregunta suelta, el notebook es la herramienta
> correcta.
"""
    ),
    # ------------------------------------------------------------------
    md(
        """
---
# Bloque 7 · Databricks: el otro problema

Kedro resuelve la **estructura**. Databricks resuelve la **escala**: qué hacer cuando los datos
ya no caben en la memoria de un computador.

Nuestro dataset son 40.680 filas y unos 20 MB en RAM. La flota real de Waymo genera del orden de
**200.000 detecciones por segmento**, y hay 798 segmentos solo en el conjunto de entrenamiento:
unos 160 millones de filas. Eso ya no lo abre pandas en un portátil.

### Qué es cada cosa

| Concepto | Qué es |
|---|---|
| **Apache Spark** | Motor que reparte el cómputo entre muchas máquinas, con DataFrames distribuidos |
| **Databricks** | Plataforma comercial con Spark gestionado, notebooks colaborativos y almacenamiento |
| **Delta Lake** | Formato sobre Parquet que añade transacciones, versionado e histórico |
| **Unity Catalog** | Catálogo con permisos y trazabilidad de origen a nivel de organización |

### El mismo análisis, en los dos mundos

| Operación | pandas (lo que hiciste) | PySpark (Databricks) |
|---|---|---|
| Leer | `pd.read_csv(ruta)` | `spark.read.csv(ruta, header=True)` |
| Ver las primeras filas | `df.head()` | `df.show(5)` |
| Filtrar | `df[df.speed_mps > 10]` | `df.filter(df.speed_mps > 10)` |
| Columna nueva | `df["kmh"] = df.speed_mps * 3.6` | `df.withColumn("kmh", df.speed_mps * 3.6)` |
| Agrupar | `df.groupby("object_type").size()` | `df.groupBy("object_type").count()` |
| Contar nulos | `df.isna().sum()` | `df.select([count(when(col(c).isNull(), c)) for c in df.columns])` |
| Guardar | `df.to_parquet(ruta)` | `df.write.format("delta").save(ruta)` |

**La sintaxis se parece. La semántica no.**
"""
    ),
    md(
        """
### La diferencia que de verdad importa: evaluación perezosa

En pandas, cada línea se ejecuta cuando la escribes. En Spark, **no pasa nada** hasta que pides
un resultado concreto.

```python
df = spark.read.csv("s3://.../detecciones/*.csv", header=True)   # no lee nada
rapidas = df.filter(df.speed_mps > 10)                            # no filtra nada
con_kmh = rapidas.withColumn("kmh", rapidas.speed_mps * 3.6)      # no calcula nada

con_kmh.count()     # AQUÍ ocurre todo, y de una sola pasada por los datos
```

Spark acumula las tres operaciones, las optimiza en conjunto y recorre los datos **una vez**. Por
eso puede procesar terabytes: nunca carga todo en memoria.

Esto tiene una consecuencia incómoda: **el error aparece lejos de donde está la causa**. Un
nombre de columna mal escrito en la línea 2 revienta en el `count()` de la línea 20. Es la queja
número uno de quien viene de pandas.

### Cuándo hace falta

| Situación | Herramienta |
|---|---|
| Los datos caben en RAM (hasta unos pocos GB) | **pandas**. Levantar un clúster para esto es tirar plata |
| No caben en RAM, pero sí en el disco de una máquina | **Parquet leído por trozos**, o Polars, o DuckDB |
| No caben en una máquina | **Spark / Databricks** |
| Caben, pero el proceso debe repetirse y mantenerse | **Kedro** (con pandas por debajo) |
"""
    ),
    md(
        """
### Kedro y Databricks no compiten: se combinan

Aquí es donde se cobra todo lo del bloque 2. Para llevar **este mismo proyecto** a Databricks no
se toca ningún nodo: se cambia el tipo de dataset en `catalog.yml`.

```yaml
# Lo que tenemos hoy — pandas, CSV en disco local
detecciones_crudas:
  type: pandas.CSVDataset
  filepath: ../datos/crudos/detecciones_waymo_like.csv

detecciones_limpias:
  type: pandas.ParquetDataset
  filepath: data/03_primary/detecciones_limpias.parquet
```

```yaml
# Lo mismo en Databricks — Spark y Delta Lake
detecciones_crudas:
  type: spark.SparkDataset
  filepath: dbfs:/mnt/waymo/01_raw/detecciones
  file_format: delta

detecciones_limpias:
  type: spark.SparkDataset
  filepath: dbfs:/mnt/waymo/03_primary/detecciones_limpias
  file_format: delta
  save_args:
    mode: overwrite
```

**Los nodos no se enteran.** Lo que sí habría que revisar es el cuerpo de las funciones: nuestros
nodos usan la API de pandas, y sobre un DataFrame de Spark hay que usar la de PySpark. Kedro
resuelve el *dónde*; el *cómo* sigue siendo tuyo.

> Ese matiz importa y conviene no vendértelo de más: separar el catálogo **no** hace tu código
> mágicamente distribuido. Lo que hace es que la migración sea un trabajo acotado y localizado
> en los nodos, en vez de una reescritura del proyecto entero.
"""
    ),
    md(
        """
### Si quieres probarlo por tu cuenta

**Databricks Free Edition** permite crear una cuenta gratuita con un clúster pequeño en
<https://databricks.com/learn/free-edition>. Alcanza de sobra para este dataset.

1. **Workspace → Import**: sube `01_alumno_exploracion.ipynb`. Databricks lee `.ipynb`.
2. Crea un clúster (el más pequeño) y espera a que arranque, unos 5 minutos.
3. Sube el CSV en **Catalog → Add data**, o léelo desde la URL raw de GitHub.
4. El notebook corre **tal cual con pandas** en el nodo maestro: 40.000 filas no necesitan Spark.
5. Para ver la diferencia, reescribe un bloque con PySpark y compara.

> **No es parte de la evaluación** y no hace falta para el proyecto. Está aquí para que sepas que
> existe y, sobre todo, para que sepas **cuándo no lo necesitas**: que es casi siempre, en un
> proyecto de esta asignatura.
"""
    ),
    # ------------------------------------------------------------------
    md(
        """
---
# Bloque 8 · Del dato al modelo, sin salir del pipeline

El `kedro run` que acabas de lanzar **no paró en la limpieza**. Siguió con el pipeline
`supervisado`, que consume `detecciones_limpias` y entrena un clasificador.

En ningún archivo está escrito que el modelamiento va después de la limpieza. Kedro lo dedujo de
que uno consume lo que el otro produce. Eso es el proceso completo de Machine Learning declarado
como un grafo:

```
CSV crudo → calidad → preprocesamiento → detecciones_limpias.parquet → supervisado → métricas
```

### La pregunta que responde

> ¿Se puede anticipar qué detecciones van a ser **difíciles** a partir de la geometría del objeto
> y de dónde está?

Si se pudiera, el equipo de percepción sabría de antemano en qué situaciones no conviene confiar
en el sensor.

**Se descartó clasificar `object_type`**, que parecía lo natural. Sobre este dataset se resuelve
al 99,98 % con cualquier partición, porque el generador sortea las dimensiones **por tipo de
objeto** y basta el largo de la caja para acertar. Un ejercicio donde todo sale perfecto no
enseña nada sobre evaluación.
"""
    ),
    code(
        """
metricas = pd.read_csv(PROYECTO / "data" / "07_model_output" / "metricas_por_clase.csv")
print("=== Métricas por clase ===")
print(metricas.to_string(index=False))

print("\\n=== Matriz de confusión (filas = real, columnas = predicho) ===")
print(pd.read_csv(PROYECTO / "data" / "07_model_output" / "matriz_confusion.csv", index_col=0))
"""
    ),
    md(
        """
### Mira la fila que nadie mira

La exactitud ronda el **90 %**. Suena bien. Ahora mira la fila de `LEVEL_2`:

| | Precisión | Recall | F1 |
|---|---|---|---|
| `LEVEL_1` (88,9 % de los datos) | 0,93 | 0,96 | 0,94 |
| **`LEVEL_2` (11,1 %)** | **0,54** | **0,40** | **0,46** |

Un recall de 0,40 significa que **el modelo se pierde el 60 % de las detecciones difíciles**, que
son exactamente las que queríamos anticipar. El 90 % de exactitud se lo debe casi entero a acertar
la clase mayoritaria, que es la fácil.

> *Un promedio global oculta a la minoría.* Es la misma idea del bloque de sesgo de la Actividad
> 1.1, ahora con un modelo entrenado delante y una cifra que la demuestra.

Por eso el pipeline guarda `metricas_por_clase.csv` y no un único número. **Un modelo no se
reporta con una cifra.**
"""
    ),
    md(
        """
---
# Bloque 9 · Dos fugas de información, y solo una se nota

El pipeline mide las dos en vez de afirmarlas. El resultado es más interesante de lo esperado.
"""
    ),
    code(
        """
print("=== Fuga por variable derivada ===")
print(pd.read_csv(PROYECTO / "data" / "07_model_output" / "fuga_de_variable.csv").to_string(index=False))

print("\\n=== Fuga por agrupación (partición) ===")
print(pd.read_csv(PROYECTO / "data" / "07_model_output" / "comparacion_particiones.csv").to_string(index=False))
"""
    ),
    md(
        """
### La primera sí se nota: `num_lidar_points`

La etiqueta `detection_difficulty` **la asigna el sensor a partir del número de puntos láser**.
Incluir esa columna entre las variables predictoras no es informativo: es contarle al modelo la
respuesta con otras palabras.

El F1-macro sube de **0,70 a 0,75**. El modelo parece mejor y no sirve para nada, porque en el
momento en que quisieras predecir la dificultad ya tendrías la dificultad.

Es el error más común en la práctica y el más difícil de ver: **no hay ningún síntoma, solo un
resultado sospechosamente bueno**.

### La segunda no se nota, y eso también hay que saber decirlo

Partir el dataset al azar por fila deja **153 segmentos a caballo** entre entrenamiento y prueba,
contra 0 al partir por segmento. La fuga existe, es medible en segmentos compartidos… y la
métrica **no se mueve** (−0,005).

No es un fallo del ejercicio: es una propiedad de este dataset. El generador sortea cada detección
de forma independiente dentro del segmento, así que la dependencia entre filas que la fuga
explotaría **no existe aquí**. En datos reales de Waymo, donde los fotogramas consecutivos siguen
al mismo objeto, sí existe.

> La conclusión no es *"partir por grupo da igual"*. Es una más incómoda y más útil: **un riesgo
> que no se manifiesta en tus datos de prueba sigue siendo un riesgo.** La partición por grupo se
> justifica por cómo se generaron los datos, no por la diferencia que se mide hoy.

Y de paso, la lección de método: el pipeline **midió** las dos, y una salió en cero. Medir y
reportar un cero es parte del trabajo; elegir solo las mediciones que confirman lo que ibas a
decir es sesgo de confirmación, el mismo del que se habla en la Actividad 1.1.
"""
    ),
    md(
        """
---
# Bloque 10 · Aprendizaje no supervisado (EA3)

El mismo `kedro run` siguió más allá del modelo. El pipeline `no_supervisado` toma
`detecciones_limpias` y hace la pregunta contraria a la del bloque 8:

> Sin decirle a nadie qué es cada objeto, ¿aparecen **grupos naturales**? ¿Y coinciden con los
> tipos que el sensor etiquetó?

Es la contracara de la EA2. Allí había una etiqueta y se medía el acierto; aquí no hay etiqueta y
hay que **justificar** que la estructura encontrada significa algo.

**`object_type` viaja en la tabla pero no entra en el agrupamiento.** Se guarda para contrastar
después, que es distinto de usarla.

### Escalar antes de agrupar no es opcional

K-medias mide distancias euclídeas. Sin escalar, `num_lidar_points` —que llega a miles— aplasta a
`box_height` —que ronda 1,7—, y el resultado sería un agrupamiento por número de puntos disfrazado
de agrupamiento por objeto.
"""
    ),
    code(
        """
D = PROYECTO / "data" / "07_model_output"

print("=== ¿Cuántos grupos? ===")
print(pd.read_csv(D / "busqueda_de_k.csv").to_string(index=False))
"""
    ),
    md(
        """
**La inercia siempre baja al añadir grupos**, así que por sí sola no decide nada: si le hicieras
caso, terminarías con un grupo por fila. La silueta sí penaliza los grupos que se solapan, y aquí
tiene su máximo en **k = 3**.

Usamos **k = 4** de todos modos. La tabla siguiente muestra por qué esa discrepancia es
interesante en vez de un error: el cuarto grupo tiene sentido de dominio aunque la métrica
prefiera tres. **La silueta propone; el conocimiento del dominio dispone** — la misma idea que con
los atípicos en la Actividad 1.3.
"""
    ),
    code(
        """
print("=== Perfil de cada grupo (medias de las variables escaladas) ===")
print(pd.read_csv(D / "perfil_de_grupos.csv").to_string(index=False))

print("\\n=== ¿Los grupos coinciden con el tipo de objeto? (% por grupo) ===")
print(pd.read_csv(D / "grupos_vs_etiqueta.csv").to_string(index=False))
"""
    ),
    md(
        """
### Cómo se lee esa tabla

**No es una evaluación.** El algoritmo nunca vio `object_type`, así que no puede "acertar" ni
"fallar". Es una comprobación de sentido:

- Hay **dos grupos que son 100 % `vehicle`**, y conviene mirar su perfil antes de llamarlo un
  error. Uno tiene 19.927 filas y medidas normales; el otro solo 573 (1,5 %) con `box_length` casi
  **cinco desviaciones típicas** por encima de la media y `box_height` otro tanto.

  Esos son los **buses**. El agrupamiento redescubrió solo los mismos atípicos legítimos que en la
  Actividad 1.3 había que aprender a *no* eliminar. No encontró un cuarto tipo de objeto: encontró
  una subestructura real dentro de uno.
- Un grupo mezcla peatones y señalética: ambos son pequeños y están quietos, así que **por
  geometría se parecen** aunque para el negocio no tengan nada que ver.
- Y un cuarto grupo se define por tener muchos más puntos láser (`num_lidar_points` a +2,3 σ): son
  los objetos **cercanos**, de cualquier tipo. El algoritmo agrupó por distancia al sensor, que es
  una propiedad de la medición, no del objeto.

> Un agrupamiento sin interpretar no es un hallazgo. "Grupo 2" no le sirve a nadie: hay que poder
> decir *"objetos grandes y rápidos"* o *"objetos pequeños y estáticos"*, y eso sale del perfil,
> no del algoritmo.
"""
    ),
    code(
        """
print("=== Reducción de dimensionalidad (PCA) ===")
varianza = pd.read_csv(D / "varianza_pca.csv")
print(varianza.to_string(index=False))
print()
n90 = int((varianza["varianza_acumulada"] < 0.90).sum() + 1)
print(f"Con {n90} de {len(varianza)} componentes se conserva el 90 % de la varianza.")
print("Eso mide cuánta redundancia hay entre las variables: si bastan pocas, varias")
print("de las originales estaban diciendo casi lo mismo.")
"""
    ),
    md(
        """
---
# Bloque 11 · Y ahora, con datos reales

Todo lo anterior corrió sobre el dataset **sintético**. El pipeline `waymo_real` hace exactamente
lo mismo sobre datos reales del Waymo Open Dataset:

```bash
python herramientas/descargar_waymo.py --muestra 40      # ~40 MB
cd kedro_mly1101 && kedro run --pipeline waymo_real
```

**No duplica ni un solo nodo.** Reutiliza `calidad`, `preprocesamiento`, `supervisado` y
`no_supervisado` remapeando su entrada: donde leían el CSV, leen la salida de la ingesta de Waymo.
Es la demostración de por qué separar el catálogo del análisis valía la pena.

### Por qué 40 segmentos y no uno

Con un solo segmento **no se puede partir en entrenamiento y prueba sin fuga**: sus ~18.000
detecciones comparten clima, hora y ubicación. El pipeline no lo apaña cayendo a una partición al
azar —sería la mala práctica que llevamos toda la sesión evitando—: **falla, y el error dice qué
descargar.**

### Tres traducciones que no son un cambio de nombre

1. **La velocidad es un vector.** Waymo da `speed.x` y `speed.y`; la rapidez es su módulo.
   Quedarse con `speed.x` da valores plausibles y equivocados.
2. **El tipo de objeto es un entero**, no una cadena. Y existe el `0` (*unknown*).
3. **El `NaN` de la dificultad NO es un dato faltante.** Waymo solo rellena
   `difficulty_level.detection` cuando la detección es difícil; vacío significa `LEVEL_1`. Son
   **15.356 `NaN` de 18.633**: tratarlos como faltantes borraría el 82 % de los datos.

   Es el **reverso exacto** del defecto de la Actividad 1.3, donde un `-1` disfraza un faltante.
   Aquí un faltante disfraza un valor.
"""
    ),
    code(
        """
RUTA_REAL = PROYECTO / "data" / "waymo" / "07_model_output" / "metricas_por_clase.csv"

if RUTA_REAL.exists():
    print("=== EA2 sobre datos REALES de Waymo ===")
    print(pd.read_csv(RUTA_REAL).to_string(index=False))
else:
    print("No hay resultados del recorrido real en esta máquina.")
    print("Para generarlos:")
    print("   python herramientas/descargar_waymo.py --muestra 40")
    print("   cd kedro_mly1101 && kedro run --pipeline waymo_real")
"""
    ),
    md(
        """
### Lo que cambia al pasar del mock a lo real

Cifras medidas el 2026-08-26 sobre 40 segmentos (530.396 detecciones):

| | Sintético | Real |
|---|---|---|
| Filas · segmentos | 40.680 · 153 | **530.396** · 40 |
| % `cyclist` | 1,94 % | **0,45 %** |
| Mediana `speed_mps` | 5,35 | **0,01** — casi todo está detenido |
| Clima | 3 categorías sucias | **100 % `sunny`** |
| Defectos de calidad encontrados | 10 | **0** |
| EA2 · exactitud | 0,897 | 0,781 |
| EA2 · **F1 de la clase minoritaria** | **0,462** | **0,089** |
| EA3 · silueta | máximo en k = 3 | **sin codo**: sube hasta k = 8 |

**Las tres últimas filas son la clase entera.**

**La limpieza no encuentra nada.** El Waymo Open Dataset está curado: los 10 defectos son
sintéticos y se inyectaron para que hubiera algo que descubrir. Lo que aprendiste a detectar
existe en el mundo real; en *este* dataset publicado, no.

**El modelo se desploma.** De 0,46 a **0,089** de F1 en la clase minoritaria: acierta el 5,9 % de
las detecciones difíciles. El problema es mucho más duro de lo que el mock sugería.

> Es la lección más incómoda del curso y la más valiosa: **un buen resultado sobre datos de
> juguete no predice nada.** El dataset sintético sirve para aprender el método; para saber si el
> método funciona hay que salir a los datos de verdad.

**Y la silueta deja de tener máximo**, así que el criterio automático para elegir `k` falla. No
existe "el k correcto": la decisión final es de dominio, no de métrica.

Por último, ese **100 % `sunny`** no es casualidad: es el sesgo de muestreo del censo —793 de 798
segmentos soleados— visible ahora en los datos con los que se entrena el modelo. El bloque de
ética de la Actividad 1.1 deja de ser una advertencia y pasa a ser una propiedad medible del
dataset que tienes delante.
"""
    ),
    md(
        """
---
# Cierre · Hacia dónde sigue esto

El grafo que acabas de ejecutar va del CSV crudo a las métricas por clase. **Ese es el proceso
completo de Machine Learning**, declarado como dependencias y ejecutado con un comando.

| Experiencia | Pipeline | Nodos | Consume | Estado |
|---|---|---|---|---|
| **EA1** · Datos | `calidad` · `preprocesamiento` | 4 + 5 | El CSV crudo | ✅ |
| **EA2** · Supervisado | `supervisado` | 8 | `detecciones_limpias` | ✅ |
| **EA3** · No supervisado | `no_supervisado` | 7 | `detecciones_limpias` | ✅ |
| — · Datos reales | `ingesta` + `waymo_real` | 26 | Los Parquet de Waymo | ✅ |
| **EFT** | Integra las tres | — | Todo el grafo | ⏳ |

Cada experiencia **añadió nodos, no reescribió el análisis previo**. Y el recorrido sobre datos
reales no duplicó ninguno: remapeó la entrada del grafo que ya existía.

---

### Tres ideas para llevarse

1. **Un notebook que funciona no es un proceso que funciona.** El notebook depende del orden en
   que ejecutaste las celdas; un pipeline declara sus dependencias y las resuelve solo.
2. **Separar el catálogo y los parámetros del análisis** es lo que permite cambiar de origen, de
   formato o de escala sin reescribir. Y es lo que permite que una decisión de limpieza la
   cambie alguien que no programa.
3. **Escalar es la última respuesta, no la primera.** Antes de un clúster hay tipos bien
   elegidos, Parquet y lectura por trozos. Un `float32` en vez de un `float64` reduce la memoria
   a la mitad, y eso es gratis.

Y una cuarta, que es la que sostiene a las otras: nada de esto habría sido posible si
`src/eda.py` se hubiera escrito con `print` y gráficos dentro. **Las funciones puras no son una
manía de estilo: son lo que permite que el mismo código sirva en un notebook, en un test y en un
pipeline de producción.**
"""
    ),
]
