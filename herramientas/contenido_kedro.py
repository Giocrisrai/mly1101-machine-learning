"""Fuente única del notebook opcional de Kedro y Databricks.

Genera ``notebooks/04_opcional_kedro_databricks.ipynb``.

Alcance decidido con el docente: **Kedro se ejecuta de verdad** (se construye un proyecto
mínimo con ``%%writefile`` y se corre con ``kedro run``), y **Databricks queda conceptual**,
porque exige una cuenta y un clúster que no se pueden pedir en clase. Lo que sí se muestra de
Databricks es el código equivalente lado a lado, que es lo que permite decidir cuándo hace falta.

El pipeline que se construye no es un ejemplo de juguete: sus nodos **reutilizan las funciones
de ``src/eda.py``** que los alumnos ya usaron en la Actividad 1.3. La idea que debe quedar es que
industrializar no significa reescribir el análisis, sino declarar sus dependencias.

Regenerar tras editar:

    python herramientas/construir_notebooks.py
"""

from __future__ import annotations

from contenido_semana01 import URL_REPO, code, md

CELDAS_KEDRO: list[dict] = [
    md(
        """
# MLY1101 · Notebook opcional
## De un notebook a un pipeline: Kedro, y qué cambiaría en Databricks

**Este notebook es opcional.** No entra en la evaluación de la EA1. Está para responder una
pregunta que aparece en cuanto un análisis deja de ser un ejercicio:

> El notebook funciona. ¿Por qué no basta con eso?

---

### El problema del notebook

Un notebook tiene un defecto que no se ve mientras trabajas solo: **el estado depende del orden
en que ejecutaste las celdas, no del orden en que están escritas.**

Puedes borrar una celda y el resultado sigue en memoria. Puedes ejecutar la 12 antes que la 7 y
todo funciona. Al día siguiente lo abres de nuevo, ejecutas de arriba abajo y falla, y nadie
sabe por qué.

En una entrega de clase eso es una molestia. En un sistema que decide algo, es un incidente.

| | Notebook | Pipeline |
|---|---|---|
| Orden de ejecución | El que hayas hecho tú | Declarado y resuelto automáticamente |
| Dónde están las rutas de archivo | Repartidas por el código | En un solo catálogo |
| Reejecutar solo lo que cambió | No | Sí |
| Probar una pieza con `pytest` | Difícil | Es una función normal |
| Correr sin que nadie mire | No | Sí |

**Kedro** aplica esa estructura a un proyecto de Python. **Databricks** resuelve otro problema
distinto: qué hacer cuando los datos ya no caben en un computador.
"""
    ),
    md(
        """
---
## 0 · Preparación

En Colab hay que instalar Kedro (no viene preinstalado). Tarda cerca de un minuto.
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
    RAIZ = REPO
else:
    RAIZ = Path("..").resolve()

sys.path.insert(0, str(RAIZ / "src"))
RUTA_DATOS = (RAIZ / "datos" / "crudos" / "detecciones_waymo_like.csv").resolve()

print("Colab:", EN_COLAB)
print("Dataset:", RUTA_DATOS, "->", RUTA_DATOS.exists())
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
# Bloque 1 · Anatomía de un proyecto Kedro

Un proyecto de Kedro son cuatro piezas, y cada una responde a una pregunta:

| Pieza | Archivo | Pregunta que responde |
|---|---|---|
| **Catálogo** | `conf/base/catalog.yml` | ¿Dónde vive cada dato y en qué formato? |
| **Nodos** | `nodes.py` | ¿Qué transformación hace cada paso? |
| **Pipeline** | `pipeline.py` | ¿Qué depende de qué? |
| **Registro** | `pipeline_registry.py` | ¿Qué pipelines existen en este proyecto? |

Lo importante es lo que **no** hay: ninguna ruta de archivo dentro del código de análisis, y
ningún orden de ejecución escrito a mano. Kedro deduce el orden de las dependencias entre nodos.

Vamos a construirlo entero, archivo por archivo.
"""
    ),
    code(
        """
PROYECTO = Path("kedro_mly1101").resolve()

for carpeta in [
    PROYECTO / "conf" / "base",
    PROYECTO / "conf" / "local",
    PROYECTO / "data",
    PROYECTO / "src" / "kedro_mly1101" / "pipelines" / "calidad",
]:
    carpeta.mkdir(parents=True, exist_ok=True)

for paquete in [
    PROYECTO / "src" / "kedro_mly1101",
    PROYECTO / "src" / "kedro_mly1101" / "pipelines",
    PROYECTO / "src" / "kedro_mly1101" / "pipelines" / "calidad",
]:
    (paquete / "__init__.py").touch()

# Desactivar la telemetría de Kedro: si no, pide consentimiento y en un notebook
# eso interrumpe la ejecución.
(PROYECTO / ".telemetry").write_text("consent: false\\n", encoding="utf-8")

print("Estructura creada en:", PROYECTO)
"""
    ),
    code(
        """
# Kedro reconoce una carpeta como proyecto por este bloque del pyproject.toml.
(PROYECTO / "pyproject.toml").write_text(
    f'''[tool.kedro]
package_name = "kedro_mly1101"
project_name = "MLY1101 - diagnostico de calidad"
kedro_init_version = "{kedro.__version__}"
''',
    encoding="utf-8",
)
print((PROYECTO / "pyproject.toml").read_text())
"""
    ),
    # ------------------------------------------------------------------
    md(
        """
---
# Bloque 2 · El catálogo: dónde vive el dato

Esta es la pieza que más cambia la forma de trabajar. En un notebook, la ruta del archivo está
escrita en medio del código:

```python
df = pd.read_csv("../datos/crudos/detecciones_waymo_like.csv")   # ¿y en producción?
```

En Kedro, el dato tiene un **nombre** y el código solo usa ese nombre. Dónde está y en qué
formato se declara aparte:

```python
def diagnosticar(detecciones_crudas):   # el nodo solo conoce el nombre
    ...
```

Consecuencia práctica: cambiar de CSV a Parquet, o de disco local a un bucket en la nube, es
editar tres líneas de YAML. **Ni una línea del análisis se toca.**
"""
    ),
    code(
        """
catalogo = f'''
# Catálogo de datos: el único lugar del proyecto donde hay rutas.

detecciones_crudas:
  type: pandas.CSVDataset
  filepath: {RUTA_DATOS}

# --- Salidas del pipeline ---------------------------------------------------

resumen_calidad:
  type: pandas.CSVDataset
  filepath: data/01_resumen_calidad.csv
  save_args:
    index: true

desbalance_clases:
  type: pandas.CSVDataset
  filepath: data/02_desbalance_clases.csv
  save_args:
    index: true

valores_imposibles:
  type: pandas.CSVDataset
  filepath: data/03_valores_imposibles.csv
  save_args:
    index: false

# Parquet para el dataset limpio: conserva los tipos, como se midió en la Act. 1.2.
detecciones_limpias:
  type: pandas.ParquetDataset
  filepath: data/04_detecciones_limpias.parquet
'''
(PROYECTO / "conf" / "base" / "catalog.yml").write_text(catalogo, encoding="utf-8")
print(catalogo)
"""
    ),
    # ------------------------------------------------------------------
    md(
        """
---
# Bloque 3 · Los nodos: funciones puras y nada más

Un nodo de Kedro es **una función normal de Python**. Recibe DataFrames, devuelve DataFrames, y
no sabe de dónde vinieron ni a dónde van.

Fíjate en lo que hacen estos nodos: **reutilizan `src/eda.py`**, exactamente las mismas funciones
que usaste en la Actividad 1.3. No hay que reescribir el análisis para industrializarlo. Ese es
el premio de haber escrito funciones puras desde el principio, sin `print` y sin gráficos dentro.
"""
    ),
    code(
        r'''
nodos = f"""
\"\"\"Nodos del pipeline de calidad de datos.

Cada funcion es pura: recibe y devuelve DataFrames, sin imprimir ni graficar.
Reutilizan src/eda.py del repositorio, las mismas funciones de la Actividad 1.3.
\"\"\"

import sys

sys.path.insert(0, r"{RAIZ.resolve() / 'src'}")

import pandas as pd

import eda


def diagnosticar(detecciones: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Radiografia por columna: tipos, cardinalidad, nulos declarados y ocultos.\"\"\"
    return eda.resumen_calidad(detecciones)


def medir_desbalance(detecciones: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Frecuencia de cada tipo de objeto, tras unificar variantes de escritura.\"\"\"
    tipos = eda.normalizar_categoria(
        detecciones["object_type"], mapa={{"peaton": "pedestrian", "ped": "pedestrian"}}
    )
    return eda.resumen_desbalance(tipos)


def auditar_dominio(detecciones: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Filas que violan reglas fisicas del dominio: son imposibles, no atipicas.\"\"\"
    reglas = {{
        "altura nula o negativa": "box_height <= 0",
        "largo negativo": "box_length < 0",
        "velocidad superior a 100 m/s": "speed_mps > 100",
        "puntos laser negativos (centinela -1)": "num_lidar_points < 0",
    }}
    return eda.valores_imposibles(detecciones, reglas)


def limpiar(detecciones: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Aplica las decisiones de preprocesamiento acordadas en la Actividad 1.3.

    Cada paso corresponde a una fila de la tabla de decisiones. Los valores
    imposibles se convierten en faltantes en vez de eliminarse: quitar la fila
    entera descarta tambien las columnas que si eran validas.
    \"\"\"
    limpio = detecciones.copy()

    # 1. El centinela -1 de num_lidar_points es un faltante disfrazado.
    limpio["num_lidar_points"] = limpio["num_lidar_points"].replace(-1, pd.NA)

    # 2. "N/D" impide que la marca de tiempo sea numerica.
    limpio["timestamp_micros"] = eda.a_numerico(limpio["timestamp_micros"])

    # 3. Unificar las variantes de escritura de las categoricas.
    limpio["object_type"] = eda.normalizar_categoria(
        limpio["object_type"], mapa={{"peaton": "pedestrian", "ped": "pedestrian"}}
    )
    limpio["weather"] = eda.normalizar_categoria(
        limpio["weather"], mapa={{"lluvia": "rain", "soleado": "sunny", "niebla": "fog"}}
    )

    # 4. Las dimensiones imposibles pasan a faltantes, no se borran las filas.
    for columna in ["box_height", "box_length", "box_width"]:
        limpio.loc[limpio[columna] <= 0, columna] = pd.NA

    # 5. Los duplicados exactos si se eliminan: una fila identica no aporta nada.
    limpio = limpio.drop_duplicates().reset_index(drop=True)

    # 6. sensor_version es constante: varianza cero, informacion cero.
    limpio = limpio.drop(columns=["sensor_version"])

    return limpio
"""
(PROYECTO / "src" / "kedro_mly1101" / "pipelines" / "calidad" / "nodes.py").write_text(
    nodos, encoding="utf-8"
)
print(nodos[:900], "...")
'''
    ),
    # ------------------------------------------------------------------
    md(
        """
---
# Bloque 4 · El pipeline: declarar dependencias, no orden

Aquí está la idea central de Kedro y conviene detenerse en ella.

**No se escribe en qué orden ejecutar los nodos.** Se declara qué necesita cada uno (`inputs`) y
qué produce (`outputs`). Kedro construye el grafo de dependencias y deduce el orden solo.

Consecuencias que se obtienen gratis:

- Los nodos independientes **pueden correr en paralelo** (`kedro run --runner=ParallelRunner`).
- Si cambia un nodo, se puede reejecutar **solo lo que depende de él**.
- El grafo se puede dibujar (`kedro viz`) y sirve para explicarle el proceso a alguien más.
- Un ciclo o una dependencia que falta es un **error al construir**, no un fallo a mitad de
  ejecución.
"""
    ),
    code(
        '''
pipeline_py = """
\\"\\"\\"Pipeline de calidad de datos.

No se declara el ORDEN de ejecucion, solo las dependencias: que necesita cada
nodo y que produce. Kedro deduce el orden a partir del grafo.
\\"\\"\\"

from kedro.pipeline import Pipeline, node

from .nodes import auditar_dominio, diagnosticar, limpiar, medir_desbalance


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=diagnosticar,
                inputs="detecciones_crudas",
                outputs="resumen_calidad",
                name="diagnosticar_calidad",
            ),
            node(
                func=medir_desbalance,
                inputs="detecciones_crudas",
                outputs="desbalance_clases",
                name="medir_desbalance",
            ),
            node(
                func=auditar_dominio,
                inputs="detecciones_crudas",
                outputs="valores_imposibles",
                name="auditar_reglas_de_dominio",
            ),
            node(
                func=limpiar,
                inputs="detecciones_crudas",
                outputs="detecciones_limpias",
                name="aplicar_decisiones_de_limpieza",
            ),
        ]
    )
"""
(PROYECTO / "src" / "kedro_mly1101" / "pipelines" / "calidad" / "pipeline.py").write_text(
    pipeline_py, encoding="utf-8"
)

registro = """
from kedro.pipeline import Pipeline

from kedro_mly1101.pipelines.calidad.pipeline import create_pipeline as calidad


def register_pipelines() -> dict[str, Pipeline]:
    pipeline_calidad = calidad()
    return {"calidad": pipeline_calidad, "__default__": pipeline_calidad}
"""
(PROYECTO / "src" / "kedro_mly1101" / "pipeline_registry.py").write_text(
    registro, encoding="utf-8"
)

print("pipeline.py y pipeline_registry.py escritos")
'''
    ),
    # ------------------------------------------------------------------
    md(
        """
### Antes de ejecutar: el orden que Kedro dedujo

En `pipeline.py` los nodos están escritos en un orden cualquiera y **nunca dijimos cuál va
primero**. Kedro construyó el grafo a partir de los `inputs` y `outputs` de cada uno. Míralo:
"""
    ),
    code(
        """
sys.path.insert(0, str(PROYECTO / "src"))

from kedro_mly1101.pipeline_registry import register_pipelines

pipeline = register_pipelines()["calidad"]

print("Orden de ejecución deducido por Kedro:\\n")
for i, nodo in enumerate(pipeline.nodes, start=1):
    print(f"  {i}. {nodo.name}")
    print(f"     recibe : {sorted(nodo.inputs)}")
    print(f"     produce: {sorted(nodo.outputs)}\\n")

print("Datos que el pipeline espera encontrar :", sorted(pipeline.inputs()))
print("Datos que el pipeline produce          :", sorted(pipeline.outputs()))
"""
    ),
    md(
        """
Los cuatro nodos dependen solo de `detecciones_crudas`, así que **son independientes entre sí** y
podrían ejecutarse en paralelo (`kedro run --runner=ParallelRunner`). Si un nodo consumiera la
salida de otro, Kedro lo colocaría después sin que nadie se lo dijera.

---
# Bloque 5 · Ejecutar

Un comando. Sin argumentos, sin rutas, sin orden.
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
print(f"Nodos ejecutados : {ejecutados} de {len(pipeline.nodes)}")
print("Pipeline completo:", "Pipeline execution completed" in salida)

assert resultado.returncode == 0, f"el pipeline falló:\\n{salida[-2000:]}"
assert ejecutados == len(pipeline.nodes), "no se ejecutaron todos los nodos"
print("\\n✅ Los cuatro nodos corrieron y dejaron sus salidas en disco.")
"""
    ),
    code(
        """
import pandas as pd

print("Archivos producidos por el pipeline:\\n")
for archivo in sorted((PROYECTO / "data").glob("*")):
    print(f"  {archivo.name:38s} {archivo.stat().st_size/1024:8.1f} KB")

print("\\n--- Reglas de dominio violadas ---")
print(pd.read_csv(PROYECTO / "data" / "03_valores_imposibles.csv").to_string(index=False))

print("\\n--- Desbalance de clases ---")
print(pd.read_csv(PROYECTO / "data" / "02_desbalance_clases.csv", index_col=0))
"""
    ),
    code(
        """
# El dataset limpio, guardado en Parquet: conserva los tipos.
limpio = pd.read_parquet(PROYECTO / "data" / "04_detecciones_limpias.parquet")
crudo = pd.read_csv(RUTA_DATOS)

print(f"Crudo  : {len(crudo):,} filas × {crudo.shape[1]} columnas")
print(f"Limpio : {len(limpio):,} filas × {limpio.shape[1]} columnas")
print(f"\\nFilas eliminadas   : {len(crudo) - len(limpio):,} (duplicados exactos)")
print(f"Columnas eliminadas: {crudo.shape[1] - limpio.shape[1]} (sensor_version, constante)")
print(f"\\nCategorías de object_type: {crudo['object_type'].nunique()} -> {limpio['object_type'].nunique()}")
print(f"Categorías de weather    : {crudo['weather'].nunique()} -> {limpio['weather'].nunique()}")
"""
    ),
    md(
        """
### Lo que acaba de pasar

El mismo análisis de la Actividad 1.3, pero ahora:

- **Se ejecuta con un comando** y sin intervención humana. Se puede programar cada noche.
- **Las rutas están en un solo archivo.** Cambiar el origen no toca el análisis.
- **Cada nodo es una función normal**, así que `pytest` puede probarla sin levantar nada.
- **El orden lo deduce Kedro** del grafo de dependencias.
- **La limpieza quedó escrita como código**, no como una lista de decisiones en un informe que
  alguien tendrá que volver a implementar.

Ese último punto es el que más se subestima. Una tabla de decisiones en Markdown se interpreta
distinto cada vez que alguien la lee. La función `limpiar()` se ejecuta igual siempre.

> **`kedro viz`** dibuja el grafo en el navegador. En local: `pip install kedro-viz` y
> `kedro viz run` dentro de la carpeta del proyecto. Es la mejor forma de explicarle un pipeline
> a alguien que no lee código.

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
# Bloque 6 · Databricks: el otro problema

Kedro resuelve la **estructura**. Databricks resuelve la **escala**: qué hacer cuando los datos
ya no caben en la memoria de un computador.

Nuestro dataset son 40.680 filas y 20 MB en RAM. La flota real de Waymo genera del orden de
**200.000 detecciones por segmento**, y hay 798 segmentos solo en el conjunto de entrenamiento:
unos 160 millones de filas. Eso ya no lo abre pandas en un portátil.

### Qué es cada cosa

| Concepto | Qué es |
|---|---|
| **Apache Spark** | Motor que reparte el cómputo entre muchas máquinas. Trabaja con DataFrames distribuidos |
| **Databricks** | Plataforma comercial que ofrece Spark gestionado, notebooks colaborativos y almacenamiento |
| **Delta Lake** | Formato de almacenamiento sobre Parquet que añade transacciones, versionado e histórico |
| **Unity Catalog** | Catálogo de datos con permisos y trazabilidad de origen a nivel de organización |

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

Esto tiene una consecuencia práctica incómoda: **el error aparece lejos de donde está la causa**.
Un nombre de columna mal escrito en la línea 2 revienta en el `count()` de la línea 20. Es la
queja número uno de quien viene de pandas.

### Cuándo hace falta

| Situación | Herramienta |
|---|---|
| Los datos caben en RAM (hasta unos pocos GB) | **pandas**. Levantar un clúster para esto es tirar plata |
| No caben, pero caben en el disco de una máquina | **Parquet leído por trozos**, o Polars, o DuckDB |
| No caben en una máquina | **Spark / Databricks** |
| Caben, pero el proceso debe repetirse y mantenerse | **Kedro** (con pandas por debajo) |

**Kedro y Databricks no compiten.** Kedro corre sobre Databricks: se cambia el tipo de dataset en
`catalog.yml` de `pandas.CSVDataset` a `spark.SparkDataset` y **los nodos siguen siendo los
mismos**. Esa es exactamente la ventaja de haber separado el catálogo del análisis.

```yaml
# El mismo catálogo, apuntando a Spark y Delta Lake en vez de a un CSV local.
detecciones_crudas:
  type: spark.SparkDataset
  filepath: dbfs:/mnt/waymo/detecciones
  file_format: delta
```
"""
    ),
    md(
        """
### Si quieres probarlo por tu cuenta

**Databricks Free Edition** (antes Community Edition) permite crear una cuenta gratuita con un
clúster pequeño en <https://databricks.com/learn/free-edition>. Alcanza de sobra para este
dataset.

Pasos, una vez dentro:

1. **Workspace → Import**: sube `01_alumno_exploracion.ipynb`. Databricks lee `.ipynb`.
2. Crea un clúster (el más pequeño) y espera a que arranque, unos 5 minutos.
3. Sube el CSV en **Catalog → Add data**, o léelo directo desde la URL raw de GitHub.
4. El notebook corre **tal cual con pandas** en el nodo maestro: 40.000 filas no necesitan Spark.
5. Para ver la diferencia, reescribe un bloque con PySpark y compara.

> **No es parte de la evaluación** y no hace falta para el proyecto. Está aquí para que sepan que
> existe y, sobre todo, para que sepan **cuándo no lo necesitan**: que es casi siempre, en un
> proyecto de esta asignatura.
"""
    ),
    md(
        """
---
## Cierre

Tres ideas para llevarse:

1. **Un notebook que funciona no es un proceso que funciona.** El notebook depende del orden en
   que ejecutaste las celdas; un pipeline declara sus dependencias y las resuelve solo.
2. **Separar el catálogo del análisis es lo que permite cambiar de escala sin reescribir.** El
   mismo nodo lee de un CSV local o de un Delta Lake en la nube: cambia el YAML, no el código.
3. **Escalar es la última respuesta, no la primera.** Antes de un clúster hay tipos bien
   elegidos, Parquet y lectura por trozos. Un `float32` en vez de un `float64` reduce la memoria
   a la mitad, y eso es gratis.

Lo que hace que este notebook funcione, por cierto, no es Kedro: es que las funciones de
`src/eda.py` se escribieron puras desde el principio —sin `print`, sin gráficos, sin estado—.
Por eso pudieron convertirse en nodos sin tocar una línea.
"""
    ),
]
