"""Fuente única del contenido de la Actividad 1.2 — Estructuras de Datos y Almacenamiento.

Indicador de logro **IL 1.2**: *utiliza estructuras de datos en Python para el almacenamiento
y manipulación eficiente de datasets.*

De este archivo salen dos notebooks:

- ``notebooks/03_alumno_estructuras.ipynb``   (versión con TODO)
- ``notebooks/03_docente_estructuras.ipynb``  (versión resuelta con pauta)

Todas las cifras de la pauta están medidas sobre ``detecciones_waymo_like.csv`` con la semilla
42. Si se cambia ``--filas`` o la semilla del generador, hay que volver a medirlas.

Regenerar los notebooks tras editar este archivo:

    python herramientas/construir_notebooks.py
"""

from __future__ import annotations

from contenido_semana01 import URL_REPO, code, md, md_docente

CELDAS_ACT12: list[dict] = [
    # ======================================================================
    # ENCUADRE
    # ======================================================================
    md(
        """
# MLY1101 · Machine Learning — Actividad 1.2
## Estructuras de Datos y Almacenamiento en Python

**Resultado de aprendizaje (RA1):** recopila, a través de un trabajo colaborativo, sets de
datos representativos y de calidad, a partir de distintas fuentes, para responder a las
necesidades del contexto de negocio, considerando aspectos éticos.

**Indicador de logro (IL 1.2):** utiliza estructuras de datos en Python para el almacenamiento
y manipulación eficiente de datasets.

---

### La idea central de hoy

Elegir una estructura de datos **no es una decisión de estilo**. Decide cuánta RAM consume el
proceso, cuánto tarda en responder y —lo más traicionero— si el resultado es correcto.

Hoy vas a comprobar cuatro cosas, midiéndolas:

1. Una lista de Python ocupa **cuatro veces** lo que el mismo dato en un arreglo de NumPy.
2. Un ciclo `for` tarda **decenas de veces** más que la misma operación vectorizada.
3. Ajustar los tipos de columna reduce la memoria del dataset **a la mitad**.
4. Confundir `.loc` con `.iloc` **no da error**: da un resultado equivocado en silencio.

El cuarto punto es el que le cuesta el fin de semana a alguien todos los años.

---

### El caso

Mismo dataset de detecciones LiDAR de las actividades 1.1 y 1.3. Ahora la pregunta no es *de
dónde salen* ni *qué tan sucios están*, sino:

> **¿Estamos manipulando estos datos de una forma que aguante cuando en vez de 40.000
> detecciones sean 40 millones?**

---

### Al final de la sesión debes entregar

- El notebook con los 15 TODO resueltos y sus autochequeos en verde.
- Una **tabla de decisiones de almacenamiento** (última celda) con el formato que elegiste para
  tu proyecto y por qué, respaldado con las cifras que mediste tú.
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Actividad 1.2
>
> **Cómo usar este documento.** Es el solucionario de `03_alumno_estructuras.ipynb`: mismo
> contenido más el código resuelto, las respuestas esperadas y los criterios de logro.
>
> **La actividad son 6 horas pedagógicas** según el programa. La distribución de abajo cubre
> el trabajo guiado; las horas restantes quedan para que apliquen lo mismo al caso oficial
> que hayan elegido (Telco, Housing o Spotify), sobre el que se rinde la Evaluación Parcial.
>
> **Distribución del bloque guiado (~2 h):**
>
> | Bloque | Min | Foco |
> |---|---|---|
> | 0 · Encuadre | 5 | Elegir estructura es una decisión de ingeniería |
> | 1 · Listas vs NumPy | 20 | Memoria y tiempo, medidos en vivo |
> | 2 · Anatomía del ndarray | 15 | `shape`, `ndim`, `dtype` y el precio de float32 |
> | 3 · Series y DataFrame | 15 | El índice explícito y la alineación automática |
> | 4 · `.loc` vs `.iloc` | 30 | El error silencioso ⭐ **el bloque imprescindible** |
> | 5 · Manipulación avanzada | 20 | `groupby`, `merge`, `pivot_table` |
> | 6 · Carga y guardado | 20 | Benchmark medido + pérdida de tipos ⭐ |
> | Cierre | 10 | Tabla de decisiones de almacenamiento |
>
> **Si el tiempo se acorta, se recorta el 5** (se puede dejar como trabajo autónomo) y se
> comprime el 2. **Nunca recortes el 4.**
>
> **Todas las cifras de esta pauta están medidas** sobre el CSV publicado (semilla 42,
> 40.680 filas). Los tiempos varían según la máquina: lo que no varía es el orden de magnitud.
"""
    ),
    md(
        """
---
## Preparación del entorno

Ejecuta esta celda primero. Funciona tanto en Google Colab como en Jupyter local.
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
    # El notebook vive en notebooks/, así que la raíz del repositorio es la carpeta superior.
    RAIZ = Path("..").resolve()

sys.path.insert(0, str(RAIZ / "src"))
RUTA_DATOS = RAIZ / "datos" / "crudos" / "detecciones_waymo_like.csv"

# Carpeta de trabajo para los archivos que vamos a exportar en el bloque 6.
SALIDAS = Path("salidas_act12")
SALIDAS.mkdir(exist_ok=True)

print("Colab:", EN_COLAB)
print("Raíz del repositorio:", RAIZ)
print("¿Existe el dataset?:", RUTA_DATOS.exists())
"""
    ),
    code(
        """
import sys
import time

import numpy as np
import pandas as pd

import formatos  # comparación de formatos de almacenamiento: src/formatos.py

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 140)

df = pd.read_csv(RUTA_DATOS)
print("pandas", pd.__version__, "| numpy", np.__version__)
print(f"Dataset: {df.shape[0]:,} filas × {df.shape[1]} columnas")
"""
    ),
    # ======================================================================
    # BLOQUE 1 — Listas vs NumPy
    # ======================================================================
    md(
        """
---
# Bloque 1 · Por qué las listas de Python no sirven para esto

Una lista de Python es un **arreglo de punteros**: guarda direcciones de memoria que apuntan a
objetos dispersos por el montón. Cada flotante de esa lista es un objeto completo de Python, con
su cabecera, su contador de referencias y su tipo.

Un `ndarray` de NumPy es un **bloque contiguo** de bytes, todos del mismo tipo. El procesador lo
recorre aprovechando la caché, y las operaciones las ejecuta código compilado en C.

| Estructura | Memoria | Velocidad | Tipos |
|---|---|---|---|
| Lista de Python | Dispersa (punteros + objetos) | Lenta (un salto por elemento) | Heterogéneos |
| Arreglo de NumPy | Bloque contiguo | Rápida (C, vectorizada) | Homogéneos y fijos |

No lo creas porque lo dice la tabla. Mídelo.
"""
    ),
    md(
        """
### ✏️ TODO 1 — Medir la memoria

Compara cuánto ocupa la columna `speed_mps` como lista de Python y como arreglo de NumPy.

*Pista: `sys.getsizeof(lista)` **solo mide el arreglo de punteros**, no los objetos apuntados. Por
eso usamos `formatos.memoria_lista_python()`, que suma ambos. Para el arreglo, `.nbytes`.*
"""
    ),
    code(
        """
velocidades_lista = df["speed_mps"].dropna().tolist()
velocidades_array = np.array(velocidades_lista, dtype="float64")

bytes_lista = formatos.memoria_lista_python(velocidades_lista)
bytes_array = velocidades_array.nbytes

print(f"Elementos: {len(velocidades_lista):,}\\n")
print(f"Lista de Python : {bytes_lista/1024**2:6.2f} MB")
print(f"Arreglo NumPy   : {bytes_array/1024**2:6.2f} MB")
print(f"Factor          : {bytes_lista/bytes_array:6.1f}×")
print()
print(f"Engaño frecuente -> sys.getsizeof(lista) dice solo "
      f"{sys.getsizeof(velocidades_lista)/1024**2:.2f} MB: "
      f"mide los punteros, no los números.")
""",
        """
# TODO 1: ¿cuánta memoria ocupa cada estructura?
velocidades_lista = df["speed_mps"].dropna().tolist()
velocidades_array = np.array(velocidades_lista, dtype="float64")

bytes_lista = formatos.memoria_lista_python(velocidades_lista)
bytes_array = velocidades_array.____          # atributo que da los bytes del bloque

print(f"Elementos: {len(velocidades_lista):,}\\n")
print(f"Lista de Python : {bytes_lista/1024**2:6.2f} MB")
print(f"Arreglo NumPy   : {bytes_array/1024**2:6.2f} MB")
print(f"Factor          : {bytes_lista/bytes_array:6.1f}×")
print()
print(f"Engaño frecuente -> sys.getsizeof(lista) dice solo "
      f"{sys.getsizeof(velocidades_lista)/1024**2:.2f} MB: "
      f"mide los punteros, no los números.")
""",
    ),
    md(
        """
### ✏️ TODO 2 — Medir el tiempo

Convierte las velocidades de m/s a km/h (multiplicar por 3,6) de las dos maneras y compara.

*Pista: la versión vectorizada no lleva ningún `for`.*
"""
    ),
    code(
        """
inicio = time.perf_counter()
kmh_loop = [v * 3.6 for v in velocidades_lista]
seg_loop = time.perf_counter() - inicio

inicio = time.perf_counter()
kmh_vectorizado = velocidades_array * 3.6
seg_vectorizado = time.perf_counter() - inicio

print(f"Ciclo de Python : {seg_loop*1000:8.2f} ms")
print(f"Vectorizado     : {seg_vectorizado*1000:8.2f} ms")
print(f"Factor          : {seg_loop/seg_vectorizado:8.0f}×")
""",
        """
# TODO 2: la misma conversión, de las dos maneras.
inicio = time.perf_counter()
kmh_loop = [____ for v in velocidades_lista]      # con ciclo
seg_loop = time.perf_counter() - inicio

inicio = time.perf_counter()
kmh_vectorizado = ____                            # sin ciclo, sobre el ndarray
seg_vectorizado = time.perf_counter() - inicio

print(f"Ciclo de Python : {seg_loop*1000:8.2f} ms")
print(f"Vectorizado     : {seg_vectorizado*1000:8.2f} ms")
print(f"Factor          : {seg_loop/seg_vectorizado:8.0f}×")
""",
    ),
    code(
        """
# Autochequeo
np.testing.assert_allclose(kmh_loop, kmh_vectorizado, rtol=1e-12,
                           err_msg="las dos versiones deben dar el mismo resultado")
assert bytes_lista > 3 * bytes_array, "revisa: la lista debería ocupar bastante más"
assert seg_vectorizado < seg_loop, "revisa: la versión vectorizada debería ser más rápida"
print("✅ Mismo resultado, mucha menos memoria y mucho menos tiempo.")
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — TODO 1 y 2
>
> **Cifras medidas** sobre las 39.893 velocidades no nulas del CSV publicado:
>
> | | Valor |
> |---|---|
> | Lista de Python | **1,22 MB** |
> | Arreglo NumPy (`float64`) | **0,30 MB** |
> | Factor de memoria | **4,0×** |
> | `sys.getsizeof(lista)` a secas | 0,30 MB ← *el engaño* |
> | Ciclo vs vectorizado | del orden de **20×** |
>
> **El detalle de `sys.getsizeof` vale la clase entera.** Da casi exactamente lo mismo que el
> ndarray, así que un alumno que lo use "comprueba" que las listas son igual de eficientes.
> Está midiendo el arreglo de punteros (8 bytes por elemento), no los objetos `float` de Python
> (24 bytes cada uno) que hay al otro lado. 8 + 24 = 32 contra 8 del `float64`: de ahí el 4×.
>
> **Sobre los tiempos:** en 40.000 elementos la diferencia son décimas de milisegundo y a alguien
> le va a parecer irrelevante. Es el momento de escalar la pregunta: *si esto fueran 40 millones
> de detecciones —un mes de flota, no 153 segmentos— el ciclo tarda medio minuto y el vectorizado
> menos de un segundo. Y eso dentro de un entrenamiento que repite la operación miles de veces.*
>
> **El factor exacto varía por máquina y por ejecución.** Si alguien obtiene 8× y otro 60×, ambos
> están bien: el `assert` solo comprueba el orden, no el valor.
>
> **Criterio de logro:** mide ambas magnitudes, obtiene el mismo resultado numérico por las dos
> vías y explica *por qué* la lista ocupa más (punteros + objetos, no un bloque contiguo).
"""
    ),
    # ======================================================================
    # BLOQUE 2 — Anatomía del ndarray
    # ======================================================================
    md(
        """
---
# Bloque 2 · Anatomía de un arreglo

Antes de entrenar cualquier modelo hay que auditar la forma de la matriz de variables. Tres
atributos bastan:

| Atributo | Qué devuelve | Para qué sirve |
|---|---|---|
| `.shape` | Tupla con el tamaño de cada dimensión | Comprobar que `X` e `y` tienen las mismas filas |
| `.ndim` | Cuántos ejes tiene | Distinguir un vector de una matriz |
| `.dtype` | El tipo exacto de los elementos | Es lo que decide cuánta RAM consume |
"""
    ),
    code(
        """
columnas_numericas = ["box_length", "box_width", "box_height", "num_lidar_points"]
X = df[columnas_numericas].to_numpy()

print("shape:", X.shape, "-> (filas, columnas)")
print("ndim :", X.ndim, "-> es una matriz 2D")
print("dtype:", X.dtype)
print(f"memoria: {X.nbytes/1024**2:.2f} MB")
"""
    ),
    md(
        """
### ✏️ TODO 3 — El precio del `dtype`

`float64` usa 8 bytes por número; `float32`, 4. La mitad de memoria. ¿Es gratis?

Convierte `X` a `float32`, mide el ahorro y **mide también el error** que se introduce.

*Pista: `.astype("float32")` y `np.abs(original - convertido).max()`.*
"""
    ),
    code(
        """
X32 = X.astype("float32")

error_max = np.abs(X - X32.astype("float64")).max()
print(f"float64: {X.nbytes/1024**2:.2f} MB")
print(f"float32: {X32.nbytes/1024**2:.2f} MB   (ahorro {100*(1 - X32.nbytes/X.nbytes):.0f} %)")
print(f"Error máximo introducido: {error_max:.2e}")
print(f"Magnitud típica del dato: {np.abs(X).mean():.2f}")
""",
        """
# TODO 3: ¿cuánto se ahorra y cuánto se paga al bajar la precisión?
X32 = X.astype("____")

error_max = np.abs(X - X32.astype("float64")).____()
print(f"float64: {X.nbytes/1024**2:.2f} MB")
print(f"float32: {X32.nbytes/1024**2:.2f} MB   (ahorro {100*(1 - X32.nbytes/X.nbytes):.0f} %)")
print(f"Error máximo introducido: {error_max:.2e}")
print(f"Magnitud típica del dato: {np.abs(X).mean():.2f}")
""",
    ),
    md(
        """
### ✏️ TODO 4 — Optimizar el DataFrame completo

Lo mismo, pero sobre todo el dataset. Dos cambios:

- las columnas de texto con **pocas categorías distintas** pasan a `category`;
- las columnas continuas pasan a `float32`.

*Pista: `.astype("category")` y `.astype("float32")`. Mide con `memory_usage(deep=True)`.*
"""
    ),
    code(
        """
CATEGORICAS = ["object_type", "weather", "time_of_day", "detection_difficulty", "sensor_version"]
CONTINUAS = ["box_center_x", "box_center_y", "box_center_z",
             "box_length", "box_width", "box_height", "speed_mps"]

df_optimizado = df.copy()
for columna in CATEGORICAS:
    df_optimizado[columna] = df_optimizado[columna].astype("category")
for columna in CONTINUAS:
    df_optimizado[columna] = df_optimizado[columna].astype("float32")

mem_antes = df.memory_usage(deep=True).sum() / 1024**2
mem_despues = df_optimizado.memory_usage(deep=True).sum() / 1024**2

print(f"Antes   : {mem_antes:6.1f} MB")
print(f"Después : {mem_despues:6.1f} MB")
print(f"Ahorro  : {100*(1 - mem_despues/mem_antes):5.0f} %")
""",
        """
# TODO 4: reduce la memoria del DataFrame sin perder ninguna fila.
CATEGORICAS = ["object_type", "weather", "time_of_day", "detection_difficulty", "sensor_version"]
CONTINUAS = ["box_center_x", "box_center_y", "box_center_z",
             "box_length", "box_width", "box_height", "speed_mps"]

df_optimizado = df.copy()
for columna in CATEGORICAS:
    df_optimizado[columna] = df_optimizado[columna].astype("____")
for columna in CONTINUAS:
    df_optimizado[columna] = df_optimizado[columna].astype("____")

mem_antes = df.memory_usage(____).sum() / 1024**2
mem_despues = df_optimizado.memory_usage(____).sum() / 1024**2

print(f"Antes   : {mem_antes:6.1f} MB")
print(f"Después : {mem_despues:6.1f} MB")
print(f"Ahorro  : {100*(1 - mem_despues/mem_antes):5.0f} %")
""",
    ),
    code(
        """
# Autochequeo
assert len(df_optimizado) == len(df), "optimizar memoria no puede perder filas"
assert mem_despues < mem_antes * 0.7, "revisa: deberías haber ahorrado más de un 30 %"
assert str(df_optimizado["object_type"].dtype) == "category", (
    "revisa: object_type debería quedar como category"
)
print(f"✅ {mem_antes:.1f} MB → {mem_despues:.1f} MB con las mismas {len(df):,} filas.")
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — TODO 3 y 4
>
> **TODO 3.** El ahorro es exactamente el 50 %. El error máximo es del orden de `1e-05` frente a
> valores cuya magnitud típica ronda las decenas: unas diez cifras significativas de margen. Para
> medidas de un sensor con precisión de centímetros, `float32` sobra.
>
> **La pregunta que hay que hacer es cuándo *no* sobra.** Respuestas: cuando se acumulan millones
> de sumas y el error se propaga; cuando se restan números muy parecidos (cancelación
> catastrófica); cuando el dato es dinero. *La precisión que necesitas la define el dominio, no
> la costumbre.*
>
> **TODO 4.** Medido sobre el CSV publicado: **20,1 MB → 8,7 MB, un 57 % de ahorro**, sin perder
> una sola fila.
>
> **De dónde sale el grueso del ahorro:** de `category`, no de `float32`. Una columna `object`
> con 40.680 cadenas de texto guarda 40.680 objetos `str` de Python; como `category` guarda un
> diccionario de 7 valores más un arreglo de enteros pequeños. Conviene mostrarlo columna a
> columna con `df.memory_usage(deep=True).sort_values()`.
>
> **Advertencia que hay que dar, porque muerde en la Actividad 1.3:** `category` es fantástico
> para memoria y **un estorbo para limpiar**. Sobre una columna `category`, un `.replace()` que
> introduzca un valor nuevo falla o se comporta raro, porque el conjunto de categorías está
> fijado. La secuencia correcta es **limpiar primero, convertir a `category` después.**
>
> **El `deep=True` no es opcional.** Sin él, pandas reporta 5,0 MB para el DataFrame original:
> cuenta los punteros, no las cadenas. Es el mismo engaño del `sys.getsizeof` del bloque 1, y
> vale la pena nombrarlo así para que quede la idea general: *casi toda herramienta que mide
> memoria te miente por defecto.*
>
> **Criterio de logro:** consigue un ahorro mayor al 30 % sin perder filas y explica que el
> grueso viene de las categóricas.
"""
    ),
    # ======================================================================
    # BLOQUE 3 — Series y DataFrame
    # ======================================================================
    md(
        """
---
# Bloque 3 · Series y DataFrame: el índice lo cambia todo

Una **Series** es un arreglo unidimensional **con índice explícito**. Esa es toda la diferencia
con un ndarray, y es una diferencia enorme: el índice es una etiqueta, y pandas la usa para
alinear operaciones automáticamente.

Un **DataFrame** es un contenedor de Series que comparten el mismo índice. De ahí que admita
columnas de tipos distintos, como una tabla de base de datos.
"""
    ),
    code(
        """
ventas = pd.Series([450, 600, 320], index=["Ene", "Feb", "Mar"])
print(ventas["Feb"], "<- acceso por etiqueta, no por posición\\n")

# El índice del DataFrame que venimos usando es el automático: 0, 1, 2, ...
print("Índice de df:", df.index[:5].tolist(), "...")
"""
    ),
    md(
        """
### ✏️ TODO 5 — La alineación automática

Suma dos Series cuyos índices **no coinciden del todo** y observa qué hace pandas.

*Pista: no da error. Mira dónde aparecen los `NaN`.*
"""
    ),
    code(
        """
a = pd.Series([10, 20, 30], index=["x", "y", "z"])
b = pd.Series([1, 2, 3], index=["y", "z", "w"])

suma = a + b
print(suma)
print()
print("NumPy, en cambio, suma por posición y no sabe de etiquetas:")
print(a.to_numpy() + b.to_numpy())
""",
        """
# TODO 5: ¿qué pasa al sumar dos Series con índices distintos?
a = pd.Series([10, 20, 30], index=["x", "y", "z"])
b = pd.Series([1, 2, 3], index=["y", "z", "w"])

suma = ____
print(suma)
print()
print("NumPy, en cambio, suma por posición y no sabe de etiquetas:")
print(a.to_numpy() + b.to_numpy())
""",
    ),
    code(
        """
# Autochequeo
assert suma.isna().sum() == 2, "revisa: ¿cuántas etiquetas aparecen en una sola de las dos Series?"
assert suma["y"] == 21, "revisa: en 'y' deberían sumarse 20 y 1"
print("✅ pandas alineó por etiqueta y puso NaN donde una de las dos no tenía valor.")
print("   NumPy habría sumado por posición y devuelto 11, 22, 33: otro resultado, sin avisar.")
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — TODO 5
>
> **Respuesta:** `w` y `x` quedan `NaN`; `y` = 21 y `z` = 32.
>
> **Ninguna de las dos herramientas se equivoca: hacen cosas distintas.** NumPy suma por
> posición y devuelve `[11, 22, 33]`. pandas suma por etiqueta y devuelve `[NaN, 21, 32, NaN]`.
> El error es del programador que no sabía cuál de las dos estaba usando.
>
> Este bloque es corto a propósito: es el **montaje** del bloque 4, donde esta misma alineación
> deja de ser una curiosidad y se convierte en un error silencioso sobre datos reales.
>
> **Criterio de logro:** predice dónde aparecen los `NaN` antes de ejecutar y explica la
> diferencia con NumPy.
"""
    ),
    # ======================================================================
    # BLOQUE 4 — .loc vs .iloc  ⭐
    # ======================================================================
    md(
        """
---
# Bloque 4 · ⭐ `.loc` contra `.iloc`

| | Selecciona por | Ejemplo |
|---|---|---|
| `.loc` | **Etiqueta** del índice y nombre de columna | `df.loc[0, "speed_mps"]` |
| `.iloc` | **Posición** entera, de 0 a n−1 | `df.iloc[0, 10]` |

En un DataFrame recién leído, la etiqueta y la posición **coinciden**: el índice es 0, 1, 2, …
Por eso `.loc` y `.iloc` parecen intercambiables y mucha gente aprende mal.

Dejan de coincidir en cuanto filtras, ordenas o eliminas filas. Y ahí empieza el problema.
"""
    ),
    code(
        """
ciclistas = df[df["object_type"] == "CYCLIST"]

print(f"Ciclistas: {len(ciclistas)}")
print("Índice del resultado:", ciclistas.index[:5].tolist(), "...")
print()
print("Fíjate: la primera fila del filtro NO tiene etiqueta 0.")
"""
    ),
    md(
        """
### ✏️ TODO 6 — Posición contra etiqueta

Obtén la **primera fila** de `ciclistas` de las dos maneras y comprueba si son la misma.

*Pista: por posición siempre es `0`. Por etiqueta hay que usar la etiqueta que realmente existe:
`ciclistas.index[0]`.*
"""
    ),
    code(
        """
por_posicion = ciclistas.iloc[0]
por_etiqueta = ciclistas.loc[ciclistas.index[0]]

print("Misma fila:", por_posicion.equals(por_etiqueta))
print()
print("Y esto, en cambio, es un error:")
try:
    ciclistas.loc[0]
except KeyError as error:
    print("  KeyError ->", error)
    print("  La etiqueta 0 no existe en este filtro: esa detección no era un ciclista.")
""",
        """
# TODO 6: la primera fila del filtro, por posición y por etiqueta.
por_posicion = ciclistas.____[0]
por_etiqueta = ciclistas.____[ciclistas.index[0]]

print("Misma fila:", por_posicion.equals(por_etiqueta))
print()
print("Y esto, en cambio, es un error:")
try:
    ciclistas.loc[0]
except KeyError as error:
    print("  KeyError ->", error)
    print("  La etiqueta 0 no existe en este filtro: esa detección no era un ciclista.")
""",
    ),
    md(
        """
### ✏️ TODO 7 — ⭐ El error que **no** avisa

El `KeyError` de arriba es un buen error: se ve, se corrige y a otra cosa.

El peligroso es este. Queremos agregar la velocidad en km/h a una tabla de ciclistas con el
índice reiniciado. El código no falla. Ejecútalo y cuenta los `NaN`.
"""
    ),
    code(
        """
kmh = ciclistas["speed_mps"] * 3.6          # conserva el índice original: 76, 194, 199, ...
tabla = ciclistas.reset_index(drop=True)    # índice nuevo: 0, 1, 2, ...

tabla["speed_kmh"] = kmh                    # pandas alinea por ETIQUETA, no por posición

print(f"Filas: {len(tabla)}")
print(f"NaN en speed_kmh: {tabla['speed_kmh'].isna().sum()}")
print(f"Filas con valor : {tabla['speed_kmh'].notna().sum()}   <- y encima están MAL")
print()
print(tabla[["object_type", "speed_mps", "speed_kmh"]].head(3))
"""
    ),
    md(
        """
Mira la última salida con calma:

- La operación **no dio error**.
- La mayoría de las filas quedó en `NaN`.
- **Unas pocas sí tienen valor, y ese valor está equivocado**: le corresponde a otra detección,
  la que originalmente llevaba esa etiqueta.

Un resultado parcialmente lleno es más peligroso que uno vacío: parece que funcionó.

Ahora arréglalo de las dos formas posibles.

*Pista: o bien despojas a la Series de su índice (`.to_numpy()` o `.values`), o bien reinicias el
índice de la Series igual que el del DataFrame.*
"""
    ),
    code(
        """
# Forma A — quitarle el índice a la Series: pandas alinea por posición.
tabla_a = ciclistas.reset_index(drop=True)
tabla_a["speed_kmh"] = kmh.to_numpy()

# Forma B — reiniciar el índice de la Series igual que el del DataFrame.
tabla_b = ciclistas.reset_index(drop=True)
tabla_b["speed_kmh"] = kmh.reset_index(drop=True)

print(f"Forma A -> NaN: {tabla_a['speed_kmh'].isna().sum()}")
print(f"Forma B -> NaN: {tabla_b['speed_kmh'].isna().sum()}")
print()
print(tabla_a[["object_type", "speed_mps", "speed_kmh"]].head(3))
""",
        """
# TODO 7: arregla la asignación de las dos formas.
# Forma A — quitarle el índice a la Series.
tabla_a = ciclistas.reset_index(drop=True)
tabla_a["speed_kmh"] = kmh.____()

# Forma B — reiniciar el índice de la Series igual que el del DataFrame.
tabla_b = ciclistas.reset_index(drop=True)
tabla_b["speed_kmh"] = kmh.____(drop=True)

print(f"Forma A -> NaN: {tabla_a['speed_kmh'].isna().sum()}")
print(f"Forma B -> NaN: {tabla_b['speed_kmh'].isna().sum()}")
print()
print(tabla_a[["object_type", "speed_mps", "speed_kmh"]].head(3))
""",
    ),
    code(
        """
# Autochequeo
assert tabla["speed_kmh"].isna().sum() > 0, (
    "revisa: la versión rota debería tener NaN; ¿reiniciaste el índice del DataFrame?"
)
assert tabla_a["speed_kmh"].isna().sum() == ciclistas["speed_mps"].isna().sum(), (
    "revisa: los únicos NaN que deberían quedar son los que ya venían en speed_mps"
)
np.testing.assert_allclose(
    tabla_a["speed_kmh"].dropna().to_numpy(),
    tabla_b["speed_kmh"].dropna().to_numpy(),
    rtol=1e-9,
    err_msg="las dos formas de arreglarlo deben dar lo mismo",
)
np.testing.assert_allclose(
    tabla_a["speed_kmh"].dropna().to_numpy(),
    (tabla_a["speed_mps"].dropna() * 3.6).to_numpy(),
    rtol=1e-9,
    err_msg="cada fila debe llevar SU propia velocidad convertida",
)
print("✅ Arreglado por las dos vías, y cada fila lleva su propio valor.")
"""
    ),
    md(
        """
### ✏️ TODO 8 — La otra cara: posiciones de columna

`.iloc` también muerde. `df.iloc[:, 10]` devuelve la columna que esté en la **posición 10**, sea
la que sea. Si alguien reordena las columnas aguas arriba, tu código sigue funcionando y
devuelve otra cosa.

Compruébalo: mira qué columna es la 10, reordena las columnas y vuelve a mirar.
"""
    ),
    code(
        """
print("Columna en posición 10 (original) :", df.columns[10])

barajado = df[sorted(df.columns)]
print("Columna en posición 10 (reordenado):", barajado.columns[10])
print()
print("¿df.iloc[:, 10] devuelve lo mismo en ambos?:",
      df.iloc[:, 10].equals(barajado.iloc[:, 10]))
print("¿df.loc[:, 'speed_mps'] devuelve lo mismo?:",
      df.loc[:, "speed_mps"].equals(barajado.loc[:, "speed_mps"]))
""",
        """
# TODO 8: ¿qué pasa con .iloc cuando cambia el orden de las columnas?
print("Columna en posición 10 (original) :", df.columns[10])

barajado = df[sorted(df.columns)]
print("Columna en posición 10 (reordenado):", barajado.columns[____])
print()
print("¿df.iloc[:, 10] devuelve lo mismo en ambos?:",
      df.iloc[:, 10].equals(barajado.____[:, 10]))
print("¿df.loc[:, 'speed_mps'] devuelve lo mismo?:",
      df.loc[:, "speed_mps"].equals(barajado.____[:, "speed_mps"]))
""",
    ),
    code(
        """
# Autochequeo
assert not df.iloc[:, 10].equals(barajado.iloc[:, 10]), (
    "revisa: al reordenar, la posición 10 debería apuntar a otra columna"
)
assert df.loc[:, "speed_mps"].equals(barajado.loc[:, "speed_mps"]), (
    "revisa: el nombre de la columna no cambia al reordenar"
)
print("✅ La posición depende del orden; la etiqueta, no.")
print("   Regla práctica: para columnas, usa SIEMPRE el nombre.")
"""
    ),
    md(
        """
**✍️ Tu respuesta al bloque 4:**

*(doble clic aquí y escribe)*

En una frase: ¿cuándo usarías `.iloc` y cuándo `.loc`? Da un ejemplo de cada uno.
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 4 (TODO 6, 7 y 8) ⭐
>
> **Este es el bloque que justifica la sesión.** Si solo alcanza para uno, es este.
>
> **Cifras medidas** sobre el CSV publicado: hay **789** detecciones `CYCLIST`, y sus primeras
> etiquetas son **76, 194, 199, 305, 312**. Tras el `reset_index`, la asignación rota deja
> **771 NaN de 789**, y **18 filas con un valor equivocado**.
>
> **De dónde salen esos 18.** Son las etiquetas originales de ciclistas que resultan ser menores
> que 789 y por tanto existen también en el índice nuevo (76, 194, 199, 305, 312, …). pandas
> encuentra la etiqueta, la considera una coincidencia legítima y copia el valor. Nadie avisa
> nada.
>
> **Cómo montarlo en clase.** Antes de ejecutar la celda del TODO 7, pregunta cuántos `NaN` van a
> salir. La sala se divide entre "ninguno" y "todos". Nadie dice 771. Ejecutar entonces tiene
> mucho más efecto que explicarlo.
>
> **La frase que debe quedar:** *un resultado parcialmente lleno es más peligroso que uno vacío,
> porque parece que funcionó.* Con 789 filas se nota. Con 4 millones y un `.head()` que sale
> perfecto porque las primeras etiquetas sí coincidieron, no se nota hasta producción.
>
> **Respuesta al TODO 8:** la posición 10 es `speed_mps`; al ordenar alfabéticamente pasa a ser
> `num_lidar_points`. `.iloc[:, 10]` cambia de columna sin decir nada; `.loc[:, "speed_mps"]` no.
>
> **Respuesta esperada del cierre:** `.iloc` cuando la posición es lo que importa de verdad —los
> primeros n registros de una serie temporal, un corte de entrenamiento y prueba por orden— y
> `.loc` en todo lo demás, especialmente para columnas, que siempre deben ir por nombre.
>
> **Error frecuente que conviene atajar:** el encadenado `df[df.x > 5]["y"] = 0`. No modifica el
> original, avisa con un `SettingWithCopyWarning` que casi nadie lee, y la forma correcta es
> `df.loc[df.x > 5, "y"] = 0`. Es exactamente el mismo malentendido de fondo.
>
> **Criterio de logro:** distingue posición de etiqueta, **explica por qué la asignación rota no
> lanza error** y arregla el caso por las dos vías.
"""
    ),
    # ======================================================================
    # BLOQUE 5 — Manipulación avanzada
    # ======================================================================
    md(
        """
---
# Bloque 5 · Manipulación avanzada

Tres operaciones cubren la mayor parte del trabajo diario:

| Operación | Qué hace | Pregunta que responde |
|---|---|---|
| `groupby` | Divide, aplica y combina | ¿Cuánto es X para cada categoría? |
| `merge` | Une dos tablas por una llave | ¿Cómo le pego el contexto a cada detección? |
| `pivot_table` | Cruza dos variables en una matriz | ¿Cómo se comporta X según A y B a la vez? |
"""
    ),
    md(
        """
### ✏️ TODO 9 — `groupby` con varias métricas

Calcula, por tipo de objeto: cantidad de detecciones, velocidad promedio, largo mediano y puntos
LiDAR medianos. Ordena de más a menos frecuente.

*Pista: `.agg(nombre=("columna", "funcion"), ...)`.*
"""
    ),
    code(
        """
resumen_por_tipo = (
    df.groupby("object_type")
    .agg(
        n=("object_type", "size"),
        velocidad_media=("speed_mps", "mean"),
        largo_mediano=("box_length", "median"),
        puntos_medianos=("num_lidar_points", "median"),
    )
    .sort_values("n", ascending=False)
    .round(2)
)
resumen_por_tipo
""",
        """
# TODO 9: resumen por tipo de objeto con cuatro métricas.
resumen_por_tipo = (
    df.groupby("____")
    .agg(
        n=("object_type", "____"),
        velocidad_media=("speed_mps", "____"),
        largo_mediano=("box_length", "____"),
        puntos_medianos=("num_lidar_points", "____"),
    )
    .sort_values("n", ascending=False)
    .round(2)
)
resumen_por_tipo
""",
    ),
    md(
        """
### ✏️ TODO 10 — `merge`: pegar el contexto

Cada segmento tiene un contexto (clima, momento del día). Constrúyelo como tabla aparte y únelo
al dataset por `segment_id`.

*Pista: `pd.merge(izquierda, derecha, on="llave", how="left")`. Comprueba que no se pierdan ni se
dupliquen filas.*
"""
    ),
    code(
        """
contexto = (
    df.groupby("segment_id")
    .agg(n_detecciones=("segment_id", "size"),
         puntos_medianos_segmento=("num_lidar_points", "median"))
    .reset_index()
)

unido = pd.merge(df, contexto, on="segment_id", how="left")

print(f"Antes del merge : {len(df):,} filas")
print(f"Después         : {len(unido):,} filas")
print(f"Columnas nuevas : {[c for c in unido.columns if c not in df.columns]}")
unido.head(3)
""",
        """
# TODO 10: pega a cada detección el contexto de su segmento.
contexto = (
    df.groupby("____")
    .agg(n_detecciones=("segment_id", "size"),
         puntos_medianos_segmento=("num_lidar_points", "median"))
    .reset_index()
)

unido = pd.merge(df, contexto, on="____", how="____")

print(f"Antes del merge : {len(df):,} filas")
print(f"Después         : {len(unido):,} filas")
print(f"Columnas nuevas : {[c for c in unido.columns if c not in df.columns]}")
unido.head(3)
""",
    ),
    code(
        """
# Autochequeo
assert len(unido) == len(df), (
    "revisa: un merge que cambia el número de filas indica llaves duplicadas en la tabla derecha"
)
assert unido["n_detecciones"].notna().all(), "revisa: quedaron detecciones sin contexto"
print(f"✅ {len(unido):,} filas, ninguna perdida ni duplicada.")
"""
    ),
    md(
        """
### ✏️ TODO 11 — `pivot_table`: cruzar dos variables

¿Los puntos LiDAR medianos dependen del clima, del momento del día, o de la combinación?
Construye la tabla cruzada.

*Pista: `df.pivot_table(index=..., columns=..., values=..., aggfunc=...)`.*
"""
    ),
    code(
        """
cruce = df.pivot_table(
    index="time_of_day",
    columns="detection_difficulty",
    values="num_lidar_points",
    aggfunc="median",
)
cruce
""",
        """
# TODO 11: puntos LiDAR medianos por momento del día y dificultad.
cruce = df.pivot_table(
    index="____",
    columns="____",
    values="num_lidar_points",
    aggfunc="____",
)
cruce
""",
    ),
    md(
        """
**✍️ Tu respuesta al TODO 11:**

*(doble clic aquí y escribe)*

¿Qué te dice esa tabla sobre la relación entre dificultad de detección y cantidad de puntos?
¿Es una relación que esperabas?
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 5
>
> **TODO 9.** Salen **7 filas**, no 4: las variantes de escritura de `object_type` siguen ahí
> (`PEDESTRIAN`, `Pedestrian`, `PEATON`, `Ped`). Es el mismo recordatorio de la Actividad 1.1:
> agrupar no limpia. Bien aprovechado, es la mejor motivación para el bloque de categorías de la
> 1.3: *sin normalizar, este `groupby` reparte a los peatones en cuatro grupos y ninguna cifra
> sirve.*
>
> **TODO 10.** El `merge` deja **40.680 filas**, las mismas. El `assert` es el punto pedagógico:
> un `merge` que cambia el número de filas casi siempre significa llaves duplicadas en la tabla
> derecha, y es un error que pasa desapercibido porque el código no falla. Enséñales a comprobar
> el largo **siempre** después de un `merge`.
>
> **TODO 11.** Los puntos medianos por dificultad muestran la relación esperada: las detecciones
> `LEVEL_2` tienen bastantes menos puntos que las `LEVEL_1`, en los tres momentos del día. Es una
> relación **generativa** del dataset, no un defecto: menos puntos ⇒ el sensor la marca como
> difícil.
>
> **La pregunta que conviene lanzar:** si "difícil" se decide a partir de los puntos, ¿puedo usar
> `detection_difficulty` como variable para predecir la calidad de la detección? No sin pensarlo:
> es casi una función de `num_lidar_points`. Meter las dos en un modelo lineal es introducir
> multicolinealidad; y si la etiqueta se derivara del objetivo, sería fuga de información. Deja
> sembrado el término: se retoma en la Actividad 1.3 y en la EA2.
>
> **Criterio de logro:** usa las tres operaciones correctamente, **verifica el largo tras el
> merge** e interpreta el cruce en términos del dominio, no solo de números.
"""
    ),
    # ======================================================================
    # BLOQUE 6 — Carga y guardado
    # ======================================================================
    md(
        """
---
# Bloque 6 · Carga y almacenamiento eficiente

| Formato | Lectura | Escritura | Parámetro que más importa |
|---|---|---|---|
| CSV | `pd.read_csv()` | `df.to_csv()` | `index=False`, `sep`, `dtype` |
| Excel | `pd.read_excel()` | `df.to_excel()` | `sheet_name`, `index=False` |
| JSON | `pd.read_json()` | `df.to_json()` | `orient` |
| Parquet | `pd.read_parquet()` | `df.to_parquet()` | `compression` |

`index=False` merece una nota: sin él, `to_csv` escribe el índice como una columna sin nombre.
Al releer, aparece como `Unnamed: 0`. Es el origen del 90 % de las columnas basura que se ven en
los datasets de internet.
"""
    ),
    md(
        """
### ✏️ TODO 12 — Medir los cuatro formatos

Exporta una muestra del dataset en los cuatro formatos y compara peso y tiempos.

Usamos una **muestra de 5.000 filas** por una razón práctica: escribir 40.680 filas con
`openpyxl` tarda del orden de un minuto, y no vamos a gastar la clase mirando una barra de
progreso. (Además, Excel admite como máximo 1.048.576 filas: no es un formato para volumen.)

*Pista: `formatos.medir_formatos(df, carpeta)` hace las cuatro escrituras y las cuatro lecturas.*
"""
    ),
    code(
        """
import pyarrow  # noqa: F401  (calienta el import para que no contamine la primera medición)

muestra = df_optimizado.head(5_000)
comparativa = formatos.medir_formatos(muestra, SALIDAS)
comparativa
""",
        """
# TODO 12: mide los cuatro formatos sobre una muestra de 5.000 filas.
import pyarrow  # noqa: F401  (calienta el import para que no contamine la primera medición)

muestra = df_optimizado.____(5_000)
comparativa = formatos.medir_formatos(____, SALIDAS)
comparativa
""",
    ),
    md(
        """
### ✏️ TODO 13 — Lo que el CSV pierde por el camino

Mira la columna `conserva_dtypes` de la tabla anterior. Solo un formato dice `True`.

Averigua exactamente **qué** se perdió al pasar por CSV.

*Pista: `formatos.columnas_con_dtype_cambiado(original, releido)`.*
"""
    ),
    code(
        """
muestra.to_csv(SALIDAS / "ida_y_vuelta.csv", index=False)
releida = pd.read_csv(SALIDAS / "ida_y_vuelta.csv")

perdidas = formatos.columnas_con_dtype_cambiado(muestra, releida)
print(f"Columnas que cambiaron de tipo: {len(perdidas)} de {muestra.shape[1]}\\n")
print(perdidas.to_string(index=False))
print()
print(f"Memoria antes de guardar : {muestra.memory_usage(deep=True).sum()/1024**2:.2f} MB")
print(f"Memoria al releer el CSV : {releida.memory_usage(deep=True).sum()/1024**2:.2f} MB")
""",
        """
# TODO 13: ¿qué tipos se perdieron al pasar por CSV?
muestra.to_csv(SALIDAS / "ida_y_vuelta.csv", index=____)
releida = pd.read_csv(SALIDAS / "ida_y_vuelta.csv")

perdidas = formatos.____(muestra, releida)
print(f"Columnas que cambiaron de tipo: {len(perdidas)} de {muestra.shape[1]}\\n")
print(perdidas.to_string(index=False))
print()
print(f"Memoria antes de guardar : {muestra.memory_usage(deep=True).sum()/1024**2:.2f} MB")
print(f"Memoria al releer el CSV : {releida.memory_usage(deep=True).sum()/1024**2:.2f} MB")
""",
    ),
    code(
        """
# Autochequeo
assert len(perdidas) > 0, "revisa: el CSV debería haber perdido tipos"
assert comparativa.set_index("formato").loc["parquet", "conserva_dtypes"], (
    "revisa: Parquet sí guarda el esquema"
)
assert not comparativa.set_index("formato").loc["csv", "conserva_dtypes"], (
    "revisa: el CSV no guarda tipos, solo texto"
)
print(f"✅ Todo el trabajo del TODO 4 ({len(perdidas)} columnas optimizadas) se perdió al guardar en CSV.")
print("   Parquet lo conservó.")
"""
    ),
    md(
        """
### ✏️ TODO 14 — El `index=False` que todos olvidan

Guarda la muestra **con** y **sin** `index=False` y compara las columnas al releer.
"""
    ),
    code(
        """
muestra.to_csv(SALIDAS / "con_indice.csv")                # sin index=False
muestra.to_csv(SALIDAS / "sin_indice.csv", index=False)

con = pd.read_csv(SALIDAS / "con_indice.csv")
sin = pd.read_csv(SALIDAS / "sin_indice.csv")

print("Columnas al releer 'con_indice.csv':", con.shape[1], "->", con.columns[0])
print("Columnas al releer 'sin_indice.csv':", sin.shape[1], "->", sin.columns[0])
""",
        """
# TODO 14: el efecto de olvidar index=False.
muestra.to_csv(SALIDAS / "con_indice.csv")                # sin index=False
muestra.to_csv(SALIDAS / "sin_indice.csv", index=____)

con = pd.read_csv(SALIDAS / "con_indice.csv")
sin = pd.read_csv(SALIDAS / "sin_indice.csv")

print("Columnas al releer 'con_indice.csv':", con.shape[1], "->", con.columns[0])
print("Columnas al releer 'sin_indice.csv':", sin.shape[1], "->", sin.columns[0])
""",
    ),
    code(
        """
# Autochequeo
assert con.shape[1] == sin.shape[1] + 1, "revisa: sin index=False aparece una columna de más"
assert con.columns[0].startswith("Unnamed"), "revisa: la columna extra debería llamarse Unnamed: 0"
print("✅ Sin index=False, cada guardado agrega una columna basura. Guarda tres veces y tendrás tres.")
"""
    ),
    md(
        """
### ✏️ TODO 15 — Tu decisión

Con las cifras que acabas de medir, completa la tabla de decisiones para **tu** proyecto.
"""
    ),
    code(
        """
decision = {
    "formato_para_datos_crudos": "csv",       # llega así, no se toca
    "formato_para_datos_procesados": "parquet",
    "formato_para_entregar_al_negocio": "excel",
    "por_que_parquet": "pesa menos de la mitad y es el único que conserva los dtypes",
}

for clave, valor in decision.items():
    print(f"{clave:35s}: {valor}")
""",
        """
# TODO 15: decide y justifica con TUS cifras.
decision = {
    "formato_para_datos_crudos": "____",
    "formato_para_datos_procesados": "____",
    "formato_para_entregar_al_negocio": "____",
    "por_que_parquet": "____",
}

for clave, valor in decision.items():
    print(f"{clave:35s}: {valor}")
""",
    ),
    code(
        """
# Autochequeo
assert all(v and not v.startswith("____") for v in decision.values()), (
    "revisa: quedaron campos sin completar"
)
assert decision["formato_para_datos_procesados"] == "parquet", (
    "revisa: ¿qué formato conservó los tipos y pesó menos?"
)
print("✅ Decisión documentada. Cópiala a la tabla del cierre.")
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 6 (TODO 12 a 15) ⭐
>
> **Cifras medidas** sobre 5.000 filas ya optimizadas (categóricas + `float32`):
>
> | Formato | Peso | vs CSV | ¿Conserva tipos? |
> |---|---|---|---|
> | **Parquet** | 0,23 MB | **0,40×** | ✅ |
> | CSV | 0,57 MB | 1,00× | ❌ |
> | Excel | 0,57 MB | 1,01× | ❌ |
> | JSON | 2,01 MB | 3,56× | ❌ |
>
> Y el CSV pierde el tipo de **11 de las 16 columnas**: las 5 `category` vuelven como `object` y
> las 6 `float32` vuelven como `float64`.
>
> **La frase del bloque:** *todo el trabajo del TODO 4 se deshace al guardar en CSV.* Ahorraron
> un 57 % de memoria y lo perdieron entero en el `to_csv`. Encadenado así, se entiende solo.
>
> **Un matiz honesto que hay que decir, porque si no alguien lo descubre y desconfía del resto:**
> Parquet **no** gana siempre. Con pocas filas (unos cientos) su encabezado —esquema y
> estadísticas por grupo de filas— pesa más que lo que ahorra, y llega a ser *más* grande que el
> CSV. La compresión columnar necesita repetición para lucirse. Hay un test que lo deja escrito:
> `tests/test_formatos.py::test_con_pocas_filas_el_parquet_puede_pesar_mas`.
>
> **Sobre Excel.** No pesa mucho menos que el CSV pese a ser binario, porque un `.xlsx` es un ZIP
> de XML, y el XML es verborrágico. Su ventaja no es técnica: es que el área de negocio lo abre.
> Esa es una razón perfectamente válida para elegirlo, y conviene decirlo así en vez de
> despreciarlo.
>
> **TODO 14.** Sin `index=False` aparece `Unnamed: 0`. El detalle que remata: si alguien guarda y
> relee tres veces, termina con `Unnamed: 0`, `Unnamed: 0.1` y `Unnamed: 0.2`. Pregunta cuántos
> datasets de Kaggle han visto con esa columna. Van a reconocerla.
>
> **TODO 15.** La respuesta esperada: crudos en el formato en que llegan (no se reescribe lo que
> llegó, es la evidencia de origen); procesados en Parquet; entrega al negocio en Excel. Lo que se
> evalúa es la justificación **con las cifras propias**, no la elección.
>
> **Criterio de logro:** ejecuta el benchmark, identifica qué se pierde y por qué, y elige
> formatos justificando con las cifras que midió.
"""
    ),
    # ======================================================================
    # CIERRE
    # ======================================================================
    md(
        """
---
# Cierre · Tabla de decisiones de almacenamiento

Esta es la entrega de la Actividad 1.2. Rellénala con las cifras que **tú** mediste y cópiala al
notebook del proyecto de equipo.

---

## Decisiones de estructura y almacenamiento

**Equipo:** `____`
**Dataset del proyecto:** `____`

### Lo que medí

| Medición | Mi cifra |
|---|---|
| Filas × columnas del dataset | |
| Memoria al cargarlo (`deep=True`) | |
| Memoria tras optimizar tipos | |
| Ahorro conseguido | |
| Peso en CSV | |
| Peso en Parquet | |
| Columnas que el CSV pierde de tipo | |

### Lo que decidí

| Etapa | Formato elegido | Por qué (con cifra) |
|---|---|---|
| Datos crudos | | |
| Datos procesados | | |
| Entrega al negocio | | |

### Las tres trampas que ahora conozco

1. `sys.getsizeof` sobre una lista **miente**, y `memory_usage()` sin `deep=True` también:
   `____`
2. Asignar una Series con índice distinto **no da error**, deja `NaN` y valores cruzados:
   `____`
3. Guardar en CSV **borra los tipos**: `____`

### Pregunta de cierre

Si tu dataset creciera de `____` filas a **cien millones**, ¿qué es lo primero que dejaría de
funcionar de tu código actual, y qué cambiarías?

`____`
"""
    ),
    md_docente(
        """
> ### 🎓 Criterios de logro — Actividad 1.2 (IL 1.2)
>
> | Nivel | Descripción |
> |---|---|
> | **Destacado (4)** | Los 15 TODO en verde; explica *por qué* la asignación desalineada no falla; la tabla de decisiones cita cifras propias y la respuesta de cierre identifica un cuello de botella real (memoria, no CPU) |
> | **Logrado (3)** | Los 15 TODO en verde; distingue `.loc` de `.iloc` con un ejemplo propio; la tabla está completa con sus mediciones |
> | **En desarrollo (2)** | Resuelve los bloques 1, 2 y 5 pero no logra explicar el error silencioso del bloque 4; la tabla tiene celdas sin cifra |
> | **Inicial (1)** | Ejecuta las celdas sin interpretar; usa `.loc` e `.iloc` indistintamente |
>
> **Lo que hay que mirar al corregir, en este orden:**
>
> 1. **El bloque 4.** Si no puede explicar por qué salieron 771 `NaN`, no entendió el índice, y
>    eso vuelve a morder en todas las semanas siguientes.
> 2. **Que las cifras sean suyas.** Copiar las de la pauta es fácil de detectar: los tiempos
>    varían por máquina, así que dos entregas con tiempos idénticos al milisegundo son la misma
>    entrega.
> 3. **La respuesta de cierre.** La buena identifica la **memoria** como el primer límite (el
>    DataFrame completo no cabe en RAM), no la velocidad. Ahí es donde entran Parquet leído por
>    trozos, Dask o Spark, y ahí conecta con Databricks del notebook 04.
>
> **Enlace con el resto de la EA1:** la Actividad 1.3 asume que ya saben filtrar sin romper el
> índice. Si este bloque queda flojo, la 1.3 se llena de `NaN` inexplicables.
"""
    ),
]
