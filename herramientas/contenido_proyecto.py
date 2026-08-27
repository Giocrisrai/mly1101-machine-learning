"""Fuente única de la plantilla de proyecto de equipo (Semana 1 completa, EA1).

Genera ``notebooks/10_proyecto_equipo_plantilla.ipynb``: una sola versión, sin solucionario.
Es la plantilla que cada equipo copia a su fork y rellena con **su** dataset.

Decisión de diseño importante: la plantilla **funciona tal cual**, sin tocar nada. Si el equipo
no ha elegido dataset todavía, cae en el de la asignatura y todo el notebook corre de principio
a fin como ejemplo trabajado. Cuando el equipo tiene el suyo, cambia **una sola variable**
(``RUTA_MI_DATASET``) y el mismo notebook se aplica a sus datos.

Ese detalle es lo que evita el problema clásico de las plantillas: que estén llenas de celdas
rotas hasta que alguien las complete, y que nadie sepa si el error es suyo o de la plantilla.

Regenerar tras editar:

    python herramientas/construir_notebooks.py
"""

from __future__ import annotations

from contenido_semana01 import URL_REPO, code, md

CELDAS_PROYECTO: list[dict] = [
    md(
        """
# MLY1101 · Plantilla de proyecto de equipo
## EA1 · Análisis y preprocesamiento de datos

Esta es la plantilla que tu equipo rellena con **su propio dataset**. Recoge lo trabajado en las
tres actividades de la Semana 1:

| Actividad | Qué aporta a este proyecto |
|---|---|
| **1.1** Fuentes y colaboración | La ficha de fuentes y el acuerdo de trabajo del equipo |
| **1.2** Estructuras y almacenamiento | La optimización de tipos y la elección de formato |
| **1.3** Análisis exploratorio | El diagnóstico de calidad y las decisiones de preprocesamiento |

---

### Cómo usar esta plantilla

1. Un integrante hace **fork** del repositorio y el resto colabora sobre él.
2. **Copia este notebook** con el nombre de tu equipo:
   `10_proyecto_<nombre_del_equipo>.ipynb`. No trabajes sobre el original: el siguiente `git
   merge upstream/main` lo sobrescribe.
3. Rellena los campos marcados con `____` y ejecuta las celdas en orden.
4. Trabaja en una rama, no en `main`, y entra a `main` por Pull Request.

> **Funciona sin configurar nada.** Si aún no eligieron dataset, la plantilla usa el de la
> asignatura y corre completa como ejemplo. Cuando tengan el suyo, cambian **una sola variable**
> en la celda de configuración y todo lo demás se aplica solo.

---

### El dataset que elijan debe cumplir cuatro condiciones

| Condición | Por qué |
|---|---|
| **Al menos 1.000 filas y 8 columnas** | Con menos, no hay nada que diagnosticar |
| **Mezcla de numéricas y categóricas** | Si es todo numérico, la mitad del análisis no aplica |
| **Tener problemas de calidad reales** | Un dataset ya limpio no permite evidenciar el RA1 |
| **Licencia que permita el uso académico** | Y saber cuál es, no suponerla |

Si el dataset elegido está impecable, es mala señal: normalmente significa que alguien ya hizo
el trabajo que ustedes deben evidenciar. Busquen datos crudos, no una versión curada.
"""
    ),
    md(
        """
---
## 0 · Configuración

Ejecuta esta celda. Si aún no tienes dataset propio, déjala tal cual.
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

# ---------------------------------------------------------------------------
# ⬇️  LA ÚNICA LÍNEA QUE TIENEN QUE CAMBIAR
#
#     None                      -> usa el dataset de la asignatura (ejemplo trabajado)
#     "mis_datos.csv"           -> un archivo que subieron a Colab
#     "https://.../datos.csv"   -> una URL pública
# ---------------------------------------------------------------------------
RUTA_MI_DATASET = None

SALIDAS = Path("salidas_proyecto")
SALIDAS.mkdir(exist_ok=True)

DATASET_DE_LA_ASIGNATURA = RAIZ / "datos" / "crudos" / "detecciones_waymo_like.csv"
ES_EJEMPLO = RUTA_MI_DATASET is None
ORIGEN = DATASET_DE_LA_ASIGNATURA if ES_EJEMPLO else RUTA_MI_DATASET

print("Colab:", EN_COLAB)
if ES_EJEMPLO:
    print("⚠️  Usando el dataset de la asignatura como EJEMPLO.")
    print("    Cambien RUTA_MI_DATASET cuando tengan el suyo.")
else:
    print("✅ Usando el dataset del equipo:", ORIGEN)
"""
    ),
    code(
        """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import eda        # diagnóstico de calidad      (Actividad 1.3)
import formatos   # comparación de formatos     (Actividad 1.2)
import fuentes    # lectura de fuentes varias   (Actividad 1.1)

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 160)
sns.set_theme(style="whitegrid")

print("pandas", pd.__version__, "| numpy", np.__version__)
"""
    ),
    # ------------------------------------------------------------------
    md(
        """
---
## 1 · Ficha del equipo y del problema

*(Doble clic para editar. Todo lo que diga `____` hay que reemplazarlo.)*

**Equipo:** `____`
**Integrantes:** `____`
**Repositorio (URL del fork):** `____`

### El problema de negocio

**Sector:** `____` *(retail, banca, salud, educación, transporte, otro)*

**Situación:** `____`
*(Dos o tres frases. Qué organización, qué está pasando, por qué le importa a alguien.)*

**Pregunta que queremos responder con datos:** `____`
*(Una sola pregunta, concreta y respondible. "Entender a los clientes" no lo es. "¿Qué clientes
tienen mayor probabilidad de no renovar en los próximos 3 meses?" sí.)*

**Tipo de aprendizaje que corresponde:** `____` *(supervisado / no supervisado / refuerzo)*

**Por qué:** `____`
*(La clave es si existe una respuesta correcta conocida para casos del pasado.)*

**Si es supervisado, ¿cuál es la variable objetivo `y`?:** `____`

**A quién le sirve la respuesta y qué decisión cambiaría:** `____`
*(Si nadie toma una decisión distinta con el resultado, el proyecto no tiene destinatario.)*
"""
    ),
    # ------------------------------------------------------------------
    md(
        """
---
## 2 · Ficha de fuentes de datos

Viene de la **Actividad 1.1**. Una tabla por fuente, las tres completas.

### Fuente 1 — la principal

| Campo | Valor |
|---|---|
| Nombre | `____` |
| URL de origen | `____` |
| Tipo | `____` *(estructurada / semiestructurada / no estructurada)* |
| Formato | `____` *(CSV / JSON / SQL / Parquet / Excel / texto / imagen)* |
| Filas × columnas | `____` |
| Licencia | `____` *(y si permite uso académico)* |
| Fecha de descarga | `____` |
| ¿Datos personales? | `____` |
| ¿Quién la recolectó y con qué propósito? | `____` |

### Fuente 2

| Campo | Valor |
|---|---|
| Nombre | `____` |
| URL de origen | `____` |
| Tipo | `____` |
| Formato | `____` |
| Filas × columnas | `____` |
| Licencia | `____` |
| Fecha de descarga | `____` |
| ¿Datos personales? | `____` |
| ¿Quién la recolectó y con qué propósito? | `____` |

### Fuente 3

| Campo | Valor |
|---|---|
| Nombre | `____` |
| URL de origen | `____` |
| Tipo | `____` |
| Formato | `____` |
| Filas × columnas | `____` |
| Licencia | `____` |
| Fecha de descarga | `____` |
| ¿Datos personales? | `____` |
| ¿Quién la recolectó y con qué propósito? | `____` |

### Lista de chequeo de privacidad

Marca cada casilla solo si la respuesta está verificada, no supuesta.

- [ ] Ninguna fuente contiene identificadores directos (nombre, RUT, correo, teléfono, patente),
      o si los contiene, están anonimizados y documentamos cómo.
- [ ] Revisamos si se puede **reidentificar** a alguien combinando columnas
      *(comuna + edad + profesión suele bastar, aunque ninguna lo sea por separado)*.
- [ ] Cada licencia permite el uso que le vamos a dar.
- [ ] Descartamos las columnas que no necesitamos *(lo que no se recolecta no se filtra)*.
- [ ] Podemos declarar el origen y la fecha de cada fuente.
"""
    ),
    # ------------------------------------------------------------------
    md(
        """
---
## 3 · Carga

Tres formas de traer datos a Colab. Usa la que corresponda a tu caso y **borra las otras dos**.
"""
    ),
    code(
        """
# --- Vía A: archivo subido a mano (la más simple para empezar) -----------------
# Descomenta estas tres líneas y ejecuta. Se abre un selector de archivos.
#
# from google.colab import files
# subidos = files.upload()
# RUTA_MI_DATASET = next(iter(subidos))

# --- Vía B: URL pública (la más reproducible) ----------------------------------
# RUTA_MI_DATASET = "https://.../mis_datos.csv"

# --- Vía C: API de Kaggle (si el dataset está ahí) -----------------------------
# 1. En kaggle.com -> Settings -> API -> "Create New Token": descarga kaggle.json
# 2. Súbelo a Colab y ejecuta:
#
# !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d USUARIO/NOMBRE-DEL-DATASET --unzip -p datos_equipo
# RUTA_MI_DATASET = "datos_equipo/archivo.csv"

print("Origen que se va a leer:", ORIGEN)
"""
    ),
    code(
        """
df = pd.read_csv(ORIGEN)

print(f"Filas: {df.shape[0]:,}   Columnas: {df.shape[1]}")
print(f"Memoria real: {df.memory_usage(deep=True).sum()/1024**2:.1f} MB")
df.head()
"""
    ),
    code(
        """
# Comprobación de las condiciones mínimas del dataset elegido.
n_numericas = df.select_dtypes(include=np.number).shape[1]
n_no_numericas = df.shape[1] - n_numericas

print(f"Numéricas: {n_numericas}   No numéricas: {n_no_numericas}")

problemas = []
if len(df) < 1_000:
    problemas.append(f"solo {len(df)} filas (se piden al menos 1.000)")
if df.shape[1] < 8:
    problemas.append(f"solo {df.shape[1]} columnas (se piden al menos 8)")
if n_numericas == 0 or n_no_numericas == 0:
    problemas.append("falta mezcla de variables numéricas y categóricas")

if problemas:
    print("\\n⚠️  El dataset no cumple:")
    for p in problemas:
        print("   -", p)
    print("\\n   Se puede seguir, pero parte del análisis no va a tener sustancia.")
else:
    print("\\n✅ El dataset cumple las condiciones mínimas.")
"""
    ),
    md(
        """
### 3.1 · ¿Qué representa una fila?

**La pregunta más importante y la que más se salta.** Antes de calcular nada, escríbelo:

**Una fila de nuestro dataset es:** `____`

*(Un cliente, una transacción, un cliente-mes, una detección de un objeto en un instante… No es
lo mismo, y define qué significa "duplicado", qué se puede promediar y cómo se arma después la
partición de entrenamiento y prueba.)*
"""
    ),
    # ------------------------------------------------------------------
    md(
        """
---
## 4 · Diagnóstico de calidad

De la **Actividad 1.3**. `eda.resumen_calidad()` entrega la radiografía completa: tipo,
cardinalidad, nulos declarados y **nulos ocultos** (valores como `-1`, `"N/D"` o `""` que
representan un faltante sin ser `NaN`).
"""
    ),
    code(
        """
df.info()
"""
    ),
    code(
        """
resumen = eda.resumen_calidad(df)
resumen
"""
    ),
    code(
        """
# Las tres preguntas que hay que responder con la tabla de arriba.
print("1. Columnas de cardinalidad casi única (son llaves, NO variables predictoras):")
print(resumen[resumen["pct_unicos"] > 90][["dtype", "n_unicos", "pct_unicos"]], "\\n")

print("2. Columnas constantes (varianza cero, información cero):")
print(resumen[resumen["n_unicos"] <= 1][["dtype", "n_unicos", "ejemplos"]], "\\n")

print("3. Columnas con algo faltante (declarado u oculto):")
faltantes = resumen[resumen["pct_faltante_total"] > 0]
print(faltantes[["n_nulos", "n_centinelas", "pct_faltante_total"]].sort_values(
    "pct_faltante_total", ascending=False))
"""
    ),
    code(
        """
# Duplicados: exactos y lógicos.
# ⬇️ CAMBIEN esta lista por las columnas que identifican una fila en SU dataset.
LLAVE = [c for c in df.columns[:2]]

print("Llave usada:", LLAVE)
eda.reporte_duplicados(df, llave=LLAVE)
"""
    ),
    md(
        """
> **Duplicado exacto** = la fila entera se repite; `drop_duplicates()` los elimina.
> **Duplicado lógico** = se repite la llave pero cambia algo más; `drop_duplicates()` **no** los
> toca y son los peligrosos: significan que la misma entidad tiene dos versiones distintas de la
> verdad, y hay que decidir cuál vale.
"""
    ),
    code(
        """
# Valores atípicos por el criterio IQR, columna numérica por columna numérica.
numericas = df.select_dtypes(include=np.number).columns.tolist()
perfil = eda.perfil_numerico(df, numericas)
perfil[["count", "mean", "std", "min", "max", "asimetria", "n_outliers_iqr"]]
"""
    ),
    code(
        """
# Visualización rápida: distribución y atípicos de las primeras variables numéricas.
a_graficar = numericas[:4]
if a_graficar:
    fig, ejes = plt.subplots(2, len(a_graficar), figsize=(4 * len(a_graficar), 7))
    ejes = np.atleast_2d(ejes)
    for i, columna in enumerate(a_graficar):
        df[columna].plot.hist(bins=40, ax=ejes[0, i], title=columna)
        df.boxplot(column=columna, ax=ejes[1, i])
    plt.tight_layout()
    plt.show()
else:
    print("No hay columnas numéricas que graficar.")
"""
    ),
    md(
        """
### 4.1 · Atípico imposible contra atípico legítimo

El criterio IQR **propone**; el conocimiento del dominio **dispone**. Antes de eliminar nada,
clasifica cada atípico que encontraste:

| Columna | Valor extremo | ¿Imposible o legítimo? | Qué haremos |
|---|---|---|---|
| `____` | `____` | `____` | `____` |
| `____` | `____` | `____` | `____` |
| `____` | `____` | `____` | `____` |

*Imposible = viola una regla del dominio (una edad de 300 años, un precio negativo, una altura
de 0 m). Se corrige o se marca como faltante.*

*Legítimo = raro pero real (el bus que mide 15 m, el cliente que compró 200 veces). **No se
elimina**: normalmente es justo el caso que interesa.*
"""
    ),
    code(
        """
# Reglas de dominio: expresiones que describen filas INVÁLIDAS.
# ⬇️ CAMBIEN estas reglas por las de SU dominio.
REGLAS = {}
for columna in numericas[:3]:
    REGLAS[f"{columna} negativo"] = f"`{columna}` < 0"

if REGLAS:
    print(eda.valores_imposibles(df, REGLAS).to_string(index=False))
else:
    print("Definan al menos una regla de dominio para su dataset.")
"""
    ),
    # ------------------------------------------------------------------
    md(
        """
---
## 5 · Estructuras y memoria

De la **Actividad 1.2**. Optimizar tipos no es cosmética: decide si el dataset cabe en RAM
cuando crezca.
"""
    ),
    code(
        """
mem_antes = df.memory_usage(deep=True).sum() / 1024**2

df_optimizado = df.copy()

# Categóricas: columnas de texto con pocos valores distintos (menos del 50 % de cardinalidad).
for columna in df.select_dtypes(include="object").columns:
    if df[columna].nunique(dropna=True) < len(df) * 0.5:
        df_optimizado[columna] = df_optimizado[columna].astype("category")

# Continuas: float64 -> float32 (la mitad de memoria, ~7 cifras significativas).
for columna in df.select_dtypes(include="float64").columns:
    df_optimizado[columna] = df_optimizado[columna].astype("float32")

mem_despues = df_optimizado.memory_usage(deep=True).sum() / 1024**2

print(f"Antes   : {mem_antes:7.2f} MB")
print(f"Después : {mem_despues:7.2f} MB")
print(f"Ahorro  : {100*(1 - mem_despues/mem_antes):6.1f} %")
assert len(df_optimizado) == len(df), "optimizar memoria no puede perder filas"
"""
    ),
    md(
        """
> **Cuidado con el orden.** Convertir a `category` **antes** de limpiar es un problema: sobre una
> columna categórica, un `.replace()` que introduzca un valor nuevo falla o se comporta raro,
> porque el conjunto de categorías queda fijado. **Limpien primero, conviertan después.**
"""
    ),
    code(
        """
# ¿Qué formato conviene para guardar el dataset procesado? Mídanlo, no lo supongan.
import pyarrow  # noqa: F401  (calienta el import para no contaminar la primera medición)

muestra = df_optimizado.head(min(5_000, len(df_optimizado)))
comparativa = formatos.medir_formatos(muestra, SALIDAS)
comparativa
"""
    ),
    code(
        """
# Y esto es lo que se pierde al pasar por CSV.
muestra.to_csv(SALIDAS / "ida_y_vuelta.csv", index=False)
releida = pd.read_csv(SALIDAS / "ida_y_vuelta.csv")

perdidas = formatos.columnas_con_dtype_cambiado(muestra, releida)
print(f"Columnas que cambiaron de tipo al releer el CSV: {len(perdidas)} de {muestra.shape[1]}")
if len(perdidas):
    print(perdidas.to_string(index=False))
"""
    ),
    # ------------------------------------------------------------------
    md(
        """
---
## 6 · Sesgo y representatividad

Un sesgo sin consecuencia medible es una frase bonita. Busquen la consecuencia.
"""
    ),
    code(
        """
# Composición del dataset según sus variables categóricas.
categoricas = df.select_dtypes(include=["object", "category"]).columns.tolist()

for columna in categoricas[:3]:
    if df[columna].nunique() <= 20:
        print(f"--- {columna} ---")
        print(eda.resumen_desbalance(df[columna]).head(8), "\\n")
"""
    ),
    code(
        """
# ¿Los datos faltantes se reparten igual entre grupos, o se concentran en uno?
# ⬇️ CAMBIEN estas dos variables por las de SU dataset.
COLUMNA_CON_NULOS = faltantes.index[0] if len(faltantes) else None
VARIABLE_DE_GRUPO = categoricas[0] if categoricas else None

if COLUMNA_CON_NULOS and VARIABLE_DE_GRUPO and df[COLUMNA_CON_NULOS].isna().any():
    tabla = eda.matriz_nulos_por_grupo(df, COLUMNA_CON_NULOS, [VARIABLE_DE_GRUPO])
    print(f"% de nulos en '{COLUMNA_CON_NULOS}' según '{VARIABLE_DE_GRUPO}':\\n")
    print(tabla.sort_values("pct_nulos", ascending=False), "\\n")

    peor = tabla["pct_nulos"].idxmax()
    mejor = tabla["pct_nulos"].idxmin()
    if tabla.loc[mejor, "pct_nulos"] > 0:
        factor = tabla.loc[peor, "pct_nulos"] / tabla.loc[mejor, "pct_nulos"]
        print(f"'{peor}' perdería {factor:.1f} veces más filas que '{mejor}' con un dropna().")
else:
    print("No hay una columna con NaN declarados para cruzar. Revisen los nulos OCULTOS:")
    print(resumen[resumen["n_centinelas"] > 0][["n_centinelas", "ejemplos"]])
"""
    ),
    md(
        """
### 6.1 · El riesgo que identificamos

**Grupo que está subrepresentado o peor medido:** `____`

**Cifra que lo respalda:** `____`
*(No vale "podría haber sesgo". Vale "el grupo X es el 4 % de las filas y concentra el 18 % de
los valores faltantes".)*

**Decisión técnica que lo empeoraría:** `____`
*(Por ejemplo: un `dropna()` global, imputar con la media general, eliminar los atípicos.)*

**Consecuencia concreta sobre una persona:** `____`
*(A quién le va a ir peor, y cómo. Este es el punto que separa una respuesta memorizada de una
entendida.)*

**Y por qué la métrica no lo mostraría:** `____`
*(Si el conjunto de prueba tiene el mismo sesgo que el de entrenamiento, el promedio se ve bien.)*

**Qué haremos al respecto:** `____`
"""
    ),
    # ------------------------------------------------------------------
    md(
        """
---
## 7 · Tabla de decisiones de preprocesamiento

El entregable central. Una fila por decisión, y **cada una con su cifra**.

| # | Problema detectado | Cifra que lo cuantifica | Decisión | Justificación | Qué se pierde |
|---|---|---|---|---|---|
| 1 | `____` | `____` | `____` | `____` | `____` |
| 2 | `____` | `____` | `____` | `____` | `____` |
| 3 | `____` | `____` | `____` | `____` | `____` |
| 4 | `____` | `____` | `____` | `____` | `____` |
| 5 | `____` | `____` | `____` | `____` | `____` |

**La columna "qué se pierde" no es opcional.** Toda decisión de limpieza descarta información.
Eliminar filas es elegir qué parte de la realidad borrar; imputar es inventar un dato que no se
midió; recortar atípicos es decidir que lo raro no existe. Escribir el costo obliga a comprobar
que valía la pena.

> **Fuga de información (*data leakage*).** Cualquier estadístico que usen para transformar los
> datos —una media para imputar, un mínimo y un máximo para escalar, las categorías de un
> codificador— tiene que calcularse **solo con los datos de entrenamiento**. Si lo calculan sobre
> el dataset completo, el conjunto de prueba filtra información al modelo y la evaluación queda
> inflada. En la próxima semana esto se resuelve con `Pipeline` de scikit-learn.
"""
    ),
    code(
        """
# Guarden el dataset procesado en el formato que eligieron (con su justificación medida arriba).
SALIDA_FINAL = SALIDAS / "dataset_procesado.parquet"
df_optimizado.to_parquet(SALIDA_FINAL, index=False)

print(f"Guardado: {SALIDA_FINAL}  ({SALIDA_FINAL.stat().st_size/1024**2:.2f} MB)")
print(f"Filas: {len(df_optimizado):,}   Columnas: {df_optimizado.shape[1]}")
print("\\nSi trabajan en Colab, descárguenlo antes de cerrar la sesión:")
print("   from google.colab import files; files.download(str(SALIDA_FINAL))")
"""
    ),
    # ------------------------------------------------------------------
    md(
        """
---
## 8 · Mini-informe de calidad de datos

Esta es la entrega evaluada. Máximo una página.

---

### Contexto

`____` *(Dos frases: qué problema, qué datos.)*

### Cinco hallazgos, cada uno con su cifra

1. `____`
2. `____`
3. `____`
4. `____`
5. `____`

> Ninguna afirmación sin una cifra. "Hay muchos nulos" no es un hallazgo; "el 23 % de la columna
> `ingreso` está vacía y se concentra en los clientes sin sucursal asignada" sí lo es.

### Tres decisiones de preprocesamiento, con su justificación

1. **Decisión:** `____` · **Porque:** `____` · **Costo:** `____`
2. **Decisión:** `____` · **Porque:** `____` · **Costo:** `____`
3. **Decisión:** `____` · **Porque:** `____` · **Costo:** `____`

### Un riesgo ético o de sesgo

`____`

### ¿Podemos confiar en estos datos para responder nuestra pregunta?

`____` *(Sí, no, o "sí con estas condiciones". Responder "no" está permitido y a veces es la
respuesta correcta y valiente: entonces expliquen qué haría falta.)*

### Lo que haríamos con más tiempo

`____`
"""
    ),
    md(
        """
---
## 9 · Autoevaluación

Antes de entregar, revisen contra la rúbrica (`docs/rubrica_ra1.md`). Marquen solo lo que
puedan demostrar señalando una celda del notebook.

| | Criterio | ¿Dónde está la evidencia? |
|---|---|---|
| ☐ | La ficha de las 3 fuentes está completa, con licencia y fecha | |
| ☐ | Está escrito qué representa una fila | |
| ☐ | Identificamos tipos de variable y columnas que no son predictoras | |
| ☐ | Cuantificamos nulos declarados **y ocultos** | |
| ☐ | Distinguimos duplicados exactos de lógicos | |
| ☐ | Clasificamos los atípicos en imposibles y legítimos | |
| ☐ | La tabla de decisiones tiene 5 filas, cada una con cifra y costo | |
| ☐ | El riesgo de sesgo tiene cifra **y** consecuencia sobre un grupo concreto | |
| ☐ | Elegimos formato de guardado con una medición propia | |
| ☐ | El notebook corre de principio a fin sin errores | |
| ☐ | El repositorio tiene commits de **todos** los integrantes | |

> **Ese último punto se revisa con `git log`.** Un repositorio con 40 commits de una sola persona
> dice más sobre el trabajo del equipo que cualquier declaración.
"""
    ),
    code(
        """
# Comprobación final: que el notebook corra completo.
print("Dataset :", ORIGEN)
print(f"Filas   : {len(df):,}   Columnas: {df.shape[1]}")
print(f"Memoria : {mem_antes:.2f} MB -> {mem_despues:.2f} MB")
print(f"Columnas con algo faltante: {len(faltantes)}")
print()
if ES_EJEMPLO:
    print("⚠️  Esto sigue siendo el EJEMPLO con el dataset de la asignatura.")
    print("    Cambien RUTA_MI_DATASET en la celda de configuración y vuelvan a ejecutar todo.")
else:
    print("✅ Notebook ejecutado completo sobre el dataset del equipo.")
"""
    ),
]
