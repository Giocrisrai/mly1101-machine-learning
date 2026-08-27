"""Fuente única del contenido de la Actividad 1.1 — Fuentes de Datos y Trabajo Colaborativo.

Indicador de logro **IL 1.1**: *identifica diversas fuentes de datos y herramientas de
trabajo colaborativo para responder a necesidades de negocio.*

De este archivo salen dos notebooks:

- ``notebooks/02_alumno_fuentes.ipynb``   (versión con TODO)
- ``notebooks/02_docente_fuentes.ipynb``  (versión resuelta con pauta)

Las funciones ``md`` / ``md_docente`` / ``code`` se reutilizan de
``contenido_semana01``: el formato de celda es el mismo para todo el repositorio.

Regenerar los notebooks tras editar este archivo:

    python herramientas/construir_notebooks.py
"""

from __future__ import annotations

from contenido_semana01 import URL_REPO, code, md, md_docente

CELDAS_ACT11: list[dict] = [
    # ======================================================================
    # ENCUADRE
    # ======================================================================
    md(
        """
# MLY1101 · Machine Learning — Actividad 1.1
## Fuentes de Datos y Trabajo Colaborativo

**Resultado de aprendizaje (RA1):** recopila, a través de un trabajo colaborativo, sets de
datos representativos y de calidad, a partir de distintas fuentes (texto plano, archivos CSV,
otros) para responder a las necesidades del contexto de negocio, considerando aspectos éticos.

**Indicador de logro (IL 1.1):** identifica diversas fuentes de datos y herramientas de trabajo
colaborativo para responder a necesidades de negocio.

---

### La idea central de hoy

En los ejemplos de clase los datos siempre llegan como un CSV ordenado. En el trabajo real casi
nunca es así:

```
   Base relacional  ─┐
   API que responde ─┤
   JSON anidado      ├──►  un DataFrame  ──►  EDA  ──►  modelo
   Texto libre      ─┤
   Planilla Excel   ─┘
```

Hoy no vamos a limpiar datos ni a entrenar nada. Vamos a **traer los mismos datos desde cuatro
tipos de fuente distintos** y comprobar que llegamos al mismo DataFrame. Después discutiremos
con qué herramientas trabaja un equipo sobre eso, y por qué la recolección nunca es neutral.

> El PPT dice que Parquet comprime mejor y que el 80 % de los datos no está estructurado. Hoy
> vamos a **medirlo**, no a citarlo.

---

### El caso

Seguimos en el equipo de percepción de una empresa de conducción autónoma, con el mismo dataset
de detecciones LiDAR de la Actividad 1.3. La diferencia es de dónde lo sacamos.

En una empresa real, esas 40.680 detecciones no están en un archivo: están en una tabla de una
base de datos, el contexto de cada segmento llega por una API en JSON anidado, y los incidentes
del turno los escribe una persona en prosa. Alguien tiene que juntar las tres cosas.

---

### Al final de la sesión debes entregar

Una **ficha de fuentes de datos** (última celda) con:

- las 3 fuentes que tu equipo usará en el proyecto, clasificadas por tipo y formato;
- el acuerdo de trabajo colaborativo del equipo (roles, ramas, revisión);
- 1 riesgo de sesgo o de privacidad identificado en al menos una de esas fuentes.
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Actividad 1.1
>
> **Cómo usar este documento.** Es el solucionario de `02_alumno_fuentes.ipynb`. Contiene el
> mismo contenido más el código resuelto de cada TODO, las respuestas esperadas y los criterios
> de logro por bloque.
>
> **La actividad son 6 horas pedagógicas** según el programa. La distribución de abajo cubre
> el trabajo guiado; las horas restantes quedan para que apliquen lo mismo al caso oficial
> que hayan elegido (Telco, Housing o Spotify), sobre el que se rinde la Evaluación Parcial.
>
> **Distribución del bloque guiado (~2 h):**
>
> | Bloque | Min | Foco |
> |---|---|---|
> | 0 · Encuadre | 10 | De dónde vienen realmente los datos |
> | 1 · Ecosistema y tipos de aprendizaje | 15 | Ubicar la asignatura; qué problema es cuál |
> | 2 · Datos estructurados: CSV, URL y **SQL** | 30 | Que SQL deje de ser una palabra del PPT ⭐ |
> | 3 · Semiestructurados: JSON anidado | 20 | `json_normalize` y `record_path` |
> | 4 · No estructurados: texto libre | 20 | Del 80 % del que todos hablan, algo concreto |
> | 5 · Herramientas del equipo | 15 | Colab, GitHub, Kedro, Databricks |
> | 6 · Ética y sesgos en la recolección | 20 | El censo real de Waymo: 793 soleados de 798 ⭐ |
> | Cierre · Ficha de fuentes | 10 | Insumo directo del proyecto de equipo |
>
> **Los dos bloques imprescindibles son el 2 y el 6.** Si el tiempo se acorta, se recorta el 3
> (basta mostrar la salida) y se acorta el 5 a la tabla comparativa.
>
> **Regla de oro de la clase, igual que en la 1.3:** ninguna afirmación sin una cifra que la
> respalde.
"""
    ),
    md(
        """
---
## Preparación del entorno

Ejecuta esta celda primero. Funciona tanto en Google Colab como en Jupyter local.

> Si estás en Colab, aparecerá el aviso *"Este cuaderno no lo ha creado Google"*. Es normal para
> cualquier notebook abierto desde GitHub: pulsa **"Ejecutar de todos modos"**.
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

print("Colab:", EN_COLAB)
print("Raíz del repositorio:", RAIZ)
print("¿Existe el dataset?:", RUTA_DATOS.exists())
"""
    ),
    code(
        """
import json
import sqlite3

import numpy as np
import pandas as pd

import eda       # utilidades de diagnóstico:   src/eda.py
import fuentes   # lectura desde fuentes varias: src/fuentes.py

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 140)

print("pandas", pd.__version__, "| numpy", np.__version__, "| sqlite", sqlite3.sqlite_version)
"""
    ),
    # ======================================================================
    # BLOQUE 1 — Ecosistema y tipos de aprendizaje
    # ======================================================================
    md(
        """
---
# Bloque 1 · Dónde estamos parados

La asignatura recorre el ciclo completo, y hoy estamos en el primer tramo:

```
Problema → DATOS → Exploración → Preprocesamiento → Modelamiento → Evaluación → Interpretación
           └ hoy ┘
```

### Los tres tipos de aprendizaje

| Tipo | Qué recibe | Qué busca | Ejemplo en este dominio |
|---|---|---|---|
| **Supervisado** | Datos etiquetados `(X, y)` | Predecir `y` para casos nuevos | Dado el tamaño y los puntos LiDAR, ¿es peatón o ciclista? |
| **No supervisado** | Solo `X`, sin etiquetas | Encontrar estructura oculta | ¿Hay grupos naturales de detecciones que se comporten distinto? |
| **Por refuerzo** | Un entorno y recompensas | Aprender una política de acción | Decidir cuándo frenar el vehículo |

La distinción no es teórica: **decide qué datos necesitas recolectar**. Un problema supervisado
exige etiquetas, y las etiquetas casi siempre las tiene que producir una persona. Eso cuesta
dinero y tiempo, y es la razón número uno por la que un proyecto de ML se cae antes de empezar.
"""
    ),
    md(
        """
### ✏️ TODO 1

Clasifica cada pregunta de negocio según el tipo de aprendizaje que corresponde. Completa el
diccionario y ejecuta el autochequeo.

*Pista: pregúntate si existe una respuesta correcta conocida para cada ejemplo del pasado. Si
existe, es supervisado.*
"""
    ),
    code(
        """
tipos_de_problema = {
    "Predecir si un objeto detectado es peatón, ciclista o vehículo": "supervisado",
    "Agrupar segmentos de conducción que se parezcan entre sí": "no supervisado",
    "Estimar la velocidad de un objeto a partir de sus cajas sucesivas": "supervisado",
    "Descubrir qué combinaciones de clima y hora producen detecciones raras": "no supervisado",
    "Decidir la maniobra del vehículo maximizando seguridad a lo largo del trayecto": "refuerzo",
}
""",
        """
# TODO 1: completa el tipo de aprendizaje de cada problema.
# Opciones: supervisado | no supervisado | refuerzo
tipos_de_problema = {
    "Predecir si un objeto detectado es peatón, ciclista o vehículo": "____",
    "Agrupar segmentos de conducción que se parezcan entre sí": "____",
    "Estimar la velocidad de un objeto a partir de sus cajas sucesivas": "____",
    "Descubrir qué combinaciones de clima y hora producen detecciones raras": "____",
    "Decidir la maniobra del vehículo maximizando seguridad a lo largo del trayecto": "____",
}
""",
    ),
    code(
        """
# Autochequeo
validos = {"supervisado", "no supervisado", "refuerzo"}
assert set(tipos_de_problema.values()) <= validos, f"solo se admiten: {validos}"
assert sum(v == "supervisado" for v in tipos_de_problema.values()) == 2, (
    "revisa: ¿en cuántos casos existe una respuesta correcta conocida para el pasado?"
)
assert sum(v == "refuerzo" for v in tipos_de_problema.values()) == 1, (
    "revisa: solo uno implica tomar decisiones secuenciales en un entorno"
)
print("✅ Los cinco problemas están bien clasificados.")
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — TODO 1
>
> **Respuesta:** supervisado / no supervisado / supervisado / no supervisado / refuerzo.
>
> **El que más se discute es el tercero.** Estimar velocidad *parece* no supervisado porque
> "solo hay cajas". Es supervisado: la velocidad medida por el sensor es la etiqueta `y`, y es
> **regresión**, no clasificación. Buen momento para introducir que supervisado se divide en
> clasificación (etiqueta categórica) y regresión (etiqueta numérica).
>
> **El cuarto también genera debate.** "Detecciones raras" suena a que hay una etiqueta de
> "rara". No la hay: nadie marcó cuáles lo son. Por eso es no supervisado (detección de
> anomalías). Si alguien hubiera revisado y marcado 5.000 detecciones como raras, pasaría a ser
> supervisado. **El tipo de problema no lo define la pregunta, lo definen los datos que tienes.**
>
> **Pregunta para el curso:** ¿cuánto costaría etiquetar a mano las 40.680 detecciones de este
> dataset? A 5 segundos cada una, son 56 horas de trabajo humano. Por eso el aprendizaje no
> supervisado no es el hermano pobre: muchas veces es lo único que el presupuesto permite.
>
> **Criterio de logro:** clasifica correctamente al menos 4 de 5 y justifica el criterio usado
> (¿existe etiqueta?), no solo el resultado.
"""
    ),
    # ======================================================================
    # BLOQUE 2 — Datos estructurados
    # ======================================================================
    md(
        """
---
# Bloque 2 · Datos estructurados

Un dato **estructurado** tiene esquema rígido: filas, columnas y un tipo por columna, definido
de antemano. Es el terreno cómodo de pandas.

| Formato | Tipo | Ventaja principal | Cuándo se usa |
|---|---|---|---|
| **CSV / TSV** | Texto plano | Universal, lo abre cualquier cosa | Intercambio, datasets pequeños y medianos |
| **JSON / XML** | Semiestructurado | Jerárquico y flexible | Respuestas de API, datos web |
| **SQL** | Estructurado | Consistencia, integridad, concurrencia | Bases corporativas, la fuente de verdad |
| **Parquet** | Binario columnar | Alta compresión y lectura por columnas | Volumen grande, análisis |
| **Excel** | Binario | Lo entiende el área de negocio | Reportes, carga manual |

Vamos a traer **los mismos datos** por tres vías y a comprobar que coinciden.
"""
    ),
    md(
        """
### Vía 1 — CSV local

La que ya conoces. Es también la que más engaña: un CSV no guarda tipos, solo texto, y pandas
tiene que **adivinar** el tipo de cada columna al leerlo.
"""
    ),
    code(
        """
df = pd.read_csv(RUTA_DATOS)
print(f"Filas: {df.shape[0]:,}   Columnas: {df.shape[1]}   Segmentos: {df['segment_id'].nunique()}")
df.head(3)
"""
    ),
    md(
        """
### ✏️ TODO 2 — Vía 2: leer por URL

En Colab no siempre vas a clonar un repositorio. Muchas veces el dato está publicado en una URL
y se lee directamente, sin descargarlo a mano.

Completa la lectura desde la URL *raw* de GitHub y comprueba que obtienes las mismas filas.

*Pista: `pd.read_csv` acepta una URL igual que acepta una ruta de archivo.*
"""
    ),
    code(
        f"""
URL_CSV = (
    "{URL_REPO.replace('github.com', 'raw.githubusercontent.com')}"
    "/main/datos/crudos/detecciones_waymo_like.csv"
)

try:
    df_url = pd.read_csv(URL_CSV)
    print(f"Leídas {{len(df_url):,}} filas desde la URL")
    print("¿Mismas dimensiones que el CSV local?:", df_url.shape == df.shape)
except Exception as error:
    # Sin conexión (o repositorio privado) el ejercicio no debe detener la clase.
    df_url = df.copy()
    print("No se pudo leer desde la URL:", type(error).__name__)
    print("Se sigue con el CSV local. La sintaxis es la misma.")
""",
        f"""
# TODO 2: lee el mismo dataset desde la URL raw de GitHub.
URL_CSV = (
    "{URL_REPO.replace('github.com', 'raw.githubusercontent.com')}"
    "/main/datos/crudos/detecciones_waymo_like.csv"
)

try:
    df_url = pd.____(____)
    print(f"Leídas {{len(df_url):,}} filas desde la URL")
    print("¿Mismas dimensiones que el CSV local?:", df_url.shape == df.shape)
except Exception as error:
    df_url = df.copy()
    print("No se pudo leer desde la URL:", type(error).__name__)
    print("Se sigue con el CSV local. La sintaxis es la misma.")
""",
    ),
    md(
        """
### Vía 3 — SQL

En una empresa, el dato **no** está en un CSV: está en una base de datos relacional, porque
varias personas escriben en ella al mismo tiempo y hace falta garantizar consistencia.

Para practicarlo no necesitamos instalar nada: `sqlite3` viene con Python y puede crear una base
**en memoria**, que existe mientras dure el notebook y desaparece al cerrarlo.
"""
    ),
    code(
        """
conexion = fuentes.a_sqlite(df, tabla="detecciones")

# ¿Qué tablas hay? La misma pregunta que le harías a una base real.
print(fuentes.consultar(conexion, "SELECT name FROM sqlite_master WHERE type='table'"))
print()
print(fuentes.consultar(conexion, "SELECT * FROM detecciones LIMIT 3"))
"""
    ),
    md(
        """
### ✏️ TODO 3 — Tu primera consulta

Escribe una consulta SQL que devuelva, **por tipo de objeto**, cuántas detecciones hay y cuál es
la velocidad promedio, ordenadas de más a menos frecuente.

*Pista: `SELECT columna, COUNT(*) AS n, AVG(otra) AS prom FROM tabla GROUP BY columna ORDER BY n DESC`.*
"""
    ),
    code(
        """
SQL = '''
SELECT object_type,
       COUNT(*)       AS n,
       AVG(speed_mps) AS velocidad_promedio
FROM detecciones
GROUP BY object_type
ORDER BY n DESC
'''
por_sql = fuentes.consultar(conexion, SQL)
por_sql
""",
        """
# TODO 3: completa la consulta SQL.
SQL = '''
SELECT object_type,
       ____        AS n,
       ____        AS velocidad_promedio
FROM detecciones
GROUP BY ____
ORDER BY n DESC
'''
por_sql = fuentes.consultar(conexion, SQL)
por_sql
""",
    ),
    md(
        """
### ✏️ TODO 4 — El mismo cálculo en pandas

Ahora haz **exactamente lo mismo** con `groupby`. Si SQL y pandas no coinciden, uno de los dos
está mal.

*Pista: `df.groupby(...).agg(n=("col", "size"), prom=("col", "mean"))`.*
"""
    ),
    code(
        """
por_pandas = (
    df.groupby("object_type")
    .agg(n=("object_type", "size"), velocidad_promedio=("speed_mps", "mean"))
    .sort_values("n", ascending=False)
    .reset_index()
)
por_pandas
""",
        """
# TODO 4: el mismo resultado del TODO 3, pero con pandas.
por_pandas = (
    df.groupby("____")
    .agg(n=("object_type", "____"), velocidad_promedio=("speed_mps", "____"))
    .sort_values("n", ascending=False)
    .reset_index()
)
por_pandas
""",
    ),
    code(
        """
# Autochequeo
assert por_sql["n"].tolist() == por_pandas["n"].tolist(), (
    "revisa: los conteos no coinciden. ¿Agrupaste por la misma columna?"
)
np.testing.assert_allclose(
    por_sql["velocidad_promedio"].to_numpy(dtype=float),
    por_pandas["velocidad_promedio"].to_numpy(dtype=float),
    rtol=1e-9,
    err_msg="los promedios no coinciden entre SQL y pandas",
)
print(f"✅ SQL y pandas dan el mismo resultado en las {len(por_sql)} categorías.")
print("   (Sí: son 7 categorías para 4 tipos de objeto. Ese es un problema de la Act. 1.3.)")
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — TODO 3 y 4
>
> **Respuesta:** ambos devuelven **7 filas**, no 4. `VEHICLE` 25.111, `PEDESTRIAN` 9.134,
> `SIGN` 3.303, `Pedestrian` 1.069, `PEATON` 811, `CYCLIST` 789, `Ped` 463.
>
> **Aprovecha el momento.** El ejercicio era sobre SQL, pero el resultado destapa el defecto de
> categorías inconsistentes que se trabaja en la Actividad 1.3. Vale la pena decirlo en voz
> alta: *cambiar de herramienta no arregla los datos*. La misma suciedad aparece en SQL, en
> pandas y en Power BI.
>
> **Detalle técnico que conviene mencionar:** `AVG(speed_mps)` en SQL ignora los `NULL`, igual
> que `.mean()` en pandas ignora los `NaN`. Por eso coinciden. Si alguien esperaba que SQL
> tratara los nulos distinto, es buen momento para aclararlo: la convención es la misma, y
> **ninguna de las dos te avisa** de que descartó filas.
>
> **Pregunta para el curso:** si SQL y pandas hacen lo mismo, ¿para qué SQL? Respuestas
> esperables: los datos ya viven ahí; el motor filtra 500 millones de filas y te entrega 3.000,
> mientras que pandas tendría que cargarlas todas en RAM; y varias personas pueden leer y
> escribir a la vez sin pisarse.
>
> **Criterio de logro:** escribe una consulta con `GROUP BY` correcta, reproduce el resultado en
> pandas y explica cuándo conviene cada herramienta.
"""
    ),
    # ======================================================================
    # BLOQUE 3 — Semiestructurados
    # ======================================================================
    md(
        """
---
# Bloque 3 · Datos semiestructurados (JSON anidado)

Una API casi nunca devuelve una tabla. Devuelve **JSON jerárquico**: diccionarios dentro de
diccionarios y listas dentro de diccionarios.

En el Waymo Open Dataset real, el contexto de cada segmento (clima, momento del día, conteos)
llega justamente así. Vamos a reconstruirlo a partir de nuestro dataset y a aplanarlo.
"""
    ),
    code(
        """
contexto = fuentes.contexto_por_segmento(df)

print(f"{len(contexto)} registros, uno por segmento. El primero se ve así:\\n")
print(json.dumps(contexto[0], indent=2, ensure_ascii=False))
"""
    ),
    md(
        """
### ✏️ TODO 5 — Aplanar el nivel superior

Fíjate en la estructura: `condiciones` es un **diccionario** dentro del registro y `objetos` es
una **lista**.

Compara qué pasa al construir el DataFrame de dos maneras:

1. con `pd.DataFrame(contexto)` — la vía ingenua;
2. con `pd.json_normalize(contexto)` — la vía correcta.

*Pista: mira las columnas que produce cada una y qué hay dentro de la celda `condiciones`.*
"""
    ),
    code(
        """
ingenuo = pd.DataFrame(contexto)
plano = pd.json_normalize(contexto)

print("pd.DataFrame       ->", list(ingenuo.columns))
print("pd.json_normalize  ->", list(plano.columns))
print()
print("Contenido de la celda 'condiciones' en la vía ingenua:")
print(" ", ingenuo.loc[0, "condiciones"], type(ingenuo.loc[0, "condiciones"]).__name__)
plano.head(3)
""",
        """
# TODO 5: compara las dos formas de construir el DataFrame.
ingenuo = pd.____(contexto)
plano = pd.____(contexto)

print("pd.DataFrame       ->", list(ingenuo.columns))
print("pd.json_normalize  ->", list(plano.columns))
print()
print("Contenido de la celda 'condiciones' en la vía ingenua:")
print(" ", ingenuo.loc[0, "condiciones"], type(ingenuo.loc[0, "condiciones"]).__name__)
plano.head(3)
""",
    ),
    code(
        """
# Autochequeo
assert "condiciones.weather" in plano.columns, (
    "revisa: json_normalize debería crear columnas con notación de punto"
)
assert isinstance(ingenuo.loc[0, "condiciones"], dict), (
    "revisa: en la vía ingenua la celda debería seguir conteniendo un diccionario"
)
print("✅ json_normalize aplanó el diccionario anidado; pd.DataFrame lo dejó dentro de la celda.")
"""
    ),
    md(
        """
### ✏️ TODO 6 — Expandir la lista

`json_normalize` aplanó `condiciones`, pero **dejó `objetos` como una lista dentro de la celda**.
Una lista en una celda no se puede filtrar, ni agrupar, ni graficar.

Para convertirla en filas hace falta `record_path` (qué lista se expande) y `meta` (qué campos
del nivel padre se arrastran).

*Pista: `pd.json_normalize(datos, record_path="lista", meta=["campo_del_padre"])`.*
"""
    ),
    code(
        """
objetos = pd.json_normalize(contexto, record_path="objetos", meta=["segment_id"])

print("Una fila por (segmento, tipo de objeto):", objetos.shape)
print("Total de detecciones reconstruido:", f"{objetos['n'].sum():,}")
objetos.head(5)
""",
        """
# TODO 6: expande la lista `objetos` a una fila por (segmento, tipo de objeto).
objetos = pd.json_normalize(contexto, record_path="____", meta=["____"])

print("Una fila por (segmento, tipo de objeto):", objetos.shape)
print("Total de detecciones reconstruido:", f"{objetos['n'].sum():,}")
objetos.head(5)
""",
    ),
    code(
        """
# Autochequeo
assert list(objetos.columns) == ["tipo", "n", "segment_id"], (
    "revisa: deberías obtener las columnas tipo, n y segment_id"
)
assert objetos["n"].sum() == len(df), (
    "revisa: al expandir la lista no se puede perder ni inventar ninguna detección"
)
print(f"✅ {len(objetos)} filas que suman exactamente las {len(df):,} detecciones originales.")
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — TODO 5 y 6
>
> **Respuesta del TODO 5:** `pd.DataFrame` produce 4 columnas y deja un `dict` **dentro** de la
> celda `condiciones`. `pd.json_normalize` produce 5 y crea `condiciones.weather` y
> `condiciones.time_of_day` como columnas de verdad.
>
> **Respuesta del TODO 6:** `record_path="objetos"`, `meta=["segment_id"]`. El `assert`
> comprueba que la suma de `n` da 40.680, es decir que no se perdió ninguna detección al
> expandir.
>
> **El error más frecuente** es aceptar la vía ingenua porque "el DataFrame se creó sin error".
> Ese es exactamente el problema: pandas **no falla**, te entrega un DataFrame con objetos de
> Python dentro. Todo se ve bien hasta que intentas `.groupby("condiciones.weather")` tres
> celdas más abajo. Vale la pena mostrar en vivo que `ingenuo["condiciones"].str.upper()`
> devuelve `NaN` en vez de fallar.
>
> **Pregunta para el curso:** ¿por qué la API no devuelve la tabla plana directamente y nos
> ahorra el trabajo? Porque la estructura anidada **no tiene pérdida**: un segmento con 3 tipos
> de objeto y otro con 5 caben en el mismo formato. La tabla plana obliga a decidir de antemano
> cuántas columnas hay.
>
> **Criterio de logro:** usa `json_normalize` con `record_path` y `meta`, y explica por qué la
> vía ingenua es insuficiente aunque no dé error.
"""
    ),
    # ======================================================================
    # BLOQUE 4 — No estructurados
    # ======================================================================
    md(
        """
---
# Bloque 4 · Datos no estructurados (texto libre)

El PPT dice que el 80 % de los datos que se generan hoy no está estructurado: texto, imágenes,
audio, video. La cifra se repite mucho y se practica poco.

Aquí tenemos un caso concreto y pequeño: los **partes de incidente** que escribe el operador al
final del turno. Son prosa, sin esquema. El dato útil —qué segmento quedó comprometido— está
enterrado en la frase.
"""
    ),
    code(
        """
partes = fuentes.generar_partes_incidente(df, n=12, semilla=42)

for i, parte in enumerate(partes[:4], start=1):
    print(f"[{i}] {parte}\\n")
"""
    ),
    md(
        """
### ✏️ TODO 7 — Extraer la estructura escondida

Los identificadores de segmento tienen la forma `seg_` seguida de cuatro dígitos. Escribe la
expresión regular que los encuentre y aplícala a todos los partes.

*Pista: `\\d` es un dígito y `{4}` significa "exactamente cuatro". Usa `re.findall`.*
"""
    ),
    code(
        """
import re

PATRON = r"seg_\\d{4}"

mencionados = []
for parte in partes:
    mencionados.extend(re.findall(PATRON, parte))

print("Menciones encontradas:", len(mencionados))
print("Segmentos distintos:", len(set(mencionados)))
print(sorted(set(mencionados)))
""",
        """
# TODO 7: escribe la expresión regular que captura los identificadores de segmento.
import re

PATRON = r"____"

mencionados = []
for parte in partes:
    mencionados.extend(re.findall(PATRON, parte))

print("Menciones encontradas:", len(mencionados))
print("Segmentos distintos:", len(set(mencionados)))
print(sorted(set(mencionados)))
""",
    ),
    md(
        """
### ✏️ TODO 8 — Cruzar lo no estructurado con lo estructurado

Aquí está el punto de todo el bloque: convertir el texto en una tabla y **cruzarla con el
DataFrame** para cuantificar el impacto.

Responde: ¿cuántas detecciones del dataset pertenecen a segmentos mencionados en algún parte de
incidente, y qué porcentaje del total representan?
"""
    ),
    code(
        """
tabla_menciones = fuentes.segmentos_comprometidos(partes)
comprometidas = df[df["segment_id"].isin(tabla_menciones["segment_id"])]

print(tabla_menciones.head())
print()
print(f"Segmentos comprometidos: {len(tabla_menciones)} de {df['segment_id'].nunique()}")
print(f"Detecciones afectadas:   {len(comprometidas):,} de {len(df):,} "
      f"({100 * len(comprometidas) / len(df):.1f} %)")
""",
        """
# TODO 8: cuantifica el impacto de los partes sobre el dataset.
tabla_menciones = fuentes.segmentos_comprometidos(partes)
comprometidas = df[df["segment_id"].____(tabla_menciones["segment_id"])]

print(tabla_menciones.head())
print()
print(f"Segmentos comprometidos: {len(tabla_menciones)} de {df['segment_id'].nunique()}")
print(f"Detecciones afectadas:   {len(comprometidas):,} de {len(df):,} "
      f"({100 * len(comprometidas) / len(df):.1f} %)")
""",
    ),
    code(
        """
# Autochequeo
assert len(tabla_menciones) > 0, "revisa: el patrón debería encontrar segmentos"
assert len(comprometidas) > 0, (
    "revisa: si el cruce da cero filas, los identificadores extraídos no existen en el dataset"
)
assert set(tabla_menciones["segment_id"]) <= set(df["segment_id"]), (
    "revisa: extrajiste identificadores que no están en el dataset"
)
print(f"✅ El texto libre se volvió una tabla cruzable: {len(comprometidas):,} detecciones marcadas.")
"""
    ),
    md(
        """
**✍️ Tu respuesta al TODO 8:**

*(doble clic aquí y escribe)*

¿Qué harías con esas detecciones antes de entrenar un modelo? ¿Las eliminarías, las marcarías
con una columna nueva, o las dejarías tal cual? Justifica.
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — TODO 7 y 8
>
> **Respuesta:** `PATRON = r"seg_\\d{4}"`. Con `n=12` y `semilla=42` se encuentran 12 menciones
> en 12 segmentos distintos, y el cruce marca del orden del 8 % de las detecciones (la cifra
> exacta depende de cuántas detecciones tenga cada segmento sorteado; que **no** sea 12/153 =
> 7,8 % exacto es en sí un hallazgo: los segmentos no tienen todos el mismo tamaño).
>
> **La respuesta esperada al TODO 8 es "marcarlas, no eliminarlas".** Se agrega una columna
> booleana `incidente_reportado` y se decide más tarde, con el modelo en la mano. Eliminar 3.000
> detecciones porque un operador escribió una nota es una decisión enorme tomada con evidencia
> mínima. Además, **el parte puede ser justamente la señal interesante**: si el LiDAR falla con
> lluvia, esas filas son las que hay que estudiar, no las que hay que botar.
>
> **Lo que hay que dejar claro del bloque:** el texto libre no se "analiza" mágicamente. Se le
> extrae una estructura (aquí, con una expresión regular; en otros casos con NLP) y **recién
> entonces** se cruza con lo demás. El 80 % no estructurado no es un dato que se use tal cual:
> es materia prima que hay que convertir.
>
> **Si sobra tiempo:** pregunta qué pasa si el operador escribe `Seg_0042` con mayúscula, o
> `segmento 42`. La regex falla en silencio y el cruce da menos filas sin avisar. Esa fragilidad
> es la razón por la que el texto libre es caro.
>
> **Criterio de logro:** escribe la regex correcta, ejecuta el cruce y argumenta la decisión
> distinguiendo entre marcar y eliminar.
"""
    ),
    # ======================================================================
    # BLOQUE 5 — Herramientas colaborativas
    # ======================================================================
    md(
        """
---
# Bloque 5 · Con qué trabaja un equipo

Ya sabes traer datos de cuatro sitios distintos. Ahora: ¿dónde vive ese trabajo cuando son cinco
personas y no una?

| Herramienta | Qué problema resuelve | Cuándo **no** usarla |
|---|---|---|
| **Google Colab** | Entorno con Python, pandas y GPU listos, sin instalar nada. Se comparte como un documento. | Cuando el proceso debe correr solo cada noche, o cuando el dato no puede salir de la empresa |
| **GitHub** | Historial de cambios, revisión entre pares, un solo lugar con la verdad del código | Para versionar datos pesados (para eso están DVC, LFS o un data lake) |
| **Kedro** | Convierte un notebook en un *pipeline* reproducible: nodos, dependencias declaradas y un catálogo de datos | Para una exploración de media hora; el andamiaje cuesta más que el análisis |
| **Databricks** | Ejecuta el mismo análisis sobre volúmenes que no caben en un computador, con Spark y almacenamiento Delta | Cuando los datos caben en RAM: pagar un clúster para 40.000 filas es tirar plata |

Las cuatro son complementarias, no alternativas. Un flujo profesional típico: se **explora** en
Colab, se **versiona** en GitHub, se **industrializa** con Kedro y se **escala** en Databricks.

> El notebook `04_opcional_kedro_databricks.ipynb` convierte el análisis de la Actividad 1.3 en
> un pipeline de Kedro que se ejecuta con un comando, y muestra qué cambiaría en Databricks.
"""
    ),
    code(
        """
# El repositorio que estás usando tiene historial de verdad. Míralo.
!git -C "{RAIZ}" log --oneline -8
"""
    ),
    md(
        """
### El flujo de trabajo del equipo

Esto es lo que se espera que hagan durante el proyecto. No es burocracia: es lo que evita que el
domingo a las 23:00 alguien sobrescriba el trabajo de otro.

```bash
# 1. Cada persona trabaja en su propia rama, nunca directo en main
git checkout -b eda-valores-nulos

# 2. Commits pequeños y con mensaje que explique el porqué
git add notebooks/analisis.ipynb
git commit -m "Documenta el patrón de nulos en la variable ingreso"

# 3. Subir y abrir un Pull Request para que alguien más lo revise
git push -u origin eda-valores-nulos
```

**Tres acuerdos que evitan el 90 % de los problemas:**

1. **Nadie hace `push` a `main`.** Todo entra por Pull Request, revisado por otra persona.
2. **Un notebook, un responsable.** Los `.ipynb` son casi imposibles de fusionar automáticamente
   porque guardan las salidas: si dos personas editan el mismo, hay conflicto seguro.
3. **Los datos pesados no se versionan.** Van a Drive o a un almacenamiento externo; en el
   repositorio queda el *código que los descarga*.
"""
    ),
    md(
        """
### ✏️ TODO 9 — El acuerdo de tu equipo

Rellena esta tabla con tu equipo. Es parte de la entrega.

| Rol | Quién | Responsabilidad concreta |
|---|---|---|
| Responsable del repositorio | | Crea el repo, aprueba los PR, mantiene el README |
| Responsable de los datos | | Documenta el origen, la licencia y la fecha de descarga |
| Responsable del EDA | | Ejecuta el diagnóstico de calidad y lo documenta |
| Responsable del informe | | Consolida los hallazgos y revisa la redacción |

**Nuestros acuerdos:**

- Rama por tarea, nombre con el formato: `____`
- Frecuencia de integración a `main`: `____`
- Dónde viven los datos que no van al repositorio: `____`
- Qué hacemos si dos personas necesitan tocar el mismo notebook: `____`
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 5
>
> **Este bloque no tiene código que corregir, tiene acuerdos que revisar.** Vale la pena pasar
> por los puestos y mirar la tabla: los equipos que dejan "Responsable de los datos" vacío son
> los que en la semana 4 no van a poder decir de dónde salió su dataset.
>
> **La celda de `git log` falla si el alumno trabaja local sin `git`.** No es grave; el objetivo
> es que vean historial real, y basta con proyectarlo desde el equipo docente.
>
> **El punto 2 del flujo merece una demostración.** Abre dos veces el mismo `.ipynb`, cambia una
> celda en cada copia y muestra el conflicto de Git: el `.ipynb` es JSON con las salidas
> incrustadas, y Git no sabe fusionarlo. Es el motivo real por el que este repositorio **genera**
> los notebooks desde archivos `.py` en vez de editarlos a mano.
>
> **Pregunta para el curso:** si Colab guarda todo automáticamente en Drive, ¿para qué GitHub?
> Porque Drive guarda la **última** versión, no la historia: no puedes volver al estado de hace
> tres días, ni ver quién cambió qué, ni revisar antes de integrar.
>
> **Sobre Kedro y Databricks:** aquí basta con que sepan qué problema resuelve cada uno. La
> pregunta que deben poder responder es *"¿cuándo NO usarlo?"*, porque es la que distingue a
> quien entendió de quien memorizó el nombre.
>
> **Criterio de logro:** el equipo define roles con responsabilidades concretas y acuerda un
> flujo de integración explícito.
"""
    ),
    # ======================================================================
    # BLOQUE 6 — Ética y sesgos
    # ======================================================================
    md(
        """
---
# Bloque 6 · La recolección no es neutral

Tres sesgos que se introducen **antes** de escribir una línea de modelo:

| Sesgo | Qué es | Cómo se ve |
|---|---|---|
| **De muestreo** | Los datos no representan proporcionalmente a la población objetivo | Un modelo excelente en promedio y pésimo con un grupo |
| **De confirmación** | Se recolecta solo lo que apoya la hipótesis previa | El análisis "confirma" lo que ya se creía |
| **De privacidad** | Se recolecta más de lo necesario, o sin consentimiento | Datos personales identificables donde no hacían falta |

### El caso medido: el Waymo Open Dataset real

No es un ejemplo hipotético. Se hizo el **censo completo** de los 798 segmentos del conjunto de
entrenamiento del Waymo Open Dataset (no una muestra: los 798), y el resultado está en
`docs/sesgo_waymo.md`:

| Condición | Segmentos | % |
|---|---|---|
| Soleado | **793** | **99,4 %** |
| Lluvia | **5** | **0,6 %** |

| Momento del día | Segmentos | % |
|---|---|---|
| Día | 647 | 81,1 % |
| Noche | 79 | 9,9 % |
| Amanecer/atardecer | 72 | 9,0 % |

| Ubicación | Segmentos | % |
|---|---|---|
| San Francisco | 409 | 51,3 % |
| Phoenix | 284 | 35,6 % |
| Otras | 105 | 13,2 % |

**Un vehículo autónomo entrenado con estos datos ha visto llover cinco veces.** Y ha aprendido a
conducir, sobre todo, en dos ciudades soleadas de Estados Unidos. Pregúntate qué pasa cuando ese
sistema llega a Santiago en junio.

Esto no es un error de Waymo: es una consecuencia de dónde están sus flotas. El error sería
**no declararlo** y presentar el modelo como si funcionara en todas partes.
"""
    ),
    code(
        """
# Nuestro dataset sintético hereda la misma forma. Mírala.
composicion = pd.DataFrame(
    {
        "detecciones": df["time_of_day"].value_counts(),
        "pct": (100 * df["time_of_day"].value_counts(normalize=True)).round(1),
    }
)
print(composicion, "\\n")

print("Tipos de objeto (tras unificar las variantes de escritura):")
tipos = eda.normalizar_categoria(
    df["object_type"], mapa={"peaton": "pedestrian", "ped": "pedestrian"}
)
print(eda.resumen_desbalance(tipos))
"""
    ),
    md(
        """
### ✏️ TODO 10 — Del sesgo a la consecuencia

Un sesgo de muestreo sin consecuencia medible es una frase bonita. Busca la consecuencia.

Calcula el **porcentaje de valores faltantes en `speed_mps` según el momento del día**. Después
responde: si elimináramos todas las filas con velocidad faltante, ¿a qué grupo estaríamos
borrando más?

*Pista: `eda.matriz_nulos_por_grupo(df, columna, grupos)`.*
"""
    ),
    code(
        """
nulos_por_momento = eda.matriz_nulos_por_grupo(df, "speed_mps", ["time_of_day"])
print(nulos_por_momento, "\\n")

peor = nulos_por_momento["pct_nulos"].idxmax()
mejor = nulos_por_momento["pct_nulos"].idxmin()
factor = nulos_por_momento.loc[peor, "pct_nulos"] / nulos_por_momento.loc[mejor, "pct_nulos"]
print(f"'{peor}' pierde {factor:.1f} veces más filas que '{mejor}' si se hace dropna().")
""",
        """
# TODO 10: ¿el dato faltante se reparte igual entre los grupos?
nulos_por_momento = eda.matriz_nulos_por_grupo(df, "____", ["____"])
print(nulos_por_momento, "\\n")

peor = nulos_por_momento["pct_nulos"].idxmax()
mejor = nulos_por_momento["pct_nulos"].idxmin()
factor = nulos_por_momento.loc[peor, "pct_nulos"] / nulos_por_momento.loc[mejor, "pct_nulos"]
print(f"'{peor}' pierde {factor:.1f} veces más filas que '{mejor}' si se hace dropna().")
""",
    ),
    code(
        """
# Autochequeo
assert peor == "Night", "revisa: ¿qué grupo concentra los valores faltantes?"
assert factor > 2, "revisa: la diferencia entre grupos debería ser grande, no marginal"
print(f"✅ El faltante NO es aleatorio: se concentra de noche ({factor:.1f}× más).")
print("   Un dropna() silencioso deja al modelo aún más ciego de noche de lo que ya estaba.")
"""
    ),
    md(
        """
### Privacidad: la lista de chequeo

Antes de usar cualquier fuente en tu proyecto, responde estas cinco preguntas. Si alguna respuesta
es "no sé", no la uses todavía.

1. **¿Contiene datos personales?** Nombre, RUT, correo, teléfono, dirección, patente, geolocalización
   fina, rostro o voz. En Chile los rige la Ley 19.628 y su reforma (Ley 21.719).
2. **¿Se puede reidentificar a alguien combinando columnas?** Comuna + edad + profesión suele bastar,
   aunque ninguna de las tres sea identificadora por sí sola.
3. **¿La licencia permite el uso que le voy a dar?** Uso académico, comercial, redistribución: son
   permisos distintos. *(El Waymo Open Dataset, por ejemplo, es de uso no comercial y no permite
   redistribuir los datos: por eso este repositorio usa un dataset sintético.)*
4. **¿Necesito todas las columnas?** Minimización: lo que no se recolecta no se filtra.
5. **¿Puedo declarar de dónde salió y cuándo?** Si no puedes documentar el origen, no puedes
   defender el resultado.
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — TODO 10 y bloque 6
>
> **Respuesta:** de noche falta el 4,08 % de las velocidades; de día, el 1,47 %; al amanecer o
> atardecer, el 0,91 %. La noche pierde **4,5 veces** más filas que el amanecer, y unas 2,8 veces
> más que el día.
>
> **La cadena de razonamiento que se busca:** de noche el LiDAR recibe menos puntos → la
> velocidad se estima peor → se registra como faltante más seguido → un `dropna()` elimina
> proporcionalmente más detecciones nocturnas → el modelo se entrena con aún menos noche de la
> poca que había → falla más de noche → y como el conjunto de prueba tiene el mismo sesgo, **la
> métrica no lo muestra**. Ese último eslabón es el que hay que subrayar.
>
> **El dato de Waymo es el que se les queda.** 793 segmentos soleados contra 5 con lluvia, sobre
> el censo completo de 798. Si en la clase entra una sola cifra, que sea esa. Funciona mejor
> preguntándolo antes de mostrarlo: *"¿cuántos de los 798 segmentos creen que tienen lluvia?"*
> Nadie dice 5.
>
> **Sobre el sesgo de confirmación**, un ejemplo del propio dominio: si el equipo decide de
> antemano que "el sensor funciona bien" y solo audita los segmentos diurnos y soleados, va a
> encontrar exactamente lo que fue a buscar. El diseño del muestreo determina la conclusión.
>
> **Cuidado con una respuesta cómoda que suele aparecer:** "entonces imputamos con la media y
> listo". Imputar con la media global mete velocidad diurna en filas nocturnas: no elimina el
> sesgo, lo disfraza y además borra la señal de que el sensor tiene problemas de noche. La
> alternativa razonable es imputar por grupo o, mejor, agregar una columna que marque el
> faltante. Eso se trabaja en la semana de preprocesamiento.
>
> **Criterio de logro:** cuantifica el sesgo con una cifra, encadena sesgo → decisión técnica →
> consecuencia sobre un grupo concreto, y propone una alternativa al `dropna()`.
"""
    ),
    # ======================================================================
    # CIERRE
    # ======================================================================
    md(
        """
---
# Cierre · Ficha de fuentes de tu proyecto

Esta es la entrega de la Actividad 1.1 y el punto de partida del proyecto de equipo. Rellénala
con tu grupo y cópiala al notebook `10_proyecto_equipo_plantilla.ipynb`.

---

## Ficha de fuentes de datos

**Equipo:** `____`
**Caso de negocio:** `____` *(retail, banca, salud, educación, transporte, otro)*
**Pregunta que queremos responder:** `____`
**Tipo de aprendizaje que corresponde:** `____` *(y por qué)*

### Fuente 1

| Campo | Valor |
|---|---|
| Nombre y origen (URL) | |
| Tipo | estructurada / semiestructurada / no estructurada |
| Formato | CSV / JSON / SQL / Parquet / Excel / texto / imagen |
| Tamaño aproximado | |
| Licencia y si permite nuestro uso | |
| Fecha de descarga | |
| ¿Contiene datos personales? | |

### Fuente 2

| Campo | Valor |
|---|---|
| Nombre y origen (URL) | |
| Tipo | |
| Formato | |
| Tamaño aproximado | |
| Licencia y si permite nuestro uso | |
| Fecha de descarga | |
| ¿Contiene datos personales? | |

### Fuente 3

| Campo | Valor |
|---|---|
| Nombre y origen (URL) | |
| Tipo | |
| Formato | |
| Tamaño aproximado | |
| Licencia y si permite nuestro uso | |
| Fecha de descarga | |
| ¿Contiene datos personales? | |

### Riesgo de sesgo o privacidad detectado

*(Un riesgo concreto, en una de las tres fuentes, con la consecuencia que tendría sobre un grupo
específico. No vale "podría haber sesgo".)*

`____`

### Acuerdo de trabajo del equipo

*(La tabla de roles y los cuatro acuerdos del TODO 9.)*

`____`
"""
    ),
    md_docente(
        """
> ### 🎓 Criterios de logro — Actividad 1.1 (IL 1.1)
>
> | Nivel | Descripción |
> |---|---|
> | **Destacado (4)** | Trae datos desde las tres naturalezas de fuente sin ayuda; la ficha identifica un riesgo de sesgo **con su consecuencia sobre un grupo concreto**; el acuerdo de equipo es específico y ejecutable |
> | **Logrado (3)** | Completa los 9 TODO con autochequeo en verde; la ficha tiene las 3 fuentes con licencia y fecha; el riesgo está identificado aunque quede general |
> | **En desarrollo (2)** | Resuelve las fuentes estructuradas pero se pierde en JSON anidado o en el texto libre; la ficha tiene vacíos en licencia u origen |
> | **Inicial (1)** | Solo lee el CSV; la ficha no distingue tipos de fuente |
>
> **Errores que conviene anticipar en voz alta antes de empezar:**
>
> 1. Confundir *formato* con *tipo*: JSON es un formato; semiestructurado es el tipo.
> 2. Copiar la URL de la página web en vez de la URL *raw* del archivo. `pd.read_csv` sobre la
>    página de GitHub descarga HTML, no datos.
> 3. Anotar "licencia: Kaggle". Kaggle no es una licencia; cada dataset tiene la suya y hay que
>    ir a buscarla.
>
> **Enlace con el resto de la EA1:** la ficha de fuentes es el insumo del proyecto de equipo. Sin
> ella, en la Actividad 1.2 no tienen sobre qué trabajar.
"""
    ),
]
