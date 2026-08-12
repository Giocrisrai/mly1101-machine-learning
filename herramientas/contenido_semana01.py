"""Fuente única del contenido de la Semana 01 (EA1) de MLY1101.

De este archivo salen los dos notebooks:

- ``notebooks/01_alumno_exploracion.ipynb``  (versión con TODO)
- ``notebooks/01_docente_solucionario.ipynb`` (versión resuelta con pauta)

Cada celda es un diccionario:

- ``md(texto)``            -> markdown en ambos notebooks
- ``md_docente(texto)``    -> markdown solo en el solucionario (respuestas, criterios)
- ``code(solucion)``       -> código idéntico en ambos (setup, celdas de apoyo)
- ``code(solucion, todo)`` -> el alumno recibe ``todo``; el docente, ``solucion``

Regenerar los notebooks tras editar este archivo:

    python herramientas/construir_notebooks.py
"""

from __future__ import annotations

URL_REPO = "https://github.com/Giocrisrai/mly1101-machine-learning"


def md(texto: str) -> dict:
    return {"tipo": "md", "fuente": texto.strip("\n")}


def md_docente(texto: str) -> dict:
    return {"tipo": "md", "fuente": texto.strip("\n"), "solo_docente": True}


def code(solucion: str, todo: str | None = None) -> dict:
    celda = {"tipo": "code", "fuente": solucion.strip("\n")}
    if todo is not None:
        celda["todo"] = todo.strip("\n")
    return celda


CELDAS: list[dict] = [
    # ======================================================================
    # BLOQUE 0 — El problema antes del algoritmo (15 min)
    # ======================================================================
    md(
        f"""
# MLY1101 · Machine Learning — Semana 01
## EA1 · Análisis y Preprocesamiento de Datos

**Resultado de aprendizaje (RA1):** implementar estrategias y técnicas de preprocesamiento
en el diseño de soluciones de Machine Learning, con un tratamiento responsable de la información.

---

### La idea central de hoy

Un proyecto de Machine Learning **no empieza eligiendo un algoritmo**. Empieza entendiendo
el problema y mirando los datos:

```
Problema → Datos → Exploración → Preprocesamiento → Modelamiento → Evaluación → Interpretación
           └────────── aquí estamos hoy ──────────┘
```

Hoy no vamos a entrenar ningún modelo. Vamos a hacer algo que decide el éxito o el fracaso
del modelo que entrenaremos más adelante: **entender y limpiar los datos**.

> Un modelo entrenado con datos que nadie revisó no es un modelo: es una opinión con decimales.

---

### El problema

Trabajas en el equipo de percepción de una empresa de conducción autónoma. El vehículo lleva
un sensor **LiDAR** que, varias veces por segundo, detecta objetos alrededor y entrega para
cada uno una *caja delimitadora* (bounding box) con su posición, tamaño y velocidad estimada.

El equipo de modelamiento quiere entrenar un clasificador que distinga **peatones, ciclistas,
vehículos y señalética**. Antes de gastar una sola hora en eso, alguien tiene que responder:

> **¿Podemos confiar en estas detecciones? ¿Qué tan sucios están los datos y qué habría que
> arreglar antes de modelar?**

Ese alguien eres tú, hoy.

---

### Sobre los datos

El archivo `detecciones_waymo_like.csv` es un **dataset sintético** generado para esta clase.
Usa **el mismo esquema** del componente `lidar_box` del
[Waymo Open Dataset v2](https://waymo.com/open/), un conjunto de datos real de conducción
autónoma. Es sintético por dos razones honestas:

1. Los datos reales de Waymo pesan varios GB y su licencia no permite redistribuirlos.
2. Nos permite garantizar que los problemas de calidad que hay que descubrir **están ahí**.

Si quieres repetir este mismo análisis sobre datos **reales** de Waymo, el notebook
`00_opcional_waymo_real.ipynb` explica cómo hacerlo: el código de este notebook funciona igual,
porque el esquema es el mismo.

---

### Al final de la sesión debes entregar

Un **mini-informe en Markdown** (última celda del notebook) con:

- 5 hallazgos sobre la calidad de los datos, cada uno respaldado con una cifra;
- 3 decisiones de preprocesamiento, cada una con su justificación;
- 1 riesgo ético o de sesgo identificado en el dataset.
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente
>
> **Cómo usar este documento.** Este es el solucionario del notebook
> `01_alumno_exploracion.ipynb`. Contiene el mismo contenido más: el código resuelto de cada
> TODO, las respuestas esperadas de cada pregunta de discusión (bloques `🎓 Pauta docente`) y
> los criterios de logro por bloque.
>
> **Distribución sugerida de las 4 horas:**
>
> | Bloque | Min | Foco |
> |---|---|---|
> | 0 · El problema | 15 | Encuadre. Que nadie abra sklearn hoy. |
> | 1 · Carga e inspección | 45 | `.info()`, `.dtypes`, memoria, primer diagnóstico |
> | 2 · Tipos de variables | 45 | Taxonomía + categorías inconsistentes |
> | 3 · Nulos y duplicados | 45 | Nulos ocultos, patrón MNAR, duplicado lógico |
> | 4 · Outliers | 45 | IQR vs z, imposible vs legítimo |
> | 5 · Decisiones | 30 | Tabla de decisiones + fuga de información |
> | 6 · Datos responsables | 20 | Sesgo de muestreo, datos personales |
> | Cierre | 15 | Mini-informe |
>
> **El dataset tiene exactamente 10 defectos inyectados**, listados en
> `src/generar_dataset.py::CATALOGO_DEFECTOS` y verificados por `pytest`. Si un grupo dice
> "los datos están limpios", tiene 10 cosas por encontrar.
>
> **Regla de oro de la clase:** ninguna afirmación sin una cifra que la respalde.
"""
    ),
    # ----------------------------------------------------------------------
    # Preparación del entorno
    # ----------------------------------------------------------------------
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

print("Colab:", EN_COLAB)
print("Raíz del repositorio:", RAIZ)
print("¿Existe el dataset?:", RUTA_DATOS.exists())
"""
    ),
    code(
        """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import eda  # utilidades de diagnóstico del repositorio: src/eda.py

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 140)
sns.set_theme(style="whitegrid")

print("pandas", pd.__version__, "| numpy", np.__version__)
"""
    ),
    # ======================================================================
    # BLOQUE 1 — Carga e inspección (45 min)
    # ======================================================================
    md(
        """
---
# Bloque 1 · Carga e inspección inicial

**Preguntas que debemos responder antes de tocar nada:**

1. ¿Cuántas filas y columnas hay? ¿Cuánta memoria ocupan?
2. ¿Qué representa **una fila**? (esta es la pregunta más importante y la que más se salta)
3. ¿El tipo de dato que pandas infirió para cada columna es el que corresponde?
"""
    ),
    code(
        """
df = pd.read_csv(RUTA_DATOS)
print(f"Filas: {df.shape[0]:,}   Columnas: {df.shape[1]}")
df.head()
"""
    ),
    md(
        """
### 📖 Diccionario de datos

| Columna | Significado |
|---|---|
| `segment_id` | Identificador del segmento de conducción (~20 s de grabación) |
| `timestamp_micros` | Instante de la detección, en microsegundos |
| `id_interno` | Identificador único de la detección |
| `object_type` | Tipo de objeto detectado |
| `box_center_x/y/z` | Centro de la caja, en metros, respecto del vehículo (x = adelante) |
| `box_length/width/height` | Dimensiones de la caja, en metros |
| `speed_mps` | Velocidad estimada del objeto, en m/s |
| `num_lidar_points` | Cantidad de puntos láser que cayeron sobre el objeto |
| `weather` | Condición climática del segmento |
| `time_of_day` | Momento del día |
| `detection_difficulty` | Dificultad de la detección según el sensor (LEVEL_1 = fácil) |
| `sensor_version` | Versión del firmware del sensor |

**Una fila = una detección de un objeto en un instante determinado.** No es un objeto, ni un
segmento, ni un vehículo. Ténlo presente: define qué significa "duplicado" más adelante.
"""
    ),
    md(
        """
### ✏️ TODO 1

Obtén, en una sola celda:

1. la estructura del DataFrame con `.info()`;
2. el uso de memoria **real** (`memory_usage(deep=True)`) en MB.
"""
    ),
    code(
        """
df.info()
memoria_mb = df.memory_usage(deep=True).sum() / 1024**2
print(f"\\nMemoria real: {memoria_mb:.1f} MB")
""",
        """
# TODO 1: estructura del DataFrame y memoria real en MB
df.____()

memoria_mb = df.memory_usage(____).sum() / 1024**2
print(f"\\nMemoria real: {memoria_mb:.1f} MB")
""",
    ),
    md(
        """
### ✏️ TODO 2 — El primer problema

Mira la salida anterior con atención. Hay una columna cuyo tipo **no es el que debería ser**.

1. Identifica qué columna es y por qué debería ser numérica.
2. Averigua qué valor la está ensuciando y cuántas filas lo tienen.

*Pista: `df["columna"].unique()` en una columna con miles de valores no sirve de mucho. Piensa
en qué le pasa a una columna numérica cuando aparece un texto.*
"""
    ),
    code(
        """
# timestamp_micros quedó como object: pandas no pudo inferir un tipo numérico.
convertidos = pd.to_numeric(df["timestamp_micros"], errors="coerce")
no_convertibles = df.loc[convertidos.isna(), "timestamp_micros"]

print("dtype actual:", df["timestamp_micros"].dtype)
print("Valores no convertibles a número:", len(no_convertibles))
print(no_convertibles.value_counts())
""",
        """
# TODO 2: ¿qué valor ensucia la columna y cuántas filas lo tienen?
convertidos = pd.to_numeric(df["____"], errors="coerce")
no_convertibles = df.loc[convertidos.isna(), "____"]

print("dtype actual:", df["____"].dtype)
print("Valores no convertibles a número:", len(no_convertibles))
print(no_convertibles.value_counts())
""",
    ),
    code(
        """
# Autochequeo
assert df["timestamp_micros"].dtype == object, "revisa: ¿estás mirando la columna correcta?"
assert len(no_convertibles) > 0, "deberías haber encontrado valores no convertibles"
print(f"✅ Hallazgo 1: {len(no_convertibles)} filas con un valor de texto en una columna numérica.")
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — TODO 2
>
> **Respuesta:** `timestamp_micros` tiene dtype `object` porque contiene el literal `"N/D"` en
> ~60 filas (0,15 %). Basta **un** valor de texto para que toda la columna deje de ser numérica.
>
> **Pregunta para el curso:** ¿por qué es peligroso? Porque `df["timestamp_micros"].mean()`
> falla, `.sort_values()` ordena alfabéticamente ("9" > "10") y cualquier resta de tiempos
> revienta. Un error de 60 filas rompe una columna de 40.000.
>
> **Error frecuente:** el alumno hace `df.dropna()` esperando que desaparezca. No desaparece:
> `"N/D"` **no es** un nulo para pandas, es un string perfectamente válido.
>
> **Criterio de logro:** identifica la columna, cuantifica el problema (60 filas) y explica por
> qué el dtype importa.
"""
    ),
    md(
        """
### Diagnóstico general

En vez de revisar columna por columna a mano, usamos `eda.resumen_calidad()`, que entrega una
radiografía completa: tipo, cardinalidad, nulos y **valores centinela** (valores que
representan un dato faltante sin ser `NaN`, como `-1` o `"N/D"`).
"""
    ),
    code(
        """
resumen = eda.resumen_calidad(df)
resumen
"""
    ),
    md(
        """
### ✏️ TODO 3

Usando la tabla anterior, responde en la celda de texto de abajo:

1. ¿Qué columna tiene **cardinalidad casi 100 %**? ¿Sirve como variable predictora? ¿Por qué?
2. ¿Qué columna es **constante**? ¿Qué aporta a un modelo?
3. ¿Qué columnas tienen nulos declarados (`NaN`) y cuáles tienen **nulos ocultos** (centinelas)?
"""
    ),
    code(
        """
print("Columnas de cardinalidad casi única (no son features):")
print(resumen[resumen["pct_unicos"] > 90][["dtype", "n_unicos", "pct_unicos"]], "\\n")

print("Columnas constantes (no aportan información):")
print(resumen[resumen["n_unicos"] <= 1][["dtype", "n_unicos", "ejemplos"]], "\\n")

print("Columnas con algo faltante (declarado u oculto):")
print(resumen[resumen["pct_faltante_total"] > 0][["n_nulos", "n_centinelas", "pct_faltante_total"]])
""",
        """
# TODO 3: filtra la tabla `resumen` para responder las tres preguntas.
# Pista: resumen[resumen["pct_unicos"] > 90], resumen[resumen["n_unicos"] <= 1], ...
print("Columnas de cardinalidad casi única (no son features):")
print(____)

print("\\nColumnas constantes (no aportan información):")
print(____)

print("\\nColumnas con algo faltante (declarado u oculto):")
print(____)
""",
    ),
    md(
        """
**✍️ Tu respuesta al TODO 3:**

*(doble clic aquí y escribe)*

1.
2.
3.
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — TODO 3
>
> 1. **`id_interno`** tiene ~98 % de valores únicos. No sirve como predictor: un identificador
>    no tiene relación causal con nada. Si se lo damos a un modelo con capacidad suficiente,
>    memoriza el identificador y el rendimiento en test se desploma. Sirve como llave, no como
>    feature. (`segment_id` es distinto: agrupa detecciones y **sí** sirve para razonar sobre
>    dependencia entre filas y para armar el split más adelante.)
> 2. **`sensor_version`** es constante (`v2.0.1`). Varianza cero ⇒ información cero. Se elimina.
>    Vale la pena preguntar al curso: ¿y si mañana llega la versión `v2.1`? Entonces sí importa,
>    y probablemente cambien las distribuciones. Documentarlo es parte del trabajo.
> 3. Nulos declarados: `speed_mps` (~2 %) y `weather` (~5 %). Nulos ocultos: `num_lidar_points`
>    con `-1` (~3 %) y `timestamp_micros` con `"N/D"`.
>
> **Punto clave del bloque:** `df.isna().sum()` **no** basta para auditar los faltantes.
"""
    ),
    # ======================================================================
    # BLOQUE 2 — Tipos de variables y categorías (45 min)
    # ======================================================================
    md(
        """
---
# Bloque 2 · Tipos de variables y categorías

El tipo que usa pandas (`int64`, `object`, …) no es lo mismo que el **tipo estadístico** de la
variable, y es el tipo estadístico el que decide qué se puede hacer con ella:

| Tipo estadístico | Definición | Ejemplo aquí | ¿Media? |
|---|---|---|---|
| **Nominal** | categorías sin orden | `object_type`, `weather` | ❌ |
| **Ordinal** | categorías con orden | `detection_difficulty` | ❌ (sí mediana) |
| **Discreta** | numérica, se cuenta | `num_lidar_points` | ✅ |
| **Continua** | numérica, se mide | `speed_mps`, `box_length` | ✅ |

`timestamp_micros` es un caso aparte: es numérica, pero su significado es **temporal**. Calcular
su promedio no tiene sentido; calcular diferencias, sí.
"""
    ),
    md(
        """
### ✏️ TODO 4

Completa el diccionario clasificando cada columna. Después ejecuta el autochequeo.
"""
    ),
    code(
        """
tipos_estadisticos = {
    "segment_id": "nominal",
    "timestamp_micros": "temporal",
    "id_interno": "identificador",
    "object_type": "nominal",
    "box_center_x": "continua",
    "box_center_y": "continua",
    "box_center_z": "continua",
    "box_length": "continua",
    "box_width": "continua",
    "box_height": "continua",
    "speed_mps": "continua",
    "num_lidar_points": "discreta",
    "weather": "nominal",
    "time_of_day": "nominal",
    "detection_difficulty": "ordinal",
    "sensor_version": "constante",
}
""",
        """
# TODO 4: completa el tipo estadístico de cada columna.
# Opciones: nominal | ordinal | discreta | continua | temporal | identificador | constante
tipos_estadisticos = {
    "segment_id": "nominal",
    "timestamp_micros": "____",
    "id_interno": "____",
    "object_type": "____",
    "box_center_x": "continua",
    "box_center_y": "____",
    "box_center_z": "____",
    "box_length": "____",
    "box_width": "____",
    "box_height": "____",
    "speed_mps": "____",
    "num_lidar_points": "____",
    "weather": "____",
    "time_of_day": "____",
    "detection_difficulty": "____",
    "sensor_version": "____",
}
""",
    ),
    code(
        """
# Autochequeo
faltantes = set(df.columns) - set(tipos_estadisticos)
assert not faltantes, f"faltan columnas por clasificar: {faltantes}"
assert "____" not in tipos_estadisticos.values(), "quedaron casilleros sin completar"
assert tipos_estadisticos["num_lidar_points"] == "discreta", "se cuentan puntos: es discreta"
assert tipos_estadisticos["detection_difficulty"] == "ordinal", "LEVEL_1 < LEVEL_2: hay orden"
print("✅ Clasificación completa y coherente.")
"""
    ),
    md(
        """
### El problema de las categorías

Ahora miremos qué categorías existen realmente en las variables nominales. Aquí es donde
aparecen los problemas que ningún `.info()` muestra.

### ✏️ TODO 5

Muestra la frecuencia de cada valor de `object_type` y de `weather`, **incluyendo los nulos**.
"""
    ),
    code(
        """
print(df["object_type"].value_counts(dropna=False), "\\n")
print(df["weather"].value_counts(dropna=False))
""",
        """
# TODO 5: frecuencias incluyendo nulos (revisa el parámetro dropna)
print(df["object_type"].value_counts(____), "\\n")
print(df["weather"].value_counts(____))
""",
    ),
    md(
        """
Cuenta las categorías que ves. ¿Cuántos tipos de objeto hay **en realidad**? ¿Cuántas condiciones
climáticas distintas existen **en realidad**?

### ✏️ TODO 6

Normaliza ambas columnas: quita espacios, unifica mayúsculas y traduce las variantes a una
forma canónica. Usa `eda.normalizar_categoria(serie, mapa)`.

*Ojo con `"RAIN "` y `" rain"`: los espacios son invisibles en pantalla pero `pandas` los cuenta
como categorías distintas.*
"""
    ),
    code(
        """
mapa_objetos = {"peaton": "pedestrian", "ped": "pedestrian"}
mapa_clima = {"soleado": "sunny", "lluvia": "rain", "niebla": "fog"}

df["object_type_limpio"] = eda.normalizar_categoria(df["object_type"], mapa_objetos)
df["weather_limpio"] = eda.normalizar_categoria(df["weather"], mapa_clima)

print(df["object_type_limpio"].value_counts(dropna=False), "\\n")
print(df["weather_limpio"].value_counts(dropna=False))
""",
        """
# TODO 6: define los mapas de equivalencia y normaliza.
# El mapa se aplica DESPUÉS de pasar a minúsculas y quitar espacios: escribe las llaves así.
mapa_objetos = {"peaton": "pedestrian", "____": "pedestrian"}
mapa_clima = {"soleado": "sunny", "____": "rain", "____": "fog"}

df["object_type_limpio"] = eda.normalizar_categoria(df["object_type"], ____)
df["weather_limpio"] = eda.normalizar_categoria(df["weather"], ____)

print(df["object_type_limpio"].value_counts(dropna=False), "\\n")
print(df["weather_limpio"].value_counts(dropna=False))
""",
    ),
    code(
        """
# Autochequeo
assert set(df["object_type_limpio"].dropna().unique()) == {"vehicle", "pedestrian", "cyclist", "sign"}, \\
    "deben quedar exactamente 4 tipos de objeto"
assert set(df["weather_limpio"].dropna().unique()) == {"sunny", "rain", "fog"}, \\
    "deben quedar exactamente 3 condiciones climáticas"
print("✅ 7 variantes de objeto → 4 categorías | 11 variantes de clima → 3 categorías")
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — TODOs 5 y 6
>
> **Lo que debe pasar:** el alumno cuenta 7 valores distintos en `object_type` y 11 en `weather`,
> y descubre que en realidad son 4 y 3.
>
> **La pregunta que hay que hacer:** *"Si el equipo de modelamiento entrena con esto tal cual,
> ¿qué pasa?"*
> Respuesta: un one-hot encoding genera 7 columnas para 4 clases reales. El modelo trata
> `PEATON` y `Pedestrian` como cosas distintas, reparte la evidencia entre categorías gemelas y
> aprende peor de cada una. Y si en producción llega `"Peaton"` (una variante nueva), no
> corresponde a ninguna columna aprendida.
>
> **El detalle de los espacios** vale la pena mostrarlo en vivo:
> `df["weather"].unique()` muestra `'RAIN '` y `'rain'` casi idénticos en pantalla. Sugerencia:
> ejecutar `[repr(v) for v in df["weather"].dropna().unique()]` para que los espacios se vean.
>
> **Advertencia metodológica:** normalizamos en columnas *nuevas* (`_limpio`) en vez de
> sobreescribir. Así el dataset original queda auditable y podemos comparar antes/después. Es
> una buena práctica que conviene explicitar.
>
> **Criterio de logro:** deja las 4 y 3 categorías correctas y explica el impacto en el modelo.
"""
    ),
    md(
        """
### Desbalance de clases

Con las categorías ya limpias, podemos ver algo que antes estaba oculto: cómo se reparten
las clases que el equipo quiere predecir.
"""
    ),
    code(
        """
desbalance = eda.resumen_desbalance(df["object_type_limpio"])
print(desbalance)

fig, ax = plt.subplots(figsize=(7, 3.5))
desbalance["pct"].sort_values().plot.barh(ax=ax, color="#4C72B0")
ax.set_xlabel("% de detecciones")
ax.set_ylabel("")
ax.set_title("Composición del dataset por tipo de objeto")
for i, valor in enumerate(desbalance["pct"].sort_values()):
    ax.text(valor + 0.7, i, f"{valor:.1f}%", va="center")
plt.tight_layout()
plt.show()
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — desbalance
>
> `CYCLIST` es ~2 % de las filas: hay ~30 vehículos por cada ciclista.
>
> **Preguntas para el curso:**
> - *"Si mi modelo predice siempre VEHICLE, ¿qué exactitud obtiene?"* → ~62 %. Y es inútil.
>   Aquí queda sembrada la discusión de EA2 sobre por qué el *accuracy* engaña.
> - *"¿Cuál es la clase donde equivocarse cuesta vidas?"* → ciclista y peatón. Justamente las
>   más difíciles de detectar y, en el caso del ciclista, la más escasa. El desbalance no es un
>   problema estadístico abstracto: es un problema de seguridad.
>
> **No corresponde todavía** hablar de SMOTE ni de `class_weight`. Basta con dejar registrado el
> hallazgo. Se retoma en EA2.
"""
    ),
    # ======================================================================
    # BLOQUE 3 — Nulos y duplicados (45 min)
    # ======================================================================
    md(
        """
---
# Bloque 3 · Datos faltantes y duplicados

Tres preguntas, en este orden:

1. ¿Cuántos faltan? (lo fácil)
2. ¿Están **escondidos** detrás de un valor válido? (lo que casi nadie revisa)
3. ¿Faltan **al azar** o siguen un patrón? (lo que decide qué podemos hacer con ellos)
"""
    ),
    md(
        """
### ✏️ TODO 7

Calcula, para cada columna, el número y el porcentaje de nulos declarados, mostrando solo las
columnas que tengan al menos uno.
"""
    ),
    code(
        """
nulos = pd.DataFrame({
    "n_nulos": df.isna().sum(),
    "pct": (100 * df.isna().mean()).round(2),
})
nulos[nulos["n_nulos"] > 0].sort_values("n_nulos", ascending=False)
""",
        """
# TODO 7: conteo y porcentaje de nulos por columna, solo las que tengan alguno.
nulos = pd.DataFrame({
    "n_nulos": df.____().sum(),
    "pct": (100 * df.____().mean()).round(2),
})
nulos[nulos["n_nulos"] > 0].sort_values("n_nulos", ascending=False)
""",
    ),
    md(
        """
### Nulos ocultos

`isna()` solo ve lo que pandas reconoce como faltante. Un dato faltante también puede estar
disfrazado de valor válido: `-1`, `0`, `-999`, `"N/D"`, `"sin dato"`.

### ✏️ TODO 8

`num_lidar_points` es un conteo de puntos láser. Por definición **no puede ser negativo**.
Averigua cuántas filas violan esa regla y qué valor usan.
"""
    ),
    code(
        """
print(df["num_lidar_points"].describe(), "\\n")
n_centinela = (df["num_lidar_points"] == -1).sum()
print(f"Filas con -1: {n_centinela:,} ({100 * n_centinela / len(df):.2f}%)")
print("Nulos que pandas ve en esa columna:", df["num_lidar_points"].isna().sum())
""",
        """
# TODO 8: ¿cuál es el mínimo de num_lidar_points? ¿tiene sentido? ¿cuántas filas lo tienen?
print(df["num_lidar_points"].____(), "\\n")
n_centinela = (df["num_lidar_points"] == ____).sum()
print(f"Filas con -1: {n_centinela:,} ({100 * n_centinela / len(df):.2f}%)")
print("Nulos que pandas ve en esa columna:", df["num_lidar_points"].isna().sum())
""",
    ),
    md(
        """
### ✏️ TODO 9

Crea las versiones corregidas de las dos columnas contaminadas, convirtiendo el valor centinela
en un `NaN` explícito:

- `num_lidar_points_limpio`: igual que la original, pero con `-1` → `NaN`.
- `timestamp_limpio`: la marca de tiempo convertida a número, con `"N/D"` → `NaN`.
"""
    ),
    code(
        """
df["num_lidar_points_limpio"] = df["num_lidar_points"].replace(-1, np.nan)
df["timestamp_limpio"] = eda.a_numerico(df["timestamp_micros"])

print(df[["num_lidar_points_limpio", "timestamp_limpio"]].isna().sum())
print("\\nTipos:", df["num_lidar_points_limpio"].dtype, "|", df["timestamp_limpio"].dtype)
""",
        """
# TODO 9: convierte los valores centinela en NaN explícitos.
df["num_lidar_points_limpio"] = df["num_lidar_points"].replace(____, np.nan)
df["timestamp_limpio"] = eda.a_numerico(df["____"])

print(df[["num_lidar_points_limpio", "timestamp_limpio"]].isna().sum())
print("\\nTipos:", df["num_lidar_points_limpio"].dtype, "|", df["timestamp_limpio"].dtype)
""",
    ),
    code(
        """
# Autochequeo
assert df["num_lidar_points_limpio"].isna().sum() > 0, "los -1 deben quedar como NaN"
assert (df["num_lidar_points_limpio"].dropna() > 0).all(), "no pueden quedar conteos negativos"
assert pd.api.types.is_numeric_dtype(df["timestamp_limpio"]), "el timestamp debe ser numérico"
print("✅ Nulos ocultos convertidos en nulos explícitos.")
"""
    ),
    md(
        """
### ¿Los nulos son aleatorios?

Esta es **la** pregunta del bloque. Tres escenarios posibles:

| Mecanismo | Significa | Consecuencia |
|---|---|---|
| **MCAR** | falta al azar puro | eliminar filas es (casi) inofensivo |
| **MAR** | la falta depende de *otras* variables observadas | se puede imputar condicionando |
| **MNAR** | la falta depende del *propio* valor faltante | eliminar **sesga** el dataset |

Veamos el caso de `speed_mps`.

### ✏️ TODO 10

Cruza el porcentaje de nulos de `speed_mps` por `detection_difficulty` y `time_of_day`. Usa
`eda.matriz_nulos_por_grupo(df, columna, [grupo1, grupo2])`.
"""
    ),
    code(
        """
patron = eda.matriz_nulos_por_grupo(df, "speed_mps", ["detection_difficulty", "time_of_day"])
print(patron, "\\n")

fig, ax = plt.subplots(figsize=(6.5, 3))
sns.heatmap(patron, annot=True, fmt=".1f", cmap="Reds", cbar_kws={"label": "% nulos"}, ax=ax)
ax.set_title("% de velocidad faltante según dificultad y momento del día")
plt.tight_layout()
plt.show()
""",
        """
# TODO 10: ¿el porcentaje de nulos es parejo entre grupos, o hay un patrón?
patron = eda.matriz_nulos_por_grupo(df, "____", ["____", "____"])
print(patron, "\\n")

fig, ax = plt.subplots(figsize=(6.5, 3))
sns.heatmap(patron, annot=True, fmt=".1f", cmap="Reds", cbar_kws={"label": "% nulos"}, ax=ax)
ax.set_title("% de velocidad faltante según dificultad y momento del día")
plt.tight_layout()
plt.show()
""",
    ),
    md(
        """
**✍️ Discusión (escribe tu respuesta):**

Si el equipo decide `df.dropna(subset=["speed_mps"])` antes de entrenar:

1. ¿Qué tipo de detecciones desaparecen del dataset?
2. ¿Qué le pasa al modelo entrenado con lo que queda cuando el auto circula **de noche**?
3. ¿Es esto MCAR, MAR o MNAR?

*(doble clic y responde)*
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — TODO 10 (el momento más importante de la clase)
>
> **Las cifras:** en `LEVEL_1` falta ~0,4 % de las velocidades sin importar la hora. En
> `LEVEL_2` de noche falta ~**34 %**. No es azar: es el sensor fallando justo cuando le cuesta.
>
> **Respuestas esperadas:**
> 1. Desaparecen casi solo las detecciones **difíciles y nocturnas**.
> 2. El modelo se entrena con un mundo más fácil y más iluminado que el real. Su desempeño
>    medido en validación será optimista y su desempeño real de noche, peor de lo esperado.
>    Y de noche es cuando más importa.
> 3. **MNAR** (o MAR según cómo se argumente, y ambas defensas son válidas si están
>    fundamentadas): la falta depende de condiciones que también afectan al valor mismo. Lo que
>    NO es, es MCAR, y eso es lo que hay que dejar claro.
>
> **La frase para cerrar el bloque:** *"Eliminar filas nunca es gratis. Siempre estás eligiendo
> qué parte de la realidad borrar."*
>
> **Extensión si sobra tiempo:** ¿qué haría el alumno en vez de eliminar? Opciones razonables:
> imputar por mediana **dentro de cada grupo** (tipo de objeto × dificultad), o agregar una
> columna indicadora `speed_faltante` que le diga al modelo que ahí no había medición. Esta
> última suele ser la mejor y casi nadie la propone sola.
>
> **Criterio de logro:** describe el patrón con cifras y conecta el faltante con un riesgo
> concreto de seguridad, no solo con "el modelo pierde datos".
"""
    ),
    md(
        """
### Duplicados

Recuerda: **una fila = una detección de un objeto en un instante**. Entonces, dos filas con el
mismo `id_interno` son, por definición, un error.

Hay dos tipos de duplicado y solo uno se resuelve con `drop_duplicates()`:

- **Duplicado exacto:** la fila completa está repetida.
- **Duplicado lógico:** se repite la *llave*, pero los demás valores difieren. `drop_duplicates()`
  no lo detecta, porque para pandas las filas son distintas.

### ✏️ TODO 11

Cuantifica ambos tipos usando `eda.reporte_duplicados(df, llave)` con la llave
`["segment_id", "timestamp_micros", "id_interno"]`, y muestra un ejemplo concreto de duplicado
lógico.
"""
    ),
    code(
        """
LLAVE = ["segment_id", "timestamp_micros", "id_interno"]
print(eda.reporte_duplicados(df, LLAVE), "\\n")

# Un ejemplo concreto: una llave repetida cuyas filas NO son idénticas.
repetidas = df[df.duplicated(subset=LLAVE, keep=False)]
for _, grupo in repetidas.groupby(LLAVE):
    if grupo.drop_duplicates().shape[0] > 1:
        display(grupo[LLAVE + ["object_type", "box_center_x", "num_lidar_points"]])
        break
""",
        """
# TODO 11: cuantifica duplicados exactos y lógicos, y muestra un ejemplo de duplicado lógico.
LLAVE = ["segment_id", "timestamp_micros", "id_interno"]
print(eda.reporte_duplicados(df, ____), "\\n")

repetidas = df[df.duplicated(subset=LLAVE, keep=False)]
for _, grupo in repetidas.groupby(LLAVE):
    if grupo.drop_duplicates().shape[0] > 1:   # el grupo tiene filas distintas entre sí
        display(grupo[LLAVE + ["object_type", "box_center_x", "num_lidar_points"]])
        break
""",
    ),
    code(
        """
# Autochequeo
reporte = eda.reporte_duplicados(df, LLAVE).iloc[0]
assert reporte["dup_exactos"] > 0, "hay duplicados exactos en el dataset"
assert reporte["dup_logicos"] > 0, "y también duplicados lógicos: drop_duplicates() no basta"
print(f"✅ {reporte['dup_exactos']} duplicados exactos + {reporte['dup_logicos']} duplicados lógicos.")
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — TODO 11
>
> **Cifras esperadas (dataset de 40.680 filas):** ~480 duplicados exactos y ~200 duplicados
> lógicos (680 por llave en total).
>
> **La demostración que conviene hacer en vivo:** ejecutar `len(df.drop_duplicates())` y mostrar
> que quedan ~200 filas con llave repetida. `drop_duplicates()` **no terminó el trabajo**.
>
> **Preguntas:**
> - *"¿Por qué existen duplicados lógicos en un sistema real?"* → reprocesamiento del mismo
>   segmento con otra versión del algoritmo, mezcla de dos exportaciones, reintentos tras una
>   caída de red. Es lo normal en un pipeline de datos, no una rareza.
> - *"¿Con cuál de las dos filas te quedas?"* → no hay respuesta única, y ese es el punto: hay
>   que **decidir y documentar** (la más reciente, la de más puntos láser, el promedio). Lo
>   inaceptable es no darse cuenta.
>
> **Criterio de logro:** distingue ambos tipos, los cuantifica y propone una regla de
> desempate justificada.
"""
    ),
    # ======================================================================
    # BLOQUE 4 — Outliers (45 min)
    # ======================================================================
    md(
        """
---
# Bloque 4 · Valores atípicos

Un valor atípico puede ser dos cosas muy distintas:

- un **error de medición** (el sensor falló) → hay que corregirlo o eliminarlo;
- un **caso real poco frecuente** (existe un bus) → eliminarlo es destruir información valiosa.

Los métodos estadísticos **no distinguen entre ambos**. Esa distinción la hace quien conoce el
dominio. Por eso este bloque no se trata de aplicar una fórmula, sino de mirar los datos.
"""
    ),
    code(
        """
numericas = ["box_center_x", "box_center_y", "box_center_z",
             "box_length", "box_width", "box_height", "speed_mps"]
eda.perfil_numerico(df, numericas)
"""
    ),
    md(
        """
### ✏️ TODO 12

Mira la fila de `speed_mps` en la tabla anterior: compara la mediana (`50%`) con el máximo.

Grafica la distribución de `speed_mps` con un histograma y un boxplot, y responde: ¿es plausible
el máximo? (1 m/s = 3,6 km/h).
"""
    ),
    code(
        """
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
df["speed_mps"].plot.hist(bins=80, ax=axes[0], color="#4C72B0")
axes[0].set_title("Distribución de speed_mps")
axes[0].set_xlabel("m/s")

df.boxplot(column="speed_mps", ax=axes[1], vert=False)
axes[1].set_title("Boxplot de speed_mps")
plt.tight_layout()
plt.show()

maximo = df["speed_mps"].max()
print(f"Máximo observado: {maximo:.1f} m/s = {maximo * 3.6:.0f} km/h")
print(f"Detecciones sobre 60 m/s (216 km/h): {(df['speed_mps'] > 60).sum()}")
""",
        """
# TODO 12: histograma y boxplot de speed_mps; luego convierte el máximo a km/h.
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
df["____"].plot.hist(bins=80, ax=axes[0], color="#4C72B0")
axes[0].set_title("Distribución de speed_mps")
axes[0].set_xlabel("m/s")

df.boxplot(column="____", ax=axes[1], vert=False)
axes[1].set_title("Boxplot de speed_mps")
plt.tight_layout()
plt.show()

maximo = df["speed_mps"].____()
print(f"Máximo observado: {maximo:.1f} m/s = {maximo * 3.6:.0f} km/h")
print(f"Detecciones sobre 60 m/s (216 km/h): {(df['speed_mps'] > 60).sum()}")
""",
    ),
    md(
        """
### El criterio del rango intercuartil (IQR)

Se marca como atípico todo valor fuera del intervalo

$$[\\,Q_1 - k\\cdot IQR,\\;\\; Q_3 + k\\cdot IQR\\,], \\qquad IQR = Q_3 - Q_1$$

con $k = 1{,}5$ para atípicos moderados y $k = 3$ para extremos.

Una alternativa es el **puntaje z**: $z = (x - \\mu)/\\sigma$, atípico si $|z| > 3$. Pero $\\mu$ y
$\\sigma$ se calculan *con* los outliers incluidos, así que un valor extremo infla $\\sigma$ y se
esconde a sí mismo. El IQR, basado en cuantiles, es más robusto.

### ✏️ TODO 13

Compara ambos criterios sobre `speed_mps`: ¿cuántos valores marca cada uno?
"""
    ),
    code(
        """
por_iqr = eda.detectar_outliers_iqr(df["speed_mps"], k=1.5)
por_z = eda.detectar_outliers_zscore(df["speed_mps"], umbral=3)
inferior, superior = eda.limites_iqr(df["speed_mps"], k=1.5)

print(f"Límites IQR: [{inferior:.2f}, {superior:.2f}] m/s")
print(f"Marcados por IQR:     {por_iqr.sum():>5}")
print(f"Marcados por z-score: {por_z.sum():>5}")
print(f"Marcados por ambos:   {(por_iqr & por_z).sum():>5}")
""",
        """
# TODO 13: compara el criterio IQR con el z-score sobre speed_mps.
por_iqr = eda.detectar_outliers_iqr(df["speed_mps"], k=____)
por_z = eda.detectar_outliers_zscore(df["speed_mps"], umbral=____)
inferior, superior = eda.limites_iqr(df["speed_mps"], k=1.5)

print(f"Límites IQR: [{inferior:.2f}, {superior:.2f}] m/s")
print(f"Marcados por IQR:     {por_iqr.sum():>5}")
print(f"Marcados por z-score: {por_z.sum():>5}")
print(f"Marcados por ambos:   {(por_iqr & por_z).sum():>5}")
""",
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — TODOs 12 y 13
>
> **Cifras:** el máximo de `speed_mps` ronda los 338 m/s ≈ **1.218 km/h**. Un peatón a velocidad
> de avión comercial. ~160 detecciones superan los 60 m/s.
>
> **Punto clave del bloque:** el IQR marca **muchos más** valores que el z-score. Y aquí viene la
> trampa: el IQR marca a los buses (grandes pero reales) y el z-score deja pasar velocidades
> imposibles, porque esos mismos valores inflaron σ. **Ningún criterio automático sabe cuál es
> cuál.**
>
> Vale la pena escribirlo en la pizarra: *el umbral estadístico propone, el conocimiento del
> dominio dispone*.
"""
    ),
    md(
        """
### ✏️ TODO 14 — Atípico imposible vs. atípico legítimo

Ahora la parte que ninguna fórmula resuelve. Revisa los valores atípicos de `box_length`:

1. ¿Cuántos son **negativos**? ¿Puede existir un objeto de largo negativo?
2. Los objetos con largo mayor a 12 m, ¿son errores? Mira su ancho, su alto y su tipo antes de
   responder.
"""
    ),
    code(
        """
atipicos_largo = eda.detectar_outliers_iqr(df["box_length"])
print(f"Atípicos de box_length según IQR: {atipicos_largo.sum()}\\n")

print("--- Largo negativo (imposible) ---")
print(f"{(df['box_length'] < 0).sum()} filas\\n")

print("--- Largo > 12 m: ¿error o realidad? ---")
grandes = df[df["box_length"] > 12]
print(grandes[["object_type_limpio", "box_length", "box_width", "box_height"]].describe().round(2))
print("\\nTipos de objeto involucrados:", grandes["object_type_limpio"].unique())
""",
        """
# TODO 14: separa los atípicos imposibles de los legítimos.
atipicos_largo = eda.detectar_outliers_iqr(df["____"])
print(f"Atípicos de box_length según IQR: {atipicos_largo.sum()}\\n")

print("--- Largo negativo (imposible) ---")
print(f"{(df['box_length'] < ____).sum()} filas\\n")

print("--- Largo > 12 m: ¿error o realidad? ---")
grandes = df[df["box_length"] > 12]
print(grandes[["object_type_limpio", "box_length", "box_width", "box_height"]].describe().round(2))
print("\\nTipos de objeto involucrados:", grandes["____"].unique())
""",
    ),
    md(
        """
**✍️ Tu conclusión:** ¿qué harías con cada uno de los dos grupos y por qué?

*(doble clic y responde)*
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — TODO 14
>
> **Los negativos (~80 filas):** físicamente imposibles. Es una falla del sensor o del
> exportador. Se tratan como faltantes (`NaN`), no se "arreglan" tomando valor absoluto: no
> sabemos si el largo real era ese número.
>
> **Los mayores a 12 m (~600 filas):** son todos `VEHICLE`, con ancho ~2,6 m y alto ~3,2 m.
> Es decir: **buses y camiones**. Son reales, son exactamente el tipo de objeto que un auto
> autónomo no puede permitirse ignorar, y el IQR los marcó como atípicos.
>
> **La pregunta decisiva para el curso:** *"Si eliminamos todos los atípicos de `box_length`,
> ¿qué acabamos de hacer?"* → Entrenar un auto autónomo que nunca vio un bus.
>
> **Regla práctica que conviene dejar escrita:**
> 1. Definir primero las reglas de dominio (qué es físicamente imposible).
> 2. Tratar lo imposible como dato faltante.
> 3. Lo raro pero posible **se conserva**, y se documenta.
>
> **Criterio de logro:** el alumno propone tratamientos **distintos** para ambos grupos y
> justifica el porqué con el dominio, no con estadística.
"""
    ),
    md(
        """
### ✏️ TODO 15 — Reglas de dominio

En lugar de confiar en un umbral estadístico, escribamos explícitamente **qué es imposible**
en este dominio. Completa el diccionario de reglas: cada valor es una expresión que describe
las filas **inválidas**.
"""
    ),
    code(
        """
reglas = {
    "largo no positivo": "box_length <= 0",
    "alto no positivo": "box_height <= 0",
    "ancho no positivo": "box_width <= 0",
    "velocidad sobre 60 m/s (216 km/h)": "speed_mps > 60",
    "conteo de puntos negativo": "num_lidar_points < 0",
    "peatón más alto que 2.5 m": "object_type_limpio == 'pedestrian' and box_height > 2.5",
}
eda.valores_imposibles(df, reglas)
""",
        """
# TODO 15: completa las reglas de dominio (describen filas INVÁLIDAS).
reglas = {
    "largo no positivo": "box_length <= 0",
    "alto no positivo": "____",
    "ancho no positivo": "____",
    "velocidad sobre 60 m/s (216 km/h)": "speed_mps > ____",
    "conteo de puntos negativo": "____",
    "peatón más alto que 2.5 m": "object_type_limpio == 'pedestrian' and box_height > 2.5",
}
eda.valores_imposibles(df, reglas)
""",
    ),
    # ======================================================================
    # BLOQUE 5 — Decisiones de preprocesamiento (30 min)
    # ======================================================================
    md(
        """
---
# Bloque 5 · De los hallazgos a las decisiones

Encontrar problemas es la mitad del trabajo. La otra mitad es **decidir qué hacer con cada uno
y dejarlo documentado**, porque cada decisión cambia los datos con los que se entrenará el
modelo.

### ✏️ TODO 16

Completa esta tabla con tus decisiones. Es el corazón de tu entrega.

| Columna | Problema detectado | Cifra | Decisión | Justificación |
|---|---|---|---|---|
| `timestamp_micros` | Valor `"N/D"` fuerza dtype texto | 60 filas | Convertir a numérico, `"N/D"` → `NaN` | Se preserva la fila; solo se pierde el instante |
| `num_lidar_points` | `-1` como nulo oculto | | | |
| `weather` | | | | |
| `object_type` | | | | |
| duplicados | | | | |
| `box_length` | | | | |
| `speed_mps` | | | | |
| `sensor_version` | | | | |

*(doble clic para editar la tabla)*
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — TODO 16 (tabla de referencia)
>
> | Columna | Problema | Cifra | Decisión razonable | Justificación |
> |---|---|---|---|---|
> | `timestamp_micros` | `"N/D"` | 60 | `to_numeric(errors="coerce")` | No se pierde la fila completa por un campo |
> | `num_lidar_points` | `-1` centinela | ~1.200 (3 %) | `-1` → `NaN` + columna indicadora | El `-1` contaminaría cualquier promedio |
> | `weather` | 11 variantes + 5 % nulos | ~2.000 nulos | Normalizar; nulo como categoría `"desconocido"` | El nulo de clima puede ser informativo |
> | `object_type` | 7 variantes para 4 clases | ~2.300 filas | Normalizar a 4 categorías | Es la variable objetivo de EA2 |
> | duplicados | 480 exactos + 200 lógicos | 1,7 % | Eliminar exactos; para lógicos, regla explícita (p. ej. conservar el de más puntos láser) | Un duplicado sesga el entrenamiento y contamina el split |
> | `box_length` | negativos vs. buses | 80 vs. 600 | Negativos → `NaN`; buses **se conservan** | Solo lo imposible es error |
> | `speed_mps` | nulos MNAR + imposibles | 787 + 160 | Imposibles → `NaN`; **no** eliminar filas; imputar por grupo + indicador | Eliminar sesga contra la noche |
> | `sensor_version` | constante | 100 % | Eliminar la columna | Varianza cero |
>
> Se acepta cualquier decisión distinta **si está justificada**. Lo que no se acepta es
> `df.dropna()` sin argumento.
"""
    ),
    md(
        """
### ✏️ TODO 17 — Aplica tus decisiones

Escribe una función que reciba el DataFrame crudo y devuelva el limpio. Que sea una función y no
celdas sueltas importa: es reproducible, se puede testear y se puede volver a aplicar a datos
nuevos.
"""
    ),
    code(
        """
def limpiar(datos: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Aplica las decisiones de preprocesamiento acordadas y devuelve una copia limpia.\"\"\"
    d = datos.copy()

    # 1. Tipos y valores centinela
    d["timestamp_micros"] = eda.a_numerico(d["timestamp_micros"])
    d["num_lidar_points"] = d["num_lidar_points"].replace(-1, np.nan)

    # 2. Categorías
    d["object_type"] = eda.normalizar_categoria(d["object_type"], {"peaton": "pedestrian", "ped": "pedestrian"})
    d["weather"] = eda.normalizar_categoria(d["weather"], {"soleado": "sunny", "lluvia": "rain", "niebla": "fog"})
    d["weather"] = d["weather"].fillna("desconocido")

    # 3. Valores físicamente imposibles -> faltantes (NO se corrigen: no sabemos el valor real)
    for columna in ["box_length", "box_width", "box_height"]:
        d.loc[d[columna] <= 0, columna] = np.nan
    d.loc[d["speed_mps"] > 60, "speed_mps"] = np.nan

    # 4. Indicador de faltante: le dice al modelo que ahí no hubo medición
    d["speed_faltante"] = d["speed_mps"].isna().astype(int)

    # 5. Duplicados: primero los exactos, después los lógicos con una regla explícita
    #    (nos quedamos con la detección que tiene más puntos láser: es la mejor medición).
    d = d.drop_duplicates()
    d = (d.sort_values("num_lidar_points", ascending=False, na_position="last")
           .drop_duplicates(subset=["segment_id", "timestamp_micros", "id_interno"], keep="first")
           .sort_index())

    # 6. Columnas sin valor predictivo
    d = d.drop(columns=["sensor_version"])

    return d


df_limpio = limpiar(pd.read_csv(RUTA_DATOS))
print(f"Crudo:  {len(df):,} filas")
print(f"Limpio: {len(df_limpio):,} filas  ({len(df) - len(df_limpio):,} eliminadas)")
df_limpio.head(3)
""",
        """
# TODO 17: completa la función de limpieza con TUS decisiones del TODO 16.
def limpiar(datos: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Aplica las decisiones de preprocesamiento acordadas y devuelve una copia limpia.\"\"\"
    d = datos.copy()

    # 1. Tipos y valores centinela
    d["timestamp_micros"] = eda.a_numerico(d["timestamp_micros"])
    d["num_lidar_points"] = d["num_lidar_points"].replace(____, np.nan)

    # 2. Categorías
    d["object_type"] = eda.normalizar_categoria(d["object_type"], ____)
    d["weather"] = eda.normalizar_categoria(d["weather"], ____)
    d["weather"] = d["weather"].fillna("desconocido")

    # 3. Valores físicamente imposibles -> faltantes
    for columna in ["box_length", "box_width", "box_height"]:
        d.loc[d[columna] <= 0, columna] = np.nan
    d.loc[d["speed_mps"] > ____, "speed_mps"] = np.nan

    # 4. Indicador de faltante
    d["speed_faltante"] = d["speed_mps"].isna().astype(int)

    # 5. Duplicados: exactos y luego lógicos, con una regla de desempate explícita
    d = d.____()
    d = (d.sort_values("num_lidar_points", ascending=False, na_position="last")
           .drop_duplicates(subset=["segment_id", "timestamp_micros", "id_interno"], keep="first")
           .sort_index())

    # 6. Columnas sin valor predictivo
    d = d.drop(columns=[____])

    return d


df_limpio = limpiar(pd.read_csv(RUTA_DATOS))
print(f"Crudo:  {len(df):,} filas")
print(f"Limpio: {len(df_limpio):,} filas  ({len(df) - len(df_limpio):,} eliminadas)")
df_limpio.head(3)
""",
    ),
    code(
        """
# Autochequeo del dataset limpio
assert df_limpio.duplicated(subset=["segment_id", "timestamp_micros", "id_interno"]).sum() == 0, \\
    "no deben quedar llaves repetidas"
assert set(df_limpio["object_type"].unique()) == {"vehicle", "pedestrian", "cyclist", "sign"}
assert "sensor_version" not in df_limpio.columns
assert (df_limpio["box_length"].dropna() > 0).all(), "no deben quedar largos imposibles"
assert df_limpio["box_length"].max() > 12, "los buses deben SEGUIR AHÍ: no son errores"
assert pd.api.types.is_numeric_dtype(df_limpio["timestamp_micros"])
print("✅ Dataset limpio y auditable. Los casos raros pero reales siguen presentes.")
"""
    ),
    md(
        """
### ⚠️ Una advertencia para las próximas semanas: la fuga de información

Fíjate en algo que **no** hicimos: no imputamos los nulos con la media ni escalamos ninguna
variable.

No es un olvido. Si calculas la media de todo el dataset y con ella rellenas los nulos, y
**después** separas entrenamiento y prueba, el conjunto de prueba ya influyó en el conjunto de
entrenamiento a través de esa media. Eso se llama **fuga de información** (*data leakage*), y su
síntoma es un modelo que rinde excelente en las pruebas y mal en producción.

El orden correcto es:

```
limpieza estructural (lo de hoy)  →  separar train/test  →  ajustar imputación y escalado SOLO con train  →  aplicar a test
```

En EA2 haremos esto con `Pipeline` y `ColumnTransformer` de scikit-learn, que existen justamente
para que este error sea difícil de cometer.
"""
    ),
    # ======================================================================
    # BLOQUE 6 — Tratamiento responsable (20 min)
    # ======================================================================
    md(
        """
---
# Bloque 6 · Tratamiento responsable de la información

Los datos de conducción autónoma se recogen **en la vía pública**, donde hay personas que nunca
dieron su consentimiento. Antes de modelar, tres preguntas:

1. **¿Hay datos personales aquí?** El dataset no tiene nombres ni rostros, pero sí posiciones de
   peatones asociadas a un instante y un segmento. Si el segmento tiene geolocalización (los
   datos reales de Waymo la tienen), la combinación *lugar + hora + trayectoria* puede
   reidentificar a una persona. Anonimizar no es solo borrar la columna "nombre".
2. **¿Cómo se recolectó?** En los datos reales, con cámaras y LiDAR en vía pública. Waymo difumina
   rostros y patentes antes de publicar. Esa decisión es parte del diseño del dataset, no un
   detalle técnico.
3. **¿A quién representa mal este dataset?** Es la pregunta del ejercicio siguiente.

### ✏️ TODO 18

Calcula la composición del dataset por clima y momento del día, y el porcentaje de detecciones
que ocurren de noche o con lluvia.
"""
    ),
    code(
        """
composicion = pd.crosstab(df_limpio["weather"], df_limpio["time_of_day"], normalize="all") * 100
print(composicion.round(2), "\\n")

pct_noche = 100 * (df_limpio["time_of_day"] == "Night").mean()
pct_lluvia = 100 * (df_limpio["weather"] == "rain").mean()
pct_dificil_noche = 100 * ((df_limpio["time_of_day"] == "Night") &
                           (df_limpio["detection_difficulty"] == "LEVEL_2")).mean()

print(f"Detecciones nocturnas: {pct_noche:.1f}%")
print(f"Detecciones con lluvia: {pct_lluvia:.1f}%")
print(f"Nocturnas Y difíciles: {pct_dificil_noche:.1f}%")
print(f"Ciclistas: {100 * (df_limpio['object_type'] == 'cyclist').mean():.1f}%")
""",
        """
# TODO 18: composición por clima y hora; luego los porcentajes de noche, lluvia y ciclistas.
composicion = pd.crosstab(df_limpio["____"], df_limpio["____"], normalize="all") * 100
print(composicion.round(2), "\\n")

pct_noche = 100 * (df_limpio["time_of_day"] == "____").mean()
pct_lluvia = 100 * (df_limpio["weather"] == "____").mean()
pct_dificil_noche = 100 * ((df_limpio["time_of_day"] == "Night") &
                           (df_limpio["detection_difficulty"] == "LEVEL_2")).mean()

print(f"Detecciones nocturnas: {pct_noche:.1f}%")
print(f"Detecciones con lluvia: {pct_lluvia:.1f}%")
print(f"Nocturnas Y difíciles: {pct_dificil_noche:.1f}%")
print(f"Ciclistas: {100 * (df_limpio['object_type'] == 'cyclist').mean():.1f}%")
""",
    ),
    md(
        """
**✍️ Discusión final (escribe tu respuesta):**

Un modelo entrenado con este dataset se instalará en autos que circulan **de noche y con lluvia**,
y que se cruzan con **ciclistas**.

1. ¿Qué situación está sub-representada en los datos?
2. ¿Qué grupo de personas corre más riesgo si el modelo falla en esa situación?
3. Nombra **una** medida concreta que propondrías antes de desplegar este modelo.

*(doble clic y responde)*
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — TODO 18 y cierre ético
>
> **Cifras:** ~20 % nocturnas, ~21 % con lluvia, ~2 % ciclistas. La intersección
> *noche + difícil* es pequeña, y encima es justo donde más faltan las velocidades (bloque 3).
>
> **Las tres respuestas esperadas:**
> 1. La conducción nocturna con mala visibilidad, y en particular las detecciones difíciles en
>    esas condiciones. Es doble carencia: hay pocos datos **y** los que hay están incompletos.
> 2. Ciclistas y peatones: los usuarios más vulnerables de la vía. Un falso negativo con un
>    vehículo es un roce; con un peatón, no.
> 3. Se acepta cualquier medida concreta y accionable: recolectar más datos nocturnos y de
>    lluvia; **evaluar el modelo por subgrupo** (métricas separadas para noche/día,
>    ciclista/vehículo) en vez de una métrica global; ponderar las clases minoritarias;
>    restringir la operación a condiciones validadas hasta tener evidencia.
>
> **La idea que debe quedar:** un promedio global oculta a las minorías. Un modelo con 97 % de
> exactitud puede tener 60 % en ciclistas nocturnos, y ese 3 % de error no está repartido al
> azar: recae sobre quienes ya son más vulnerables. Esto conecta directamente con el RA1
> ("tratamiento responsable de la información") y se retoma en EA2 con las métricas por clase.
>
> **Conexión con la realidad, si hay tiempo:** vale la pena mencionar que los sistemas de
> conducción autónoma reales se validan por condición operacional (ODD, *operational design
> domain*) precisamente por esto.
"""
    ),
    # ======================================================================
    # CIERRE
    # ======================================================================
    md(
        """
---
# 📝 Entrega: mini-informe

Completa la celda siguiente. Es lo que entregas al final de la sesión.

**Reglas:**
- Cada hallazgo debe incluir **una cifra**. "Hay datos sucios" no es un hallazgo; "el 3 % de
  `num_lidar_points` usa `-1` como nulo oculto" sí lo es.
- Cada decisión debe incluir **una justificación**, no solo qué hiciste.
"""
    ),
    md(
        """
## Informe de calidad de datos — EA1

**Estudiante:**
**Fecha:**

### Contexto
*(¿qué problema se quiere resolver y qué representa una fila del dataset?)*

### 5 hallazgos

| # | Hallazgo | Cifra | Impacto en el modelo |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |

### 3 decisiones de preprocesamiento

| # | Decisión | Justificación | Qué se pierde |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |

### 1 riesgo ético o de sesgo


### Conclusión
*(en 3 líneas: ¿está este dataset listo para entrenar un modelo? ¿qué falta?)*
"""
    ),
    md(
        """
---
### ✅ Antes de cerrar

- [ ] Todas las celdas ejecutan sin error (Kernel → Restart & Run All).
- [ ] Los autochequeos muestran ✅.
- [ ] El mini-informe está completo, con cifras.
- [ ] Las celdas de discusión tienen tu respuesta escrita.

### Lo que viene

- **Semana 2:** técnicas de preprocesamiento aplicadas (imputación, codificación, escalado) y
  cómo evitar la fuga de información con `Pipeline`.
- **EA2:** aprendizaje supervisado — regresión y clasificación. Ahí veremos por qué ese 2 % de
  ciclistas es un problema serio.
- **EA3:** aprendizaje no supervisado — segmentación y reducción de dimensionalidad.

### ¿Quieres hacerlo con datos reales?

El notebook `00_opcional_waymo_real.ipynb` explica cómo bajar un fragmento del Waymo Open
Dataset real y correr **este mismo análisis** sobre él. El esquema es el mismo; el código, casi
idéntico.
"""
    ),
    md_docente(
        """
> ### 🎓 Criterios de logro de la sesión (resumen)
>
> | Nivel | Descripción |
> |---|---|
> | **Logrado** | Encuentra al menos 7 de los 10 defectos, los cuantifica, distingue outlier imposible de legítimo, identifica el patrón MNAR y justifica cada decisión de preprocesamiento. |
> | **En desarrollo** | Encuentra los defectos evidentes (nulos declarados, duplicados exactos) pero no los ocultos; aplica criterios estadísticos sin cuestionarlos. |
> | **Inicial** | Ejecuta el notebook sin interpretar; usa `dropna()` y `drop_duplicates()` sin justificar; el informe no tiene cifras. |
>
> **Los 10 defectos inyectados** (`src/generar_dataset.py::CATALOGO_DEFECTOS`):
>
> 1. `timestamp_micros` con `"N/D"` → dtype `object`
> 2. `num_lidar_points` con `-1` como nulo oculto (~3 %)
> 3. `weather` con 11 variantes para 3 categorías + 5 % nulos
> 4. `object_type` con 7 variantes para 4 categorías
> 5. Duplicados exactos (~1,2 %) y lógicos (~0,5 %)
> 6. Outliers imposibles: velocidad hasta 338 m/s, alto 0, largo negativo
> 7. Outliers legítimos: buses de 12 a 18 m (no eliminar)
> 8. Nulos MNAR en `speed_mps` (34 % en LEVEL_2 nocturno)
> 9. Desbalance: `CYCLIST` ~2 %
> 10. `sensor_version` constante e `id_interno` casi único
>
> **Señal de alerta durante la clase:** si un grupo termina el bloque 3 en 15 minutos,
> probablemente está ejecutando sin leer. Pregúntale por el porcentaje de nulos de `speed_mps`
> **en LEVEL_2 nocturno**: si no lo tiene, no miró el patrón.
"""
    ),
]
