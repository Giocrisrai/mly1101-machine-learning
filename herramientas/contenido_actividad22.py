"""Fuente única del contenido de la Actividad 2.2 — Modelamiento supervisado.

Indicador de logro **IL 2.2**: *construye modelos de aprendizaje supervisado para problemas
de regresión y clasificación según la naturaleza del caso.*

Alcance declarado: esta actividad cubre **clasificación**. La regresión, que el IL2.2 también
menciona, se trabaja sobre los casos oficiales (House Prices) en la Evaluación Parcial 2.

De este archivo salen dos notebooks:

- ``notebooks/05_alumno_supervisado.ipynb``   (versión con TODO)
- ``notebooks/05_docente_supervisado.ipynb``  (versión resuelta con pauta)

**Los notebooks reutilizan los nodos del pipeline** (``kedro_mly1101/``), no una copia.
Esos módulos solo importan pandas y ``src/eda.py``, así que funcionan en Colab sin
instalar Kedro. Es la misma regla de siempre: una sola verdad sobre los datos.

Todas las cifras de la pauta están medidas sobre ``detecciones_waymo_like.csv`` con la
semilla 42. Si cambia la semilla o ``--filas``, hay que volver a medirlas.

Regenerar tras editar:

    python herramientas/construir_notebooks.py
"""

from __future__ import annotations

from contenido_semana01 import URL_REPO, code, md, md_docente

CELDAS_ACT22: list[dict] = [
    # ======================================================================
    # ENCUADRE
    # ======================================================================
    md(
        """
# MLY1101 · Machine Learning — Actividad 2.2
## Modelamiento supervisado: predecir qué detecciones van a fallar

**Resultado de aprendizaje (RA2):** aplica modelos estadísticos al conjunto de datos
procesados para interpretarlos, utilizando metodologías ágiles, con la finalidad de obtener
conocimientos relevantes que permitan responder a las necesidades del contexto de negocio,
considerando aspectos éticos.

**Indicador de logro (IL 2.2):** construye modelos de aprendizaje supervisado para problemas
de regresión y clasificación según la naturaleza del caso.

> **Alcance de esta sesión:** trabajamos **clasificación**. La regresión se aborda sobre los
> casos oficiales de la asignatura (*House Prices*) en la Evaluación Parcial 2.

---

### De dónde venimos

En la EA1 respondimos *"¿podemos confiar en estos datos?"*. Ahora que están limpios, la
pregunta cambia:

```
Problema → Datos → Exploración → Preprocesamiento → MODELAMIENTO → Evaluación → Interpretación
                                                    └──── aquí estamos hoy ────┘
```

### La pregunta de hoy

> **¿Se puede anticipar qué detecciones van a ser difíciles**, a partir de la geometría del
> objeto y de dónde está?

Si se pudiera, el equipo de percepción sabría **de antemano** en qué situaciones no conviene
confiar en el sensor. Eso vale más que un modelo que clasifica objetos: es información
accionable sobre la propia incertidumbre del sistema.

---

### Por qué no clasificamos el tipo de objeto

Parecía lo natural, y es lo que casi todos proponen. Lo probamos: **se resuelve al 99,98 %**
con cualquier configuración. El generador sortea las dimensiones por tipo de objeto, así que
basta el largo de la caja para acertar.

Un ejercicio donde todo sale perfecto no enseña nada sobre evaluación. Y aprender a evaluar
es exactamente lo que se evalúa hoy.

---

### La idea central de hoy

> **Un modelo no se reporta con una cifra.**

Vas a entrenar un modelo que alcanza casi un 90 % de exactitud. También vas a descubrir que
un modelo que responde **siempre lo mismo, sin mirar los datos**, alcanza un 88,96 %.

Esos dos números juntos son la sesión completa.

---

### Al final de la sesión debes entregar

Un **informe de modelamiento** (última celda) con:

- la justificación de cómo partiste el dataset, con la cifra que lo respalda;
- la comparación de tu modelo contra el baseline, en **dos** métricas;
- el análisis por clase, señalando explícitamente a quién falla el modelo;
- una decisión argumentada: ¿pondrías este modelo en producción?
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Actividad 2.2
>
> **Cómo usar este documento.** Es el solucionario de `05_alumno_supervisado.ipynb`: mismo
> contenido más el código resuelto, las respuestas esperadas y los criterios de logro.
>
> **La actividad son 6 horas pedagógicas** según el programa. La distribución de abajo cubre
> unas 4 h de trabajo guiado; las 2 h restantes quedan para que apliquen lo mismo al caso
> oficial que hayan elegido (Telco, Housing o Spotify), que es sobre lo que se evalúa la
> Parcial 2.
>
> **Distribución del bloque guiado:**
>
> | Bloque | Min | Foco |
> |---|---|---|
> | 0 · Encuadre | 15 | Del diagnóstico a la predicción |
> | 1 · Del dato limpio al problema supervisado | 30 | `X`, `y`, y qué NO puede ser variable |
> | 2 · La partición ⭐ | 45 | Por qué no al azar; partir por grupo |
> | 3 · El baseline ⭐⭐ | 30 | **El bloque que da vuelta la clase** |
> | 4 · Entrenar | 30 | Un modelo es tres líneas; lo difícil es lo de al lado |
> | 5 · Evaluar por clase ⭐⭐ | 45 | Matriz de confusión y a quién le falla |
> | 6 · Fuga de información ⭐ | 30 | La variable que cuenta la respuesta |
> | 7 · Interpretar | 20 | Qué aprendió, y si tiene sentido |
> | Cierre · Informe | 15 | La decisión: ¿producción sí o no? |
>
> **Los bloques imprescindibles son el 3 y el 5.** Si el tiempo se acorta, se recorta el 7 y
> se comprime el 6 mostrando solo la tabla.
>
> **La sesión entera cuelga de una sorpresa**, y conviene no arruinarla: en el bloque 3 el
> baseline trivial saca **88,96 %** y el modelo del bloque 4 saca **89,65 %**. Siete décimas.
> No lo anticipes: deja que lo descubran ellos.
>
> **Regla de oro, la misma de la EA1:** ninguna afirmación sin una cifra que la respalde.
"""
    ),
    md(
        """
---
## Preparación del entorno

Ejecuta esta celda primero. Funciona en Google Colab y en Jupyter local.

> Fíjate en algo: no vamos a reescribir la limpieza de la EA1. Vamos a **importar las mismas
> funciones** que ya usamos entonces, que además son las que corren en el pipeline del
> repositorio. Una sola verdad sobre los datos.
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

sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "kedro_mly1101" / "src"))

RUTA_DATOS = RAIZ / "datos" / "crudos" / "detecciones_waymo_like.csv"
RUTA_PARAMETROS = RAIZ / "kedro_mly1101" / "conf" / "base" / "parameters.yml"

print("Colab:", EN_COLAB)
print("¿Existe el dataset?:", RUTA_DATOS.exists())
"""
    ),
    code(
        """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

import eda  # utilidades de diagnóstico de la EA1

# Los nodos del pipeline del repositorio. Solo necesitan pandas: no hace falta
# instalar Kedro para usarlos.
from kedro_mly1101.pipelines.preprocesamiento import nodes as limpieza
from kedro_mly1101.pipelines.supervisado import nodes as modelo

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 140)
sns.set_theme(style="whitegrid")

PARAMETROS = yaml.safe_load(RUTA_PARAMETROS.read_text(encoding="utf-8"))
print("Parámetros disponibles:", sorted(PARAMETROS))
"""
    ),
    # ======================================================================
    # BLOQUE 1
    # ======================================================================
    md(
        """
---
# Bloque 1 · Del dato limpio al problema supervisado

Un problema supervisado necesita tres cosas, y ninguna es el algoritmo:

| Pieza | Qué es | Aquí |
|---|---|---|
| **`X`** | Las variables predictoras | Geometría y posición del objeto |
| **`y`** | La etiqueta que se quiere predecir | `detection_difficulty` |
| **El grupo** | La unidad que no se puede partir | `segment_id` |

Esa tercera pieza no aparece en los tutoriales y es la que decide si el resultado es honesto.
La veremos en el bloque 2.
"""
    ),
    code(
        """
crudo = pd.read_csv(RUTA_DATOS)

# La limpieza de la EA1, aplicada paso a paso con los mismos nodos del pipeline.
paso = limpieza.normalizar_categorias(crudo, PARAMETROS["mapas_categorias"])
paso = limpieza.descubrir_faltantes(paso, PARAMETROS["centinelas"])
paso = limpieza.marcar_imposibles(paso, PARAMETROS["reglas_dominio"])
limpio = limpieza.quitar_duplicados_y_constantes(paso, PARAMETROS["columnas_a_descartar"])

print(f"Crudo  : {crudo.shape[0]:,} filas × {crudo.shape[1]} columnas")
print(f"Limpio : {limpio.shape[0]:,} filas × {limpio.shape[1]} columnas")
limpio.head(3)
"""
    ),
    md(
        """
### ✏️ TODO 1 — ¿Está desbalanceado?

Antes de elegir cualquier métrica hay que saber cómo se reparte la etiqueta. Calcula la
frecuencia absoluta y relativa de `detection_difficulty`.

*Pista: `eda.resumen_desbalance()` de la EA1 hace exactamente esto.*
"""
    ),
    code(
        """
distribucion = eda.resumen_desbalance(limpio["detection_difficulty"])
distribucion
""",
        """
# TODO 1: ¿cómo se reparte la etiqueta que queremos predecir?
distribucion = eda.____(limpio["____"])
distribucion
""",
    ),
    code(
        """
# Autochequeo
minoritaria = distribucion["pct"].idxmin()
pct_minoritaria = distribucion.loc[minoritaria, "pct"]
assert pct_minoritaria < 20, "revisa: ¿estás mirando la columna detection_difficulty?"
print(f"✅ Clase minoritaria: {minoritaria} con {pct_minoritaria:.1f} % de las filas.")
print("   Recuerda esa cifra: vuelve en el bloque 3 y decide toda la sesión.")
"""
    ),
    md(
        """
### ✏️ TODO 2 — Qué **no** puede ser una variable

Tres columnas de este dataset no pueden entrar en `X`, cada una por un motivo distinto.
Completa el diccionario diciendo por qué.

*Pista: piensa en qué pasaría si el modelo las usara. Una es un identificador, otra define los
grupos de la partición, y la tercera es de donde el sensor **deriva** la etiqueta.*
"""
    ),
    code(
        """
por_que_no_son_variables = {
    "id_interno": "identificador: casi único, el modelo lo memoriza y no generaliza",
    "segment_id": "es la unidad de agrupación para partir, no una propiedad del objeto",
    "num_lidar_points": "de aquí deriva la etiqueta: usarla es contarle la respuesta al modelo",
}
for columna, motivo in por_que_no_son_variables.items():
    print(f"{columna:20s} -> {motivo}")
""",
        """
# TODO 2: ¿por qué cada una de estas columnas NO puede ser una variable predictora?
por_que_no_son_variables = {
    "id_interno": "____",
    "segment_id": "____",
    "num_lidar_points": "____",
}
for columna, motivo in por_que_no_son_variables.items():
    print(f"{columna:20s} -> {motivo}")
""",
    ),
    md(
        """
### ✏️ TODO 3 — Armar la tabla de modelamiento

Usa `modelo.preparar_variables()` con los parámetros del pipeline. Comprueba después qué
columnas quedaron.
"""
    ),
    code(
        """
CONFIG = PARAMETROS["modelo"]
FUGA = PARAMETROS["fuga"]

tabla = modelo.preparar_variables(limpio, CONFIG, FUGA)

print("Variables predictoras:", CONFIG["variables"])
print("Etiqueta             :", CONFIG["objetivo"])
print("Grupo                :", CONFIG["grupo"])
print(f"\\nTabla de modelamiento: {tabla.shape[0]:,} filas × {tabla.shape[1]} columnas")
tabla.head(3)
""",
        """
# TODO 3: arma la tabla de modelamiento con los parámetros del pipeline.
CONFIG = PARAMETROS["____"]
FUGA = PARAMETROS["fuga"]

tabla = modelo.____(limpio, CONFIG, FUGA)

print("Variables predictoras:", CONFIG["variables"])
print("Etiqueta             :", CONFIG["objetivo"])
print("Grupo                :", CONFIG["grupo"])
print(f"\\nTabla de modelamiento: {tabla.shape[0]:,} filas × {tabla.shape[1]} columnas")
tabla.head(3)
""",
    ),
    code(
        """
# Autochequeo
assert CONFIG["objetivo"] in tabla.columns, "falta la etiqueta"
assert CONFIG["grupo"] in tabla.columns, "falta la llave de agrupación"
assert "num_lidar_points" not in CONFIG["variables"], (
    "revisa: num_lidar_points NO debe estar entre las variables predictoras"
)
assert "num_lidar_points" in tabla.columns, (
    "pero sí debe viajar en la tabla: la necesitamos en el bloque 6 para medir la fuga"
)
print("✅ Tabla lista. Fíjate en la sutileza: num_lidar_points está en la TABLA")
print("   pero no entre las VARIABLES. Volvemos a eso en el bloque 6.")
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 1
>
> **TODO 1.** `LEVEL_1` 88,89 % · `LEVEL_2` 11,11 %. Sobre el dataset limpio son 40.200 filas.
> Que digan la cifra en voz alta: **11 %**. La van a necesitar en 40 minutos.
>
> **TODO 2** es el que más discusión da, y vale la pena dejar hablar:
>
> - `id_interno`: cardinalidad casi única. Un modelo con capacidad suficiente lo memoriza y el
>   rendimiento en test se desploma. Es una llave, no una feature. Ya salió en la EA1.
> - `segment_id`: aquí muchos dicen "pero podría ser útil, hay segmentos más difíciles". Tienen
>   razón en la intuición y por eso mismo **no** puede ser variable: si el modelo aprende a
>   reconocer segmentos, en producción llegan segmentos nuevos que nunca vio y el modelo no
>   tiene nada que decir. Es la definición de no generalizar.
> - `num_lidar_points`: la dificultad **se deriva** de esta columna. Usarla no es informativo,
>   es contarle la respuesta. Se llama *fuga de la variable objetivo* y se mide en el bloque 6.
>
> **Si alguien pregunta por `weather` o `time_of_day`:** son legítimas y podrían usarse. No están
> en la lista porque son constantes dentro de un segmento, así que aportan la misma información
> que el segmento — y ya vimos por qué eso es problemático. Es una buena pregunta y merece
> reconocerse como tal.
>
> **Criterio de logro:** distingue las tres razones (identificador / unidad de agrupación /
> derivada de la etiqueta) sin confundirlas entre sí.
"""
    ),
    # ======================================================================
    # BLOQUE 2 — LA PARTICIÓN
    # ======================================================================
    md(
        """
---
# Bloque 2 · ⭐ La partición: dónde se gana o se pierde la honestidad

Todo el mundo sabe que hay que separar entrenamiento y prueba. Casi nadie se pregunta **cómo**.

La respuesta por defecto —`train_test_split` al azar— es correcta solo si las filas son
**independientes entre sí**. Aquí no lo son:

> Una fila es **una detección**. Un segmento son **~20 segundos de grabación** con decenas de
> detecciones que comparten clima, momento del día, ubicación y estado del sensor.

Si partes al azar, detecciones del mismo segmento caen en entrenamiento **y** en prueba. El
modelo puede reconocer el segmento en vez del objeto, y la métrica de prueba mide memoria, no
capacidad de generalizar.

Es **fuga de información por agrupación**, y no da ningún síntoma: el modelo no falla, saca
mejor nota.
"""
    ),
    md(
        """
### ✏️ TODO 4 — Partir por grupo

Usa `modelo.particionar()`, que agrupa por `segment_id`. Después comprueba cuántos segmentos
quedan a caballo entre las dos partes.
"""
    ),
    code(
        """
marcada = modelo.particionar(tabla, CONFIG)

entrena = marcada[marcada["particion"] == "entrenamiento"]
prueba = marcada[marcada["particion"] == "prueba"]

segmentos_entrena = set(entrena[CONFIG["grupo"]])
segmentos_prueba = set(prueba[CONFIG["grupo"]])

print(f"Entrenamiento: {len(entrena):,} filas ({len(entrena)/len(marcada):.1%})  "
      f"en {len(segmentos_entrena)} segmentos")
print(f"Prueba       : {len(prueba):,} filas ({len(prueba)/len(marcada):.1%})  "
      f"en {len(segmentos_prueba)} segmentos")
print(f"\\nSegmentos en AMBAS partes: {len(segmentos_entrena & segmentos_prueba)}")
""",
        """
# TODO 4: parte el dataset agrupando por segmento.
marcada = modelo.____(tabla, CONFIG)

entrena = marcada[marcada["particion"] == "entrenamiento"]
prueba = marcada[marcada["particion"] == "____"]

segmentos_entrena = set(entrena[CONFIG["grupo"]])
segmentos_prueba = set(prueba[CONFIG["grupo"]])

print(f"Entrenamiento: {len(entrena):,} filas ({len(entrena)/len(marcada):.1%})  "
      f"en {len(segmentos_entrena)} segmentos")
print(f"Prueba       : {len(prueba):,} filas ({len(prueba)/len(marcada):.1%})  "
      f"en {len(segmentos_prueba)} segmentos")
print(f"\\nSegmentos en AMBAS partes: {len(segmentos_entrena & segmentos_prueba)}")
""",
    ),
    md(
        """
### ✏️ TODO 5 — La partición ingenua, para comparar

Ahora la vía por defecto: al azar por fila. Cuenta los segmentos compartidos.
"""
    ),
    code(
        """
al_azar = modelo.particionar_al_azar(tabla, CONFIG)

az_entrena = set(al_azar.loc[al_azar["particion"] == "entrenamiento", CONFIG["grupo"]])
az_prueba = set(al_azar.loc[al_azar["particion"] == "prueba", CONFIG["grupo"]])

print(f"Partición por grupo   -> segmentos compartidos: {len(segmentos_entrena & segmentos_prueba)}")
print(f"Partición al azar     -> segmentos compartidos: {len(az_entrena & az_prueba)}")
""",
        """
# TODO 5: la partición al azar por fila, para poder comparar.
al_azar = modelo.____(tabla, CONFIG)

az_entrena = set(al_azar.loc[al_azar["particion"] == "entrenamiento", CONFIG["grupo"]])
az_prueba = set(al_azar.loc[al_azar["particion"] == "prueba", CONFIG["grupo"]])

print(f"Partición por grupo   -> segmentos compartidos: {len(segmentos_entrena & segmentos_prueba)}")
print(f"Partición al azar     -> segmentos compartidos: {len(az_entrena & az_prueba)}")
""",
    ),
    code(
        """
# Autochequeo
assert len(segmentos_entrena & segmentos_prueba) == 0, (
    "revisa: al partir por grupo NINGÚN segmento puede estar en las dos partes"
)
assert len(az_entrena & az_prueba) > 0, (
    "revisa: la partición al azar debería dejar segmentos compartidos"
)
print(f"✅ Por grupo: 0 segmentos compartidos.  Al azar: {len(az_entrena & az_prueba)}.")
"""
    ),
    md(
        """
### ✏️ TODO 6 — ¿Y cuánto daño hace?

Aquí viene lo interesante, y no es lo que esperas. Mide el efecto en vez de suponerlo.
"""
    ),
    code(
        """
comparacion = modelo.comparar_particiones(marcada, al_azar, CONFIG)
comparacion
""",
        """
# TODO 6: entrena con las dos particiones y compara.
comparacion = modelo.____(marcada, al_azar, CONFIG)
comparacion
""",
    ),
    md(
        """
**✍️ Tu respuesta al TODO 6:**

*(doble clic aquí y escribe)*

La diferencia entre las dos particiones es prácticamente nula. Entonces, ¿da igual cómo se
parta? Justifica tu respuesta.
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 2 ⭐
>
> **Cifras medidas:** por grupo, 29.946 filas de entrenamiento (74,5 %) en **114 segmentos** y
> 10.254 de prueba en **39**, con **0 compartidos**. Al azar, **153 segmentos compartidos** —
> todos.
>
> **El TODO 6 es una trampa deliberada, y hay que sostenerla.** La diferencia de exactitud es
> de **−0,005**: nada. Alguien va a decir, con toda lógica, *"entonces da igual"*.
>
> **No des la respuesta enseguida. Deja que lo defiendan.** Y después:
>
> *"Miren la columna `segmentos_compartidos`. La fuga está ahí: 153 contra 0. Lo que no
> aparece es su efecto. ¿Por qué?"*
>
> La razón es concreta: este dataset es **sintético** y el generador sortea cada detección de
> forma independiente dentro del segmento. La dependencia entre filas que la fuga explotaría
> **no existe aquí**. En datos reales de Waymo, donde los fotogramas consecutivos siguen al
> mismo objeto, sí existe.
>
> **La respuesta correcta es "no da igual", y el argumento no es la métrica:**
>
> > Un riesgo que no se manifiesta en tus datos de prueba sigue siendo un riesgo. La partición
> > por grupo se justifica por **cómo se generaron los datos**, no por la diferencia que se mide
> > hoy. Si eliges tu método por el resultado que te da, no estás midiendo: estás buscando la
> > respuesta que querías.
>
> **Este es el momento más valioso de la sesión para un alumno bueno.** Le estás enseñando que
> la evidencia empírica no siempre alcanza, y que hay decisiones que se toman por diseño.
>
> **Criterio de logro:** obtiene 0 segmentos compartidos, reconoce que la diferencia es nula y
> **aun así** defiende la partición por grupo con un argumento de diseño, no de métrica.
"""
    ),
    # ======================================================================
    # BLOQUE 3 — EL BASELINE
    # ======================================================================
    md(
        """
---
# Bloque 3 · ⭐⭐ El baseline: contra qué compites

Antes de entrenar nada, una pregunta incómoda:

> **¿Qué exactitud saca un modelo que no mira los datos?**

Un *baseline* es un modelo deliberadamente estúpido. `DummyClassifier` de scikit-learn tiene
varias estrategias; la más brutal es `most_frequent`: **responde siempre la clase mayoritaria**,
pase lo que pase.

No es un ejercicio retórico. Es la referencia sin la cual **ninguna métrica significa nada**.
"""
    ),
    md(
        """
### ✏️ TODO 7 — Antes de ejecutar, apuesta

Escribe tu predicción **antes** de correr la celda siguiente:

**Creo que el baseline `most_frequent` sacará una exactitud de:** `____ %`

*(No hagas trampa. Escríbelo primero.)*
"""
    ),
    code(
        """
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report

X_entrena = entrena[CONFIG["variables"]]
y_entrena = entrena[CONFIG["objetivo"]]
X_prueba = prueba[CONFIG["variables"]]
y_prueba = prueba[CONFIG["objetivo"]]

tonto = DummyClassifier(strategy="most_frequent", random_state=42)
tonto.fit(X_entrena, y_entrena)

reporte_tonto = classification_report(
    y_prueba, tonto.predict(X_prueba), output_dict=True, zero_division=0
)

print(f"Baseline 'responder siempre lo mismo':")
print(f"   exactitud : {reporte_tonto['accuracy']:.4f}")
print(f"   F1-macro  : {reporte_tonto['macro avg']['f1-score']:.4f}")
""",
        """
# TODO 7: entrena el baseline que responde siempre la clase mayoritaria.
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report

X_entrena = entrena[CONFIG["variables"]]
y_entrena = entrena[CONFIG["objetivo"]]
X_prueba = prueba[CONFIG["variables"]]
y_prueba = prueba[CONFIG["objetivo"]]

tonto = DummyClassifier(strategy="____", random_state=42)
tonto.fit(X_entrena, y_entrena)

reporte_tonto = classification_report(
    y_prueba, tonto.predict(X_prueba), output_dict=True, zero_division=0
)

print(f"Baseline 'responder siempre lo mismo':")
print(f"   exactitud : {reporte_tonto['accuracy']:.4f}")
print(f"   F1-macro  : {reporte_tonto['macro avg']['f1-score']:.4f}")
""",
    ),
    md(
        """
### ✏️ TODO 8 — ¿De dónde sale ese número?

El baseline saca casi un 89 % de exactitud **sin mirar una sola variable**. Comprueba que no es
casualidad: compara esa exactitud con la proporción de la clase mayoritaria en el conjunto de
prueba.
"""
    ),
    code(
        """
pct_mayoritaria = (y_prueba == y_prueba.value_counts().idxmax()).mean()

print(f"Exactitud del baseline          : {reporte_tonto['accuracy']:.4f}")
print(f"Proporción de la clase mayoritaria: {pct_mayoritaria:.4f}")
print(f"\\n¿Son el mismo número?: {np.isclose(reporte_tonto['accuracy'], pct_mayoritaria)}")
""",
        """
# TODO 8: ¿por qué el baseline saca justo ese número?
pct_mayoritaria = (y_prueba == y_prueba.value_counts().____()).mean()

print(f"Exactitud del baseline          : {reporte_tonto['accuracy']:.4f}")
print(f"Proporción de la clase mayoritaria: {pct_mayoritaria:.4f}")
print(f"\\n¿Son el mismo número?: {np.isclose(reporte_tonto['accuracy'], pct_mayoritaria)}")
""",
    ),
    code(
        """
# Autochequeo
assert np.isclose(reporte_tonto["accuracy"], pct_mayoritaria), (
    "revisa: la exactitud del baseline ES la proporción de la clase mayoritaria"
)
print("✅ La exactitud del baseline no es un resultado: es la proporción de la clase")
print("   mayoritaria, con otro nombre. En un problema desbalanceado, la exactitud")
print("   te dice cuán desbalanceado está, no cuán bueno es tu modelo.")
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 3 ⭐⭐
>
> **Este es el bloque que da vuelta la clase. No lo recortes nunca.**
>
> **Cifras medidas:** el baseline `most_frequent` obtiene **exactitud 0,8896** y
> **F1-macro 0,4708**.
>
> **El TODO 7 funciona solo si apuestan de verdad.** Insiste en que lo escriban. Las respuestas
> típicas rondan el 50 % —"es tonto, acertará la mitad"—. Cuando sale 89 %, la sala se calla.
> Es el mejor momento pedagógico de las cuatro horas y se pierde si lo adelantas.
>
> **La revelación del TODO 8** es que ese 0,8896 **es exactamente** la proporción de `LEVEL_1`
> en el conjunto de prueba. No hay ningún aprendizaje detrás: es aritmética.
>
> La frase que debe quedar, dicha tal cual:
>
> > *En un problema desbalanceado, la exactitud te dice cuán desbalanceado está el dataset, no
> > cuán bueno es tu modelo.*
>
> **Prepáralos para el bloque 4** sin adelantar la cifra: *"En el próximo bloque van a entrenar
> un modelo de verdad. Apunten ese 0,8896: es la vara."*
>
> **Si alguien pregunta por `strategy="stratified"`** (responder al azar respetando las
> proporciones), vale la pena mostrarlo: da exactitud **0,8061**, peor, pero F1-macro
> **0,5007**, mejor. Dos baselines que se ordenan al revés según la métrica. Es un excelente
> anticipo del bloque 5.
>
> **Criterio de logro:** explica por qué la exactitud del baseline coincide con la proporción de
> la clase mayoritaria, y qué implica eso para elegir métricas.
"""
    ),
    # ======================================================================
    # BLOQUE 4 — ENTRENAR
    # ======================================================================
    md(
        """
---
# Bloque 4 · Entrenar un modelo de verdad

Ahora sí. Un **bosque aleatorio**: muchos árboles de decisión entrenados sobre muestras
distintas, que votan.

Fíjate en algo mientras lo escribes: **entrenar son tres líneas**. Todo lo difícil de esta
sesión está alrededor —cómo partiste, contra qué comparas, cómo mides—, no aquí.

### Un detalle que no es decorativo: `class_weight="balanced"`

`LEVEL_2` es el 11 % de los datos. Sin compensar, el modelo aprende que **ignorar la clase
minoritaria casi no penaliza** el acierto global. Y en un sistema de percepción de un vehículo,
la clase minoritaria es justo la que importa.
"""
    ),
    md(
        """
### ✏️ TODO 9 — Entrenar

Usa `modelo.entrenar()`, que aplica los parámetros del pipeline.
"""
    ),
    code(
        """
clasificador = modelo.entrenar(marcada, CONFIG)

print("Modelo:", type(clasificador).__name__)
print("Árboles:", clasificador.n_estimators, "| profundidad máxima:", clasificador.max_depth)
print("Variables que vio:", list(clasificador.feature_names_in_))
""",
        """
# TODO 9: entrena el clasificador con la partición POR GRUPO.
clasificador = modelo.____(marcada, CONFIG)

print("Modelo:", type(clasificador).__name__)
print("Árboles:", clasificador.n_estimators, "| profundidad máxima:", clasificador.max_depth)
print("Variables que vio:", list(clasificador.feature_names_in_))
""",
    ),
    md(
        """
### ✏️ TODO 10 — El momento de la verdad

Compara tu modelo con el baseline. En **exactitud** y en **F1-macro**.
"""
    ),
    code(
        """
reporte = classification_report(
    y_prueba, clasificador.predict(X_prueba), output_dict=True, zero_division=0
)

resultados = pd.DataFrame(
    [
        {"modelo": "baseline (siempre lo mismo)",
         "exactitud": round(reporte_tonto["accuracy"], 4),
         "f1_macro": round(reporte_tonto["macro avg"]["f1-score"], 4)},
        {"modelo": "bosque aleatorio",
         "exactitud": round(reporte["accuracy"], 4),
         "f1_macro": round(reporte["macro avg"]["f1-score"], 4)},
    ]
)
resultados["mejora"] = (resultados["exactitud"] - resultados.loc[0, "exactitud"]).round(4)
resultados
""",
        """
# TODO 10: compara tu modelo con el baseline en DOS métricas.
reporte = classification_report(
    y_prueba, clasificador.____(X_prueba), output_dict=True, zero_division=0
)

resultados = pd.DataFrame(
    [
        {"modelo": "baseline (siempre lo mismo)",
         "exactitud": round(reporte_tonto["accuracy"], 4),
         "f1_macro": round(reporte_tonto["macro avg"]["____"], 4)},
        {"modelo": "bosque aleatorio",
         "exactitud": round(reporte["accuracy"], 4),
         "f1_macro": round(reporte["macro avg"]["____"], 4)},
    ]
)
resultados["mejora"] = (resultados["exactitud"] - resultados.loc[0, "exactitud"]).round(4)
resultados
""",
    ),
    code(
        """
# Autochequeo
ganancia_exactitud = reporte["accuracy"] - reporte_tonto["accuracy"]
ganancia_f1 = reporte["macro avg"]["f1-score"] - reporte_tonto["macro avg"]["f1-score"]

assert ganancia_exactitud < 0.02, (
    "revisa: la ganancia en exactitud debería ser pequeñísima. ¿Partiste por grupo?"
)
assert ganancia_f1 > 0.15, "revisa: en F1-macro la ganancia sí debería ser grande"
print(f"✅ Ganancia en exactitud: {ganancia_exactitud:+.4f}  ({ganancia_exactitud*100:+.2f} puntos)")
print(f"   Ganancia en F1-macro : {ganancia_f1:+.4f}")
print()
print("   Dos métricas sobre el MISMO modelo, con conclusiones opuestas.")
print("   Una dice que no sirvió de nada. La otra, que mejoró muchísimo.")
print("   El bloque 5 explica cuál tiene razón.")
"""
    ),
    md(
        """
**✍️ Tu respuesta al TODO 10:**

*(doble clic aquí y escribe)*

Tu modelo mejora la exactitud del baseline en menos de un punto porcentual. ¿Fue un fracaso?
¿Qué mirarías para decidirlo?
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 4
>
> **Cifras medidas:**
>
> | Modelo | Exactitud | F1-macro |
> |---|---|---|
> | Baseline (siempre `LEVEL_1`) | 0,8896 | 0,4708 |
> | Bosque aleatorio | **0,8965** | **0,7025** |
> | **Diferencia** | **+0,0069** | **+0,2317** |
>
> **Siete décimas de punto en exactitud.** Después de entrenar 200 árboles.
>
> **Deja que la decepción se instale unos segundos.** Es real y es útil: así se siente mirar la
> métrica equivocada. Varios van a concluir que el modelo no sirve. Es la conclusión correcta
> *dada esa métrica*, y ahí está la lección.
>
> Después señala la otra columna: **+0,23 en F1-macro**, casi un 50 % de mejora relativa. El
> mismo modelo, los mismos datos, la misma partición. Dos conclusiones opuestas.
>
> **La pregunta que abre el bloque 5:** *"¿Cuál de las dos métricas está describiendo mejor lo
> que hace este modelo? Para responder eso hay que mirar qué pasa con cada clase por
> separado."*
>
> **Sobre `class_weight="balanced"`**, si hay tiempo: quítalo y vuelve a entrenar. La exactitud
> sube un poco y el recall de `LEVEL_2` se hunde. Es la demostración de que **optimizar la
> exactitud en un problema desbalanceado significa abandonar a la minoría**.
>
> **Criterio de logro:** entrena correctamente y **no concluye nada** a partir de una sola
> métrica; identifica que hace falta el análisis por clase.
"""
    ),
    # ======================================================================
    # BLOQUE 5 — EVALUAR POR CLASE
    # ======================================================================
    md(
        """
---
# Bloque 5 · ⭐⭐ Evaluar por clase: ¿a quién le falla?

Un promedio global oculta a la minoría. Para saber qué hace de verdad un modelo hay que mirar
clase por clase:

| Métrica | Qué responde | Fórmula |
|---|---|---|
| **Precisión** | De lo que predije como `LEVEL_2`, ¿cuánto lo era? | VP / (VP + FP) |
| **Recall** | De todo lo que era `LEVEL_2`, ¿cuánto encontré? | VP / (VP + FN) |
| **F1** | La media armónica de las dos | 2·P·R / (P + R) |

**Cuál importa depende del negocio, no de la estadística.** Aquí queremos anticipar detecciones
difíciles para no confiarnos: si se nos escapan, el sistema confía en una detección mala. Eso
apunta al **recall**.
"""
    ),
    md(
        """
### ✏️ TODO 11 — El reporte por clase
"""
    ),
    code(
        """
metricas = modelo.evaluar_por_clase(clasificador, marcada, CONFIG)
metricas
""",
        """
# TODO 11: métricas por clase, no solo el promedio.
metricas = modelo.____(clasificador, marcada, CONFIG)
metricas
""",
    ),
    md(
        """
### ✏️ TODO 12 — La matriz de confusión

Las filas son lo que **era**; las columnas, lo que el modelo **dijo**. La diagonal son los
aciertos.
"""
    ),
    code(
        """
confusion = modelo.matriz_confusion(clasificador, marcada, CONFIG)
print(confusion, "\\n")

fig, ejes = plt.subplots(figsize=(5, 4))
sns.heatmap(confusion, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ejes)
ejes.set_xlabel("Predicho")
ejes.set_ylabel("Real")
ejes.set_title("Matriz de confusión")
plt.tight_layout()
plt.show()
""",
        """
# TODO 12: la matriz de confusión.
confusion = modelo.____(clasificador, marcada, CONFIG)
print(confusion, "\\n")

fig, ejes = plt.subplots(figsize=(5, 4))
sns.heatmap(confusion, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ejes)
ejes.set_xlabel("Predicho")
ejes.set_ylabel("Real")
ejes.set_title("Matriz de confusión")
plt.tight_layout()
plt.show()
""",
    ),
    md(
        """
### ✏️ TODO 13 — Poner número al daño

De la matriz, calcula cuántas detecciones difíciles se le escaparon al modelo y qué porcentaje
del total de difíciles representan.
"""
    ),
    code(
        """
difciles_perdidas = confusion.loc["LEVEL_2", "LEVEL_1"]
difciles_totales = confusion.loc["LEVEL_2"].sum()
falsas_alarmas = confusion.loc["LEVEL_1", "LEVEL_2"]

print(f"Detecciones difíciles en el conjunto de prueba : {difciles_totales:,}")
print(f"Que el modelo NO detectó como difíciles        : {difciles_perdidas:,}"
      f"  ({100*difciles_perdidas/difciles_totales:.1f} %)")
print(f"Falsas alarmas (fáciles marcadas como difíciles): {falsas_alarmas:,}")
""",
        """
# TODO 13: ¿cuántas detecciones difíciles se le escaparon?
difciles_perdidas = confusion.loc["____", "____"]
difciles_totales = confusion.loc["LEVEL_2"].sum()
falsas_alarmas = confusion.loc["____", "____"]

print(f"Detecciones difíciles en el conjunto de prueba : {difciles_totales:,}")
print(f"Que el modelo NO detectó como difíciles        : {difciles_perdidas:,}"
      f"  ({100*difciles_perdidas/difciles_totales:.1f} %)")
print(f"Falsas alarmas (fáciles marcadas como difíciles): {falsas_alarmas:,}")
""",
    ),
    code(
        """
# Autochequeo
recall_l2 = metricas.set_index("clase").loc["LEVEL_2", "recall"]
assert np.isclose(1 - difciles_perdidas / difciles_totales, recall_l2, atol=1e-3), (
    "revisa: lo que NO se escapó, sobre el total, ES el recall de LEVEL_2"
)
assert recall_l2 < 0.5, "revisa: el recall de la clase minoritaria debería ser bajo"
print(f"✅ Recall de LEVEL_2: {recall_l2:.4f}")
print(f"   El modelo encuentra {recall_l2:.0%} de las detecciones difíciles.")
print(f"   Se le escapan {100*difciles_perdidas/difciles_totales:.0f} de cada 100.")
"""
    ),
    md(
        """
### ✏️ TODO 14 — La decisión

**✍️ Tu respuesta:**

*(doble clic aquí y escribe)*

1. Con estas cifras, ¿pondrías este modelo a decidir si el vehículo confía o no en una
   detección? Justifica.
2. ¿Qué preferirías: subir el recall a costa de más falsas alarmas, o al revés? ¿Por qué, en
   **este** dominio?
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 5 ⭐⭐
>
> **Cifras medidas:**
>
> | Clase | Precisión | Recall | F1 | Soporte |
> |---|---|---|---|---|
> | `LEVEL_1` | 0,9282 | 0,9578 | 0,9428 | 9.122 |
> | **`LEVEL_2`** | **0,5422** | **0,4028** | **0,4622** | 1.132 |
> | exactitud | | | **0,8965** | |
>
> Matriz de confusión: de **1.132** detecciones difíciles, el modelo encuentra **456** y se le
> escapan **676**. Y produce **385** falsas alarmas.
>
> **La frase que resuelve el suspenso de los bloques 3 y 4:**
>
> > *El modelo saca 89,65 % de exactitud y se pierde el 60 % de las detecciones difíciles, que
> > eran justo las que queríamos anticipar. La exactitud no estaba equivocada: estaba
> > respondiendo otra pregunta.*
>
> **Ahora se entiende el F1-macro.** Promedia el F1 de cada clase **sin ponderar por tamaño**,
> así que la clase con 1.132 filas pesa lo mismo que la de 9.122. Por eso detectó una mejora que
> la exactitud no vio: el modelo sí aprendió algo sobre `LEVEL_2` —pasó de 0 aciertos a 456—,
> pero es poco en términos absolutos.
>
> **El TODO 14 no tiene una respuesta correcta, y hay que decirlo.** Lo que se evalúa es el
> razonamiento. Las buenas respuestas:
>
> - **"No lo pondría en producción tal cual"**, porque un recall de 0,40 significa que el
>   sistema confiaría en 6 de cada 10 detecciones malas. Correcta y bien argumentada.
> - **"Lo pondría, pero solo para levantar una alerta, no para decidir"**. Excelente: distingue
>   entre un modelo que decide y uno que asiste. Es la distinción que hace la industria.
> - **"Subiría el recall aunque suba la falsa alarma"**, porque el costo es asimétrico: una
>   falsa alarma es que el vehículo sea prudente de más; un falso negativo es que confíe en una
>   detección mala. **Nivel destacado.** Si alguien llega solo a que se puede mover el umbral de
>   decisión con `predict_proba`, mencionar que eso se retoma en el RA3 (ajuste y validación)
>   y en el EFT.
>
> **Respuesta que hay que corregir:** *"el modelo es malo"*. No es malo ni bueno en abstracto:
> es insuficiente **para este uso**. Para priorizar qué segmentos revisa un humano, un recall de
> 0,40 con 0,54 de precisión puede ser perfectamente útil. **La utilidad la define el uso.**
>
> **Criterio de logro:** lee la matriz de confusión, cuantifica los falsos negativos y toma una
> decisión argumentada en términos del **costo del error en este dominio**, no en términos de si
> el número "es alto".
"""
    ),
    # ======================================================================
    # BLOQUE 6 — FUGA
    # ======================================================================
    md(
        """
---
# Bloque 6 · ⭐ La fuga que sí se nota

En el bloque 2 vimos una fuga que existía pero no movía la métrica. Esta es la contraria: se
mide y duele.

`detection_difficulty` **la asigna el sensor a partir de `num_lidar_points`**. Meter esa columna
entre las predictoras no es informativo: es contarle al modelo la respuesta con otras palabras.

Y aquí está lo traicionero: **el modelo mejora**. No hay ningún síntoma, solo un resultado
sospechosamente bueno.
"""
    ),
    md(
        """
### ✏️ TODO 15 — Medir la fuga
"""
    ),
    code(
        """
efecto_fuga = modelo.comparar_fuga_de_variable(marcada, CONFIG, FUGA)
efecto_fuga
""",
        """
# TODO 15: ¿cuánto infla la métrica incluir la variable de la que sale la etiqueta?
efecto_fuga = modelo.____(marcada, CONFIG, FUGA)
efecto_fuga
""",
    ),
    code(
        """
# Autochequeo
inflacion = efecto_fuga.loc[1, "inflacion_f1_macro"]
assert inflacion > 0, "revisa: incluir la variable derivada debería SUBIR la métrica"
print(f"✅ Incluir num_lidar_points sube el F1-macro en {inflacion:+.4f}.")
print("   El modelo parece mejor y no sirve para nada: en el momento en que")
print("   quisieras predecir la dificultad, ya tendrías la dificultad.")
"""
    ),
    md(
        """
### ✏️ TODO 16 — La pregunta que detecta la fuga

**✍️ Tu respuesta:**

*(doble clic aquí y escribe)*

Existe **una sola pregunta** que hay que hacerle a cada variable predictora para detectar este
tipo de fuga. No es estadística. Escríbela con tus palabras.

*Pista: piensa en el momento en que el modelo se usaría de verdad.*
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 6 ⭐
>
> **Cifras medidas:**
>
> | Variables | Exactitud | F1-macro |
> |---|---|---|
> | Sin la variable derivada (correcto) | 0,8965 | **0,7025** |
> | Incluyendo `num_lidar_points` | 0,9264 | **0,7543** |
>
> **+0,052 de F1-macro y +3 puntos de exactitud.** Gratis, y falso.
>
> **La pregunta del TODO 16 es la respuesta que hay que llevarse de la sesión:**
>
> > **¿Voy a tener esta variable, con este valor, en el momento en que necesite hacer la
> > predicción?**
>
> Si la respuesta es no, la variable no puede estar. Aquí no: para saber `num_lidar_points` hay
> que haber procesado la detección, y en ese momento ya sabes la dificultad. **La variable
> llega tarde.**
>
> Formulaciones equivalentes que también valen: *"¿esta variable existe antes que la etiqueta o
> después?"*, *"¿esta columna se calculó usando la respuesta?"*.
>
> **Es el error más común en la industria y el más difícil de ver**, porque no hay ningún
> síntoma. Ejemplos que funcionan bien para aterrizarlo:
>
> - Predecir si un cliente se va a ir, usando *"motivo de baja"*.
> - Predecir si una transacción es fraude, usando *"monto reembolsado"*.
> - Predecir si un paciente será hospitalizado, usando *"días de hospitalización"*.
>
> Todos parecen absurdos escritos así. Ninguno lo parece dentro de una tabla con 200 columnas
> que alguien exportó de un sistema.
>
> **Si sobra tiempo**, conecta con el bloque 2: son dos fugas de naturaleza distinta. La de
> agrupación es **temporal/estructural** (filas dependientes); esta es de **la variable objetivo**
> (información del futuro). Que en este dataset una se mida y la otra no, no las hace desiguales
> en importancia.
>
> **Criterio de logro:** formula la pregunta de disponibilidad temporal y la aplica a una
> variable propia distinta de la del ejercicio.
"""
    ),
    # ======================================================================
    # BLOQUE 7 — INTERPRETAR
    # ======================================================================
    md(
        """
---
# Bloque 7 · ¿Qué aprendió el modelo?

Un modelo que acierta y no se puede explicar sirve para poco: nadie firma una decisión que no
entiende. El bosque aleatorio permite ver **qué variables usó más**.

Cuidado con la interpretación: la importancia dice **cuánto usó** el modelo cada variable, no
que esa variable *cause* nada.
"""
    ),
    md(
        """
### ✏️ TODO 17 — Importancia de variables
"""
    ),
    code(
        """
importancia = (
    pd.Series(clasificador.feature_importances_, index=CONFIG["variables"])
    .sort_values(ascending=False)
    .round(4)
)
print(importancia.to_string(), "\\n")

importancia.sort_values().plot.barh(figsize=(6, 3.5), title="¿Qué usó el modelo?")
plt.tight_layout()
plt.show()
""",
        """
# TODO 17: ¿qué variables usó más el modelo?
importancia = (
    pd.Series(clasificador.____, index=CONFIG["variables"])
    .sort_values(ascending=False)
    .round(4)
)
print(importancia.to_string(), "\\n")

importancia.sort_values().plot.barh(figsize=(6, 3.5), title="¿Qué usó el modelo?")
plt.tight_layout()
plt.show()
""",
    ),
    md(
        """
**✍️ Tu respuesta al TODO 17:**

*(doble clic aquí y escribe)*

Mira las dos variables más importantes. ¿Tiene sentido de dominio que sean esas? Explica el
mecanismo físico que lo justificaría.
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 7
>
> **Cifras medidas:**
>
> | Variable | Importancia |
> |---|---|
> | `box_center_x` | **0,5202** |
> | `speed_mps` | **0,2613** |
> | `box_length` | 0,0458 |
> | resto | < 0,045 cada una |
>
> **Dos variables concentran el 78 % de la importancia**, y las dos tienen una explicación
> física limpia:
>
> - **`box_center_x`** es la distancia longitudinal al vehículo. Cuanto más lejos está un
>   objeto, **menos puntos láser caen sobre él** —el dataset respeta una caída aproximadamente
>   cuadrática con la distancia— y menos puntos significa detección más difícil.
> - **`speed_mps`**: un objeto en movimiento se difumina entre barridos del sensor.
>
> **El modelo aprendió "lejos y en movimiento = difícil".** Es exactamente el mecanismo que
> genera la etiqueta, descubierto desde los datos. Vale la pena decirlo así: *no memorizó,
> encontró la relación.*
>
> **La advertencia que hay que dar** es sobre correlación y causa. La importancia dice **cuánto
> usó** el modelo cada variable, no que la cause. Si dos variables están correlacionadas, el
> bosque reparte la importancia entre ellas de forma arbitraria, y una variable importante puede
> desaparecer del ranking solo porque entró otra parecida.
>
> **Pregunta para el curso, si sobra tiempo:** ¿podríamos usar esto para mejorar el sensor en
> vez de para predecir? Sí: dice que el problema está en los objetos lejanos y móviles. **Un
> modelo interpretable no solo predice: informa dónde invertir.** Es el mejor argumento a favor
> de la interpretabilidad que se puede dar en clase.
>
> **Criterio de logro:** identifica las dos variables dominantes, propone un mecanismo de
> dominio plausible y **no** confunde importancia con causalidad.
"""
    ),
    # ======================================================================
    # CIERRE
    # ======================================================================
    md(
        """
---
# Cierre · Informe de modelamiento

Esta es la entrega de la Actividad 2.2. Máximo una página.

> Es el mismo esqueleto que pide el informe técnico del EFT, en pequeño: por eso conviene
> tomárselo en serio ahora.

---

### El problema

**Qué queríamos predecir:** `____`
**Por qué le importa a alguien:** `____`
**Tipo de problema:** `____` *(clasificación binaria / multiclase / regresión)*

### Cómo partí los datos

**Estrategia:** `____`
**Segmentos compartidos entre entrenamiento y prueba:** `____`
**Por qué no partí al azar:** `____`

> Ojo con esta última: la diferencia medida fue casi nula. Si tu argumento es "porque da mejor
> resultado", vuelve a leer el bloque 2.

### Contra qué comparé

| Modelo | Exactitud | F1-macro |
|---|---|---|
| Baseline (`____`) | `____` | `____` |
| Mi modelo (`____`) | `____` | `____` |

**Por qué la exactitud engaña aquí:** `____`

### A quién le falla

| Clase | Precisión | Recall | F1 | Soporte |
|---|---|---|---|---|
| `____` | | | | |
| `____` | | | | |

**De cada 100 detecciones difíciles, el modelo encuentra:** `____`
**Falsas alarmas producidas:** `____`

### Fuga de información

**Variable que excluí y por qué:** `____`
**Cuánto inflaba la métrica:** `____`
**La pregunta que uso para detectar este error:** `____`

### Qué aprendió el modelo

**Las dos variables más importantes:** `____`
**Mecanismo de dominio que lo explica:** `____`

### La decisión

**¿Pondrías este modelo en producción?** `____`

*(Responder "no" está permitido y a veces es lo correcto. Lo que se evalúa es el argumento, y
que esté en términos del costo del error en este dominio: qué pasa si el vehículo confía en una
detección mala, y qué pasa si es prudente de más.)*

**Qué haría falta para cambiar esa respuesta:** `____`
"""
    ),
    md_docente(
        """
> ### 🎓 Criterios de logro — Actividad 2.2 (IL 2.2)
>
> | Nivel | Descripción |
> |---|---|
> | **Destacado (4)** | Todo lo del nivel 3, y además: defiende la partición por grupo **con un argumento de diseño pese a que la métrica no lo respalda**; razona el umbral en términos de costo asimétrico del error; aplica la pregunta de fuga a una variable propia |
> | **Logrado (3)** | Los 17 TODO en verde; compara contra el baseline en dos métricas y explica por qué la exactitud engaña; lee la matriz de confusión y cuantifica los falsos negativos; toma una decisión argumentada sobre producción |
> | **En desarrollo (2)** | Entrena y evalúa correctamente pero concluye desde la exactitud sola; identifica que hay desbalance sin extraer consecuencias; la decisión final es "el modelo es bueno/malo" sin referirse al uso |
> | **Inicial (1)** | Ejecuta las celdas sin interpretar; no distingue la partición por grupo de la partición al azar |
>
> **Qué mirar al corregir, en este orden:**
>
> 1. **El bloque 3.** Si no puede explicar por qué el baseline saca 88,96 %, no entendió el
>    desbalance, y todo lo demás lo va a leer mal.
> 2. **La decisión del cierre.** Un "sí, tiene 90 % de exactitud" es nivel *Inicial*, aunque
>    todos los TODO estén en verde. Un "no, porque se pierde 6 de cada 10 difíciles" es
>    *Logrado*. Un "depende de si decide o solo alerta" es *Destacado*.
> 3. **La justificación de la partición.** Es el único punto donde se puede distinguir a quien
>    razona de quien optimiza el número.
>
> **Errores frecuentes que conviene anticipar en voz alta:**
>
> - Reportar solo la exactitud. Es el error que la sesión entera existe para evitar.
> - Confundir precisión con recall. Ayuda anclarlas a la pregunta: *precisión = de lo que dije,
>   ¿cuánto acerté?*; *recall = de lo que había, ¿cuánto encontré?*
> - Llamar "fuga" a cualquier cosa. Fuga es información que **no estará disponible** en el
>   momento de predecir. Que una variable sea muy predictiva no la hace fuga.
>
> **Enlace con el resto:** el `detecciones_limpias.parquet` que produce el pipeline es la entrada
> tanto de esta actividad como de la **2.3** (no supervisado), que hace la pregunta contraria. Las
> dos pertenecen al **RA2** y se evalúan juntas en la **Parcial 2**, sobre uno de los casos
> oficiales. La **2.4** retoma la interpretación de métricas que aquí se introduce.
"""
    ),
]
