"""Fuente única del contenido de la Actividad 1.4 — Impacto ético, sesgos y privacidad.

Indicador de logro **IL 1.4**: *evalúa el impacto ético y los sesgos en los datos
recopilados, garantizando el cumplimiento de estándares de privacidad en el manejo de
información.*

De este archivo salen dos notebooks:

- ``notebooks/07_alumno_etica.ipynb``   (versión con TODO)
- ``notebooks/07_docente_etica.ipynb``  (versión resuelta con pauta)

**5 horas pedagógicas** según el programa.

Todas las cifras están medidas: las del censo de Waymo salen de ``docs/sesgo_waymo.md``
(los 798 segmentos del split de training, no una muestra) y las del dataset sintético, de
``detecciones_waymo_like.csv`` con la semilla 42.

Regenerar tras editar:

    python herramientas/construir_notebooks.py
"""

from __future__ import annotations

from contenido_semana01 import URL_REPO, code, md, md_docente

CELDAS_ACT14: list[dict] = [
    md(
        """
# MLY1101 · Machine Learning — Actividad 1.4
## Impacto ético, sesgos y privacidad

**Resultado de aprendizaje (RA1):** recopila, a través de un trabajo colaborativo, sets de datos
representativos y de calidad, a partir de distintas fuentes, para responder a las necesidades del
contexto de negocio, **considerando aspectos éticos**.

**Indicador de logro (IL 1.4):** evalúa el impacto ético y los sesgos en los datos recopilados,
garantizando el cumplimiento de estándares de privacidad en el manejo de información.

---

### Por qué esta actividad no es el apéndice del curso

La ética de los datos se suele dejar para la última diapositiva, en forma de advertencias
generales que nadie puede aplicar. Hoy vamos a hacer lo contrario: **medir**.

Un sesgo sin una cifra que lo respalde es una opinión. Un riesgo de privacidad sin un caso
concreto es un trámite. En esta sesión, cada afirmación ética va a tener un número al lado.

> **La idea central:** un algoritmo no es neutral. Replica y **amplifica** las asimetrías que
> había en los datos con los que se entrenó, y lo hace en silencio, porque la métrica promedio
> se ve bien.

---

### Las tres cosas que vamos a medir

| Bloque | Pregunta | Cómo se responde |
|---|---|---|
| **Sesgo de muestreo** | ¿Los datos representan al mundo donde va a operar el sistema? | Censo completo de las condiciones de grabación |
| **Sesgo de procesamiento** | ¿Mis decisiones de limpieza perjudican a un grupo? | Composición del dataset antes y después |
| **Privacidad** | ¿Se puede reidentificar a alguien con estos datos? | Cardinalidad de combinaciones de columnas |

---

### Al final de la sesión debes entregar

Una **ficha del dataset** (*datasheet*) con la evaluación de impacto: quién está
subrepresentado y con qué cifra, qué decisión técnica lo empeoraría, sobre quién recae la
consecuencia, y qué datos personales hay o podrían reconstruirse.
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Actividad 1.4
>
> **La actividad son 5 horas pedagógicas.** La distribución de abajo cubre unas 3 h de trabajo
> guiado; las 2 h restantes quedan para que completen la ficha del dataset sobre el caso oficial
> elegido (Telco, Housing o Spotify), que es el que llevan a la Evaluación Parcial 1.
>
> **Distribución del bloque guiado:**
>
> | Bloque | Min | Foco |
> |---|---|---|
> | 0 · Encuadre | 15 | Un algoritmo no es neutral |
> | 1 · Sesgo de muestreo ⭐⭐ | 40 | El censo real: 793 soleados de 798 |
> | 2 · Sesgo de representación | 30 | Quién es minoría dentro del dataset |
> | 3 · Sesgo de procesamiento ⭐⭐ | 45 | Tu limpieza también sesga |
> | 4 · Privacidad y reidentificación ⭐ | 35 | Tres columnas inocentes |
> | 5 · La ficha del dataset | 25 | Documentar es parte del trabajo |
> | Cierre | 10 | Evaluación de impacto |
>
> **Los imprescindibles son el 1 y el 3.** El bloque 3 es el que convierte la ética en una
> decisión de ingeniería y no en un discurso.
>
> **Esta sesión funciona con preguntas, no con afirmaciones.** Cada bloque tiene una cifra que
> conviene **preguntar antes de mostrar**. Están señaladas en las pautas.
>
> **Regla de oro, la de siempre:** ninguna afirmación sin una cifra que la respalde. Aquí más que
> nunca, porque es el terreno donde es más fácil hablar bonito sin decir nada.
"""
    ),
    md(
        """
---
## Preparación del entorno
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
RUTA_DATOS = RAIZ / "datos" / "crudos" / "detecciones_waymo_like.csv"

print("Colab:", EN_COLAB, "| dataset:", RUTA_DATOS.exists())
"""
    ),
    code(
        """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import eda

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 140)
sns.set_theme(style="whitegrid")

df = pd.read_csv(RUTA_DATOS)
print(f"{df.shape[0]:,} detecciones en {df['segment_id'].nunique()} segmentos")
"""
    ),
    # ======================================================================
    # BLOQUE 1
    # ======================================================================
    md(
        """
---
# Bloque 1 · ⭐⭐ Sesgo de muestreo: ¿a quién no vio el sensor?

**Sesgo de muestreo** es que los datos no representen proporcionalmente al mundo donde el
sistema va a operar. No es un error de cálculo: es una consecuencia de **dónde y cuándo** se
recolectó.

No vamos a discutirlo en abstracto. El Waymo Open Dataset publica las condiciones de grabación
de todos sus segmentos, y en este repositorio está el **censo completo** —los 798 segmentos del
conjunto de entrenamiento, no una muestra— en `docs/sesgo_waymo.md`.

### ✏️ TODO 1 — Antes de mirar, apuesta

El Waymo Open Dataset tiene **798 segmentos** de entrenamiento, grabados por una flota de
vehículos autónomos en Estados Unidos.

**¿Cuántos crees que se grabaron con lluvia?** `____`

*(Escríbelo antes de seguir. No hagas trampa.)*
"""
    ),
    code(
        """
# El censo real, medido sobre los 798 segmentos (ver docs/sesgo_waymo.md).
censo = pd.DataFrame(
    [
        {"condicion": "clima", "valor": "soleado", "segmentos": 793},
        {"condicion": "clima", "valor": "lluvia", "segmentos": 5},
        {"condicion": "momento", "valor": "día", "segmentos": 647},
        {"condicion": "momento", "valor": "noche", "segmentos": 79},
        {"condicion": "momento", "valor": "amanecer/atardecer", "segmentos": 72},
        {"condicion": "lugar", "valor": "San Francisco", "segmentos": 409},
        {"condicion": "lugar", "valor": "Phoenix", "segmentos": 284},
        {"condicion": "lugar", "valor": "otras", "segmentos": 105},
    ]
)
censo["pct"] = censo.groupby("condicion")["segmentos"].transform(lambda s: 100 * s / s.sum()).round(1)
censo
"""
    ),
    md(
        """
### ✏️ TODO 2 — Ponerlo en palabras que signifiquen algo

**✍️ Tu respuesta:**

*(doble clic aquí y escribe)*

1. Completa la frase con la cifra: *"Un vehículo autónomo entrenado con estos datos ha visto
   llover ____ veces."*
2. ¿Qué pasa cuando ese sistema se despliega en Santiago en junio?
3. **La pregunta difícil:** ¿es esto un error de Waymo?
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 1 ⭐⭐
>
> **Cifras del censo** (798 segmentos, no una muestra):
>
> | Condición | | |
> |---|---|---|
> | Clima | soleado **793 (99,4 %)** | lluvia **5 (0,6 %)** |
> | Momento | día 647 (81,1 %) · noche 79 (9,9 %) · amanecer 72 (9,0 %) | |
> | Lugar | San Francisco 409 (51,3 %) · Phoenix 284 (35,6 %) · otras 105 (13,2 %) | |
>
> **El TODO 1 solo funciona si apuestan.** Insiste en que escriban un número. Las respuestas
> típicas van de 100 a 300. **Son cinco.** Es el dato que se llevan de la sesión, y se pierde si
> lo muestras antes de preguntarlo.
>
> **Respuesta 1:** *"ha visto llover cinco veces"*. Dicho así, en singular, cambia la percepción
> mucho más que decir "el 0,6 %".
>
> **Respuesta 3 — la que separa la reflexión del reflejo.** **No es un error de Waymo.** Es una
> consecuencia lógica de dónde están sus flotas: San Francisco y Phoenix, dos ciudades soleadas.
> Recolectar lluvia habría exigido operar en otro clima.
>
> **El error sería no declararlo** y presentar el modelo como si funcionara en todas partes. Y
> Waymo **sí lo declara**: por eso pudimos hacer este censo.
>
> La conclusión que hay que dejar:
>
> > *El sesgo no se elimina: se documenta y se acota el dominio de uso. Un dataset sesgado y
> > declarado es utilizable; uno sesgado y silencioso, no.*
>
> **Si alguien pregunta por los otros dos ejes**, vale la pena señalar que el 81 % diurno y el
> 87 % en dos ciudades son sesgos igual de reales, solo que menos llamativos que el 793/5.
>
> **Criterio de logro:** interpreta el censo, formula la consecuencia sobre un contexto de
> despliegue concreto, y distingue entre **tener** un sesgo y **ocultarlo**.
"""
    ),
    # ======================================================================
    # BLOQUE 2
    # ======================================================================
    md(
        """
---
# Bloque 2 · ¿Quién es minoría dentro del dataset?

El sesgo de muestreo es sobre las **condiciones**. Ahora miremos los **sujetos**: qué objetos
aparecen y en qué proporción.

### ✏️ TODO 3 — La composición

Calcula la frecuencia de cada tipo de objeto, **después de unificar las variantes de escritura**
(si no, los peatones quedan repartidos en cuatro grafías y ninguna cifra sirve).
"""
    ),
    code(
        """
tipos = eda.normalizar_categoria(
    df["object_type"], mapa={"peaton": "pedestrian", "ped": "pedestrian"}
)
composicion = eda.resumen_desbalance(tipos)
composicion
""",
        """
# TODO 3: composición por tipo de objeto, tras unificar variantes.
tipos = eda.____(
    df["object_type"], mapa={"peaton": "pedestrian", "ped": "pedestrian"}
)
composicion = eda.____(tipos)
composicion
""",
    ),
    md(
        """
### ✏️ TODO 4 — Del porcentaje a la persona

**✍️ Tu respuesta:**

*(doble clic aquí y escribe)*

1. ¿Qué tipo de objeto es la minoría, y con qué porcentaje?
2. En la vía pública, ¿quién es más vulnerable ante un error de detección: el ocupante de un
   vehículo o esa minoría? ¿Por qué?
3. Une las dos respuestas en una frase.
"""
    ),
    code(
        """
# Autochequeo
minoritaria = composicion["pct"].idxmin()
pct = composicion.loc[minoritaria, "pct"]
assert pct < 5, "revisa: ¿normalizaste las categorías antes de contar?"
print(f"✅ Clase minoritaria: {minoritaria} con {pct:.2f} % de las detecciones.")
print(f"   Razón respecto de la mayoritaria: {composicion.loc[minoritaria, 'ratio_vs_mayoritaria']:.4f}")
print("   Es decir, por cada ciclista hay unos", int(1/composicion.loc[minoritaria, 'ratio_vs_mayoritaria']), "vehículos.")
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 2
>
> **Cifras medidas** sobre el dataset sintético: `vehicle` 61,73 % · `pedestrian` 26,22 % ·
> `sign` 8,12 % · **`cyclist` 1,94 %**. Uno por cada ~32 vehículos.
>
> **Sobre los datos reales es peor:** en los 40 segmentos descargados, los ciclistas son el
> **0,45 %**. El dataset sintético es más benigno que la realidad.
>
> **La cadena que se busca en el TODO 4**, y hay que dejarla decir a ellos:
>
> 1. La minoría del dataset son los **ciclistas** (1,94 %).
> 2. En la vía, un ciclista es de los usuarios **más vulnerables**: sin carrocería, sin airbag.
> 3. Por lo tanto: *el grupo del que el modelo tiene menos ejemplos es exactamente el que más
>    daño sufre si el modelo se equivoca.*
>
> **Esa coincidencia no es casual y vale la pena nombrarla.** Las minorías de un dataset suelen
> ser minorías en el mundo, y las minorías suelen ser las que menos protección tienen. Ocurre
> igual en banca (clientes sin historial crediticio), en salud (enfermedades raras) y en
> contratación (perfiles atípicos).
>
> **Adelanto útil hacia el RA2:** con el 1,94 %, un modelo que **ignore a los ciclistas por
> completo** puede sacar más del 98 % de exactitud. Se mide en la Actividad 2.2, y ahí se
> entiende del todo. Aquí basta con sembrarlo.
>
> **Criterio de logro:** identifica la minoría con su cifra y conecta subrepresentación con
> vulnerabilidad, sin quedarse en "hay pocos ciclistas".
"""
    ),
    # ======================================================================
    # BLOQUE 3
    # ======================================================================
    md(
        """
---
# Bloque 3 · ⭐⭐ Tu limpieza también sesga

Los dos bloques anteriores son sobre datos que llegaron así. Este es sobre **lo que tú les
haces**.

`dropna()` es la operación más inocente del análisis de datos. Elimina filas incompletas. Nadie
la discute.

Vamos a ver a quién elimina.
"""
    ),
    md(
        """
### ✏️ TODO 5 — ¿Cuánto se pierde en total?

Empieza por la cifra que cualquiera reportaría.
"""
    ),
    code(
        """
perdidas = df["speed_mps"].isna().sum()
print(f"Filas con speed_mps faltante: {perdidas:,} de {len(df):,} "
      f"({100*perdidas/len(df):.2f} % del dataset)")
""",
        """
# TODO 5: ¿cuántas filas se perderían con un dropna sobre speed_mps?
perdidas = df["speed_mps"].____().sum()
print(f"Filas con speed_mps faltante: {perdidas:,} de {len(df):,} "
      f"({100*perdidas/len(df):.2f} % del dataset)")
""",
    ),
    md(
        """
Menos del 2 %. En cualquier informe eso se describiría como *"se eliminaron unas pocas filas
incompletas"* y nadie preguntaría más.

### ✏️ TODO 6 — La misma cifra, por grupo

Ahora calcula el porcentaje de faltantes **según el momento del día**.
"""
    ),
    code(
        """
por_momento = eda.matriz_nulos_por_grupo(df, "speed_mps", ["time_of_day"])
print(por_momento.sort_values("pct_nulos", ascending=False), "\\n")

peor = por_momento["pct_nulos"].idxmax()
mejor = por_momento["pct_nulos"].idxmin()
factor = por_momento.loc[peor, "pct_nulos"] / por_momento.loc[mejor, "pct_nulos"]
print(f"'{peor}' pierde {factor:.1f} veces más filas que '{mejor}'.")
""",
        """
# TODO 6: ¿el faltante se reparte igual entre los grupos?
por_momento = eda.____(df, "speed_mps", ["____"])
print(por_momento.sort_values("pct_nulos", ascending=False), "\\n")

peor = por_momento["pct_nulos"].idxmax()
mejor = por_momento["pct_nulos"].idxmin()
factor = por_momento.loc[peor, "pct_nulos"] / por_momento.loc[mejor, "pct_nulos"]
print(f"'{peor}' pierde {factor:.1f} veces más filas que '{mejor}'.")
""",
    ),
    md(
        """
### ✏️ TODO 7 — El efecto sobre la composición

Compara cómo se reparte el dataset por momento del día **antes y después** del `dropna()`.
"""
    ),
    code(
        """
antes = df["time_of_day"].value_counts(normalize=True).mul(100)
despues = df.dropna(subset=["speed_mps"])["time_of_day"].value_counts(normalize=True).mul(100)

efecto = pd.DataFrame({"antes_%": antes.round(2), "tras_dropna_%": despues.round(2)})
efecto["cambio_pp"] = (efecto["tras_dropna_%"] - efecto["antes_%"]).round(2)
efecto["pct_del_grupo_perdido"] = (
    100 * (df["time_of_day"].value_counts() - df.dropna(subset=["speed_mps"])["time_of_day"].value_counts())
    / df["time_of_day"].value_counts()
).round(2)
efecto
""",
        """
# TODO 7: ¿cómo cambia la composición del dataset tras el dropna?
antes = df["time_of_day"].value_counts(normalize=True).mul(100)
despues = df.____(subset=["speed_mps"])["time_of_day"].value_counts(normalize=True).mul(100)

efecto = pd.DataFrame({"antes_%": antes.round(2), "tras_dropna_%": despues.round(2)})
efecto["cambio_pp"] = (efecto["tras_dropna_%"] - efecto["antes_%"]).round(2)
efecto["pct_del_grupo_perdido"] = (
    100 * (df["time_of_day"].value_counts() - df.dropna(subset=["speed_mps"])["time_of_day"].value_counts())
    / df["time_of_day"].value_counts()
).round(2)
efecto
""",
    ),
    code(
        """
# Autochequeo
assert factor > 2, "revisa: la diferencia entre grupos debería ser grande, no marginal"
assert efecto.loc["Night", "cambio_pp"] < 0, "la noche debería PERDER peso tras el dropna"
print(f"✅ El dropna borra el {100*perdidas/len(df):.2f} % del dataset...")
print(f"   ...pero el {efecto.loc['Night', 'pct_del_grupo_perdido']:.2f} % de las detecciones nocturnas,")
print(f"   contra el {efecto.loc['Dawn/Dusk', 'pct_del_grupo_perdido']:.2f} % de las del amanecer.")
print()
print("   Una operación que en el informe aparece como 'se limpiaron los datos'")
print("   acaba de sesgar el dataset contra la condición peor medida.")
"""
    ),
    md(
        """
### ✏️ TODO 8 — El eslabón que casi nadie ve

**✍️ Tu respuesta:**

*(doble clic aquí y escribe)*

Ordena esta cadena y complétala:

```
De noche el LiDAR recibe menos puntos
        ↓
la velocidad se estima peor y se registra como faltante más seguido
        ↓
un dropna() elimina proporcionalmente más detecciones nocturnas
        ↓
        ____
        ↓
        ____
        ↓
y la métrica de evaluación NO lo muestra. ¿Por qué?
```

La última pregunta es la importante.
"""
    ),
    md(
        """
### ✏️ TODO 9 — La alternativa

Alguien propone: *"entonces imputamos la velocidad con la media y listo"*.

**✍️ Tu respuesta:**

*(doble clic aquí y escribe)*

¿Resuelve el problema? ¿Qué le harías tú en cambio?
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 3 ⭐⭐
>
> **Este es el bloque que convierte la ética en ingeniería. No lo recortes.**
>
> **Cifras medidas:**
>
> | | Faltantes | % del grupo perdido | Cambio en su peso |
> |---|---|---|---|
> | Noche | 332 | **4,08 %** | −0,44 pp |
> | Día | 418 | 1,47 % | +0,34 pp |
> | Amanecer | 37 | 0,91 % | +0,10 pp |
> | **Total** | **787** | **1,93 % del dataset** | |
>
> **El montaje es la secuencia TODO 5 → TODO 6.** Primero sale **1,93 %**, y varios lo van a
> descartar como irrelevante — con razón, si esa fuera toda la información. Después sale que la
> noche pierde **4,5 veces más** que el amanecer.
>
> Merece decirse tal cual:
>
> > *La cifra que reportarías —"eliminamos el 2 % de filas incompletas"— es verdadera y es
> > inútil. Oculta exactamente lo que había que mirar.*
>
> **Respuesta al TODO 8**, los dos eslabones que faltan:
>
> 4. → el modelo se entrena con **aún menos noche** de la poca que había;
> 5. → falla más de noche, justo donde el sensor ya era peor.
>
> **Y la última pregunta, que es la clave:** la métrica no lo muestra porque **el conjunto de
> prueba tiene el mismo sesgo que el de entrenamiento**. Si evalúas sobre datos igual de
> escasos en noche, el promedio se ve estupendo. Solo aparece si evalúas **por subgrupo**, que
> es exactamente lo que se hace en la Actividad 2.2.
>
> **Respuesta al TODO 9 — atájala, porque aparece siempre.** Imputar con la media global mete
> velocidad **diurna** en filas **nocturnas**: no elimina el sesgo, lo **disfraza**, y encima
> borra la señal de que el sensor tiene problemas de noche.
>
> Alternativas razonables, en orden de calidad:
>
> 1. **Agregar una columna que marque el faltante** (`velocidad_no_medida`). No se inventa nada
>    y el modelo puede aprender que ese hecho es informativo. Es la mejor.
> 2. **Imputar por grupo** (la mediana de cada `time_of_day`). Mejor que la media global, pero
>    sigue inventando un valor.
> 3. **Eliminar, declarándolo** y midiendo el efecto sobre la composición. Aceptable si se
>    documenta; inaceptable en silencio.
>
> **La frase del bloque:** *eliminar filas nunca es gratis: siempre estás eligiendo qué parte de
> la realidad borrar.*
>
> **Criterio de logro:** completa la cadena de cinco eslabones, explica por qué la métrica no lo
> revela, y propone una alternativa mejor que la media global justificando por qué.
"""
    ),
    # ======================================================================
    # BLOQUE 4
    # ======================================================================
    md(
        """
---
# Bloque 4 · ⭐ Privacidad: tres columnas inocentes

Este dataset no tiene nombres, ni RUT, ni correos. ¿Está entonces libre de riesgo de privacidad?

La respuesta correcta no es sí ni no: es **depende de qué se puede reconstruir combinando
columnas**. A eso se le llama **reidentificación**, y es el motivo por el que "anonimizar"
borrando la columna del nombre casi nunca basta.

### Marco legal en Chile

La **Ley 19.628** sobre protección de la vida privada, reformada por la **Ley 21.719** (2024),
que crea la Agencia de Protección de Datos Personales. Dos principios que aplican directamente
a lo que hacemos hoy:

- **Minimización:** recolectar solo lo necesario para la finalidad declarada. *Lo que no se
  recolecta no se filtra.*
- **Finalidad:** los datos se usan para aquello que se declaró al obtenerlos, no para lo que se
  nos ocurra después.
"""
    ),
    md(
        """
### ✏️ TODO 10 — ¿Cuántas columnas hacen falta para señalar a uno solo?

Calcula, para varias combinaciones de columnas, qué porcentaje de los grupos resultantes
contiene **una sola detección**. Un grupo de tamaño 1 es una fila señalada de forma única.
"""
    ),
    code(
        """
combinaciones = [
    ["segment_id"],
    ["segment_id", "time_of_day"],
    ["segment_id", "timestamp_micros"],
    ["segment_id", "timestamp_micros", "object_type"],
]

filas = []
for columnas in combinaciones:
    tamanos = df.groupby(columnas).size()
    filas.append(
        {
            "combinacion": " + ".join(columnas),
            "n_columnas": len(columnas),
            "grupos": len(tamanos),
            "grupos_de_una_fila": int((tamanos == 1).sum()),
            "pct_unicos": round(100 * (tamanos == 1).mean(), 1),
        }
    )
pd.DataFrame(filas)
""",
        """
# TODO 10: ¿cuántas columnas hacen falta para aislar una sola detección?
combinaciones = [
    ["segment_id"],
    ["segment_id", "time_of_day"],
    ["segment_id", "timestamp_micros"],
    ["segment_id", "timestamp_micros", "object_type"],
]

filas = []
for columnas in combinaciones:
    tamanos = df.____(columnas).size()
    filas.append(
        {
            "combinacion": " + ".join(columnas),
            "n_columnas": len(columnas),
            "grupos": len(tamanos),
            "grupos_de_una_fila": int((tamanos == ____).sum()),
            "pct_unicos": round(100 * (tamanos == 1).mean(), 1),
        }
    )
pd.DataFrame(filas)
""",
    ),
    code(
        """
# Autochequeo
tres = df.groupby(["segment_id", "timestamp_micros", "object_type"]).size()
assert (tres == 1).mean() > 0.5, "revisa la combinación de tres columnas"
print(f"✅ Con tres columnas, ninguna identificadora por sí sola,")
print(f"   el {100*(tres==1).mean():.1f} % de las combinaciones señala UNA sola detección.")
print()
print("   Ninguna de las tres es un dato personal. Las tres juntas son un identificador.")
"""
    ),
    md(
        """
### ✏️ TODO 11 — La lista de chequeo, aplicada

**✍️ Tu respuesta:**

*(doble clic aquí y escribe)*

1. Este dataset describe objetos, no personas. Pero **un peatón detectado es una persona**.
   ¿Qué se podría reconstruir sobre ella si se cruzara con otra fuente?
2. Aplicando **minimización**: ¿qué columna quitarías si el objetivo es solo clasificar el tipo
   de objeto?
3. El notebook opcional trabaja con datos **reales** de Waymo, cuya licencia prohíbe
   redistribuirlos. ¿Por qué crees que existe esa restricción?
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 4 ⭐
>
> **Cifras medidas:**
>
> | Combinación | Grupos | De una sola fila | % |
> |---|---|---|---|
> | `segment_id` | 153 | 0 | 0,0 % |
> | `segment_id` + `time_of_day` | 459 | 0 | 0,0 % |
> | `segment_id` + `timestamp_micros` | 22.353 | 10.675 | 47,8 % |
> | **`segment_id` + `timestamp` + `object_type`** | 30.878 | 23.303 | **75,5 %** |
>
> **La demostración es el salto de 0 % a 75,5 %.** Ninguna de las tres columnas es un dato
> personal. Las tres juntas aíslan tres de cada cuatro detecciones.
>
> Vale la pena preguntarlo antes: *"¿cuál de estas columnas les parece un dato personal?"*
> Ninguna lo es. Y ese es el punto.
>
> > *La reidentificación no necesita un nombre. Necesita suficientes columnas.*
>
> El ejemplo clásico fuera de este dominio: **comuna + fecha de nacimiento + sexo** identifica a
> la mayoría de las personas de un país, y ninguno de los tres es identificador por separado.
>
> **Respuestas esperadas al TODO 11:**
>
> 1. Con la posición y la marca de tiempo se puede reconstruir **la trayectoria** de un peatón.
>    Cruzada con una dirección o un horario conocido, eso es *dónde estuvo una persona y
>    cuándo*. Es información de ubicación, de las más sensibles que existen.
> 2. **`timestamp_micros`** es la respuesta más limpia: para clasificar el tipo de objeto por su
>    geometría, la marca de tiempo no aporta nada, y es la que habilita la trayectoria.
>    `segment_id` también vale como respuesta si se argumenta bien, aunque hace falta para
>    particionar sin fuga (Actividad 2.2) — señalar esa tensión es nivel destacado.
> 3. La licencia de Waymo restringe la redistribución en parte por esto: las nubes de puntos y
>    las imágenes contienen **personas, matrículas y fachadas** de vía pública real. No es
>    burocracia: es que quien aparece en esos datos no dio su consentimiento.
>
> **El cierre del bloque, si hay tiempo:** que noten que este repositorio usa un dataset
> **sintético** justamente por esto, y que esa decisión está documentada. La coherencia entre lo
> que se enseña y lo que se hace es parte de la clase.
>
> **Criterio de logro:** interpreta el salto de cardinalidad, identifica la trayectoria como el
> riesgo concreto y aplica minimización a una columna justificando la elección.
"""
    ),
    # ======================================================================
    # BLOQUE 5 + CIERRE
    # ======================================================================
    md(
        """
---
# Bloque 5 · La ficha del dataset

Documentar no es el trámite de después: es lo que hace utilizable un dataset sesgado. Un
*datasheet* responde las preguntas que alguien necesitará dentro de dos años, cuando quien lo
armó ya no esté.

Esta es la entrega de la Actividad 1.4. Rellénala **para el caso oficial que eligió tu equipo**
(Telco, Housing o Spotify); el ejemplo de arriba es la referencia de cómo se hace.

---

## Ficha del dataset

**Dataset:** `____` · **Equipo:** `____` · **Fecha:** `____`

### Origen

| Campo | Valor |
|---|---|
| Quién lo recolectó | `____` |
| Con qué propósito original | `____` |
| Cómo se recolectó (muestreo, censo, scraping…) | `____` |
| Periodo y lugar | `____` |
| Licencia y usos permitidos | `____` |

### Representatividad

**Población que el sistema debería cubrir:** `____`
**Población que el dataset realmente cubre:** `____`

| Grupo | % en el dataset | % esperado en el mundo | Brecha |
|---|---|---|---|
| `____` | | | |
| `____` | | | |

**Grupo subrepresentado, con cifra:** `____`

> No vale "podría haber sesgo". Vale *"el grupo X es el 4 % de las filas y concentra el 18 % de
> los valores faltantes"*.

### Evaluación de impacto

**Decisión técnica que empeoraría el sesgo:** `____`
**Consecuencia concreta, y sobre quién recae:** `____`
**Por qué la métrica promedio no lo mostraría:** `____`
**Qué haremos al respecto:** `____`

### Privacidad

- [ ] Identificadores directos: `____`
- [ ] **Combinación de columnas que permite reidentificar:** `____` *(con su % de únicos)*
- [ ] Columnas que quitamos por minimización: `____`
- [ ] Licencia verificada, no supuesta: `____`
- [ ] Podemos declarar origen y fecha: `____`

### Usos para los que este dataset **no** sirve

`____`

*(La sección más útil de la ficha y la que nunca se escribe. Un dataset acotado y declarado es
profesional; uno presentado como universal, no.)*
"""
    ),
    md_docente(
        """
> ### 🎓 Criterios de logro — Actividad 1.4 (IL 1.4)
>
> | Nivel | Descripción |
> |---|---|
> | **Destacado (4)** | Todo lo del nivel 3, y además: distingue entre **tener** un sesgo y **ocultarlo**; propone una alternativa al `dropna()` mejor que la media global y la justifica; identifica la tensión entre minimizar `segment_id` y necesitarlo para particionar sin fuga |
> | **Logrado (3)** | Los 11 TODO en verde; la ficha identifica un grupo subrepresentado **con cifra**, encadena sesgo → decisión → consecuencia sobre un grupo concreto, y reporta una combinación de columnas reidentificadora |
> | **En desarrollo (2)** | Reporta los sesgos con cifras pero no llega a la consecuencia; la sección de privacidad se limita a "no hay datos personales" |
> | **Inicial (1)** | Aborda la ética de forma genérica, sin anclarla en los datos analizados |
>
> **Qué mirar al corregir, en este orden:**
>
> 1. **Que haya cifras.** Es el criterio que separa esta actividad de un ensayo. *"El dataset
>    podría estar sesgado"* es **Inicial** por bien redactado que esté.
> 2. **La cadena del bloque 3.** Si no puede explicar por qué la métrica no revela el problema,
>    no entendió el mecanismo y va a repetir el error en la Parcial.
> 3. **La sección "usos para los que no sirve".** Casi nadie la escribe. Quien la escribe bien
>    está pensando como profesional.
>
> **Errores frecuentes que conviene anticipar:**
>
> - Confundir sesgo con desbalance. El desbalance es una propiedad del dataset; el sesgo es una
>   **discrepancia entre el dataset y el mundo** donde se va a usar.
> - Creer que sin nombres no hay riesgo de privacidad. El bloque 4 existe para eso.
> - Proponer "imputar con la media" como solución a todo.
>
> **Enlace con el resto:** esta actividad cierra el **RA1**. Lo que aquí se documenta —la ficha
> del dataset— es un insumo directo de la **Evaluación Parcial 1** (*Comprensión y preparación de
> los datos*, 30 %), que se rinde sobre el caso oficial elegido. Y la evaluación por subgrupo que
> aquí se anticipa se ejecuta en la **Actividad 2.2**.
"""
    ),
]
