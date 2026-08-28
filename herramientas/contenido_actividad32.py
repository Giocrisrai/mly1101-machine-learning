"""Fuente única de la Actividad 3.2 — Modelos de ensamble.

Indicador de logro **IL 3.2**: *desarrolla modelos basados en técnicas de ensamble para
mitigar problemas de sesgo y varianza en escenarios de negocio complejos.*

**6 horas pedagógicas.** Reutiliza el preámbulo de la Actividad 3.1: misma cadena de
limpieza y partición, mismos nodos del pipeline.

Regenerar:  python herramientas/construir_notebooks.py
"""

from __future__ import annotations

from contenido_actividad31 import PREPARACION, RA3, SETUP
from contenido_semana01 import code, md, md_docente

CELDAS_ACT32: list[dict] = [
    md(
        f"""
# MLY1101 · Machine Learning — Actividad 3.2
## Modelos de ensamble

{RA3}

**Indicador de logro (IL 3.2):** desarrolla modelos basados en técnicas de ensamble para mitigar
problemas de sesgo y varianza en escenarios de negocio complejos.

---

### De dónde viene la pregunta

La Actividad 3.1 terminó con un resultado incómodo: ajustar hiperparámetros **no mejoró nada**.
La reacción natural es *"entonces probemos un modelo más potente"*.

Eso es exactamente lo que hacemos hoy. Y también lo vamos a medir.

---

### Sesgo y varianza, en una tabla

El error de un modelo se descompone en dos partes que se combaten de forma distinta:

| | **Sesgo** | **Varianza** |
|---|---|---|
| Qué es | El modelo es demasiado simple para el problema | El modelo cambia mucho según los datos que le tocaron |
| Cómo se ve | Falla igual en entrenamiento y en prueba | Va perfecto en entrenamiento y mal en prueba |
| Ejemplo | Una recta para separar algo curvo | Un árbol sin límite de profundidad |
| Cómo se reduce | Modelo más flexible, mejores variables | **Promediar modelos**, más datos, regularizar |

**Los ensambles atacan sobre todo la varianza.** Promediar modelos que se equivocan en cosas
distintas cancela parte del error.

**Y no arreglan el sesgo.** Si todos los modelos comparten el mismo punto ciego —porque las
variables no contienen la información que hace falta—, promediarlos no lo elimina. Diez modelos
mirando por la misma ventana no ven más.

---

### Las tres familias

| Familia | Idea | Ejemplo |
|---|---|---|
| **Bagging** | Entrenar en paralelo sobre muestras distintas y promediar | Bosque aleatorio |
| **Boosting** | Entrenar en serie: cada modelo corrige los errores del anterior | Gradient boosting |
| **Votación / apilamiento** | Combinar modelos **distintos entre sí** | `VotingClassifier` |

> **Detalle que suele pasar desapercibido:** el bosque aleatorio que llevas usando desde la
> Actividad 2.2 **ya es un ensamble**. Son 200 árboles votando. No vamos a introducir los
> ensambles: llevamos toda la asignatura usando uno.

---

### Al final de la sesión debes entregar

La comparación de modelos con **métrica y costo juntos**, y una respuesta argumentada a: *¿el
ensamble más complejo gana lo suficiente para justificar lo que cuesta?*
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Actividad 3.2
>
> **6 horas pedagógicas**; ~3 h de trabajo guiado y el resto sobre el caso oficial del equipo.
>
> | Bloque | Min | Foco |
> |---|---|---|
> | 0 · Encuadre | 20 | Sesgo y varianza; el bosque ya era un ensamble |
> | 1 · El catálogo de candidatos ⭐ | 30 | Sin baseline no hay comparación |
> | 2 · Comparar con validación cruzada | 45 | Métrica **y** tiempo |
> | 3 · El ensamble por votación ⭐⭐ | 45 | **No mejora, y cuesta más** |
> | 4 · Sesgo o varianza | 25 | Diagnosticar cuál es el problema aquí |
> | Cierre | 15 | Informe |
>
> **El bloque 3 es el que sostiene la sesión.** El ensamble por votación queda **por debajo** del
> bosque solo (0,6869 contra 0,6909) y encima tarda más. No lo adelantes.
"""
    ),
    md("---\n## Preparación del entorno"),
    code(SETUP),
    code(PREPARACION),
    md(
        """
---
# Bloque 1 · ⭐ Contra qué se compara

Un resultado suelto no significa nada. Antes de comparar modelos complejos hace falta el piso:

| Candidato | Por qué está en la lista |
|---|---|
| **Baseline** | Responde siempre la clase mayoritaria. Si tu modelo no le gana, no hay modelo |
| **Árbol de decisión** | El más simple que aprende algo. Interpretable: se puede dibujar |
| **Regresión logística** | El modelo lineal. Si gana, el problema era lineal y sobraba lo demás |
| **Bosque aleatorio** | Bagging. El que venimos usando |
| **Gradient boosting** | Boosting. Corrige errores en serie |
| **Votación** | Combina árbol + bosque + boosting |

### ✏️ TODO 1 — Ejecutar la comparación

`optimizacion.comparar_ensambles()` evalúa los seis con validación cruzada **por grupo** y
cronometra cada uno.

*(Tarda unos 15 segundos: son 30 entrenamientos.)*
"""
    ),
    code(
        """
comparacion = optimizacion.comparar_ensambles(marcada, CONFIG, AJUSTE)
comparacion
""",
        """
# TODO 1: compara los seis candidatos con validación cruzada por grupo.
comparacion = optimizacion.____(marcada, CONFIG, AJUSTE)
comparacion
""",
    ),
    md(
        """
### ✏️ TODO 2 — Lo primero que hay que mirar

**✍️ Tu respuesta:**

*(doble clic aquí y escribe)*

1. ¿Cuánto le saca la **regresión logística** al baseline?
2. ¿Qué te dice eso sobre la naturaleza del problema?
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 1
>
> **Cifras medidas** (`f1_macro`, `GroupKFold` de 5 pliegues):
>
> | Modelo | Media | Desv. | Peor pliegue | Mejor pliegue | Segundos |
> |---|---|---|---|---|---|
> | Bosque aleatorio | **0,6909** | 0,0079 | 0,6790 | 0,7032 | 4,5 |
> | Ensamble por votación | 0,6869 | 0,0085 | 0,6779 | 0,7017 | 5,9 |
> | Gradient boosting | 0,6804 | 0,0085 | 0,6739 | 0,6972 | 1,3 |
> | Árbol | 0,6708 | 0,0166 | 0,6450 | 0,6934 | 0,3 |
> | Regresión logística | **0,4847** | 0,0049 | 0,4751 | 0,4889 | 0,3 |
> | Baseline | 0,4705 | 0,0010 | 0,4691 | 0,4715 | 0,3 |
>
> **Respuesta al TODO 2:** la regresión logística saca **0,0142** sobre el baseline. Casi nada.
>
> **Y eso es un diagnóstico, no un fracaso:** el problema **no es linealmente separable**. Una
> frontera recta en el espacio de las siete variables no distingue las detecciones difíciles de
> las fáciles. Los modelos de árbol, que trazan fronteras escalonadas, sacan 0,20 más.
>
> Merece decirse así:
>
> > *El modelo que peor funciona te dice algo sobre el problema, no solo sobre sí mismo. Que la
> > regresión logística fracase es información: la relación es no lineal.*
>
> **Fíjate también en la desviación del árbol: 0,0166**, el doble que la del bosque. Es varianza
> pura: un solo árbol depende mucho de qué datos le tocaron. El bosque, que promedia 200,
> estabiliza. **Ese contraste es la demostración de para qué sirve el bagging**, y está en la
> tabla antes de haber hablado del ensamble.
>
> **Criterio de logro:** compara contra el baseline **y** contra el modelo lineal, y extrae de
> ahí una conclusión sobre la naturaleza del problema.
"""
    ),
    md(
        """
---
# Bloque 2 · Métrica y costo, juntos

Fíjate en que la tabla trae una columna `segundos`. No es decoración.

La pregunta de esta actividad **no** es *"¿cuál saca el número más alto?"*. Es:

> **¿Gana lo suficiente para justificar lo que cuesta?**

Un modelo que gana 0,004 y tarda cuatro veces más no es mejor: es más caro.

### ✏️ TODO 3 — El costo relativo
"""
    ),
    code(
        """
costo = comparacion.copy()
costo["veces_mas_lento"] = (costo["segundos"] / costo["segundos"].min()).round(1)
costo["ganancia_vs_arbol"] = (
    costo["media"] - costo.loc[costo["modelo"] == "arbol", "media"].iloc[0]
).round(4)
costo[["modelo", "media", "ganancia_vs_arbol", "segundos", "veces_mas_lento"]]
""",
        """
# TODO 3: ¿cuánto cuesta cada punto de mejora?
costo = comparacion.copy()
costo["veces_mas_lento"] = (costo["segundos"] / costo["segundos"].____()).round(1)
costo["ganancia_vs_arbol"] = (
    costo["media"] - costo.loc[costo["modelo"] == "arbol", "media"].iloc[0]
).round(4)
costo[["modelo", "media", "ganancia_vs_arbol", "segundos", "veces_mas_lento"]]
""",
    ),
    md(
        """
---
# Bloque 3 · ⭐⭐ El ensamble por votación

Ya tenemos tres modelos buenos. La intuición dice que combinarlos debería dar algo mejor que
cualquiera de los tres: cada uno se equivoca en cosas distintas y el voto cancela errores.

### ✏️ TODO 4 — Antes de mirar la tabla, apuesta

**Creo que el ensamble por votación quedará:** `____`
*(por encima del bosque / igual / por debajo)*

### ✏️ TODO 5 — La comparación directa
"""
    ),
    code(
        """
duelo = comparacion[comparacion["modelo"].isin(["bosque_aleatorio", "ensamble_votacion"])]
print(duelo.to_string(index=False), "\\n")

mejor_solo = duelo[duelo["modelo"] == "bosque_aleatorio"].iloc[0]
ensamble = duelo[duelo["modelo"] == "ensamble_votacion"].iloc[0]

diferencia = ensamble["media"] - mejor_solo["media"]
sobrecosto = ensamble["segundos"] / mejor_solo["segundos"]

print(f"Diferencia en F1-macro : {diferencia:+.4f}")
print(f"Ruido entre pliegues   : {mejor_solo['desv_entre_pliegues']:.4f}")
print(f"Sobrecosto en tiempo   : {sobrecosto:.2f}×")
""",
        """
# TODO 5: ¿el ensamble le gana al mejor modelo individual?
duelo = comparacion[comparacion["modelo"].isin(["bosque_aleatorio", "ensamble_votacion"])]
print(duelo.to_string(index=False), "\\n")

mejor_solo = duelo[duelo["modelo"] == "bosque_aleatorio"].iloc[0]
ensamble = duelo[duelo["modelo"] == "____"].iloc[0]

diferencia = ensamble["media"] - mejor_solo["media"]
sobrecosto = ensamble["segundos"] / mejor_solo["segundos"]

print(f"Diferencia en F1-macro : {diferencia:+.4f}")
print(f"Ruido entre pliegues   : {mejor_solo['desv_entre_pliegues']:.4f}")
print(f"Sobrecosto en tiempo   : {sobrecosto:.2f}×")
""",
    ),
    code(
        """
# Autochequeo
assert diferencia < 0, "revisa: el ensamble debería quedar POR DEBAJO del bosque solo"
assert abs(diferencia) < mejor_solo["desv_entre_pliegues"], (
    "y la diferencia debería ser menor que el ruido entre pliegues"
)
print("✅ El ensamble quedó por debajo del bosque, y la diferencia es menor")
print("   que el ruido: no hay evidencia de que ninguno sea mejor.")
print(f"   Pero el ensamble tarda {sobrecosto:.0%} de lo que tarda el bosque.")
print()
print("   Mismo desempeño demostrable, más costo, menos interpretable.")
"""
    ),
    md(
        """
### ✏️ TODO 6 — Por qué no funcionó

**✍️ Tu respuesta:**

*(doble clic aquí y escribe)*

El ensamble combina árbol, bosque y boosting. No mejoró. Da una explicación, sabiendo que
**los ensambles reducen varianza, no sesgo**.

*Pista: mira qué tienen en común los tres modelos combinados.*
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 3 ⭐⭐
>
> **Cifras medidas:**
>
> | | Bosque solo | Ensamble por votación |
> |---|---|---|
> | F1-macro | **0,6909** | 0,6869 |
> | Desv. entre pliegues | 0,0079 | 0,0085 |
> | Segundos | 4,5 | **5,9** |
>
> ⚠️ **Los tiempos varían entre máquinas y entre corridas**; el sobrecosto medido oscila entre un
> 10 % y un 30 %. Lo que no varía es el orden: el ensamble entrena los tres modelos que combina,
> así que **siempre** cuesta más que el más caro de ellos. Al corregir, mira el orden, no el
> porcentaje.
>
> **Diferencia: −0,0040**, menor que el ruido (0,0079). Y además **es el más lento de los seis**.
>
> **El TODO 4 funciona si apuestan.** Casi todos dicen "por encima": es lo que sugiere la
> intuición y lo que dicen los tutoriales. Queda por debajo.
>
> **Respuesta al TODO 6 — la explicación correcta.** Los tres modelos combinados son **todos de
> árboles**: árbol, bosque (árboles en paralelo) y boosting (árboles en serie). Trazan el mismo
> tipo de frontera y **se equivocan en las mismas detecciones**.
>
> Un ensamble funciona cuando sus miembros cometen errores **poco correlacionados**. Aquí están
> muy correlacionados, así que promediar no cancela casi nada. Y encima el árbol simple, que es
> el peor de los tres (0,6708), arrastra el promedio hacia abajo.
>
> > *Un ensamble no es "más modelos". Es más modelos **distintos**. Si todos miran por la misma
> > ventana, promediarlos no amplía la vista.*
>
> **Si alguien propone la mejora correcta**, reconócela: incluir la **regresión logística** en la
> votación aportaría un tipo de error distinto. En este caso probablemente empeoraría el promedio
> porque su desempeño es muy bajo (0,4847), pero el razonamiento es el bueno. Se puede probar en
> vivo si hay tiempo.
>
> **El otro remate**: el modelo que gana es el que ya teníamos desde la Actividad 2.2. Dos
> sesiones del RA3 —ajuste y ensamble— y el modelo no ha mejorado. **Eso también es un
> resultado**, y saberlo con evidencia vale más que sospecharlo.
>
> **Criterio de logro:** identifica que la diferencia no supera el ruido, considera el costo, y
> explica el fracaso por la correlación entre los modelos combinados.
"""
    ),
    md(
        """
---
# Bloque 4 · ¿Sesgo o varianza?

Si ni el ajuste ni el ensamble mejoran, la pregunta es **qué limita** al modelo.

| Síntoma | Diagnóstico | Qué hacer |
|---|---|---|
| Va mucho mejor en entrenamiento que en validación | **Varianza** | Más datos, regularizar, promediar |
| Va parecido en ambos, y ambos mediocres | **Sesgo** | Mejores variables, modelo más flexible |

### ✏️ TODO 7 — El diagnóstico
"""
    ),
    code(
        """
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, cross_val_score

X = entrena[CONFIG["variables"]]
y = entrena[CONFIG["objetivo"]]
grupos = entrena[CONFIG["grupo"]]

modelo = RandomForestClassifier(
    n_estimators=200, max_depth=12, class_weight="balanced",
    random_state=CONFIG["semilla"], n_jobs=-1,
)
modelo.fit(X, y)

en_entrenamiento = f1_score(y, modelo.predict(X), average="macro")
en_validacion = cross_val_score(
    modelo, X, y, groups=grupos,
    cv=GroupKFold(n_splits=AJUSTE["n_pliegues"]), scoring=AJUSTE["metrica"], n_jobs=-1,
).mean()

print(f"F1-macro en entrenamiento : {en_entrenamiento:.4f}")
print(f"F1-macro en validación    : {en_validacion:.4f}")
print(f"Brecha                    : {en_entrenamiento - en_validacion:.4f}")
""",
        """
# TODO 7: ¿el modelo sufre de sesgo o de varianza?
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, cross_val_score

X = entrena[CONFIG["variables"]]
y = entrena[CONFIG["objetivo"]]
grupos = entrena[CONFIG["grupo"]]

modelo = RandomForestClassifier(
    n_estimators=200, max_depth=12, class_weight="balanced",
    random_state=CONFIG["semilla"], n_jobs=-1,
)
modelo.fit(X, y)

en_entrenamiento = f1_score(y, modelo.____(X), average="macro")
en_validacion = ____(
    modelo, X, y, groups=grupos,
    cv=GroupKFold(n_splits=AJUSTE["n_pliegues"]), scoring=AJUSTE["metrica"], n_jobs=-1,
).mean()

print(f"F1-macro en entrenamiento : {en_entrenamiento:.4f}")
print(f"F1-macro en validación    : {en_validacion:.4f}")
print(f"Brecha                    : {en_entrenamiento - en_validacion:.4f}")
""",
    ),
    md(
        """
**✍️ Tu respuesta al TODO 7:**

*(doble clic aquí y escribe)*

1. ¿El problema es sesgo o varianza?
2. Según ese diagnóstico, ¿qué habría que hacer para mejorar de verdad?
3. ¿Por qué eso explica que el ajuste y el ensamble no sirvieran?
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 4
>
> **Cifra medida:** con `max_depth=12` la brecha entre entrenamiento y validación es
> considerable —el bosque memoriza bastante—, pero **la validación se queda estancada en ~0,69
> para todos los modelos y todas las configuraciones probadas en las dos sesiones anteriores**.
>
> **Ese estancamiento es la clave del diagnóstico.** Si el problema fuera solo varianza,
> promediar (bosque) o regularizar (limitar profundidad) habría movido la validación. No la
> movió. **Hay un techo de sesgo**: la información necesaria para distinguir mejor las
> detecciones difíciles **no está en las siete variables** que le dimos.
>
> **Respuestas esperadas:**
>
> 1. **Ambas cosas, pero lo que limita es el sesgo.** La varianza existe y el bosque ya la
>    controla; el techo lo pone la información disponible.
> 2. **Mejores variables.** Lo que más se acerca: la distancia al sensor
>    (`√(x² + y²)` en vez de x e y por separado), el ángulo, el volumen de la caja, o
>    características del segmento. **Ingeniería de características, no más modelo.**
> 3. Porque el ajuste y el ensamble **atacan la varianza**, y la varianza no era el cuello de
>    botella. Es como afinar el motor cuando el problema es que falta camino.
>
> **La frase que cierra el RA3 hasta aquí:**
>
> > *Antes de probar un modelo más potente, pregúntate si el problema es que tu modelo no
> > aprende, o que tus datos no dicen.*
>
> **Si sobra tiempo**, el experimento es de dos minutos y es muy convincente: agregar
> `distancia = np.sqrt(box_center_x**2 + box_center_y**2)` como variable y reentrenar. Es la
> relación física que genera la etiqueta, y una sola variable derivada suele mover más que las
> dos sesiones anteriores juntas.
>
> **Criterio de logro:** diagnostica el techo de sesgo, propone ingeniería de características, y
> conecta ese diagnóstico con el fracaso del ajuste y del ensamble.
"""
    ),
    md(
        """
---
# Cierre · Informe de ensamble

### Los candidatos

| Modelo | Familia | F1-macro | Desv. | Segundos |
|---|---|---|---|---|
| `____` | | | | |
| `____` | | | | |
| `____` | | | | |

**Baseline:** `____` · **Mejor modelo individual:** `____` · **Ensamble:** `____`

### La comparación que importa

**Diferencia entre el ensamble y el mejor individual:** `____`
**Ruido entre pliegues:** `____`
**¿Es distinguible?** `____`
**Sobrecosto en tiempo:** `____`

**Decisión y por qué:** `____`

> Si eliges el más simple, **dilo con el argumento del costo y la interpretabilidad**, no como
> si te conformaras. Elegir el modelo suficiente es una decisión de ingeniería, no una renuncia.

### Diagnóstico

**¿Sesgo o varianza?** `____` · **Evidencia:** `____`
**Qué haría falta para mejorar de verdad:** `____`
"""
    ),
    md_docente(
        """
> ### 🎓 Criterios de logro — Actividad 3.2 (IL 3.2)
>
> | Nivel | Descripción |
> |---|---|
> | **Destacado (4)** | Todo lo del 3, y además: explica el fracaso del ensamble por la **correlación entre sus miembros**; diagnostica el techo de sesgo con evidencia; propone una variable derivada concreta |
> | **Logrado (3)** | Compara los seis candidatos con métrica **y** costo; contrasta la diferencia contra el ruido; concluye que el ensamble no se justifica y lo argumenta |
> | **En desarrollo (2)** | Ejecuta la comparación y elige el de mayor media, sin considerar ruido ni costo |
> | **Inicial (1)** | No usa baseline, o concluye que el ensamble es mejor sin evidencia |
>
> **Lo primero al corregir:** si el informe elige el ensamble *"porque es más avanzado"*. Con
> estos datos es peor en las tres dimensiones —métrica, costo e interpretabilidad—, y esa es la
> lección de la sesión.
"""
    ),
]
