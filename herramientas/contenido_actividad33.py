"""Fuente única de la Actividad 3.3 — Robustez y selección de modelos.

Indicadores de logro **IL 3.3** (*evalúa la capacidad de generalización mediante validación
cruzada y métricas avanzadas*) e **IL 3.4** (*sustenta la selección de la solución analítica
óptima mediante la comparación cuantitativa de modelos*).

**11 horas pedagógicas.** Es la actividad de cierre del RA3 y la que prepara el EFT.

Regenerar:  python herramientas/construir_notebooks.py
"""

from __future__ import annotations

from contenido_actividad31 import PREPARACION, RA3, SETUP
from contenido_semana01 import code, md, md_docente

CELDAS_ACT33: list[dict] = [
    md(
        f"""
# MLY1101 · Machine Learning — Actividad 3.3
## Robustez y selección de modelos

{RA3}

**Indicadores de logro:**

- **IL 3.3** · evalúa la capacidad de generalización del modelo mediante esquemas de validación
  cruzada y métricas de desempeño avanzadas según la naturaleza del problema.
- **IL 3.4** · sustenta la selección de la solución analítica óptima mediante la comparación
  cuantitativa de modelos, asegurando el cumplimiento de los objetivos del negocio.

---

### La pregunta que cierra la asignatura

Llevas tres actividades midiendo modelos. Ninguna ha respondido lo único que importa al final:

> **¿Cuál eliges, y cómo lo defiendes?**

No es una pregunta de estadística. La estadística te dice si dos modelos son distinguibles; la
elección la haces tú, con criterios que la métrica no contiene.

---

### La idea central, y es incómoda

> **Casi todas las comparaciones de modelos que verás en internet no distinguen nada.**

Alguien reporta 0,847 contra 0,843 y concluye que el primero es mejor. Si la variabilidad entre
particiones es 0,012, esa diferencia es ruido: con otra semilla se habría invertido el orden.

Hoy vas a aprender a hacer esa comparación bien. Y vas a descubrir que el modelo que llevas
usando desde la Actividad 2.2 **no es distinguible** del ensamble que construiste ayer.

---

### Al final de la sesión debes entregar

La **tabla de selección sustentada**: métrica, estabilidad, costo e interpretabilidad juntos,
con la decisión argumentada. Es el esqueleto del apartado de modelamiento del EFT.
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Actividad 3.3
>
> **11 horas pedagógicas**, la segunda actividad más larga del programa. La distribución de abajo
> cubre ~4 h de trabajo guiado; el resto es para aplicar el esquema completo al caso oficial del
> equipo, que es literalmente lo que se entrega en la Parcial 3 y en el EFT.
>
> | Bloque | Min | Foco |
> |---|---|---|
> | 0 · Encuadre | 15 | Elegir no es una pregunta de estadística |
> | 1 · La variabilidad entre pliegues ⭐⭐ | 50 | Qué diferencias se pueden distinguir |
> | 2 · Métricas avanzadas | 45 | Más allá del F1: curva PR y umbral |
> | 3 · La tabla de selección ⭐⭐ | 50 | Métrica, costo e interpretabilidad juntos |
> | 4 · Defender la decisión | 30 | Lo que se evalúa en el EFT |
> | Cierre | 15 | Informe de selección |
>
> **Los dos imprescindibles son el 1 y el 3.** El 2 se puede acortar mostrando solo la curva.
"""
    ),
    md("---\n## Preparación del entorno"),
    code(SETUP),
    code(PREPARACION),
    md(
        """
---
# Bloque 1 · ⭐⭐ ¿Qué diferencias se pueden distinguir?

Cada pliegue de la validación cruzada da una puntuación distinta, porque le tocan segmentos
distintos. Esa dispersión **no es un defecto de la medición**: es la incertidumbre real de tu
estimación.

Y da la regla práctica: **si dos modelos difieren menos que esa dispersión, no puedes afirmar
que uno sea mejor.**

### ✏️ TODO 1 — Ver la dispersión, no solo la media
"""
    ),
    code(
        """
comparacion = optimizacion.comparar_ensambles(marcada, CONFIG, AJUSTE)
print(comparacion.to_string(index=False), "\\n")

fig, ejes = plt.subplots(figsize=(7, 3.5))
ejes.errorbar(
    comparacion["media"], comparacion["modelo"],
    xerr=comparacion["desv_entre_pliegues"], fmt="o", capsize=4,
)
ejes.set_xlabel("F1-macro (media ± desviación entre pliegues)")
ejes.set_title("Si las barras se solapan, no hay evidencia de diferencia")
plt.tight_layout()
plt.show()
""",
        """
# TODO 1: compara los modelos mostrando la dispersión, no solo la media.
comparacion = optimizacion.____(marcada, CONFIG, AJUSTE)
print(comparacion.to_string(index=False), "\\n")

fig, ejes = plt.subplots(figsize=(7, 3.5))
ejes.errorbar(
    comparacion["media"], comparacion["modelo"],
    xerr=comparacion["____"], fmt="o", capsize=4,
)
ejes.set_xlabel("F1-macro (media ± desviación entre pliegues)")
ejes.set_title("Si las barras se solapan, no hay evidencia de diferencia")
plt.tight_layout()
plt.show()
""",
    ),
    md(
        """
### ✏️ TODO 2 — Poner número a la distinguibilidad
"""
    ),
    code(
        """
robustez = optimizacion.analizar_robustez(comparacion)
robustez
""",
        """
# TODO 2: ¿qué modelos son distinguibles del mejor?
robustez = optimizacion.____(comparacion)
robustez
""",
    ),
    code(
        """
# Autochequeo
indistinguibles = robustez[~robustez["distinguible_del_mejor"]]["modelo"].tolist()
assert len(indistinguibles) >= 2, (
    "revisa: debería haber al menos dos modelos indistinguibles entre sí"
)
print("✅ Modelos que NO se pueden distinguir del mejor:", indistinguibles)
print()
print("   Traducido: con la evidencia que tienes, afirmar que uno es mejor")
print("   que el otro no está respaldado. Cualquiera de los dos es defendible.")
"""
    ),
    md(
        """
### ✏️ TODO 3 — La consecuencia

**✍️ Tu respuesta:**

*(doble clic aquí y escribe)*

1. El bosque saca más media que el ensamble. ¿Puedes afirmar que es mejor? ¿Por qué?
2. Si dos modelos son indistinguibles, **¿con qué criterio eliges?**
3. Alguien te muestra una comparación de modelos sin desviaciones, solo medias. ¿Qué le pides?
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 1 ⭐⭐
>
> **Cifras medidas:**
>
> | Modelo | Media | Diferencia vs mejor | Ruido del mejor | ¿Distinguible? |
> |---|---|---|---|---|
> | Bosque aleatorio | 0,6909 | — | 0,0079 | — |
> | Ensamble por votación | 0,6869 | 0,0040 | 0,0079 | **No** |
> | Gradient boosting | 0,6804 | 0,0105 | 0,0079 | Sí |
> | Árbol | 0,6708 | 0,0201 | 0,0079 | Sí |
> | Regresión logística | 0,4847 | 0,2062 | 0,0079 | Sí |
> | Baseline | 0,4705 | 0,2204 | 0,0079 | Sí |
>
> **El gráfico de barras de error es la herramienta de la sesión.** Las barras del bosque y del
> ensamble se solapan; las del boosting ya no. Se ve antes de leer ninguna tabla.
>
> **Respuestas esperadas:**
>
> 1. **No.** La diferencia (0,0040) es la mitad del ruido (0,0079). Con otra semilla el orden
>    podría invertirse. Lo correcto es decir *"son indistinguibles con esta evidencia"*.
> 2. **Con los criterios que no son la métrica:** costo de cómputo, interpretabilidad, facilidad
>    de mantener, latencia en producción. Ese es el bloque 3, y es el corazón del IL3.4.
> 3. **Las desviaciones, o el esquema de validación.** Una tabla de medias sin dispersión no
>    permite concluir nada, y pedirla es la señal de que se entendió la sesión.
>
> **El matiz honesto que conviene declarar**, porque un alumno bueno lo va a preguntar: comparar
> la diferencia contra una desviación es una **regla práctica**, no un test estadístico formal.
> Lo riguroso sería una prueba pareada sobre los pliegues o intervalos de confianza. La regla
> basta para frenar la conclusión apresurada, que es el 90 % del problema en la práctica.
>
> **Criterio de logro:** interpreta el solapamiento, concluye que bosque y ensamble son
> indistinguibles, e identifica que la elección se traslada a criterios no métricos.
"""
    ),
    md(
        """
---
# Bloque 2 · Métricas avanzadas: más allá de un número

El F1 fija implícitamente un umbral de decisión en 0,5. Pero un clasificador no devuelve una
etiqueta: devuelve una **probabilidad**, y el umbral lo eliges tú.

Mover el umbral intercambia precisión por recall. Cuál conviene **lo decide el costo del error
en el dominio**, no la estadística.

### ✏️ TODO 4 — La curva precisión-recall

*(Se usa la curva PR y no la ROC porque con clases desbalanceadas la ROC se ve optimista: su eje
depende de los verdaderos negativos, que aquí son abundantes y fáciles.)*
"""
    ),
    code(
        """
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score

prueba = marcada[marcada["particion"] == "prueba"]
X_entrena, y_entrena = entrena[CONFIG["variables"]], entrena[CONFIG["objetivo"]]
X_prueba, y_prueba = prueba[CONFIG["variables"]], prueba[CONFIG["objetivo"]]

modelo = RandomForestClassifier(
    n_estimators=200, max_depth=12, class_weight="balanced",
    random_state=CONFIG["semilla"], n_jobs=-1,
).fit(X_entrena, y_entrena)

positiva = "LEVEL_2"
probabilidades = modelo.predict_proba(X_prueba)[:, list(modelo.classes_).index(positiva)]
es_positiva = (y_prueba == positiva).astype(int)

print(f"Average precision (LEVEL_2): {average_precision_score(es_positiva, probabilidades):.4f}")
print(f"Proporción de LEVEL_2       : {es_positiva.mean():.4f}  <- el piso de un clasificador al azar")

PrecisionRecallDisplay.from_predictions(es_positiva, probabilidades, name="bosque aleatorio")
plt.title("Precisión vs recall para la clase minoritaria")
plt.tight_layout()
plt.show()
""",
        """
# TODO 4: la curva precisión-recall de la clase minoritaria.
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score

prueba = marcada[marcada["particion"] == "prueba"]
X_entrena, y_entrena = entrena[CONFIG["variables"]], entrena[CONFIG["objetivo"]]
X_prueba, y_prueba = prueba[CONFIG["variables"]], prueba[CONFIG["objetivo"]]

modelo = RandomForestClassifier(
    n_estimators=200, max_depth=12, class_weight="balanced",
    random_state=CONFIG["semilla"], n_jobs=-1,
).fit(X_entrena, y_entrena)

positiva = "LEVEL_2"
probabilidades = modelo.____(X_prueba)[:, list(modelo.classes_).index(positiva)]
es_positiva = (y_prueba == positiva).astype(int)

print(f"Average precision (LEVEL_2): {average_precision_score(es_positiva, probabilidades):.4f}")
print(f"Proporción de LEVEL_2       : {es_positiva.mean():.4f}  <- el piso de un clasificador al azar")

PrecisionRecallDisplay.from_predictions(es_positiva, probabilidades, name="bosque aleatorio")
plt.title("Precisión vs recall para la clase minoritaria")
plt.tight_layout()
plt.show()
""",
    ),
    md(
        """
### ✏️ TODO 5 — Elegir el umbral con criterio de negocio

En la Actividad 2.2 el modelo encontraba el **40 %** de las detecciones difíciles. Veamos qué
umbral haría falta para encontrar el 70 %, y qué se paga por ello.
"""
    ),
    code(
        """
from sklearn.metrics import precision_recall_curve

precision, recall, umbrales = precision_recall_curve(es_positiva, probabilidades)

objetivos = [0.40, 0.55, 0.70, 0.85]
filas = []
for objetivo in objetivos:
    i = int(np.argmin(np.abs(recall[:-1] - objetivo)))
    predicho = (probabilidades >= umbrales[i]).astype(int)
    filas.append(
        {
            "recall_objetivo": objetivo,
            "umbral": round(float(umbrales[i]), 3),
            "recall_real": round(float(recall[i]), 3),
            "precision": round(float(precision[i]), 3),
            "falsas_alarmas": int(((predicho == 1) & (es_positiva == 0)).sum()),
        }
    )
pd.DataFrame(filas)
""",
        """
# TODO 5: ¿qué umbral hace falta para cada nivel de recall, y qué cuesta?
from sklearn.metrics import precision_recall_curve

precision, recall, umbrales = precision_recall_curve(es_positiva, probabilidades)

objetivos = [0.40, 0.55, 0.70, 0.85]
filas = []
for objetivo in objetivos:
    i = int(np.argmin(np.abs(recall[:-1] - objetivo)))
    predicho = (probabilidades >= umbrales[i]).astype(int)
    filas.append(
        {
            "recall_objetivo": objetivo,
            "umbral": round(float(umbrales[i]), 3),
            "recall_real": round(float(recall[i]), 3),
            "precision": round(float(precision[i]), 3),
            "falsas_alarmas": int(((predicho == 1) & (es_positiva == ____)).sum()),
        }
    )
pd.DataFrame(filas)
""",
    ),
    md(
        """
**✍️ Tu respuesta al TODO 5:**

*(doble clic aquí y escribe)*

En este dominio, una **falsa alarma** significa que el vehículo desconfía de una detección que
era buena: es prudente de más. Un **falso negativo** significa que confía en una detección mala.

¿Qué umbral elegirías? Justifica con el costo de cada tipo de error, no con la métrica.
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 2
>
> **Cifra de referencia:** la proporción de `LEVEL_2` en la prueba es **0,110**, que es el
> *average precision* que sacaría un clasificador al azar. El modelo saca bastante más, así que
> **sí aprendió algo**, aunque el F1 al umbral 0,5 sea modesto.
>
> **Ese contraste vale la pena señalarlo:** el mismo modelo que en la Actividad 2.2 parecía flojo
> (recall 0,40) tiene capacidad de ordenamiento útil. Lo que estaba mal elegido era **el
> umbral**, no el modelo.
>
> **La tabla del TODO 5 es la que conecta con el negocio.** Subir el recall del 40 % al 70 %
> exige bajar el umbral, y eso multiplica las falsas alarmas. Los números concretos dependen de
> la corrida; lo que no cambia es la forma del intercambio.
>
> **Respuestas esperadas:**
>
> - **Umbral bajo, recall alto**, porque el costo es **asimétrico**: una falsa alarma es que el
>   vehículo frene de más; un falso negativo es que confíe en una detección mala. En seguridad,
>   la asimetría es clara. **Nivel destacado** si lo argumenta así.
> - **"Depende de cuántas falsas alarmas tolera el sistema"** — también correcta, y más madura:
>   si el vehículo desconfía del 30 % de las detecciones, se vuelve inútil por prudente.
> - *"El umbral que maximiza el F1"* — **insuficiente**. El F1 pondera precisión y recall por
>   igual, y en este dominio no valen lo mismo. Es exactamente la respuesta que la sesión busca
>   corregir.
>
> **La frase:** *el umbral no es un hiperparámetro que se optimiza: es una decisión de negocio
> que se documenta.*
>
> **Criterio de logro:** interpreta la curva PR contra la proporción de la clase positiva y elige
> un umbral argumentando con el costo asimétrico del error.
"""
    ),
    md(
        """
---
# Bloque 3 · ⭐⭐ La tabla de selección

Aquí está el **IL 3.4**: *sustentar* la selección. No reportar el máximo, **sustentar**.

Un modelo que gana por un margen indistinguible del ruido, tarda quince veces más y no se puede
explicar **no es la solución óptima**: es la que sacó el número más alto una vez.

### ✏️ TODO 6 — Las cuatro dimensiones juntas
"""
    ),
    code(
        """
seleccion = optimizacion.tabla_de_seleccion(comparacion, robustez, CONFIG)
seleccion
""",
        """
# TODO 6: métrica, estabilidad, costo e interpretabilidad en una sola tabla.
seleccion = optimizacion.____(comparacion, robustez, CONFIG)
seleccion
""",
    ),
    md(
        """
### ✏️ TODO 7 — La decisión

**✍️ Tu respuesta:**

*(doble clic aquí y escribe)*

Elige un modelo y defiéndelo. Tu argumento debe referirse a **las cuatro columnas**, no solo a
la primera.

**Modelo elegido:** `____`

| Criterio | Cómo lo justifica |
|---|---|
| Desempeño | `____` |
| Estabilidad | `____` |
| Costo | `____` |
| Interpretabilidad | `____` |

**Y la pregunta de control:** ¿qué modelo elegirías si el sistema tuviera que responder en
milisegundos dentro del vehículo?
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 3 ⭐⭐
>
> **Cifras medidas:**
>
> | Modelo | Media | Desv. | ¿Distinguible? | Segundos | Veces más lento | Interpretabilidad |
> |---|---|---|---|---|---|---|
> | Bosque aleatorio | 0,6909 | 0,0079 | — | 4,5 | **15,0×** | media |
> | Ensamble | 0,6869 | 0,0085 | No | 5,9 | 19,7× | **baja** |
> | Gradient boosting | 0,6804 | 0,0085 | Sí | 1,3 | 4,3× | media |
> | Árbol | 0,6708 | 0,0166 | Sí | 0,3 | **1,0×** | **alta** |
> | Regresión logística | 0,4847 | 0,0049 | Sí | 0,3 | 1,0× | alta |
>
> ⚠️ **Los segundos varían entre máquinas**; las medias y las desviaciones, no. Al corregir, el
> orden de costo es lo que importa, no el valor absoluto.
>
> **No hay una única respuesta correcta, y hay que decirlo.** Lo que se evalúa es el argumento.
> Tres defensas válidas:
>
> - **Bosque aleatorio.** Mejor media, estabilidad buena (0,0079, la mitad que el árbol), y el
>   ensamble no le gana. El costo de 4,5 s es irrelevante si se entrena una vez al día.
>   **Es la respuesta más común y es correcta.**
> - **Gradient boosting.** Pierde 0,0105 —distinguible, pero pequeño— y es **3,5 veces más
>   rápido** que el bosque. Si el reentrenamiento es frecuente o los datos crecen, es la
>   decisión de ingeniería. **Nivel destacado** si argumenta el intercambio.
> - **Árbol de decisión.** Pierde 0,0201 y tiene el doble de varianza, pero **se puede dibujar y
>   explicar a un comité**. Defendible si el requisito es auditabilidad —banca, salud—.
>   **Destacado** si nombra ese contexto.
>
> **El ensamble no es defendible.** Peor media, más varianza, más lento y menos interpretable.
> Pierde en las cuatro columnas. Si alguien lo elige "porque es más avanzado", ahí está la
> lección.
>
> **La pregunta de control** cambia la respuesta y por eso está: en tiempo real dentro del
> vehículo, 200 árboles por detección y varias detecciones por milisegundo no es viable. La
> respuesta apunta al **árbol** o al boosting, o a destilar el bosque en algo más liviano.
>
> > *La solución óptima no existe en abstracto. Existe la óptima **para un uso**, y por eso hay
> > que declarar el uso antes de elegir.*
>
> **Criterio de logro:** elige y defiende usando las cuatro dimensiones, y cambia su respuesta
> de forma coherente cuando cambia la restricción.
"""
    ),
    md(
        """
---
# Bloque 4 · Defender la decisión

En el EFT vas a presentar esto y te van a hacer preguntas cruzadas. Estas son las cinco que
más se repiten. Prepara tu respuesta.

### ✏️ TODO 8 — El ensayo

**✍️ Tus respuestas:**

*(doble clic aquí y escribe)*

1. *"¿Por qué este modelo y no el que sacó mejor número?"* → `____`
2. *"¿Cómo sabes que no está sobreajustado?"* → `____`
3. *"Si te doy el doble de datos, ¿mejoraría?"* → `____`
4. *"¿Qué pasa si los datos de producción no se parecen a los de entrenamiento?"* → `____`
5. *"¿Cuánto de tu mejora es real y cuánto es azar?"* → `____`
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 4
>
> **Este bloque es un ensayo del EFT**, que se evalúa con defensa individual y preguntas
> cruzadas. Vale la pena hacerlo en voz alta, en parejas, no por escrito.
>
> **Las respuestas que buscamos:**
>
> 1. Porque la diferencia **no supera la variabilidad entre pliegues**, así que no hay evidencia
>    de que sea mejor; y el que elegí gana en costo o interpretabilidad, que sí son
>    distinguibles.
> 2. Por el esquema de validación: **`GroupKFold` con 0 segmentos compartidos**, y la brecha
>    entre entrenamiento y validación. Mencionar que la prueba **no** se usó para elegir nada.
> 3. **Probablemente poco.** El diagnóstico de la Actividad 3.2 fue techo de **sesgo**, no de
>    varianza: más datos de las mismas variables no rompen ese techo. Lo que haría falta son
>    **variables nuevas**.
> 4. Es el sesgo de muestreo de la Actividad 1.4: el censo de Waymo son **793 segmentos soleados
>    de 798**. El modelo no tiene nada que decir sobre lluvia. Hay que **acotar el dominio de uso
>    y declararlo**, y monitorizar la deriva.
> 5. La diferencia contra el baseline (**+0,23 de F1-macro**) es real: veintiocho veces el ruido.
>    Las diferencias entre los modelos buenos, en cambio, están **dentro** del ruido.
>
> **Fíjate en que las cinco respuestas recorren toda la asignatura:** RA1 (sesgo de muestreo),
> RA2 (partición y evaluación por clase) y RA3 (validación y selección). Ese recorrido es
> exactamente lo que el EFT pide demostrar, y por eso conviene cerrar la sesión señalándolo.
>
> **Criterio de logro:** responde las cinco conectando con evidencia producida por él mismo, no
> con generalidades.
"""
    ),
    md(
        """
---
# Cierre · Informe de selección

Este informe **es** el apartado de modelamiento del EFT. Guárdalo.

### El problema

**Qué se predice:** `____` · **Métrica principal y por qué:** `____`
**Costo del error** (qué pasa con un falso positivo y con un falso negativo): `____`

### Esquema de validación

| Campo | Valor |
|---|---|
| Tipo | `____` |
| Pliegues | `____` |
| Agrupación | `____` |
| Segmentos compartidos | `____` |
| ¿Se usó la prueba para elegir algo? | `____` |

### Comparación cuantitativa

| Modelo | F1-macro | Desv. | ¿Distinguible del mejor? | Segundos | Interpretabilidad |
|---|---|---|---|---|---|
| Baseline | | | | | |
| `____` | | | | | |
| `____` | | | | | |

**Modelos indistinguibles entre sí:** `____`

### La decisión

**Modelo elegido:** `____`

| Criterio | Justificación |
|---|---|
| Desempeño | `____` |
| Estabilidad | `____` |
| Costo | `____` |
| Interpretabilidad | `____` |

**Umbral de decisión elegido y por qué:** `____`

### Límites, dichos por mí antes de que me los pregunten

**Qué mejora sería real y cuál sería ruido:** `____`
**En qué condiciones NO debe usarse este modelo:** `____`
**Qué haría falta para la próxima mejora:** `____`

> Esa última sección es la que separa un informe técnico de un informe de ventas. Un modelo
> presentado con sus límites es utilizable; uno presentado como universal, no.
"""
    ),
    md_docente(
        """
> ### 🎓 Criterios de logro — Actividad 3.3 (IL 3.3 e IL 3.4)
>
> | Nivel | Descripción |
> |---|---|
> | **Destacado (4)** | Todo lo del 3, y además: elige el umbral con el **costo asimétrico** del dominio; cambia coherentemente su elección al cambiar la restricción; declara los límites del modelo sin que se los pregunten |
> | **Logrado (3)** | Valida con `GroupKFold`, reporta media **y** dispersión, identifica qué modelos son indistinguibles, y sustenta la elección con las cuatro dimensiones |
> | **En desarrollo (2)** | Compara con validación cruzada pero elige por la media más alta, sin considerar dispersión ni costo |
> | **Inicial (1)** | Reporta un único número por modelo; elige el máximo sin justificar |
>
> **Qué mirar al corregir, en este orden:**
>
> 1. **¿Hay dispersión en la tabla?** Sin ella no hay comparación, solo un ranking de ruido.
> 2. **¿La justificación usa las cuatro columnas?** Si solo habla de la métrica, es
>    *En desarrollo* por completa que esté la tabla.
> 3. **¿Declara los límites?** Es lo que el EFT evalúa en la defensa, y casi nadie lo escribe.
>
> **Enlace con el EFT:** este informe es el apartado de modelamiento y optimización que pide la
> Evaluación Final Transversal. La diferencia es que allí va sobre uno de los casos oficiales
> —Telco, Housing o Spotify— y acompañado del análisis no supervisado del RA2 y del análisis de
> sesgos del RA1. **La estructura es esta.**
"""
    ),
]
