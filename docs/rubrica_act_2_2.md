# Pauta de la Actividad 2.2 · Modelamiento supervisado

**Asignatura:** MLY1101 Machine Learning · **Notebook:** `05_alumno_supervisado.ipynb`
**Horas del programa:** 6 · **Entregable:** notebook con los 17 TODO + informe de modelamiento.

**RA2:** *Aplica modelos estadísticos al conjunto de datos procesados para interpretarlos,
utilizando metodologías ágiles, con la finalidad de obtener conocimientos relevantes que
permitan responder a las necesidades del contexto de negocio, considerando aspectos éticos.*

**IL 2.2:** *Construye modelos de aprendizaje supervisado para problemas de regresión y
clasificación según la naturaleza del caso.*

> ⚠️ **Esto es una pauta formativa, no el instrumento sumativo.** La evaluación calificada del
> RA2 es la **Evaluación Parcial 2** (*Construcción e interpretación de modelos*, 40 % de la
> ponderación parcial), que se rinde sobre uno de los casos oficiales —Telco, Housing o
> Spotify— y usa la rúbrica institucional de indicadores ponderados por % de logro. Esta pauta
> sirve para acompañar y retroalimentar la actividad.

> **Alcance:** la actividad cubre **clasificación**. La regresión, que el IL2.2 también
> menciona, se trabaja en la Parcial 2 sobre *House Prices*.

> La escala, la conversión a nota y la calculadora son **las mismas del RA1**: ver
> [`rubrica_ra1.md`](rubrica_ra1.md) y `herramientas/calcular_nota.py`. Aquí solo cambian los
> indicadores.

---

## Las cinco dimensiones de la pauta

| Dim. | Qué evalúa | Dónde se evidencia | Peso |
|---|---|---|---|
| **D1** | Plantea el problema supervisado: `X`, `y`, y qué **no** puede ser variable | Bloque 1 (TODO 1–3) | 15 % |
| **D2** | Diseña una partición honesta y la justifica | Bloque 2 (TODO 4–6) | 25 % |
| **D3** | Evalúa contra un baseline y con métricas pertinentes | Bloques 3–5 (TODO 7–13) | 30 % |
| **D4** | Reconoce la fuga de información y sabe detectarla | Bloque 6 (TODO 15–16) | 15 % |
| **D5** | Interpreta el modelo y decide sobre su uso | Bloque 7 + informe | 15 % |

**D3 es la de mayor peso: es el núcleo de la sesión.**

---

## D1 · Planteamiento del problema (15 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Distingue las tres razones de exclusión sin confundirlas (identificador · unidad de agrupación · derivada de la etiqueta) y detecta que `weather`/`time_of_day` son constantes dentro del segmento |
| **3** | Arma correctamente `X`, `y` y el grupo; explica por qué `id_interno` y `num_lidar_points` quedan fuera |
| **2** | Arma la tabla pero justifica las exclusiones con "no aporta" sin distinguir motivos |
| **1** | Incluye el identificador o la etiqueta entre las variables |

**Cifra de referencia:** `LEVEL_1` 87,67 % · `LEVEL_2` 12,33 %, sobre **530.396** filas (0 nulos).

---

## D2 · La partición (25 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Obtiene 0 segmentos compartidos, **reconoce que la diferencia medida es nula y aun así defiende la partición por grupo** con un argumento de diseño, no de métrica |
| **3** | Parte por grupo, verifica 0 segmentos compartidos y explica qué es la fuga por agrupación |
| **2** | Parte por grupo porque se lo pidieron, sin poder explicar qué problema resuelve |
| **1** | Parte al azar, o no verifica los segmentos compartidos |

**Cifras de referencia:** 384.280 filas de entrenamiento (72,5 %) en **30 segmentos**, 146.116 de
prueba en **10**, **0 compartidos**. Al azar: **40 compartidos**.

> ⚠️ **El discriminador de esta dimensión.** Si el argumento del alumno es *"partí por grupo
> porque da mejor resultado"*, no llega a **3**: la medición dice lo contrario. La respuesta
> correcta es que un riesgo que no se manifiesta en los datos de prueba sigue siendo un riesgo, y
> que el método se elige por cómo se generaron los datos, no por el número que produce.

---

## D3 · Evaluación (30 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Todo lo del nivel 3, y además: razona el equilibrio precisión/recall en términos del **costo asimétrico del error en este dominio**; observa que `most_frequent` y `stratified` se ordenan al revés según la métrica |
| **3** | Entrena el baseline, explica por qué su exactitud coincide con la proporción de la clase mayoritaria, compara en dos métricas, lee la matriz de confusión y cuantifica los falsos negativos |
| **2** | Entrena y reporta métricas correctas pero concluye desde la exactitud sola |
| **1** | Reporta solo la exactitud, o no usa baseline |

**Cifras de referencia:**

| Modelo | Exactitud | F1-macro |
|---|---|---|
| Baseline `most_frequent` | **0,8172** | (clase mayoritaria en prueba) |
| Bosque aleatorio | **0,7805** | **0,4822** |

Por clase: `LEVEL_2` precisión **0,1847**, recall **0,0588**, F1 **0,0893** sobre 26.713 casos.
Matriz: **1.572** difíciles encontradas, **25.141** perdidas, **6.937** falsas alarmas.

---

## D4 · Fuga de información (15 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Formula la pregunta de disponibilidad temporal **y la aplica a una variable distinta** de la del ejercicio; distingue esta fuga de la del bloque 2 |
| **3** | Mide la inflación, explica por qué `num_lidar_points` no puede usarse y formula la pregunta correcta |
| **2** | Reproduce la medición sin poder explicar por qué esa variable es problemática |
| **1** | Llama "fuga" a cualquier variable muy predictiva |

**Cifra de referencia:** incluir `num_lidar_points` es fuga (la etiqueta se deriva de ahí).
Se mide en el bloque 6 de la actividad; no recites de memoria un delta del hilo viejo.

**La pregunta que se busca:** *¿voy a tener esta variable, con este valor, en el momento en que
necesite hacer la predicción?*

---

## D5 · Interpretación y decisión (15 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Propone un mecanismo físico para las variables dominantes; distingue entre un modelo que **decide** y uno que **asiste**; señala qué haría falta para cambiar su decisión |
| **3** | Identifica las dos variables dominantes, no confunde importancia con causalidad, y decide sobre producción argumentando con el costo del error |
| **2** | Interpreta la importancia pero la decisión final es "el modelo es bueno/malo" sin referirse al uso |
| **1** | No interpreta, o justifica la decisión solo con la exactitud |

**Cifras de referencia:** salen del bosque de la corrida; no recites un ranking del hilo viejo.
Mecanismo esperado: a mayor distancia, menos puntos láser; en movimiento, el objeto se difumina
entre barridos.

---

## Qué mirar al corregir, en este orden

1. **El bloque 3.** Si no puede explicar por qué el baseline saca ~81,7 % (proporción de
   `LEVEL_1` en prueba), no entendió el desbalance y todo lo demás lo va a leer mal.
2. **La decisión del informe.** *"Sí, tiene 78 % de exactitud"* es **Inicial** aunque todos los
   TODO estén en verde. *"No, se pierde 94 de cada 100 difíciles"* es **Logrado**. *"Depende de si
   decide o solo alerta"* es **Destacado**.
3. **La justificación de la partición.** Es el único punto que distingue a quien razona de quien
   optimiza el número.

## Retroalimentación sugerida

1. *"Reportaste la exactitud. ¿Cuánto sacaba el modelo que no mira los datos?"*
2. *"Dices que el modelo funciona. ¿Para quién, y para qué decisión?"*
3. *"Tu partición es la correcta. ¿La habrías elegido igual si el número hubiera salido peor?"*
