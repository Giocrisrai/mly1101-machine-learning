# Rúbrica de evaluación — EA2 · Aprendizaje supervisado

**Asignatura:** MLY1101 Machine Learning · **Notebook:** `05_alumno_supervisado.ipynb`
**Entregable:** notebook con los 17 TODO resueltos + informe de modelamiento.

**RA2:** *Implementa modelos de aprendizaje supervisado para resolver problemas de predicción,
evaluando su desempeño con métricas pertinentes al contexto del negocio.*

> La escala, la conversión a nota y la calculadora son **las mismas de la EA1**: ver
> [`rubrica_ea1.md`](rubrica_ea1.md) y `herramientas/calcular_nota.py`. Aquí solo cambian los
> indicadores.

---

## Las cinco dimensiones

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

**Cifra de referencia:** `LEVEL_1` 88,89 % · `LEVEL_2` 11,11 %, sobre 40.200 filas limpias.

---

## D2 · La partición (25 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Obtiene 0 segmentos compartidos, **reconoce que la diferencia medida es nula y aun así defiende la partición por grupo** con un argumento de diseño, no de métrica |
| **3** | Parte por grupo, verifica 0 segmentos compartidos y explica qué es la fuga por agrupación |
| **2** | Parte por grupo porque se lo pidieron, sin poder explicar qué problema resuelve |
| **1** | Parte al azar, o no verifica los segmentos compartidos |

**Cifras de referencia:** 29.946 filas de entrenamiento (74,5 %) en **114 segmentos**, 10.254 de
prueba en **39**, **0 compartidos**. Al azar: **153 compartidos**. Diferencia de exactitud entre
ambas: **−0,005**.

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
| Baseline `most_frequent` | **0,8896** | 0,4708 |
| Baseline `stratified` | 0,8061 | 0,5007 |
| Bosque aleatorio | **0,8965** | **0,7025** |

Por clase: `LEVEL_2` precisión **0,5422**, recall **0,4028**, F1 **0,4622** sobre 1.132 casos.
Matriz de confusión: **456** difíciles encontradas, **676** perdidas, **385** falsas alarmas.

---

## D4 · Fuga de información (15 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Formula la pregunta de disponibilidad temporal **y la aplica a una variable distinta** de la del ejercicio; distingue esta fuga de la del bloque 2 |
| **3** | Mide la inflación, explica por qué `num_lidar_points` no puede usarse y formula la pregunta correcta |
| **2** | Reproduce la medición sin poder explicar por qué esa variable es problemática |
| **1** | Llama "fuga" a cualquier variable muy predictiva |

**Cifra de referencia:** incluir `num_lidar_points` sube el F1-macro de **0,7025 a 0,7543**
(+0,052) y la exactitud a 0,9264.

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

**Cifras de referencia:** `box_center_x` **0,5202** y `speed_mps` **0,2613** concentran el 78 %
de la importancia. Mecanismo esperado: a mayor distancia, menos puntos láser; en movimiento, el
objeto se difumina entre barridos.

---

## Qué mirar al corregir, en este orden

1. **El bloque 3.** Si no puede explicar por qué el baseline saca 88,96 %, no entendió el
   desbalance y todo lo demás lo va a leer mal.
2. **La decisión del informe.** *"Sí, tiene 90 % de exactitud"* es **Inicial** aunque todos los
   TODO estén en verde. *"No, se pierde 6 de cada 10 difíciles"* es **Logrado**. *"Depende de si
   decide o solo alerta"* es **Destacado**.
3. **La justificación de la partición.** Es el único punto que distingue a quien razona de quien
   optimiza el número.

## Retroalimentación sugerida

1. *"Reportaste la exactitud. ¿Cuánto sacaba el modelo que no mira los datos?"*
2. *"Dices que el modelo funciona. ¿Para quién, y para qué decisión?"*
3. *"Tu partición es la correcta. ¿La habrías elegido igual si el número hubiera salido peor?"*
