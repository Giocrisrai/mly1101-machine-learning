# Rúbrica de evaluación — EA3 · Aprendizaje no supervisado

**Asignatura:** MLY1101 Machine Learning · **Notebook:** `06_alumno_no_supervisado.ipynb`
**Entregable:** notebook con los 11 TODO resueltos + informe de segmentación.

**RA3:** *Aplica técnicas de aprendizaje no supervisado para descubrir patrones y estructuras en
conjuntos de datos, interpretando los resultados en función del contexto del negocio.*

> Escala, conversión a nota y calculadora: las mismas de la EA1
> ([`rubrica_ea1.md`](rubrica_ea1.md)).

---

## Las cinco dimensiones

| Dim. | Qué evalúa | Dónde se evidencia | Peso |
|---|---|---|---|
| **D1** | Prepara los datos para un algoritmo de distancias | Bloque 1 (TODO 1–2) | 15 % |
| **D2** | Elige el número de grupos con criterio | Bloque 2 (TODO 3–4) | 25 % |
| **D3** | **Interpreta** los grupos: les da nombre y sentido | Bloque 3 (TODO 5–7) | 30 % |
| **D4** | Contrasta con la etiqueta sin confundirlo con una evaluación | Bloque 4 (TODO 8–9) | 15 % |
| **D5** | Reduce dimensionalidad y comunica honestamente | Bloque 5 + informe | 15 % |

**D3 es la de mayor peso.** Ejecutar K-medias es una línea; interpretarlo es la asignatura.

---

## D1 · Preparación (15 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Cuantifica la diferencia de escalas, explica el efecto sobre la distancia euclídea y nota que la etiqueta viaja en la tabla **sin** entrar al agrupamiento |
| **3** | Escala correctamente y explica por qué hace falta en un algoritmo basado en distancias |
| **2** | Escala porque se lo pidieron, sin poder explicar el efecto |
| **1** | Agrupa sin escalar |

**Cifra de referencia:** la razón entre la mayor y la menor desviación típica es de ~3 órdenes de
magnitud (`num_lidar_points` frente a `box_height`).

---

## D2 · Elección de `k` (25 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Reconoce que ningún criterio decide solo y **suspende la decisión hasta ver el contenido de los grupos**; defiende un `k` distinto al de la silueta máxima con argumento de dominio |
| **3** | Prueba varios `k`, explica que la inercia siempre baja, identifica el máximo de la silueta y justifica su elección |
| **2** | Elige un `k` razonable pero justifica solo con un criterio |
| **1** | Elige el `k` de inercia mínima, o no justifica |

**Cifras de referencia:**

| k | Inercia | Silueta |
|---|---|---|
| 2 | 109.384 | 0,4498 |
| **3** | 84.047 | **0,4730** ← máximo |
| 4 | 67.035 | 0,4630 |
| 8 | 39.057 | 0,3300 |

> **El discriminador.** *"Elegí k = 8 porque la inercia es mínima"* es un error de concepto, no
> de cálculo: la inercia siempre baja. *"Necesito ver qué hay dentro antes de decidir"* es
> **Destacado**.

---

## D3 · Interpretación de los grupos (30 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Identifica el grupo pequeño como objetos de gran tamaño (**buses**) y lo conecta con los atípicos legítimos de la EA1; reconoce que el grupo de muchos puntos agrupa por una propiedad de la **medición**, no del objeto |
| **3** | Nombra los cuatro grupos con lenguaje de dominio, apoyándose en el perfil |
| **2** | Describe los grupos con los números del perfil, sin traducirlos a lenguaje de dominio |
| **1** | Deja los grupos como "grupo 0, grupo 1" |

**Cifras de referencia** (medias en desviaciones típicas):

| Grupo | `box_length` | `box_height` | `speed_mps` | `num_lidar_points` | n | % |
|---|---|---|---|---|---|---|
| 0 | −1,03 | +0,30 | −0,93 | −0,16 | 13.787 | 36,4 % |
| 1 | +0,53 | −0,31 | +0,59 | −0,31 | 19.927 | 52,6 % |
| 2 | +0,21 | −0,20 | +0,19 | **+2,30** | 3.624 | 9,6 % |
| **3** | **+4,90** | **+4,68** | +0,57 | +0,05 | **573** | **1,51 %** |

> **El momento clave de la sesión.** El grupo 3 son los buses: los mismos atípicos legítimos que
> en la EA1 se aprendió a **no** eliminar. Si el alumno hace esa conexión sin ayuda, es
> **Destacado**.

---

## D4 · Contraste con la etiqueta (15 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Explica los dos grupos de vehículos como subestructura real; atribuye la confusión peatón/señalética a las **variables elegidas** y **propone una variable** que los separaría |
| **3** | Interpreta el cruce con cifras y **sin llamarlo acierto** |
| **2** | Lee el cruce pero lo trata como una medida de exactitud del algoritmo |
| **1** | No contrasta, o concluye que el algoritmo "se equivocó" |

**Cifras de referencia** (% de cada tipo dentro de cada grupo):

| Grupo | `cyclist` | `pedestrian` | `sign` | `vehicle` |
|---|---|---|---|---|
| 0 | 4,87 | **73,10** | 21,85 | 0,18 |
| 1 | 0,00 | 0,00 | 0,00 | **100,00** |
| 2 | 1,71 | 16,83 | 2,07 | **79,39** |
| 3 | 0,00 | 0,00 | 0,00 | **100,00** |

> ⚠️ *"El algoritmo acertó un 73 %"* es un **error de concepto**: no había nada que acertar. Baja
> esta dimensión a **En desarrollo** aunque el resto esté bien.

**Variable esperada para separar peatón de señalética:** la variación de posición entre
fotogramas (un peatón se mueve; una señal, nunca). Eso es ingeniería de características.

---

## D5 · Reducción de dimensionalidad y comunicación (15 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Acompaña el gráfico de su varianza explicada **sin que se lo pidan**; concluye sobre la utilidad de negocio de la segmentación, no sobre su calidad estadística |
| **3** | Reporta cuántas componentes conservan el 90 %, interpreta la redundancia y declara la varianza del gráfico 2D |
| **2** | Produce el gráfico sin reportar cuánta información se pierde |
| **1** | Interpreta el gráfico como si fuera el espacio real |

**Cifras de referencia:** **3 de 5** componentes conservan el 90 %. La proyección 2D conserva el
**70,2 %**, así que casi el 30 % de la información no está en el dibujo.

---

## Qué mirar al corregir, en este orden

1. **Los nombres de los grupos.** Si dicen "grupo 0, grupo 1", no hay hallazgo. Es lo primero
   que revela si entendieron para qué sirve esto.
2. **Cómo describen el cruce.** Llamarlo "acierto" es error de concepto.
3. **La conclusión final.** *"Los grupos están bien definidos"* no responde la pregunta. La
   pregunta es **qué decisión habilitan**.

## Retroalimentación sugerida

1. *"¿Cómo le explicarías el grupo 3 a alguien que no vio la tabla?"*
2. *"Dices que el algoritmo acertó. ¿Contra qué, si nunca vio las etiquetas?"*
3. *"Tu gráfico separa muy bien. ¿Cuánta información no está ahí?"*
