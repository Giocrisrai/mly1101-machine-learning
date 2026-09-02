# Pauta de la Actividad 2.4 · Interpretación y métricas de desempeño

**Asignatura:** MLY1101 Machine Learning · **Notebook:** `13_alumno_interpretacion.ipynb`
**Horas del programa:** 5 · **Entregable:** notebook con los TODO + memo Waymo + memo del
caso oficial.

**RA2:** *Aplica modelos estadísticos al conjunto de datos procesados para interpretarlos,
utilizando metodologías ágiles, con la finalidad de obtener conocimientos relevantes que
permitan responder a las necesidades del contexto de negocio, considerando aspectos éticos.*

**IL 2.4:** *Interpreta los resultados del desempeño del modelo, traduciendo métricas
técnicas a conocimientos para la organización.*

> ⚠️ **Pauta formativa, no instrumento sumativo.** La evaluación calificada es la
> **Parcial 2**, sobre un caso oficial. Esta pauta retroalimenta la actividad.

> Escala y calculadora: las mismas del RA1 ([`rubrica_ra1.md`](rubrica_ra1.md)).

---

## Las cinco dimensiones

| Dim. | Qué evalúa | Dónde se evidencia | Peso |
|---|---|---|---|
| **D1** | Traduce la matriz a frecuencia ("de cada 100") | Bloque 1 (TODO 1) | 25 % |
| **D2** | Asigna costo asimétrico y decide el uso | Bloque 2 (TODO 2) | 25 % |
| **D3** | Interpreta el no supervisado sin recitar la silueta | Bloque 3 (TODO 3) | 15 % |
| **D4** | Reporta regresión en unidades, no solo R² | Bloque 4 (TODO 4) | 20 % |
| **D5** | Escribe un memo que la organización puede usar | Bloque 5 | 15 % |

**D1 y D2 son el núcleo: sin ellas el IL2.4 no se evidencia.**

---

## D1 · De cada 100 (25 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Frase canónica **y** nombra las 385 falsas alarmas, que `por_cada_cien` no incluye |
| **3** | TP 456, FN 676 → 40 y 60; explica por qué la exactitud ~90 % no entra al memo |
| **2** | Lee la matriz pero deja el resultado en "recall = 0,40" |
| **1** | Reporta solo la exactitud, o no identifica `LEVEL_2` |

Cifras (semilla 42, mismas de [`rubrica_act_2_2.md`](rubrica_act_2_2.md)): TP **456**,
FN **676**, FP **385**. De cada 100 difíciles: **40** / **60**. Dummy `most_frequent`:
exactitud **0,8896**.

## D2 · Costo y decisión de uso (25 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Declara la razón FN:FP como supuesto y la recalcula; conecta con el umbral de la carta 2.1 (F1 ≥ 0,60 **no cumplido**) |
| **3** | FN más caro que FP; costo esperado coherente; recomienda **no** usar el modelo como permiso para confiar en el sensor |
| **2** | Calcula el costo pero recomienda "usar el modelo" sin condición |
| **1** | Trata FN y FP como intercambiables |

Referencia con 10:1: **7.145** unidades; los FN son el **94,6 %** del costo.

## D3 · Agrupamiento en lenguaje de negocio (15 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Buses + conexión con los atípicos de 1.3 + "si los hubiéramos tirado, este grupo no existiría" |
| **3** | Nombra el grupo pequeño como objetos largos / buses, sin recitar la silueta |
| **2** | Describe el perfil numérico |
| **1** | "k = 3 es el óptimo" |

## D4 · Regresión en unidades (20 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Distingue MAE (promedio) de RMSE (colas) y escribe la frase House Prices / Spotify en la unidad del caso |
| **3** | `metricas_minimas("regresion")` = MAE y RMSE; rechaza R² como único reporte |
| **2** | Calcula MAE/RMSE pero concluye con R² |
| **1** | No distingue clasificación de regresión |

## D5 · Memo (15 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Titular sin nombre de métrica; el caso oficial distingue el costo de los dos errores |
| **3** | Memo Waymo usable + tabla del caso oficial con métricas mínimas del tipo de problema |
| **2** | Memo que todavía habla de F1 como conclusión |
| **1** | En blanco, o pega el `classification_report` |

---

## Lo que no se evalúa hoy

Entrenar un modelo nuevo, ajustar hiperparámetros, elegir `k`. Si el notebook es una
copia de la 2.2, la actividad no se evidenció.
