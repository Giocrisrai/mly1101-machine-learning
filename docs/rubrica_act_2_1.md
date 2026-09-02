# Pauta de la Actividad 2.1 · Gestión de proyectos con CRISP-DM

**Asignatura:** MLY1101 Machine Learning · **Notebook:** `12_alumno_crispdm.ipynb`
**Horas del programa:** 6 · **Entregable:** notebook con los TODO + carta Waymo validada +
carta del caso oficial (Telco, House Prices o Spotify).

**RA2:** *Aplica modelos estadísticos al conjunto de datos procesados para interpretarlos,
utilizando metodologías ágiles, con la finalidad de obtener conocimientos relevantes que
permitan responder a las necesidades del contexto de negocio, considerando aspectos éticos.*

**IL 2.1:** *Implementa metodologías de trabajo (como CRISP-DM) para estructurar el
desarrollo del modelo.*

> ⚠️ **Esto es una pauta formativa, no el instrumento sumativo.** La evaluación calificada del
> RA2 es la **Evaluación Parcial 2**, sobre un caso oficial, con la rúbrica institucional.
> Esta pauta sirve para acompañar y retroalimentar la actividad.

> La escala, la conversión a nota y la calculadora son **las mismas del RA1**: ver
> [`rubrica_ra1.md`](rubrica_ra1.md) y `herramientas/calcular_nota.py`.

---

## Las cinco dimensiones de la pauta

| Dim. | Qué evalúa | Dónde se evidencia | Peso |
|---|---|---|---|
| **D1** | Distingue las seis fases y no trata CRISP-DM como cascada | Bloques 1 y 5 (TODO 1, 2, 7) | 20 % |
| **D2** | Mapea el RA1 sobre el ciclo (lo ya hecho vs. el hueco) | Bloque 2 (TODO 3) | 20 % |
| **D3** | Formula pregunta de negocio y criterio **medible** | Bloque 3 (TODO 4–5) | 25 % |
| **D4** | Cierra una carta usable (`validar_carta` vacío, con cifras) | Bloque 4 (TODO 6) | 20 % |
| **D5** | Traslada el método al caso oficial | Bloque 6 | 15 % |

**D3 es la de mayor peso: es el hueco que esta sesión existe para cerrar.**

---

## D1 · Fases y ciclo (20 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Orden canónico y, al ver exactitud alta + F1 de minoría bajo, vuelve a **negocio** (o a datos con argumento de desbalance); no propone "otro algoritmo" como primer movimiento |
| **3** | Las seis fases en orden; usa `RETORNOS`; anota 2.3 en el RA2 |
| **2** | Recita las fases pero las trata como cascada rígida |
| **1** | Pone el no supervisado en el RA3, o empieza el proyecto eligiendo el modelo |

## D2 · Retroceso sobre el RA1 (20 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Separa diagnóstico (datos) de decisión (preparación) y pone censo/`sunny` o reidentificación en negocio |
| **3** | Los seis hallazgos caen en una clave de `crispdm.FASES`; nota que el RA1 no modeló |
| **2** | Mapea a medias: mezcla preparación con modelado |
| **1** | No conecta las actividades 1.1–1.4 con ninguna fase |

Cifras de referencia (CSV sucio, semilla 42): **40.680** filas, **153** segmentos,
`CYCLIST` **1,94 %** (antes de unificar `object_type`).

## D3 · Pregunta y criterio medible (25 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Pregunta operativa (qué hace el vehículo si la detección no es confiable) y umbral anclado a un riesgo del RA1 |
| **3** | Pregunta de negocio, no de algoritmo; criterio con dígito; no usa exactitud global como éxito |
| **2** | Pregunta razonable pero criterio vago, o umbral sin relación con el dominio |
| **1** | "Qué modelo usamos" o "que prediga bien" |

## D4 · Carta Waymo (20 %)

| Nivel | Criterio observable |
|---|---|
| **4** | `validar_carta` vacío **y** los riesgos traen cifras (MNAR, desbalance, censo) |
| **3** | Carta que pasa el validador; `proxima_fase = modelado`; fuentes identificables |
| **2** | Completa los campos pero el criterio es un truco que solo busca un dígito (`F1 ≥ 0`) |
| **1** | La carta no corre o está en blanco |

## D5 · Caso oficial (15 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Distingue el costo de falso positivo y falso negativo en Telco / Housing / Spotify; despliegue = reproducibilidad, no un API |
| **3** | Pregunta de negocio + criterio con cifra + riesgos del **caso oficial**, no copiados de Waymo |
| **2** | Plantilla rellena con generalidades |
| **1** | En blanco, o pega la carta Waymo cambiando el título |

---

## Lo que no se evalúa hoy

Entrenar un modelo, elegir `k`, ajustar hiperparámetros. Eso es 2.2, 2.3 y RA3. Si aparece
un notebook con `RandomForestClassifier` "porque ya lo vimos", la sesión no cumplió: volvieron
a empezar por el algoritmo.
