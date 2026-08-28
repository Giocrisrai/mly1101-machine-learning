# Pauta del RA3 · Optimización y ensamble de modelos avanzados

**Asignatura:** MLY1101 Machine Learning
**Actividades:** 3.1 Ajuste (6 h) · 3.2 Ensamble (6 h) · 3.3 Robustez y selección (11 h)

**RA3:** *Elabora soluciones avanzadas de aprendizaje automático mediante la optimización de
hiperparámetros, técnicas de ensamble y validación cruzada, para garantizar la precisión y
generalización del modelo frente a objetivos de negocio complejos.*

> ⚠️ **Pauta formativa, no instrumento sumativo.** La evaluación calificada del RA3 es la
> **Evaluación Parcial 3** (*Optimización y comparación de soluciones*, **30 %** de la
> ponderación parcial), sobre uno de los casos oficiales. Escala y conversión a nota: las mismas
> del RA1 ([`rubrica_ra1.md`](rubrica_ra1.md)).

---

## Correspondencia con los indicadores oficiales

| IL | Descripción | Actividad | Notebook |
|---|---|---|---|
| **IL3.1** | Aplica estrategias de ajuste de hiperparámetros | 3.1 | `08_alumno_hiperparametros` |
| **IL3.2** | Desarrolla modelos de ensamble para mitigar sesgo y varianza | 3.2 | `09_alumno_ensamble` |
| **IL3.3** | Evalúa la generalización mediante validación cruzada y métricas avanzadas | 3.3 | `11_alumno_seleccion` |
| **IL3.4** | Sustenta la selección de la solución óptima mediante comparación cuantitativa | 3.3 | `11_alumno_seleccion` |

---

## El hilo que atraviesa las tres actividades

Las tres sesiones producen el **mismo resultado incómodo**, y esa repetición es deliberada:

| Actividad | Lo que se intenta | Lo que se mide | Conclusión |
|---|---|---|---|
| 3.1 | Ajustar 12 configuraciones | **−0,0006** de F1-macro | El ajuste no mejora nada |
| 3.2 | Combinar tres modelos | **−0,0040**, y más lento | El ensamble tampoco |
| 3.3 | Distinguir cuál es mejor | Diferencias **< ruido (0,0079)** | No se puede distinguir |

**La conclusión del RA3 no es "estas técnicas no sirven".** Es:

> Estas técnicas atacan la **varianza**, y aquí el cuello de botella es el **sesgo**: la
> información necesaria no está en las variables. Saberlo **con evidencia** vale más que
> sospecharlo, y es lo que permite dejar de gastar tiempo por la vía equivocada.

**Un alumno que reporte "mejoré el modelo con ajuste de hiperparámetros" no ha entendido la
experiencia**, por bien ejecutado que esté el código.

---

## D1 · Esquema de validación (25 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Todo lo del 3, y además: distingue la brecha validación/prueba (sesgo conservador) de una fuga; explica por qué un optimismo de cero no exonera la trampa de ajustar en prueba |
| **3** | Valida con `GroupKFold`, verifica 0 segmentos compartidos, justifica la métrica por el desbalance y **no usa la prueba para elegir nada** |
| **2** | Usa validación cruzada pero sin agrupar por segmento, o no justifica la métrica |
| **1** | Ajusta mirando la prueba, o valida sobre el conjunto de entrenamiento |

**Cifras de referencia:** 5 pliegues, 29.946 filas, 114 segmentos, **0 compartidos**. Brecha media
validación vs prueba: **+0,0118**. Margen de la trampa: **0,0135**.

---

## D2 · Ajuste de hiperparámetros (20 %)

| Nivel | Criterio observable |
|---|---|
| **4** | **No reporta la ganancia como mejora** al no superar el ruido; ubica las mejoras de primer orden fuera del ajuste |
| **3** | Justifica el espacio de búsqueda y la estrategia; compara contra los valores por defecto **y** contra la desviación entre pliegues |
| **2** | Ejecuta la búsqueda y reporta la mejor configuración como una mejora, sin contrastarla |
| **1** | No compara contra los valores por defecto |

**Cifras de referencia:** por defecto **0,6970** (desv. 0,0087) · mejor de 12 combinaciones
**0,6964** (desv. 0,0091) · ganancia **−0,0006**.

---

## D3 · Ensamble y diagnóstico sesgo/varianza (20 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Explica el fracaso del ensamble por la **correlación entre sus miembros** (los tres son de árboles); diagnostica el techo de sesgo con evidencia y propone una variable derivada concreta |
| **3** | Compara los seis candidatos con métrica **y** costo; concluye que el ensamble no se justifica |
| **2** | Compara medias sin considerar costo ni ruido |
| **1** | Concluye que el ensamble es mejor, o no incluye baseline |

**Cifras de referencia:** bosque **0,6909** · ensamble **0,6869** · boosting 0,6804 · árbol 0,6708
(desv. **0,0166**, el doble) · logística **0,4847** · baseline 0,4705.

> La regresión logística apenas supera al baseline: **el problema no es linealmente separable**.
> Reconocerlo es nivel destacado.

---

## D4 · Selección sustentada (25 %) — IL3.4

| Nivel | Criterio observable |
|---|---|
| **4** | Elige el umbral con el **costo asimétrico** del dominio; cambia coherentemente su elección al cambiar la restricción (p. ej. tiempo real); declara los límites del modelo sin que se los pregunten |
| **3** | Identifica qué modelos son indistinguibles y sustenta la elección con **las cuatro dimensiones**: desempeño, estabilidad, costo e interpretabilidad |
| **2** | Elige por la media más alta, sin considerar dispersión ni costo |
| **1** | Reporta un único número por modelo |

**Cifras de referencia:** bosque y ensamble **no son distinguibles** (0,0040 < 0,0079). Boosting
pierde 0,0105 pero es ~3,5× más rápido. El árbol es el más interpretable y el más barato.

> ⚠️ **Elegir el ensamble no es defendible:** pierde en las cuatro dimensiones. Si el informe lo
> elige *"porque es más avanzado"*, ahí está el error que la experiencia previene.

---

## D5 · Comunicación y defensa (10 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Responde las cinco preguntas de defensa conectando con evidencia propia de los tres RA |
| **3** | El informe declara límites, condiciones de no uso y próxima mejora |
| **2** | Informe completo pero sin sección de límites |
| **1** | Solo resultados, sin interpretación |

---

## Qué mirar al corregir, en este orden

1. **¿Hay dispersión en las tablas?** Sin ella no hay comparación, solo un ranking de ruido. Es
   lo primero y lo más discriminante.
2. **¿El informe afirma haber mejorado el modelo?** Con estos datos es falso. Es el error central
   que las tres sesiones previenen.
3. **¿La justificación de la elección usa las cuatro dimensiones?** Si solo habla de la métrica,
   no cumple el IL3.4 por completa que esté la tabla.
4. **¿Declara los límites?** Casi nadie lo escribe, y es lo que el EFT evalúa en la defensa.

## Retroalimentación sugerida

1. *"Tu mejor modelo saca 0,69 y el segundo 0,687. ¿Con qué evidencia afirmas que es mejor?"*
2. *"Dices que el ajuste mejoró el modelo. ¿Cuánto, comparado con la variabilidad entre pliegues?"*
3. *"Elegiste el modelo más complejo. Si tuviera que responder en milisegundos, ¿elegirías igual?"*

---

## Enlace con el EFT

El informe de la Actividad 3.3 **es** el apartado de modelamiento y optimización que pide la
Evaluación Final Transversal. La diferencia es que allí va sobre uno de los casos oficiales
—Telco, Housing o Spotify— acompañado del análisis no supervisado del RA2 y del análisis de
sesgos del RA1. **La estructura es la misma.**
