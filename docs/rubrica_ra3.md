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
| 3.1 | Ajustar hiperparámetros | **+0,0789** de F1-macro (0,5104 → 0,5893) | La búsqueda **sí** supera el ruido (0,0424) |
| 3.2 | Combinar modelos | GB **0,594** · ensamble **0,5938** | No distinguible: el ensamble no suma |
| 3.3 | Elegir el mejor | F1 `LEVEL_2` sigue en **0,0893** | Subir el macro no salva a la clase difícil |

**La conclusión del RA3 no es "el ajuste no sirve".** Es:

> El F1-macro se puede mover y aun así el modelo se pierde el 94 % de las detecciones
> difíciles. Elegir "el mejor" sin mirar `LEVEL_2` es el error de la exactitud, otra vez.

**Un alumno que reporte solo "mejoré el F1-macro" no ha cerrado la experiencia.**

---

## D1 · Esquema de validación (25 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Todo lo del 3, y además: distingue la brecha validación/prueba (sesgo conservador) de una fuga; explica por qué un optimismo de cero no exonera la trampa de ajustar en prueba |
| **3** | Valida con `GroupKFold`, verifica 0 segmentos compartidos, justifica la métrica por el desbalance y **no usa la prueba para elegir nada** |
| **2** | Usa validación cruzada pero sin agrupar por segmento, o no justifica la métrica |
| **1** | Ajusta mirando la prueba, o valida sobre el conjunto de entrenamiento |

**Cifras de referencia:** 5 pliegues, 384.280 filas de entrenamiento, **30** segmentos,
**0 compartidos**.

---

## D2 · Ajuste de hiperparámetros (20 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Contrasta la ganancia contra el ruido **y** mira si `LEVEL_2` se movió; no vende el macro solo |
| **3** | Justifica el espacio de búsqueda y la estrategia; compara contra los valores por defecto **y** contra la desviación entre pliegues |
| **2** | Ejecuta la búsqueda y reporta la mejor configuración como una mejora, sin contrastarla |
| **1** | No compara contra los valores por defecto |

**Cifras de referencia (v2, 2026-09-08):** por defecto **0,5104** → búsqueda **0,5893** ·
ganancia **+0,0789** (supera ruido 0,0424).

---

## D3 · Ensamble y diagnóstico sesgo/varianza (20 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Explica el fracaso del ensamble por la **correlación entre sus miembros** (los tres son de árboles); diagnostica el techo de sesgo con evidencia y propone una variable derivada concreta |
| **3** | Compara los seis candidatos con métrica **y** costo; concluye que el ensamble no se justifica |
| **2** | Compara medias sin considerar costo ni ruido |
| **1** | Concluye que el ensamble es mejor, o no incluye baseline |

**Cifras de referencia:** boosting **0,594** · ensamble **0,5938** · árbol 0,5586 · bosque
**0,5544** (desv. 0,0375) · logística **0,5056** · baseline 0,4728. El árbol es el de más
varianza (0,0542).

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

**Cifras de referencia:** boosting y ensamble **no son distinguibles** (0,0002 < 0,0282). El
bosque sí lo es (cae 0,0396) y cuesta ~14× el boosting. El árbol es el más interpretable.

> ⚠️ **Elegir el ensamble no es defendible:** misma media que el boosting, mucho más lento y
> menos interpretable. Si el informe lo elige *"porque es más avanzado"*, ahí está el error que
> la experiencia previene.

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
2. **¿El informe vende el F1-macro como si `LEVEL_2` estuviera resuelto?** El macro subió; las
   difíciles siguen en 0,0893. Es el error central que las tres sesiones previenen.
3. **¿La justificación de la elección usa las cuatro dimensiones?** Si solo habla de la métrica,
   no cumple el IL3.4 por completa que esté la tabla.
4. **¿Declara los límites?** Casi nadie lo escribe, y es lo que el EFT evalúa en la defensa.

## Retroalimentación sugerida

1. *"Tu mejor modelo saca 0,594 y el segundo 0,5938. ¿Con qué evidencia afirmas que es mejor?"*
2. *"Dices que el ajuste resolvió el modelo. El macro subió: ¿y `LEVEL_2`?"*
3. *"Elegiste el modelo más complejo. Si tuviera que responder en milisegundos, ¿elegirías igual?"*

---

## Enlace con el EFT

El informe de la Actividad 3.3 **es** el apartado de modelamiento y optimización que pide la
Evaluación Final Transversal. La diferencia es que allí va sobre uno de los casos oficiales
—Telco, Housing o Spotify— acompañado del análisis no supervisado del RA2 y del análisis de
sesgos del RA1. **La estructura es la misma.**
