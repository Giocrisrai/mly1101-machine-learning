# Guion de sala · Evaluación Parcial 2 (6 h)

**MLY1101** · Construcción e interpretación · el **mismo** caso que la EP1.
**Instrumento:** IE5 20 % · IE6 30 % · IE7 30 % · IE8 20 %. Presentación 10 min,
defensa individual.

Material: copia de `15_<equipo>_evaluacion.ipynb` (bloques EP2). Nota:
`python herramientas/calcular_nota.py --instrumento ep2 --ie …`

Formativa 2 ya rendida. RA2 = supervisado **y** no supervisado. El RA3 (ajuste,
ensamble, CV) **no** se adelanta hoy.

## Antes de entrar

- Carta CRISP-DM de la Act. 2.1 actualizada al caso oficial (no a Waymo).
- KPI de la EP1 a la vista. Si no hay KPI, no hay IE8.
- **No** proyectar el notebook institucional de Telco.
- Housing y Spotify no corren el esqueleto Telco del 15 docente: lo adaptan.

## Coreografía

| Bloque | Min | Qué preguntas antes de mostrar |
|---|---|---|
| 0 · Mapa | 20 | *“¿Hoy es RA2 o RA3?”* Si dicen “el clustering es lo avanzado”, paras. |
| 1 · Split (IE5) | 40 | *“¿Qué unidad no puede quedar en train y test?”* Telco: cliente. Housing: `PID`. Spotify: álbum/artista, no la pista. |
| 2 · Dos supervisados (IE6) | 80 | *“¿Por qué estos dos y no diez?”* Clasificación: métrica de la clase cara. Regresión: MAE en pesos, no solo \(R^2\). |
| 3 · Un no supervisado (IE7) | 70 | *“¿Qué decisión cambia si ves esos grupos?”* Un segundo RF no cumple IE7. |
| 4 · Frase de negocio (IE8) | 50 | El titular no puede decir *accuracy* ni *F1* sin traducir. |
| 5 · Presentación | 70 | Preguntas cruzadas: quien “solo hizo el KMeans” explica el recall. |

Si recortas, recorta el segundo gráfico. **No** recortes IE7 ni IE8.

## Trampas por caso

| Caso | Lo que tiene que salir en la defensa |
|---|---|
| Telco | `customerID` fuera de \(X\). `class_weight` o umbral; exactitud ~78 % no demuestra nada. KMeans no predice `Churn`. |
| Housing | `PID`/`Order` no son precio. MAE/RMSE en unidades de `SalePrice`. |
| Spotify | `popularity` no va en \(X\) si es el target. 114 mil filas: muestra para el clustering si Colab se queda sin RAM. |

## Qué es un 0

| IE | Cero si… |
|---|---|
| IE5 | Empiezan eligiendo el algoritmo; no hay carta ni split declarado. |
| IE6 | Un solo supervisado, o los dos son la misma receta con otro `random_state`. |
| IE7 | “El no supervisado es el RA3” o un tercer clasificador disfrazado. |
| IE8 | *“El F1 es 0,46 así que está bien.”* |

La EP3 es hiperparámetros. Hoy no hay `GridSearchCV` en el pizarrón.
