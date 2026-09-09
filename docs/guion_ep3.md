# Guion de sala · Evaluación Parcial 3 (6 h)

**MLY1101** · Optimización y comparación · el **mismo** caso que EP1 y EP2.
**Instrumento:** IE9 20 % · IE10 30 % · IE11 20 % · IE12 30 %. Presentación 10 min,
defensa individual.

Material: el mismo notebook 15, bloques EP3. Nota:
`python herramientas/calcular_nota.py --instrumento ep3 --ie …`

Formativa 3 ya rendida. RA3 = hiperparámetros, ensamble y validación cruzada.
**No** es “más k-medias”.

## Antes de entrar

- Tienen los dos supervisados y el no supervisado de la EP2. Hoy se **ajustan**,
  no se inventa un cuarto dataset.
- El test de la EP2 se guarda: la búsqueda vive en **train + CV**.
- **No** proyectar el notebook institucional de Telco.

## Coreografía

| Bloque | Min | Qué preguntas antes de mostrar |
|---|---|---|
| 0 · Encuadre | 15 | *“Si el default ya cumple el KPI, ¿fallaste?”* No: IE12 es justificar. |
| 1 · Hiperparámetros (IE9) | 70 | *“¿Qué estás tocando, un parámetro o un hiperparámetro?”* `GridSearchCV` / `RandomizedSearchCV` solo sobre `X_train`. |
| 2 · Ensamble (IE10) | 70 | *“Cien copias del mismo árbol, ¿bajan varianza?”* Voting / bagging / boosting con diversidad. |
| 3 · CV (IE11) | 60 | Test se toca **una** vez. Quien elige el modelo con el test se lleva 0. |
| 4 · ¿Ganó? (IE12) | 55 | Ganancia vs ruido (`std` del CV). Si no supera, entregan el default con la tabla. |
| 5 · Presentación | 70 | Defensa: cualquiera explica por qué **no** se cambiaron al modelo fancy. |

Si recortas, recorta un gráfico de importancia. **No** recortes IE11 ni IE12.

## Qué es un 0

| IE | Cero si… |
|---|---|
| IE9 | Buscan `C` / `max_depth` en el conjunto de prueba. |
| IE10 | “Ensamble” = el mismo modelo dos veces, o un solo árbol más profundo. |
| IE11 | Un split único presentado como generalización; o el test metido en la CV. |
| IE12 | *“Ganó porque 0,52 > 0,51”* sin barra de ruido. |

## Puente al EFT

Los **doce** IE vuelven. 12 h, 40 % de la nota final, defensa individual.
Calculadora: `python herramientas/calcular_nota.py --instrumento eft --ie …`
(IE1–IE4 al 5 %, IE5–IE12 al 10 %). Ética (IE4) no se “ya hizo en la EP1”: se
actualiza con lo que hizo el modelo.

No copies el zip ni el notebook Duoc de Telco a GitHub.
