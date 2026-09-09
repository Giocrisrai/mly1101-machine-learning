# Guion de sala · Evaluación Final Transversal (12 h)

**MLY1101** · 40 % de la nota final · el **mismo** caso que EP1–EP3.
**Instrumento:** IE1–IE4 al **5 %** cada uno; IE5–IE12 al **10 %** cada uno.
Presentación 10 min, grupal con **defensa individual** y preguntas cruzadas.

Material: `15_<equipo>_evaluacion.ipynb` (EP1+EP2+EP3 ya corridas) + informe
Markdown + `data/ notebooks/ models/ images/ README.md`. Nota:

```bash
python herramientas/calcular_nota.py --instrumento eft --ie \
  80 80 80 60  80 80 60 80  80 60 80 80
```

**No** proyectar el notebook institucional de Telco (anexo EFT). No hay dataset
nuevo: quien cambia de Telco a Spotify en la semana 18 empieza de cero.

## Antes de entrar

- Las tres parciales cerradas sobre el mismo caso.
- Cada integrante puede explicar **cualquier** bloque, no solo “el suyo”.
- Ética (IE4) se **actualiza** con lo que hizo el modelo, no se copia de la EP1.

## Coreografía (12 h pedagógicas)

| Bloque | Horas | Qué preguntas |
|---|---|---|
| 0 · Mapa de 12 IE | 1 | *“¿Qué falta para el 100 de cada IE?”* Abre la pauta, no el código. |
| 1 · Repo profesional | 2 | ¿Corre en un clon limpio? ¿El README dice el caso, el KPI y cómo preparar el CSV? |
| 2 · Cerrar huecos EP1–3 | 4 | Calidad, dos supervisados, un no supervisado, ajuste, CV. Sin `GridSearch` en el test. |
| 3 · Frase de negocio + ética | 2 | Titular sin *accuracy*. Quién pierde si el umbral es el default. |
| 4 · Ensayo de defensa | 2 | Sorteo: A explica el KMeans de B. Si no puede, IE5/IE8 en rojo. |
| 5 · Presentación | 1 | 10 min + cruzadas. Reloj visible. |

Si recortas, recorta un gráfico. **No** recortes defensa ni ética.

## Mínimo que la pauta institucional exige

- Dos supervisados comparados, acordes al problema (clasificación o regresión).
- Un no supervisado que aporte al negocio (no un tercer clasificador).
- Optimización y validación (RA3).
- Sesgos, limitaciones, privacidad.

## Qué es un 0 en el EFT

| IE | Cero si… |
|---|---|
| IE1–IE4 | El CSV “está en el notebook de una persona”; 0 nulos como luz verde; IDs en X. |
| IE6 | Un solo supervisado. |
| IE7 | “El clustering es el RA3.” |
| IE11 | Hiperparámetros elegidos con el conjunto de prueba. |
| IE12 | *“Ganó por 0,01”* sin ruido. |

Calculadora de un equipo: `docs/ejemplo_notas_ep1.csv` es solo EP1; para el EFT
arma un CSV con IE1…IE12.
