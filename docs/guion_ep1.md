# Guion de sala · Evaluación Parcial 1 (5 h)

**MLY1101** · Comprensión y preparación · casos oficiales (no Waymo).
**Instrumento:** IE1 10 % · IE2 30 % · IE3 40 % · IE4 20 %. Presentación 10 min,
grupal con **defensa individual**.

Material del alumno: `notebooks/15_alumno_evaluacion.ipynb` + el CSV de su caso
(después de `preparar_casos_oficiales.py`). Pauta de nota:
`python herramientas/calcular_nota.py --instrumento ep1 --ie …`

## Antes de entrar

- Formativa 1 ya rendida. Nadie abre un `RandomForest` hoy.
- **No** proyectar el notebook institucional de Telco (anexo EFT).
- Tres carpetas listas en `datos/evaluaciones/{telco,housing,spotify}/`.
- Un equipo = un caso hasta el EFT. Cambiar de caso a mitad es otro proyecto.

## Coreografía

| Bloque | Min | Qué preguntas antes de mostrar |
|---|---|---|
| 0 · Caso y KPI | 40 | *“¿Qué decisión de negocio toma alguien con su tabla?”* Si responden con el nombre de un algoritmo, vuelven a escribir. |
| 1 · Cargar (TODO 1) | 30 | El `FileNotFoundError` es la lección de IE1: sin el zip oficial no hay evaluación. |
| 2 · Calidad (TODO 2) | 70 | Telco: *“¿TotalCharges es número?”* Housing: nulos reales. Spotify: `popularity` no es feature si es el target. |
| 3 · Ética | 50 | Quién está subrepresentado; `customerID`; no tirar la clase rara. |
| 4 · Informe + repo | 50 | Markdown, notebook ejecutable, `data/ notebooks/ README.md`. |
| 5 · Presentación | 60 | 10 min + preguntas cruzadas. El que “solo hizo el gráfico” responde igual el KPI. |

Si recortas, recorta el pulido de Markdown. **No** recortes ética (IE4 = 20 %) ni
calidad (IE3 = 40 %).

## Cifras de pauta (2026-09-09, zip institucional)

| Caso | Forma | Lo que tiene que salir en la defensa |
|---|---|---|
| Telco | 7.043 × 21 | `Churn` Yes 1.869 (26,54 %); `TotalCharges` object, 11 vacíos, `tenure = 0` |
| Housing | 2.930 × 82 | `SalePrice`; no tratar `PID` como feature de precio |
| Spotify | 114.000 × 21 | `Unnamed: 0`; fuga si el split parte pistas del mismo álbum al azar |

## Qué es un 0 en IE3

Entrenar “para ir adelantados”. La EP2 es la de los modelos.
