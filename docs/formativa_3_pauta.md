# Pauta docente · Formativa 3

**No proyectar.** El error caro: tratar un 0,01 de ganancia como victoria.

En el hilo Waymo v2 (pauta de actividades, **no** de esta prueba): default 0,5104 →
búsqueda 0,5893, ganancia **0,0789**, ruido 0,0282. GB 0,594 vs ensamble 0,5938:
no distinguible. Úsalo en la devolución si ya corrieron la Act. 3.3; no lo exijas
de memoria aquí.

| Ítem | Clave |
|---|---|
| 1 | **B.** No supervisado es RA2 (IL2.3). |
| 2 | Partición del nodo = parámetro (se aprende). `max_depth` = hiperparámetro (lo fijas o lo buscas). |
| 3 | Infla el test (el modelo “vio” el hold-out al elegir). Bien: CV **solo** en train; test se toca una vez. |
| 4 | Test único: estimación en datos nunca usados para elegir. CV: estabilidad del *pipeline* en el train / comparación justa entre configuraciones. |
| 5 | No. Sin diversidad no hay reducción de varianza; es el mismo modelo 100 veces. |
| 6 | **No** te cambias. La ganancia está dentro del ruido: el default es la solución honesta. |
| 7 | IE9–C, IE10–A, IE11–D, IE12–B. Pesos oficiales: 20 / 30 / 20 / 30. |
| 8 | Entregas la comparación cuantitativa (tabla default vs búsqueda vs ruido) y **justificas** quedarte con el default. Eso es IE12, no un fracaso. |
| 9 | La del KPI (recall / F1 de la clase cara / MAE en pesos). Exactitud es vanidad si no es el criterio de negocio. |
| 10 | Tres de: defensa individual, los 12 IE, ética (IE4), dos supervisados + un no supervisado, informe Markdown, estructura `data/ notebooks/ models/`. |

**Cierre:** RA3 no es “el algoritmo más fancy”. Es demostrar que la mejora no es ruido.
