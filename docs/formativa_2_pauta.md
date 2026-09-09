# Pauta docente · Formativa 2

**No proyectar.** Tras 2.4. Si alguien responde con “el no supervisado es el RA3”,
es el error que el programa ya cometió una vez: márcalo y corrige en voz alta.

| Ítem | Clave |
|---|---|
| 1 | **B.** RA2 = supervisado **y** no supervisado. RA3 = hiperparámetros / ensamble / CV. |
| 2 | Supervisado (hay etiqueta). `KMeans` arma grupos; no entrega `P(Churn)`. Puede **apoyar** (segmentos) pero no sustituye al clasificador. |
| 3 | Un clasificador que predice siempre la mayoría se acerca a 88 % de exactitud y recall de la minoría = 0. 78 % no demuestra que vio la clase cara. |
| 4 | Recall = 0,40. Para F1 hace falta precisión (o TP/FP). Negocio: “de cada 100 casos caros, dejamos 60 sin cazar.” En Waymo v2 el F1 de `LEVEL_2` medido fue **0,0893** (recall 0,0588): úsalo solo si ya lo vieron; no es requisito de memoria. |
| 5 | Negocio: la métrica no responde al KPI (optimizamos exactitud y el jefe quería recall). Datos: el split por fila filtra el mismo cliente. |
| 6 | Falla **IE7** (no supervisado). Dos supervisados cubren IE6; el segundo RF no es clustering. |
| 7 | El \(R^2\) no es un error en pesos. MAE / RMSE (unidades de `SalePrice`) sí. |
| 8 | Aceptar FN (no ver al que se va / no ver al peatón) **o** FP (gastar ofertas / frenar de más) según el KPI escrito. Lo que no se acepta: “dan igual.” |
| 9 | Titular con acción y costo. Ej.: “El modelo deja fuera a 6 de cada 10 abandonos; no sirve para la campaña de retención tal como está.” |
| 10 | No clasifica. Sirve para ver estructura / visualizar grupos (Act. 2.3). En v2, 2 componentes explicaron **0,7447**: dato de pauta, no de memorización. |

**Cierre:** métrica que no se puede decir en la reunión no es resultado. La EP2 se juega en IE6+IE7+IE8 juntos.
