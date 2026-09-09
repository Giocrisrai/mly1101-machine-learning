# Pauta docente · Formativa 1

**No proyectar.** 1 h. Cada ítem ~6 min. Aprobación formativa: 6/10 bien fundamentados
(no es la rúbrica institucional 100/80/60/30/0).

| Ítem | Clave | Qué no aceptar |
|---|---|---|
| 1 | **B** | A (Colab no es backup del equipo). D es teatro jurídico. |
| 2 | **No.** `object` + 0 NaN suele esconder `""` o `" "`. `value_counts(dropna=False)` / `sort_values` / `to_numeric(errors="coerce")`. En Telco (EP1, no hoy) son 11 `TotalCharges` vacíos con `tenure = 0`. | “Está limpia porque no hay nulos.” |
| 3 | Centinela, no se sabe sin diccionario. `resumen_calidad` los cuenta aparte de NaN (`n_centinelas`). | Borrar la fila en silencio. |
| 4 | \(x < Q_1 - 1{,}5\,\mathrm{IQR}\) o \(x > Q_3 + 1{,}5\,\mathrm{IQR}\). Si “todos” son outlier, el criterio no describe el fenómeno: no borres la columna por receta. | “Siempre se eliminan outliers.” |
| 5 | Fuga temporal / de unidad: la misma entidad queda en train y test. En Waymo era `segment_id`; en Telco será el cliente. | “El random_state=42 lo arregla.” |
| 6 | **Muestreo.** El modelo no vio lluvia: el error en producción no aparece en el hold-out soleado. | Decir “procesamiento” (eso sería filtrar noches a mano). |
| 7 | Sobre el grupo ya minoritario (ciclista, peatón, cliente que se va). Limpieza que borra a quien el negocio dice proteger. | “Da igual, son pocas filas.” |
| 8 | Sí hay riesgo: cuasi-identificadores. Mirar `nunique` de combinaciones (comuna × sexo × fecha). | “Sin RUT no hay privacidad que mirar.” |
| 9 | 0 % nulos y aun así: tipos sucios, desbalance, sesgo de muestreo, identificadores, fuga al partir. Waymo v2: 0 % nulos y `LEVEL_2` sigue siendo el problema. | Citar un modelo (aún no es RA2). |
| 10 | RA1–B, RA2–C, RA3–A. | RA2 = solo supervisado; RA3 = no supervisado. |

**Devolución en pizarra (5 min):** el error caro de esta formativa es tratar “0 nulos” como luz verde. La EP1 se juega en IE2–IE4, no en entrenar de noche.
