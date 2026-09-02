# Guion de clase — Actividad 1.3 · EDA (RA1)

**Asignatura:** MLY1101 Machine Learning · **Duración:** 4 h (taller) · **Modalidad:** laboratorio

Material: `notebooks/01_alumno_exploracion.ipynb` (alumnos) y
`notebooks/01_docente_solucionario.ipynb` (pauta, en tu pantalla).

---

## Antes de entrar a la sala

- [ ] Abrir el solucionario y ejecutarlo completo una vez (Kernel → Restart & Run All).
- [ ] Tener el enlace de Colab del notebook del alumno listo para pegar en el chat/pizarra.
- [ ] Verificar que el repositorio esté publicado y accesible sin cuenta.
- [ ] Tener a mano el número clave del día: **34 %** (nulos de velocidad en LEVEL_2 nocturno).
- [ ] Avisar al curso que Colab mostrará *"Advertencia: Este cuaderno no lo ha creado Google"* al
      ejecutar la primera celda, y que hay que pulsar **"Ejecutar de todos modos"**. Es el aviso
      estándar de cualquier notebook abierto desde GitHub. Si no se advierte, medio curso se
      detiene en esa pantalla.

---

## Bloque 0 · El problema antes del algoritmo (15 min)

**Objetivo:** que nadie abra scikit-learn hoy.

Presentación de la asignatura y de la ruta:

```
Problema → Datos → Exploración → Preprocesamiento → Modelamiento → Evaluación → Interpretación
```

Pregunta de apertura al curso, antes de mostrar nada:

> *"Van a construir el sistema que decide si eso que está al frente del auto es un peatón o una
> bolsa de basura. ¿Por dónde empiezan?"*

Casi siempre responden con un algoritmo (redes neuronales, YOLO, etc.). Ahí se instala la idea
del semestre: **empezamos por el problema y por los datos**. El algoritmo es la penúltima
decisión, no la primera.

Cerrar con la frase: *un modelo entrenado con datos que nadie revisó no es un modelo, es una
opinión con decimales*.

**Advertir explícitamente** que el dataset es sintético y por qué (licencia de Waymo + garantía
pedagógica). La honestidad sobre el origen de los datos es parte de lo que se enseña.

---

## Bloque 1 · Carga e inspección (45 min)

**Objetivo:** que `.info()` deje de ser un trámite.

- TODO 1–3.
- **Momento clave:** el dtype `object` de `timestamp_micros`. 60 filas de 40.680 (0,15 %)
  arruinan una columna entera.
- Demostración recomendada en vivo: intentar `df["timestamp_micros"].mean()` y mostrar el error.
  Luego `df["timestamp_micros"].sort_values().head()` para mostrar el orden alfabético.
- Error frecuente: creer que `dropna()` limpia el `"N/D"`. No lo hace.

**Pregunta obligatoria antes de avanzar:** *"¿qué representa una fila?"* Si el curso no lo tiene
claro, el bloque de duplicados no va a funcionar.

---

## Bloque 2 · Tipos de variables y categorías (45 min)

**Objetivo:** distinguir el tipo de pandas del tipo estadístico.

- TODO 4–6.
- **Momento clave:** `object_type` tiene 7 valores distintos para 4 clases reales; `weather`
  tiene 11 para 3.
- Truco para mostrar los espacios invisibles:
  ```python
  [repr(v) for v in df["weather"].dropna().unique()]
  ```
  `'RAIN '` y `'rain'` se ven idénticos en una tabla; `repr` los delata.
- Pregunta: *"¿qué pasa si entrenamos así?"* → one-hot de 7 columnas para 4 clases, evidencia
  repartida entre categorías gemelas, y una variante nueva en producción que no corresponde a
  ninguna columna aprendida.
- Cierre del bloque con el desbalance: `CYCLIST` ≈ 2 %. *"Un modelo que siempre dice VEHICLE
  acierta 62 % y es inútil."* Semilla para la Actividad 2.2.

---

## Bloque 3 · Nulos y duplicados (45 min) ⭐

**Este es el bloque más importante de la sesión.** Si el tiempo se acorta, recórtalo de otro lado.

- TODO 7–11.
- **Momento clave 1 — nulos ocultos:** `num_lidar_points` mínimo `-1`. Un conteo de puntos láser
  no puede ser negativo. `isna()` no lo ve.
- **Momento clave 2 — el patrón MNAR:** el mapa de calor. En `LEVEL_1` falta 0,4 % de las
  velocidades; en `LEVEL_2` nocturno, **34 %**. El sensor falla justo cuando le cuesta.

  Preguntar, en este orden:
  1. *"Si hago `dropna(subset=['speed_mps'])`, ¿qué filas desaparecen?"*
  2. *"¿Cómo se ve el mundo para un modelo entrenado con lo que queda?"*
  3. *"¿Y cuándo circula el auto de noche?"*

  Frase de cierre: **"Eliminar filas nunca es gratis: siempre estás eligiendo qué parte de la
  realidad borrar."**

- **Momento clave 3 — duplicado lógico:** ejecutar `len(df.drop_duplicates())` y mostrar que
  quedan ~200 filas con llave repetida. `drop_duplicates()` **no** terminó el trabajo.
  Preguntar con cuál de las dos filas se quedan: no hay respuesta única, y ese es el punto.

**Señal de alerta:** si un grupo termina este bloque en 15 minutos, está ejecutando sin leer.
Pregúntale el porcentaje de nulos en LEVEL_2 nocturno. Si no lo tiene, no miró el patrón.

---

## Bloque 4 · Valores atípicos (45 min)

**Objetivo:** que entiendan que ninguna fórmula distingue un error de un caso raro.

- TODO 12–15.
- **Momento clave:** 338 m/s = 1.218 km/h (un peatón a velocidad de avión) frente a un bus de
  15 m. Ambos son "outliers" para el IQR. Solo uno es un error.
- Mostrar que el z-score marca **menos** valores que el IQR y explicar por qué: los propios
  extremos inflan σ y se esconden a sí mismos.
- Escribir en la pizarra: **el umbral estadístico propone, el conocimiento del dominio dispone.**
- La pregunta que cierra el bloque: *"si eliminamos todos los atípicos de `box_length`, ¿qué
  acabamos de hacer?"* → entrenar un auto autónomo que nunca vio un bus.

Regla práctica que deben anotar:
1. Definir primero qué es físicamente imposible.
2. Lo imposible se trata como dato faltante (no se "arregla" inventando el valor).
3. Lo raro pero posible se conserva y se documenta.

---

## Bloque 5 · Decisiones de preprocesamiento (30 min)

**Objetivo:** pasar del diagnóstico a la decisión documentada.

- TODO 16 (tabla de decisiones) y 17 (función `limpiar`).
- Insistir en que sea una **función** y no celdas sueltas: reproducible, testeable, reaplicable.
- Aceptar cualquier decisión **justificada**. Lo que no se acepta es `df.dropna()` sin argumento.
- **Cerrar con la fuga de información:** hacer notar que no imputamos ni escalamos, y que no fue
  un olvido. Orden correcto: limpieza estructural → split → ajustar imputación/escalado solo con
  train → aplicar a test. Anunciar `Pipeline` para la Actividad 2.2 sin entrar en detalle.

---

## Bloque 6 · Tratamiento responsable (20 min)

**Objetivo:** conectar el RA1 con una consecuencia concreta.

- TODO 18.
- Cifras del dataset de clase: ~20 % nocturnas, ~21 % con lluvia, ~2 % ciclistas.
- Las tres preguntas: qué está sub-representado, quién corre riesgo, qué medida concreta propones.

**El remate con datos reales.** Cuando el curso ya discutió el sesgo del dataset sintético,
muestra lo que pasa en el Waymo Open Dataset real. No es una muestra: es el **censo de los 798
segmentos** de entrenamiento (`herramientas/analizar_sesgo_waymo.py`, agosto 2026):

| | Dataset de clase | Waymo real (censo de 798) |
|---|---|---|
| Segmentos con lluvia | 21 % | **0,6 %** (5 de 798) |
| Segmentos nocturnos | 20 % | **9,9 %** (79 de 798) |
| Concentración geográfica | — | **87 %** en San Francisco y Phoenix |
| Peatones + ciclistas de día | — | **27,05 %** |
| Peatones + ciclistas de noche | — | **14,11 %** |

La frase para lanzarlo: *"Yo inventé el dataset de la clase y le puse 21 % de lluvia. ¿Cuánta
lluvia creen que tiene el dataset real de una de las empresas de conducción autónoma más
grandes del mundo?"* La respuesta —**5 grabaciones de 798**— suele producir silencio.

Y el remate del remate: el 87 % de los datos viene de San Francisco y Phoenix, dos ciudades de
clima seco. *"La falta de lluvia no es mala suerte: es consecuencia de dónde decidieron grabar.
¿Qué pasa cuando ese auto llega a Valdivia?"*

Y el dato que cierra el argumento: de noche, la proporción de peatones y ciclistas cae a la
mitad. Menos ejemplos para aprender justo cuando detectarlos es más difícil y equivocarse
cuesta más.
- **La idea que debe quedar:** un promedio global oculta a las minorías. Un modelo con 97 % de
  exactitud puede tener 60 % en ciclistas nocturnos, y ese error no se reparte al azar: recae
  sobre quienes ya son más vulnerables.
- Mencionar la reidentificación: no hay nombres en el dataset, pero *lugar + hora + trayectoria*
  puede identificar a una persona. Anonimizar no es borrar la columna "nombre".

---

## Cierre (15 min)

- Mini-informe: 5 hallazgos con cifras, 3 decisiones con justificación, 1 riesgo ético.
- Regla: ninguna afirmación sin una cifra. *"Hay datos sucios"* no es un hallazgo.
- Anunciar la Semana 2 (preprocesamiento aplicado y `Pipeline`) y el notebook opcional de Waymo
  real para quien quiera ir más allá.

---

## Si el tiempo se acorta

| Prioridad | Bloque |
|---|---|
| 🔴 Imprescindible | 3 (nulos y duplicados) y 4 (outliers) |
| 🟡 Importante | 5 (decisiones) — al menos la tabla, aunque no se programe la función |
| 🟢 Comprimible | 2 (se puede dar el mapa de normalización resuelto) y 6 (puede quedar de tarea) |

## Si sobra tiempo

- Pedir imputación por grupo (tipo de objeto × dificultad) en vez de una media global.
- Discutir la columna indicadora `speed_faltante`: casi nadie la propone y suele ser la mejor
  solución.
- **La unidad de análisis.** En el Waymo real, agregar todas las detecciones sugiere que de día
  se detecta peor que de noche (13,19 % vs. 7,04 % de detecciones difíciles). Calculado por
  segmento, la diferencia se evapora (4,81 % vs. 4,25 %): un solo segmento diurno atípico, con
  53,81 % de difíciles y muchísimas filas, arrastraba el promedio.
  Es la antesala perfecta de la Actividad 2.2: si el split de entrenamiento/prueba se hace por fila, las
  detecciones de un mismo segmento caen a ambos lados, el modelo reconoce la escena en vez de
  aprender el objeto, y la evaluación miente. **El split va por `segment_id`.**
  Está desarrollado en `notebooks/00_opcional_waymo_real.ipynb`, Paso 7.
- Abrir el notebook de Waymo real y mostrar el esquema original: los nombres jerárquicos del
  tipo `[LiDARBoxComponent].box.center.x` explican por qué renombrar es parte del trabajo.
