# Guion de clase — Actividades 1.1 y 1.2

Complemento de las pautas que ya están dentro de los solucionarios. Aquí no se repiten las
respuestas: está la **coreografía de la sala**, que es lo que no cabe en un notebook.

- Respuestas y criterios por TODO → `02_docente_fuentes.ipynb` y `03_docente_estructuras.ipynb`
- Rúbrica y niveles de logro → [`rubrica_ea1.md`](rubrica_ea1.md)
- Guion de la Actividad 1.3 (EDA) → [`guion_clase_semana01.md`](guion_clase_semana01.md)

---

## Antes de entrar a la sala

| Cosa | Por qué |
|---|---|
| Abre tú los dos notebooks del alumno en Colab y ejecútalos hasta el primer TODO | Si el clon del repositorio falla, lo descubres tú y no treinta personas a la vez |
| Avisa del aviso de Colab | *"Este cuaderno no lo ha creado Google"* detiene a media sala si nadie lo anticipó. Hay que pulsar **"Ejecutar de todos modos"** |
| Ten a mano el número: **793 de 798** | Es la cifra de la Actividad 1.1 que se les queda |
| Ten a mano el otro número: **771 de 789** | Es la de la Actividad 1.2 |

---

# Actividad 1.1 · Fuentes de datos y trabajo colaborativo

**2 horas · IL 1.1 · `02_alumno_fuentes.ipynb`**

| Bloque | Min | Acumulado |
|---|---|---|
| 0 · Encuadre | 10 | 0:10 |
| 1 · Ecosistema y tipos de aprendizaje | 15 | 0:25 |
| 2 · Estructurados: CSV, URL y SQL ⭐ | 30 | 0:55 |
| 3 · Semiestructurados: JSON anidado | 20 | 1:15 |
| 4 · No estructurados: texto libre | 20 | 1:35 |
| 5 · Herramientas del equipo | 15 | 1:50 |
| 6 · Ética y sesgos ⭐ | 20 | 2:10 |
| Cierre · Ficha de fuentes | 10 | 2:20 |

> Suma 2:20 a propósito: **siempre se pierden veinte minutos** entre que arranca Colab y que
> alguien pregunta algo. Si vas bien de tiempo, el bloque 5 se puede estirar.

### Bloque 0 — El encuadre (10 min)

Abre con una pregunta, no con la agenda:

> *"En todos los ejercicios que han hecho, los datos llegaron como un CSV ordenado. ¿De dónde
> salió ese CSV?"*

Casi nadie lo ha pensado. Ese es el punto: alguien lo exportó de alguna parte, tomó decisiones al
hacerlo, y ninguna de esas decisiones está documentada.

### Bloque 2 — SQL (30 min) ⭐

**El bloque que más rinde y el que menos esperan.** Muchos creen que SQL es "otra asignatura".

Momento clave: cuando ejecuten el TODO 3 y el TODO 4 y vean que **salen 7 categorías, no 4**.
Detén la clase ahí y pregunta:

> *"El ejercicio era sobre SQL. ¿Por qué salieron siete tipos de objeto si solo hay cuatro?"*

La conclusión que hay que dejar dicha en voz alta: **cambiar de herramienta no arregla los
datos.** La misma suciedad aparece en SQL, en pandas y en Power BI.

Si alguien pregunta *"¿y entonces para qué SQL, si pandas hace lo mismo?"*, es la mejor pregunta
del bloque. Respuestas: los datos ya viven ahí; el motor filtra 500 millones de filas y te
entrega 3.000, mientras pandas tendría que cargarlas todas en RAM; y varias personas escriben a
la vez sin pisarse.

### Bloque 3 — JSON anidado (20 min)

**Si vas apretado de tiempo, este es el que se recorta.** Basta con proyectar la salida del
solucionario y explicar la diferencia entre `pd.DataFrame` y `pd.json_normalize`.

Si lo haces completo, el momento que vale es mostrar en vivo que la vía ingenua **no falla**:

```python
ingenuo["condiciones"].str.upper()   # devuelve NaN, no da error
```

*"Pandas no les va a avisar. Se ve bien hasta tres celdas más abajo."*

### Bloque 4 — Texto libre (20 min)

El momento útil es al final, cuando decidan qué hacer con las detecciones marcadas. La respuesta
que suele aparecer primero es "eliminarlas". Contrapregunta:

> *"¿Van a botar tres mil detecciones porque un operador escribió una nota? ¿Y si el parte es
> justamente la señal interesante — que el LiDAR falla con lluvia?"*

Lo correcto es **marcarlas con una columna nueva**, no eliminarlas.

Si sobra tiempo: *"¿qué pasa si el operador escribe `Seg_0042` con mayúscula?"* La regex falla en
silencio y el cruce devuelve menos filas sin avisar. Esa fragilidad es por qué el texto libre es
caro.

### Bloque 6 — Ética (20 min) ⭐

**No muestres la cifra: pregúntala primero.**

> *"El Waymo Open Dataset tiene 798 segmentos de entrenamiento. ¿Cuántos creen que se grabaron
> con lluvia?"*

Deja que respondan. Van a decir cien, doscientos, cincuenta. **Son cinco.**

> *"Un vehículo autónomo entrenado con estos datos ha visto llover cinco veces. ¿Qué pasa cuando
> ese sistema llega a Santiago en junio?"*

Y remata con el punto que no es obvio: **la métrica no lo mostraría.** Si el conjunto de prueba
tiene el mismo sesgo que el de entrenamiento, el promedio se ve estupendo.

Atajá una respuesta cómoda que siempre aparece: *"entonces imputamos con la media y listo"*.
Imputar con la media global mete velocidad diurna en filas nocturnas: no elimina el sesgo, lo
disfraza, y encima borra la señal de que el sensor tiene problemas de noche.

---

# Actividad 1.2 · Estructuras de datos y almacenamiento

**2 horas · IL 1.2 · `03_alumno_estructuras.ipynb`**

| Bloque | Min | Acumulado |
|---|---|---|
| 0 · Encuadre | 5 | 0:05 |
| 1 · Listas vs NumPy | 20 | 0:25 |
| 2 · Anatomía del ndarray | 15 | 0:40 |
| 3 · Series y DataFrame | 15 | 0:55 |
| 4 · `.loc` vs `.iloc` ⭐⭐ | 30 | 1:25 |
| 5 · Manipulación avanzada | 20 | 1:45 |
| 6 · Carga y guardado ⭐ | 20 | 2:05 |
| Cierre · Tabla de decisiones | 10 | 2:15 |

> **El bloque 4 no se recorta nunca.** Si hay que sacrificar algo, es el 5, que funciona bien
> como trabajo autónomo.

### Bloque 1 — El engaño de `sys.getsizeof` (20 min)

Vale la clase entera. `sys.getsizeof(lista)` da casi exactamente lo mismo que el `ndarray`, así
que un alumno que lo use "comprueba" que las listas son igual de eficientes. Está midiendo el
arreglo de punteros (8 bytes por elemento), no los objetos `float` de Python (24 bytes cada uno)
que hay al otro lado.

Alguien va a decir que la diferencia de tiempo es irrelevante. Tiene razón con 40.000 elementos.
Escala la pregunta:

> *"Con 40 millones —un mes de flota, no 153 segmentos— el ciclo tarda medio minuto y el
> vectorizado menos de un segundo. Y eso dentro de un entrenamiento que repite la operación miles
> de veces."*

### Bloque 4 — El error silencioso (30 min) ⭐⭐

**Este bloque es el que justifica la sesión completa.**

Montaje, en este orden exacto:

1. Muestran que el filtro de ciclistas deja el índice en `76, 194, 199, 305, 312, …`
2. `ciclistas.loc[0]` da `KeyError`. *"Este es un buen error: se ve y se corrige."*
3. **Antes de ejecutar la celda del TODO 7, pregunta cuántos `NaN` van a salir.**

La sala se divide entre "ninguno" y "todos". **Nadie dice 771.**

4. Ejecutar. Son 771 de 789. Y 18 filas **sí tienen valor, y está equivocado**.

La frase que debe quedar, dicha tal cual:

> *"Un resultado parcialmente lleno es más peligroso que uno vacío, porque parece que funcionó."*

Explica de dónde salen los 18: son las etiquetas originales de ciclistas menores que 789, que por
casualidad existen también en el índice nuevo. Pandas encuentra la etiqueta, la considera una
coincidencia legítima y copia el valor de otra detección.

> *"Con 789 filas se nota. Con cuatro millones y un `.head()` que sale perfecto porque las
> primeras etiquetas sí coincidieron, no se nota hasta producción."*

Si sobra tiempo, engancha con el `SettingWithCopyWarning`: `df[df.x > 5]["y"] = 0` no modifica el
original, avisa con un warning que nadie lee, y lo correcto es `df.loc[df.x > 5, "y"] = 0`. Es el
mismo malentendido de fondo.

### Bloque 6 — Lo que el CSV se lleva (20 min) ⭐

El encadenamiento es lo que hace el punto. En el TODO 4 ahorraron un 57 % de memoria
(20,1 → 8,7 MB). En el TODO 13 descubren que **lo perdieron entero** al guardar en CSV: 11 de 16
columnas vuelven con otro tipo.

Di el matiz honesto, o alguien lo descubrirá solo y desconfiará del resto: **Parquet no gana
siempre**. Con unos cientos de filas su encabezado pesa más de lo que ahorra y llega a ser más
grande que el CSV. La compresión columnar necesita repetición.

Y no desprecies Excel. No pesa mucho menos que el CSV porque un `.xlsx` es un ZIP de XML, que es
verborrágico. Su ventaja no es técnica: **es que el área de negocio lo abre**. Esa es una razón
perfectamente válida para elegirlo.

Cierre del bloque, con la pregunta del TODO 14: *"¿cuántos datasets de Kaggle han visto con una
columna `Unnamed: 0`?"* La van a reconocer.

---

## Qué recortar, en orden

Si el tiempo se va —y se va—, este es el orden en que conviene sacrificar:

| Orden | Qué se recorta | Cómo |
|---|---|---|
| 1.º | Act. 1.2, bloque 5 (`groupby` / `merge` / `pivot`) | Trabajo autónomo; ya lo vieron en la 1.3 |
| 2.º | Act. 1.1, bloque 3 (JSON anidado) | Proyectar la salida del solucionario y explicar la diferencia |
| 3.º | Act. 1.1, bloque 5 (herramientas) | Dejar solo la tabla comparativa |
| 4.º | Act. 1.2, bloque 2 (`float32`) | Mencionar el 50 % de ahorro sin el ejercicio |

**Nunca:** el bloque 2 y el 6 de la Actividad 1.1, ni el bloque 4 y el 6 de la 1.2.

---

## Después de clase

1. Los equipos deben salir con la **ficha de fuentes** empezada. Sin ella no tienen sobre qué
   trabajar en la 1.2.
2. Recuérdales copiar `10_proyecto_equipo_plantilla.ipynb` con el nombre de su equipo **antes**
   de rellenarla. El siguiente `git merge upstream/main` sobrescribe el original.
3. Revisa que cada equipo tenga asignado un **responsable de los datos**. Los que dejan ese
   casillero vacío son los que en la semana 4 no van a poder decir de dónde salió su dataset.
