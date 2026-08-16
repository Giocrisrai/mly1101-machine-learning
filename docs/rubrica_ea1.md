# Rúbrica de evaluación — EA1 · Análisis y Preprocesamiento de Datos

**Asignatura:** MLY1101 Machine Learning · **Experiencia:** EA1 (20 h)
**Entregable evaluado en la Semana 1:** notebook desarrollado + mini-informe de calidad de datos.

> Esta rúbrica cubre lo trabajado en la **Semana 1**. La EA1 completa abarca 20 horas, así que
> corresponde ampliarla cuando se sumen las semanas siguientes (preprocesamiento aplicado,
> codificación y escalado).

---

## Resultado de aprendizaje

**RA1:** *Implementa estrategias y técnicas de preprocesamiento en el diseño de soluciones de
Machine Learning, considerando el tratamiento responsable de la información.*

| Indicador | Dónde se evidencia | Peso |
|---|---|---|
| IL1 · Explora un conjunto de datos e identifica su estructura y tipos de variables | Bloques 1 y 2 (TODO 1–6) | 20 % |
| IL2 · Identifica problemas de calidad de datos y los cuantifica | Bloques 3 y 4 (TODO 7–15) | 30 % |
| IL3 · Fundamenta decisiones de preprocesamiento | Bloque 5 (TODO 16–17) + informe | 25 % |
| IL4 · Reconoce implicancias éticas y sesgos en los datos | Bloque 6 (TODO 18) + informe | 15 % |
| IL5 · Comunica el análisis con orden y documentación | Notebook completo + informe | 10 % |

---

## Escala

| Nivel | Puntaje | Descripción general |
|---|---|---|
| **Destacado** | 4 | Va más allá de lo pedido: anticipa consecuencias, propone alternativas y las justifica |
| **Logrado** | 3 | Cumple el indicador de forma correcta y completa |
| **En desarrollo** | 2 | Cumple parcialmente; hay omisiones o justificaciones débiles |
| **Inicial** | 1 | Evidencia mínima o incorrecta |
| **No entregado** | 0 | No hay evidencia |

### Conversión a nota

Se usa la **exigencia del 60 %**, que es la norma institucional: el Reglamento Académico de
Duoc UC establece la escala de 1,0 a 7,0 con un decimal, y que **la nota 4,0 corresponde a haber
alcanzado el 60 % del logro de aprendizaje** de la asignatura. La escala es lineal a ambos lados
de ese punto.

Sea `P` el puntaje ponderado (0 a 4), `Pmax = 4` y `U = 0,6 · Pmax = 2,4`:

```
P < 2,4 :  nota = 1,0 + 3,0 · P / 2,4
P ≥ 2,4 :  nota = 4,0 + 3,0 · (P − 2,4) / 1,6
```

Referencia rápida con puntaje uniforme en los cinco indicadores:

| Nivel en todos los indicadores | Puntaje | Logro | Nota |
|---|---|---|---|
| Destacado (4) | 4,0 | 100 % | **7,0** |
| Logrado (3) | 3,0 | 75 % | **5,1** |
| En desarrollo (2) | 2,0 | 50 % | **3,5** |
| Inicial (1) | 1,0 | 25 % | **2,2** |
| No entregado (0) | 0,0 | 0 % | **1,0** |

**Calculadora incluida.** Para no hacer esto a mano:

```bash
python herramientas/calcular_nota.py 3 4 2 3 3      # IL1 IL2 IL3 IL4 IL5 → nota 5,2
python herramientas/calcular_nota.py --csv docs/ejemplo_notas.csv   # el curso completo
python herramientas/calcular_nota.py 3 4 2 3 3 --exigencia 0.5   # otra regla, si aplicara
```

El CSV debe tener la cabecera `nombre,IL1,IL2,IL3,IL4,IL5` (hay un ejemplo en
[`docs/ejemplo_notas.csv`](ejemplo_notas.csv)); la salida incluye promedio del curso y porcentaje
de aprobación, y marca con ⚠ a quienes reprueban. La fórmula está cubierta por 14 tests en
`tests/test_calcular_nota.py`.

> El parámetro `--exigencia` existe por si se evalúa una actividad con otra regla, pero para la
> EA1 el 60 % es el valor que corresponde según el Reglamento Académico.

**Coherencia con la ponderación de la asignatura.** El mismo reglamento fija que la nota final se
compone del promedio de las evaluaciones parciales ponderado por 0,60 y el examen final por 0,40.
La EA1 es una de esas evaluaciones parciales; la EFT es el 40 % restante, tal como indica la
coordinación de la línea.

---

## IL1 · Exploración e identificación de variables (20 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Además de clasificar correctamente las variables, distingue el tipo de pandas del tipo estadístico y explica por qué `id_interno` y `sensor_version` no son variables predictoras, con el argumento correcto (cardinalidad y varianza cero) |
| **3** | Clasifica correctamente las 16 columnas y describe qué representa una fila del dataset |
| **2** | Clasifica la mayoría, pero confunde discreta con continua o no distingue nominal de ordinal |
| **1** | Ejecuta `.info()` y `.describe()` sin interpretarlos |

**Evidencia mínima esperada:** TODO 1–4 completos, y una respuesta correcta a *"¿qué representa
una fila?"* (una detección de un objeto en un instante, no un objeto ni un segmento).

---

## IL2 · Identificación y cuantificación de problemas de calidad (30 %)

Este es el indicador de mayor peso: es el núcleo de la sesión.

| Nivel | Criterio observable |
|---|---|
| **4** | Encuentra 9 o 10 de los defectos; detecta el patrón MNAR de `speed_mps` y lo describe con cifras por subgrupo; distingue duplicado exacto de lógico; separa outlier imposible de legítimo con argumento de dominio |
| **3** | Encuentra al menos 7 defectos, incluidos **los dos nulos ocultos** (`-1` y `"N/D"`) y **el duplicado lógico**, y los cuantifica |
| **2** | Encuentra los defectos evidentes (nulos declarados, duplicados exactos, categorías inconsistentes) pero no los ocultos |
| **1** | Reporta que "hay datos sucios" sin cifras |

**Los 10 defectos** (referencia para la corrección):

| # | Defecto | Cifra esperada |
|---|---|---|
| 1 | `timestamp_micros` con `"N/D"` → dtype `object` | 60 filas (0,15 %) |
| 2 | `num_lidar_points` con `-1` como nulo oculto | 1.198 filas (2,9 %) |
| 3 | `weather`: 11 variantes para 3 categorías + nulos | 2.075 nulos (5,1 %) |
| 4 | `object_type`: 7 variantes para 4 categorías | 2.343 filas afectadas |
| 5 | Duplicados exactos y lógicos | 480 exactos + 200 lógicos |
| 6 | Outliers imposibles | 157 sobre 60 m/s · 122 con alto 0 · 80 con largo negativo |
| 7 | Outliers legítimos (buses) | 608 con largo > 12 m |
| 8 | Nulos MNAR en `speed_mps` | 787 nulos; **33,8 %** en LEVEL_2 nocturno vs. 0,4 % en LEVEL_1 |
| 9 | Desbalance de clases | `CYCLIST` 1,9 % |
| 10 | `sensor_version` constante · `id_interno` 98,3 % único | — |

**Criterio de corrección:** se acepta una tolerancia razonable en las cifras (el alumno puede
redondear o contar de otra manera), pero **no** se acepta una afirmación sin cifra.

---

## IL3 · Fundamentación de las decisiones de preprocesamiento (25 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Cada decisión indica qué se gana, **qué se pierde** y qué alternativa se descartó; propone una columna indicadora de faltante o imputación por grupo; identifica el riesgo de fuga de información |
| **3** | La tabla de decisiones está completa, cada decisión tiene justificación, y la función `limpiar()` es coherente con lo declarado |
| **2** | Aplica transformaciones correctas pero las justifica con "para limpiar los datos" |
| **1** | Usa `dropna()` y `drop_duplicates()` sin criterio |

**Señal de alerta en la corrección:** si el dataset limpio del alumno ya **no tiene buses**
(`box_length.max() < 12`), eliminó outliers legítimos. Eso baja el indicador a *En desarrollo*
aunque el resto esté bien, porque revela que aplicó un criterio estadístico sin mirar el dominio.

**Verificación rápida del entregable:**

```python
limpio = alumno_df
assert limpio["box_length"].max() > 12          # los buses siguen ahí
assert limpio.duplicated(subset=LLAVE).sum() == 0
assert set(limpio["object_type"].unique()) == {"vehicle", "pedestrian", "cyclist", "sign"}
```

---

## IL4 · Tratamiento responsable de la información (15 %)

| Nivel | Criterio observable |
|---|---|
| **4** | Conecta el sesgo de muestreo con un riesgo concreto de seguridad, identifica a quién afecta y propone una medida accionable (evaluación por subgrupo, recolección dirigida, restricción del dominio de operación) |
| **3** | Identifica la sub-representación de la conducción nocturna y de los ciclistas, y nombra al grupo en riesgo |
| **2** | Menciona la ética de forma genérica, sin anclarla en los datos analizados |
| **1** | No aborda el punto o lo reduce a "hay que cuidar los datos personales" |

**Ideas que corresponde reconocer como correctas:**
- Un promedio global oculta a las minorías: 97 % de exactitud global puede convivir con 60 % en
  ciclistas nocturnos.
- El error no se reparte al azar: recae sobre los usuarios más vulnerables de la vía.
- La reidentificación no requiere nombres: *lugar + hora + trayectoria* puede bastar.
- Eliminar filas con velocidad faltante sesga el dataset justamente contra las condiciones
  difíciles.

---

## IL5 · Comunicación y documentación (10 %)

| Nivel | Criterio observable |
|---|---|
| **4** | El notebook se lee como un informe: cada bloque tiene su conclusión escrita, los gráficos están rotulados y el mini-informe podría entregarse a alguien que no estuvo en clase |
| **3** | Notebook ejecutable de principio a fin, celdas Markdown con las respuestas, informe completo |
| **2** | Código correcto pero sin interpretación escrita; celdas de discusión en blanco |
| **1** | Solo código; el notebook no ejecuta de corrido |

**Requisito formal:** el notebook debe ejecutar completo con *Kernel → Restart & Run All*. Un
notebook que no ejecuta limita este indicador a *Inicial*, independiente del contenido.

---

## Retroalimentación sugerida

Tres frases que suelen ser las más útiles al devolver el trabajo:

1. *"Encontraste el problema, pero no lo cuantificaste: ¿cuántas filas son y qué porcentaje?"*
2. *"Eliminaste esas filas. ¿Qué parte de la realidad quedó fuera del dataset al hacerlo?"*
3. *"Tu decisión es defendible, pero no está escrito qué se pierde con ella."*
