"""Fuente única de la Actividad 3.1 — Ajuste de hiperparámetros.

Indicador de logro **IL 3.1**: *aplica estrategias de ajuste de hiperparámetros para
maximizar el rendimiento y la eficiencia de los modelos seleccionados.*

**6 horas pedagógicas.** Cifras medidas sobre ``detecciones_waymo_like.csv``, semilla 42.

Regenerar:  python herramientas/construir_notebooks.py
"""

from __future__ import annotations

from contenido_semana01 import URL_REPO, code, md, md_docente

RA3 = """**Resultado de aprendizaje (RA3):** elabora soluciones avanzadas de aprendizaje automático
mediante la optimización de hiperparámetros, técnicas de ensamble y validación cruzada, para
garantizar la precisión y generalización del modelo frente a objetivos de negocio complejos."""

SETUP = f"""
import sys
from pathlib import Path

EN_COLAB = "google.colab" in sys.modules

if EN_COLAB:
    REPO = Path("mly1101-machine-learning")
    if not REPO.exists():
        !git clone -q {URL_REPO}.git {{REPO}}
    RAIZ = REPO.resolve()
else:
    RAIZ = Path("..").resolve()

sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "kedro_mly1101" / "src"))

RUTA_DATOS = RAIZ / "datos" / "crudos" / "detecciones_waymo_like.csv"
RUTA_PARAMETROS = RAIZ / "kedro_mly1101" / "conf" / "base" / "parameters.yml"
print("Colab:", EN_COLAB, "| dataset:", RUTA_DATOS.exists())
"""

PREPARACION = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from kedro_mly1101.pipelines.preprocesamiento import nodes as limpieza
from kedro_mly1101.pipelines.supervisado import nodes as supervisado
from kedro_mly1101.pipelines.optimizacion import nodes as optimizacion

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 150)
sns.set_theme(style="whitegrid")

PARAMETROS = yaml.safe_load(RUTA_PARAMETROS.read_text(encoding="utf-8"))
CONFIG, FUGA, AJUSTE = PARAMETROS["modelo"], PARAMETROS["fuga"], PARAMETROS["ajuste"]

# La misma cadena de siempre: limpieza del RA1 -> partición del RA2.
crudo = pd.read_csv(RUTA_DATOS)
paso = limpieza.normalizar_categorias(crudo, PARAMETROS["mapas_categorias"])
paso = limpieza.descubrir_faltantes(paso, PARAMETROS["centinelas"])
paso = limpieza.marcar_imposibles(paso, PARAMETROS["reglas_dominio"])
limpio = limpieza.quitar_duplicados_y_constantes(paso, PARAMETROS["columnas_a_descartar"])

marcada = supervisado.particionar(
    supervisado.preparar_variables(limpio, CONFIG, FUGA), CONFIG
)
entrena = marcada[marcada["particion"] == "entrenamiento"]

print(f"Entrenamiento: {len(entrena):,} filas en {entrena[CONFIG['grupo']].nunique()} segmentos")
print(f"Métrica de trabajo: {AJUSTE['metrica']}  ·  pliegues: {AJUSTE['n_pliegues']}")
"""

CELDAS_ACT31: list[dict] = [
    md(
        f"""
# MLY1101 · Machine Learning — Actividad 3.1
## Ajuste de hiperparámetros

{RA3}

**Indicador de logro (IL 3.1):** aplica estrategias de ajuste de hiperparámetros para maximizar
el rendimiento y la eficiencia de los modelos seleccionados.

---

### Dónde estamos

El RA2 dejó un modelo que funciona: F1-macro de **0,70**, con un recall de 0,40 en la clase
minoritaria. La pregunta del RA3 es si se puede hacer mejor, y **cómo saber si de verdad
mejoró**.

Hoy: los hiperparámetros. Los parámetros que el modelo **no** aprende de los datos y que hay
que elegir desde fuera: cuántos árboles, qué profundidad, cuántas muestras por hoja.

---

### La idea central, y probablemente te va a decepcionar

> **El ajuste de hiperparámetros da mejoras de segundo orden.**

Las mejoras de primer orden vienen de otra parte: de las variables que elegiste, de cómo
partiste los datos, de haber definido bien el problema. Si el modelo va mal, ajustar
hiperparámetros casi nunca lo salva.

Al final de la sesión vas a haber probado 12 configuraciones distintas y vas a comparar la
mejor contra los valores por defecto. **Guarda tu expectativa** de cuánto vas a ganar.

---

### Dónde se ajusta, que es lo que de verdad se evalúa

Ajustar exige comparar configuraciones, y comparar exige medir. ¿Medir dónde?

- **En entrenamiento:** no sirve. Más complejidad siempre puntúa mejor ahí.
- **En prueba:** es hacer trampa. Si eliges la configuración que mejor puntúa en la prueba, esa
  puntuación deja de estimar el futuro.
- **En validación cruzada dentro del entrenamiento:** correcto. Y aquí, por lo mismo del RA2,
  los pliegues tienen que respetar el segmento.

---

### Al final de la sesión debes entregar

El informe de ajuste: qué espacio exploraste, con qué esquema de validación, **cuánto ganaste**
y si esa ganancia supera la variabilidad entre pliegues.
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Actividad 3.1
>
> **6 horas pedagógicas.** La distribución de abajo cubre ~3 h de trabajo guiado; el resto es
> para aplicar el mismo esquema al caso oficial del equipo, que va a la Parcial 3.
>
> | Bloque | Min | Foco |
> |---|---|---|
> | 0 · Encuadre | 15 | Qué es un hiperparámetro y qué no |
> | 1 · Validación cruzada por grupo ⭐ | 40 | Dónde se mide, y por qué GroupKFold |
> | 2 · La búsqueda | 40 | Rejilla vs aleatoria |
> | 3 · ¿Cuánto ganamos? ⭐⭐ | 40 | **El resultado decepciona, y ese es el punto** |
> | 4 · La trampa de ajustar en prueba ⭐ | 35 | Más sutil que la fuga del RA2 |
> | Cierre | 10 | Informe de ajuste |
>
> **El bloque 3 es el que sostiene la sesión y no se recorta.** El ajuste sale **peor** que los
> valores por defecto: −0,0006. No lo adelantes.
"""
    ),
    md("---\n## Preparación del entorno"),
    code(SETUP),
    code(PREPARACION),
    md(
        """
---
# Bloque 1 · ⭐ Dónde se mide: validación cruzada por grupo

Para comparar configuraciones hace falta una estimación de desempeño **que no use la prueba**.
Se saca partiendo el entrenamiento en `k` pliegues: se entrena con `k−1` y se mide en el que
queda, `k` veces.

**Y los pliegues tienen que respetar el segmento**, exactamente por lo mismo que la partición
de la Actividad 2.2: las detecciones de un segmento comparten contexto. `GroupKFold` lo
garantiza.

### ✏️ TODO 1 — Comprobar que los pliegues no rompen segmentos
"""
    ),
    code(
        """
from sklearn.model_selection import GroupKFold

X = entrena[CONFIG["variables"]]
y = entrena[CONFIG["objetivo"]]
grupos = entrena[CONFIG["grupo"]]

cv = GroupKFold(n_splits=AJUSTE["n_pliegues"])

filas = []
for numero, (idx_entrena, idx_valida) in enumerate(cv.split(X, y, groups=grupos), start=1):
    seg_entrena = set(grupos.iloc[idx_entrena])
    seg_valida = set(grupos.iloc[idx_valida])
    filas.append(
        {
            "pliegue": numero,
            "filas_entrena": len(idx_entrena),
            "filas_valida": len(idx_valida),
            "segmentos_valida": len(seg_valida),
            "segmentos_compartidos": len(seg_entrena & seg_valida),
        }
    )
pd.DataFrame(filas)
""",
        """
# TODO 1: ¿los pliegues respetan el segmento?
from sklearn.model_selection import GroupKFold

X = entrena[CONFIG["variables"]]
y = entrena[CONFIG["objetivo"]]
grupos = entrena[CONFIG["grupo"]]

cv = ____(n_splits=AJUSTE["n_pliegues"])

filas = []
for numero, (idx_entrena, idx_valida) in enumerate(cv.split(X, y, groups=____), start=1):
    seg_entrena = set(grupos.iloc[idx_entrena])
    seg_valida = set(grupos.iloc[idx_valida])
    filas.append(
        {
            "pliegue": numero,
            "filas_entrena": len(idx_entrena),
            "filas_valida": len(idx_valida),
            "segmentos_valida": len(seg_valida),
            "segmentos_compartidos": len(seg_entrena & seg_valida),
        }
    )
pd.DataFrame(filas)
""",
    ),
    code(
        """
# Autochequeo
tabla = pd.DataFrame(filas)
assert (tabla["segmentos_compartidos"] == 0).all(), (
    "revisa: ningún pliegue puede compartir segmentos. ¿Pasaste groups= al split?"
)
print(f"✅ {len(tabla)} pliegues, 0 segmentos compartidos en todos.")
print("   La estimación que salga de aquí es honesta: cada pliegue evalúa")
print("   sobre segmentos que el modelo nunca vio.")
"""
    ),
    md(
        """
### ✏️ TODO 2 — La métrica del ajuste

Antes de buscar hay que decidir **qué se está maximizando**. Mira el parámetro
`AJUSTE["metrica"]` y responde.

**✍️ Tu respuesta:**

*(doble clic aquí y escribe)*

¿Por qué se optimiza `f1_macro` y no `accuracy`? *(Pista: revisa lo que descubriste en el
bloque 3 de la Actividad 2.2.)*
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 1
>
> **Cifras medidas:** 5 pliegues sobre 29.946 filas de entrenamiento en 114 segmentos,
> **0 segmentos compartidos** en todos.
>
> **Respuesta al TODO 2:** porque el problema está desbalanceado 89/11 y la exactitud del
> baseline trivial ya es 0,8896. Optimizar exactitud llevaría al buscador a configuraciones que
> **abandonan la clase minoritaria**, que es justo la que interesa. `f1_macro` pondera las dos
> clases por igual.
>
> **La frase:** *elegir la métrica del ajuste es elegir qué error te importa. Se decide antes de
> buscar, no después de ver los resultados.*
>
> **Criterio de logro:** verifica los 0 segmentos compartidos y justifica la métrica en términos
> del desbalance.
"""
    ),
    md(
        """
---
# Bloque 2 · Buscar: rejilla contra búsqueda aleatoria

| | Rejilla (`GridSearchCV`) | Aleatoria (`RandomizedSearchCV`) |
|---|---|---|
| Qué prueba | **Todas** las combinaciones | `n_iter` combinaciones al azar |
| Costo | Producto de las opciones: explota | El que tú decidas |
| Ventaja | Exhaustiva en su rejilla | Con el mismo presupuesto explora más regiones |

Con 4 × 6 × 4 × 3 = **288 combinaciones**, cada una con 5 pliegues, la rejilla exigiría 1.440
entrenamientos. La búsqueda aleatoria hace 12 × 5 = 60.

> **Por qué la aleatoria suele bastar:** casi siempre solo un par de hiperparámetros importan de
> verdad. La rejilla gasta la mayor parte del presupuesto variando los que dan igual.

### ✏️ TODO 3 — Lanzar la búsqueda
"""
    ),
    code(
        """
busqueda = optimizacion.buscar_hiperparametros(marcada, CONFIG, AJUSTE)
busqueda.head(6)
""",
        """
# TODO 3: busca hiperparámetros con validación cruzada por grupo.
busqueda = optimizacion.____(marcada, CONFIG, AJUSTE)
busqueda.head(6)
""",
    ),
    md(
        """
### ✏️ TODO 4 — Leer la tabla con desconfianza

Mira las columnas `mean_test_score` y `std_test_score`.

**✍️ Tu respuesta:**

*(doble clic aquí y escribe)*

Compara la diferencia entre la primera y la segunda configuración con la desviación típica
entre pliegues de la primera. ¿Puedes afirmar que la primera es mejor?
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 2
>
> **Cifras medidas** (12 combinaciones, 5 pliegues, `f1_macro`):
>
> | Rango | F1-macro | Desv. | `n_estimators` | `max_depth` | `min_samples_leaf` | `max_features` |
> |---|---|---|---|---|---|---|
> | 1 | **0,6964** | 0,0091 | 400 | 16 | 1 | sqrt |
> | 2 | 0,6933 | 0,0076 | 50 | sin límite | 1 | log2 |
> | 3 | 0,6922 | 0,0078 | 100 | 4 | 20 | sqrt |
> | 4 | 0,6917 | 0,0085 | 100 | 12 | 5 | 0,5 |
>
> **Respuesta al TODO 4:** la diferencia entre la 1.ª y la 2.ª es **0,0031**; la desviación entre
> pliegues de la 1.ª es **0,0091**, casi tres veces mayor. **No hay evidencia de que la primera
> sea mejor.** El "ranking" ordena ruido.
>
> Es la primera vez en el curso que se cuestiona un ranking, y conviene decirlo así:
>
> > *Que scikit-learn te devuelva las configuraciones ordenadas no significa que ese orden
> > signifique algo. Ordenar siempre se puede; distinguir, no siempre.*
>
> **Fíjate además en el desorden del ranking:** 50 árboles sin límite de profundidad queda
> segundo, y 100 árboles con profundidad 4 queda tercero. No hay un patrón claro, que es
> exactamente lo que se espera cuando las diferencias son ruido.
>
> **Criterio de logro:** compara la diferencia contra la desviación y concluye que no son
> distinguibles.
"""
    ),
    md(
        """
---
# Bloque 3 · ⭐⭐ ¿Cuánto ganamos de verdad?

Ya tenemos la mejor configuración de 12. Ahora la comparación que importa: **contra no haber
ajustado nada**.

### ✏️ TODO 5 — Antes de ejecutar, apuesta

**Creo que el ajuste mejorará el F1-macro en:** `____`

*(Escríbelo. Otra vez.)*
"""
    ),
    code(
        """
ganancia = optimizacion.comparar_ajuste_contra_defecto(marcada, CONFIG, AJUSTE, busqueda)
ganancia
""",
        """
# TODO 5: ¿cuánto ganó el ajuste sobre los valores por defecto?
ganancia = optimizacion.____(marcada, CONFIG, AJUSTE, busqueda)
ganancia
""",
    ),
    code(
        """
# Autochequeo
delta = ganancia.loc[1, "ganancia"]
ruido = ganancia.loc[0, "desv_entre_pliegues"]
print(f"Ganancia del ajuste : {delta:+.4f}")
print(f"Ruido entre pliegues: {ruido:.4f}")
print()
assert abs(delta) < ruido, (
    "revisa: la ganancia debería quedar por debajo del ruido entre pliegues"
)
print("✅ La ganancia del ajuste es MENOR que la variabilidad entre pliegues.")
print("   Traducido: 12 configuraciones, 60 entrenamientos, y no hay evidencia")
print("   de haber mejorado nada.")
"""
    ),
    md(
        """
### ✏️ TODO 6 — Entonces, ¿el ajuste no sirve?

**✍️ Tu respuesta:**

*(doble clic aquí y escribe)*

1. ¿Fue inútil esta sesión?
2. Si el ajuste da tan poco, ¿de dónde vienen las mejoras grandes en un proyecto de ML?
3. ¿En qué situación **sí** esperarías que el ajuste diera una mejora importante?
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 3 ⭐⭐
>
> **Cifras medidas:**
>
> | Configuración | F1-macro | Desv. entre pliegues |
> |---|---|---|
> | Valores por defecto | **0,6970** | 0,0087 |
> | Mejor de la búsqueda | **0,6964** | 0,0091 |
> | **Ganancia** | **−0,0006** | |
>
> **El ajuste salió peor.** Y la diferencia (0,0006) es catorce veces menor que el ruido
> (0,0087), así que ni siquiera es "peor": es **indistinguible**.
>
> **Deja que la decepción se instale.** Es real y es la lección. Alguien va a preguntar si algo
> se hizo mal: no. Se hizo bien, y el resultado es que no había nada que ganar por ahí.
>
> **Respuestas esperadas al TODO 6:**
>
> 1. **No fue inútil: ahora sabes que no hay nada ahí.** Antes lo suponías. Descartar una vía con
>    evidencia es un resultado, aunque no sea el que esperabas. Y además tienes el argumento para
>    defender que el modelo por defecto es suficiente, en vez de gastar semanas en ajustarlo.
> 2. **De las variables** (ingeniería de características), **de más datos**, de haber definido
>    mejor el problema y de la calidad de las etiquetas. En el RA1 y el RA2, no aquí.
> 3. Cuando los valores por defecto están **lejos** de lo razonable: redes neuronales, SVM con
>    kernel, boosting con tasa de aprendizaje mal puesta. `RandomForestClassifier` tiene
>    valores por defecto muy sensatos, y por eso mueve poco.
>
> **La frase de la sesión:**
>
> > *Ajustar hiperparámetros es lo último que hay que hacer, y lo primero que todos quieren
> > hacer, porque es lo único que se puede automatizar.*
>
> **Criterio de logro:** reconoce que la ganancia no supera el ruido, no concluye que el ajuste
> "sirvió", y ubica las mejoras de primer orden fuera del ajuste.
"""
    ),
    md(
        """
---
# Bloque 4 · ⭐ La trampa: ajustar mirando la prueba

Nadie *entrena* con la prueba. Pero mucha gente **elige** mirándola: prueba varias
configuraciones, ve cuál puntúa mejor en el conjunto de prueba y reporta ese número.

El modelo nunca vio esos datos. ¿Cuál es el problema?

Que la puntuación que reportas ya no es una estimación del desempeño futuro: es **el máximo de
una muestra**, y el máximo de una muestra siempre es optimista.

### ✏️ TODO 7 — Medirlo
"""
    ),
    code(
        """
fuga = optimizacion.medir_fuga_por_ajustar_en_prueba(marcada, CONFIG, AJUSTE)
fuga
""",
        """
# TODO 7: ¿cuánto se infla la métrica al elegir mirando la prueba?
fuga = optimizacion.____(marcada, CONFIG, AJUSTE)
fuga
""",
    ),
    md(
        """
### ✏️ TODO 8 — Tres cosas distintas en una tabla

La tabla mide tres cosas que se confunden con facilidad. Explica cada una.

**✍️ Tu respuesta:**

*(doble clic aquí y escribe)*

1. **Optimismo:** salió `____`. ¿Significa que la trampa es inofensiva?
2. **Margen de la trampa:** `____`. ¿Qué representa?
3. **Brecha validación vs prueba:** la validación cruzada puntúa sistemáticamente **más bajo**
   que la prueba. ¿Es esto una fuga? ¿Por qué pasa?
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 4 ⭐
>
> **Cifras medidas** (seis profundidades, `RandomForest` de 100 árboles):
>
> | `max_depth` | Validación cruzada | En prueba | Brecha |
> |---|---|---|---|
> | 4 | 0,6918 | 0,7027 | +0,0109 |
> | 6 | 0,6919 | 0,7022 | +0,0103 |
> | 8 | 0,6906 | 0,7009 | +0,0103 |
> | 12 | 0,6914 | 0,7013 | +0,0099 |
> | 16 | 0,6956 | 0,7077 | +0,0121 |
> | **sin límite** | **0,6970** | **0,7144** | +0,0174 |
>
> | Resumen | Valor |
> |---|---|
> | Optimismo del criterio tramposo | **0,0** |
> | Margen de la trampa (mejor − peor en prueba) | **0,0135** |
> | Brecha media validación vs prueba | **+0,0118** |
>
> **El optimismo salió CERO, y hay que explicarlo bien o se saca la conclusión contraria.**
> Ambos criterios eligieron la misma configuración (*sin límite*), así que esta vez la trampa no
> pagó. **Eso no la vuelve inofensiva: significa que la suerte no la premió.**
>
> **Respuestas al TODO 8:**
>
> 1. No. Un cero aquí es una coincidencia de esta corrida, no una propiedad del método. Es el
>    mismo razonamiento del bloque 2 de la Actividad 2.2: un riesgo que no se manifestó sigue
>    siendo un riesgo.
> 2. **0,0135** es lo que separa la mejor configuración de la peor *en la prueba*: el tamaño del
>    premio por elegir mal. Es más del doble de la ganancia que se buscaba con todo el ajuste
>    (0,0006). Dicho de otro modo: **haciendo trampa se gana veinte veces más que ajustando
>    bien**, y por eso la tentación es real.
> 3. **No es fuga.** La validación cruzada entrena con 4/5 de los datos, así que estima el
>    desempeño de un modelo entrenado con **menos** datos del que finalmente entregas. Es un
>    sesgo **conservador y conocido**, y por eso es un criterio de selección seguro: se equivoca
>    hacia abajo.
>
> **La distinción que hay que dejar clara:** *"me da un número más bajo"* y *"me engaña"* no son
> lo mismo. La validación cruzada hace lo primero; ajustar en prueba hace lo segundo.
>
> **Criterio de logro:** distingue las tres magnitudes, explica por qué un optimismo de cero no
> exonera al método, e identifica la brecha validación/prueba como sesgo conservador y no fuga.
"""
    ),
    md(
        """
---
# Cierre · Informe de ajuste

**Modelo ajustado:** `____` · **Métrica optimizada:** `____` · **Por qué esa métrica:** `____`

### Esquema de validación

| Campo | Valor |
|---|---|
| Tipo de validación | `____` |
| Pliegues | `____` |
| Variable de agrupación | `____` |
| Segmentos compartidos entre pliegues | `____` |

### Espacio explorado

| Hiperparámetro | Valores | Por qué ese rango |
|---|---|---|
| `____` | | |
| `____` | | |

**Estrategia** (rejilla o aleatoria) **y por qué:** `____`
**Combinaciones probadas:** `____` de `____` posibles

### El resultado

| | F1-macro | Desv. entre pliegues |
|---|---|---|
| Valores por defecto | `____` | `____` |
| Mejor configuración | `____` | `____` |
| **Ganancia** | `____` | |

**¿La ganancia supera la variabilidad entre pliegues?** `____`
**Conclusión:** `____`

> Si tu ganancia no supera el ruido, **dilo**. Reportar una mejora que no puedes distinguir del
> azar es el error que esta sesión existe para evitar.

### Dónde buscaría la próxima mejora

`____`

*(Y por qué ahí y no en más ajuste.)*
"""
    ),
    md_docente(
        """
> ### 🎓 Criterios de logro — Actividad 3.1 (IL 3.1)
>
> | Nivel | Descripción |
> |---|---|
> | **Destacado (4)** | Todo lo del 3, y además: **no reporta la ganancia como mejora** al no superar el ruido; explica por qué un optimismo de cero no exonera la trampa; distingue la brecha validación/prueba de una fuga |
> | **Logrado (3)** | Valida con `GroupKFold` y 0 segmentos compartidos; justifica la métrica por el desbalance; compara contra los valores por defecto y contra la desviación entre pliegues |
> | **En desarrollo (2)** | Ejecuta la búsqueda y reporta la mejor configuración, pero presenta la ganancia como mejora sin contrastarla con el ruido |
> | **Inicial (1)** | Ajusta sobre la prueba, o valida sin agrupar por segmento |
>
> **Lo primero que hay que mirar al corregir:** si el informe dice *"el ajuste mejoró el modelo"*.
> Con estos datos es falso, y es el error central que la sesión previene.
"""
    ),
]
