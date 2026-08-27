"""Fuente única del contenido de la Actividad 2.3 — Modelamiento no supervisado.

Indicador de logro **IL 2.3**: *elabora algoritmos de aprendizaje no supervisado para
descubrir patrones ocultos en los datos.*

Es la actividad más larga del programa: **12 horas pedagógicas**.

De este archivo salen dos notebooks:

- ``notebooks/06_alumno_no_supervisado.ipynb``   (versión con TODO)
- ``notebooks/06_docente_no_supervisado.ipynb``  (versión resuelta con pauta)

Igual que la Actividad 2.2, **reutiliza los nodos del pipeline** en vez de duplicarlos. Cifras
medidas sobre ``detecciones_waymo_like.csv`` con la semilla 42.

Regenerar tras editar:

    python herramientas/construir_notebooks.py
"""

from __future__ import annotations

from contenido_semana01 import URL_REPO, code, md, md_docente

CELDAS_ACT23: list[dict] = [
    md(
        """
# MLY1101 · Machine Learning — Actividad 2.3
## Modelamiento no supervisado: encontrar estructura sin etiquetas

**Resultado de aprendizaje (RA2):** aplica modelos estadísticos al conjunto de datos
procesados para interpretarlos, utilizando metodologías ágiles, con la finalidad de obtener
conocimientos relevantes que permitan responder a las necesidades del contexto de negocio,
considerando aspectos éticos.

**Indicador de logro (IL 2.3):** elabora algoritmos de aprendizaje no supervisado para
descubrir patrones ocultos en los datos.

---

### La contracara de la Actividad 2.2

En la 2.2 había una etiqueta y se medía el acierto. Hoy **no hay etiqueta**, así que no se
puede acertar ni fallar. Y eso cambia todo:

| | Act. 2.2 · Supervisado | Act. 2.3 · No supervisado |
|---|---|---|
| Entrada | `X` e `y` | Solo `X` |
| Qué busca | Predecir `y` | Estructura oculta |
| Cómo se sabe si funcionó | Se mide contra `y` | **Hay que argumentarlo** |
| Riesgo típico | Fuga de información | **Encontrar patrones que no existen** |

Esa última fila es la sesión de hoy. K-medias **siempre** devuelve grupos: si le pides cuatro
sobre ruido puro, te da cuatro. Que existan no significa que signifiquen algo.

---

### La pregunta de hoy

> Sin decirle a nadie qué es cada objeto, ¿aparecen **grupos naturales** en las detecciones?
> ¿Y coinciden con los tipos que el sensor etiquetó?

---

### La idea central

> **Un grupo sin nombre no es un hallazgo.**

"Grupo 2" no le sirve a nadie. El trabajo empieza cuando el algoritmo termina: hay que mirar el
perfil de cada grupo y poder decir *"objetos grandes y rápidos"*, *"objetos cercanos al sensor"*.
Eso no lo hace el algoritmo. Lo haces tú, y es lo que se evalúa.

---

### Al final de la sesión debes entregar

Un **informe de segmentación** con:

- la justificación del número de grupos elegido, con **dos** criterios y qué haces si discrepan;
- el perfil e interpretación de cada grupo, con nombre propio;
- el contraste con la etiqueta conocida, y qué te dice sobre lo que descubriste;
- una conclusión honesta sobre si la estructura encontrada es útil para el negocio.
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Actividad 2.3
>
> **Es la actividad más larga del programa: 12 horas pedagógicas.** La distribución de abajo
> cubre unas 4 h de trabajo guiado sobre el dataset de la asignatura. Las 8 h restantes son para
> aplicar lo mismo al caso oficial elegido (Telco, Housing o Spotify) y para profundizar en las
> alternativas que se mencionan al cierre: agrupamiento jerárquico, DBSCAN y selección de
> variables.
>
> **Distribución del bloque guiado:**
>
> | Bloque | Min | Foco |
> |---|---|---|
> | 0 · Encuadre | 15 | Sin etiqueta no se puede "acertar" |
> | 1 · Escalar ⭐ | 40 | Sin escalar, agrupas por la unidad de medida |
> | 2 · Cuántos grupos ⭐⭐ | 50 | Inercia vs silueta, y qué hacer si discrepan |
> | 3 · Agrupar y **perfilar** ⭐⭐ | 50 | Un grupo sin nombre no es un hallazgo |
> | 4 · Contrastar con la etiqueta | 35 | No es una evaluación, es una comprobación de sentido |
> | 5 · Reducción de dimensionalidad | 35 | PCA y cuánta información se pierde al dibujar |
> | Cierre · Informe | 15 | ¿Sirve para el negocio? |
>
> **Los imprescindibles son el 2 y el 3.** Si el tiempo se acorta, se comprime el 5.
>
> **El momento fuerte de la sesión está en el bloque 3**, cuando descubran que uno de los grupos
> son los **buses** — los mismos atípicos legítimos que en la EA1 aprendieron a no eliminar. No
> lo adelantes.
>
> **Regla de oro, la de siempre:** ninguna afirmación sin una cifra que la respalde.
"""
    ),
    md(
        """
---
## Preparación del entorno

Como en la Actividad 2.2, no reescribimos nada: importamos los mismos nodos del pipeline.
"""
    ),
    code(
        f"""
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
    ),
    code(
        """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from kedro_mly1101.pipelines.preprocesamiento import nodes as limpieza
from kedro_mly1101.pipelines.no_supervisado import nodes as grupos

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 140)
sns.set_theme(style="whitegrid")

PARAMETROS = yaml.safe_load(RUTA_PARAMETROS.read_text(encoding="utf-8"))
CONFIG = PARAMETROS["agrupamiento"]

crudo = pd.read_csv(RUTA_DATOS)
paso = limpieza.normalizar_categorias(crudo, PARAMETROS["mapas_categorias"])
paso = limpieza.descubrir_faltantes(paso, PARAMETROS["centinelas"])
paso = limpieza.marcar_imposibles(paso, PARAMETROS["reglas_dominio"])
limpio = limpieza.quitar_duplicados_y_constantes(paso, PARAMETROS["columnas_a_descartar"])

print(f"Datos limpios: {limpio.shape[0]:,} filas")
print("Variables para agrupar:", CONFIG["variables"])
print("Etiqueta de contraste (NO se usa para agrupar):", CONFIG["etiqueta_de_contraste"])
"""
    ),
    # ======================================================================
    # BLOQUE 1 — ESCALAR
    # ======================================================================
    md(
        """
---
# Bloque 1 · ⭐ Escalar no es opcional

K-medias agrupa por **distancia euclídea**. Y la distancia suma los cuadrados de las diferencias
de cada variable, **sin importar en qué unidad estén**.

Mira nuestras variables:

| Variable | Rango típico |
|---|---|
| `box_height` | ~0,5 a 4 metros |
| `num_lidar_points` | 0 a **varios miles** |

Una diferencia de 500 puntos láser pesa cientos de veces más que una diferencia de 2 metros de
altura. Sin escalar, el "agrupamiento por objeto" sería en realidad **un agrupamiento por número
de puntos** con otro nombre.
"""
    ),
    md(
        """
### ✏️ TODO 1 — Ver el problema antes de arreglarlo

Compara la desviación típica de cada variable **antes** de escalar.
"""
    ),
    code(
        """
antes = limpio[CONFIG["variables"]].describe().T[["mean", "std", "min", "max"]]
print(antes.round(2).to_string())
print(f"\\nRazón entre la mayor y la menor desviación típica: "
      f"{antes['std'].max() / antes['std'].min():,.0f}×")
""",
        """
# TODO 1: ¿qué tan distintas son las escalas de las variables?
antes = limpio[CONFIG["variables"]].____().T[["mean", "std", "min", "max"]]
print(antes.round(2).to_string())
print(f"\\nRazón entre la mayor y la menor desviación típica: "
      f"{antes['std'].____() / antes['std'].min():,.0f}×")
""",
    ),
    md(
        """
### ✏️ TODO 2 — Escalar

`grupos.preparar_matriz()` selecciona las variables, descarta filas incompletas y las
estandariza: media 0 y desviación típica 1.
"""
    ),
    code(
        """
matriz = grupos.preparar_matriz(limpio, CONFIG)

despues = matriz[CONFIG["variables"]].describe().T[["mean", "std"]]
print(despues.round(4).to_string())
print(f"\\nFilas: {len(matriz):,}")
""",
        """
# TODO 2: escala las variables.
matriz = grupos.____(limpio, CONFIG)

despues = matriz[CONFIG["variables"]].describe().T[["mean", "std"]]
print(despues.round(4).to_string())
print(f"\\nFilas: {len(matriz):,}")
""",
    ),
    code(
        """
# Autochequeo
assert np.allclose(matriz[CONFIG["variables"]].mean(), 0, atol=1e-9), "las medias deben ser 0"
assert np.allclose(matriz[CONFIG["variables"]].std(ddof=0), 1, atol=1e-9), "las desviaciones, 1"
assert CONFIG["etiqueta_de_contraste"] in matriz.columns, (
    "la etiqueta debe viajar en la tabla, aunque no se use para agrupar"
)
print("✅ Todas las variables en la misma escala. Ahora una diferencia de 1 significa")
print("   lo mismo en cualquiera de ellas: una desviación típica.")
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 1
>
> **Cifras medidas:** la razón entre la mayor y la menor desviación típica ronda los **tres
> órdenes de magnitud** (`num_lidar_points` frente a `box_height`).
>
> **Ejercicio de 2 minutos que vale la pena hacer en vivo:** pregunta qué par de detecciones
> está "más cerca" — dos objetos idénticos que difieren en 300 puntos láser, o un peatón y un
> bus con el mismo número de puntos. Sin escalar, K-medias considera al peatón y al bus
> **vecinos**. Se entiende de inmediato.
>
> **La sutileza que importa:** la etiqueta viaja en la tabla pero **no está en
> `CONFIG["variables"]`**, así que no entra en el agrupamiento. Que un dato esté disponible no
> significa que se use. Conviene señalarlo porque es exactamente el mismo cuidado que en la Actividad 2.2
> con `num_lidar_points`.
>
> **Si alguien pregunta por `MinMaxScaler`:** también sirve, y es preferible cuando los datos
> tienen límites naturales conocidos. `StandardScaler` es más robusto ante atípicos —que aquí
> hay, y legítimos: los buses—. Buena pregunta, merece reconocerse.
>
> **Criterio de logro:** cuantifica la diferencia de escalas y explica por qué afecta a un
> algoritmo basado en distancias.
"""
    ),
    # ======================================================================
    # BLOQUE 2 — CUÁNTOS GRUPOS
    # ======================================================================
    md(
        """
---
# Bloque 2 · ⭐⭐ ¿Cuántos grupos?

K-medias necesita que le digas `k` de antemano. Y no hay una respuesta correcta: hay dos
criterios que a veces se contradicen.

| Criterio | Qué mide | Su trampa |
|---|---|---|
| **Inercia** | Suma de distancias al centro de su grupo | **Siempre baja** al añadir grupos |
| **Silueta** | Qué tan separados están los grupos entre sí | Puede no tener máximo claro |

La inercia por sí sola no decide nada: si le hicieras caso, terminarías con un grupo por fila e
inercia cero.
"""
    ),
    md(
        """
### ✏️ TODO 3 — Probar varios `k`
"""
    ),
    code(
        """
busqueda = grupos.elegir_k(matriz, CONFIG)
print(busqueda.to_string(index=False))

fig, ejes = plt.subplots(1, 2, figsize=(10, 3.5))
ejes[0].plot(busqueda["k"], busqueda["inercia"], marker="o")
ejes[0].set_title("Inercia (siempre baja)")
ejes[0].set_xlabel("k")
ejes[1].plot(busqueda["k"], busqueda["silueta"], marker="o", color="darkorange")
ejes[1].set_title("Silueta (más alto es mejor)")
ejes[1].set_xlabel("k")
plt.tight_layout()
plt.show()
""",
        """
# TODO 3: prueba varios números de grupos y mide los dos criterios.
busqueda = grupos.____(matriz, CONFIG)
print(busqueda.to_string(index=False))

fig, ejes = plt.subplots(1, 2, figsize=(10, 3.5))
ejes[0].plot(busqueda["k"], busqueda["____"], marker="o")
ejes[0].set_title("Inercia (siempre baja)")
ejes[0].set_xlabel("k")
ejes[1].plot(busqueda["k"], busqueda["____"], marker="o", color="darkorange")
ejes[1].set_title("Silueta (más alto es mejor)")
ejes[1].set_xlabel("k")
plt.tight_layout()
plt.show()
""",
    ),
    code(
        """
# Autochequeo
assert busqueda["inercia"].is_monotonic_decreasing, (
    "la inercia SIEMPRE baja al añadir grupos: por eso sola no sirve"
)
mejor_k = int(busqueda.loc[busqueda["silueta"].idxmax(), "k"])
print(f"✅ La silueta tiene su máximo en k = {mejor_k}.")
print(f"   Pero el pipeline usa k = {CONFIG['k']}. Esa discrepancia es el ejercicio.")
"""
    ),
    md(
        """
### ✏️ TODO 4 — La discrepancia

La silueta prefiere un número de grupos y el pipeline usa otro. **No es un error:** es la
situación normal.

**✍️ Tu respuesta:**

*(doble clic aquí y escribe)*

1. ¿Qué `k` elegirías tú **solo con estos dos gráficos**?
2. ¿Qué información te haría cambiar de opinión?
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 2 ⭐⭐
>
> **Cifras medidas:**
>
> | k | Inercia | Silueta |
> |---|---|---|
> | 2 | 109.384 | 0,4498 |
> | **3** | 84.047 | **0,4730** ← máximo |
> | 4 | 67.035 | 0,4630 |
> | 5 | 57.121 | 0,3550 |
> | 6 | 49.716 | 0,3413 |
> | 8 | 39.057 | 0,3300 |
>
> **La inercia baja monótonamente**, como siempre. El "codo" está entre 3 y 4 y es discutible:
> conviene mostrar que dos personas razonables lo ven en sitios distintos. Ese es justamente el
> problema del método del codo, que casi todos los tutoriales presentan como si fuera objetivo.
>
> **La silueta prefiere k = 3. Nosotros usamos k = 4 a propósito.** Deja la discrepancia abierta
> aquí: se resuelve sola en el bloque 3, cuando vean qué contiene el cuarto grupo. Si adelantas
> la respuesta, pierdes el efecto.
>
> **Respuestas esperadas al TODO 4:**
>
> - "k = 3, porque es donde la silueta es máxima" — correcta y suficiente. Nivel *Logrado*.
> - "Necesito ver qué hay dentro de cada grupo" — **exactamente**. Nivel *Destacado*: entendió
>   que la métrica no puede decidir sola.
> - "k = 8, porque la inercia es la más baja" — error de concepto. Es el momento de repetir que
>   la inercia siempre baja; con k igual al número de filas vale cero y el resultado es inútil.
>
> **La frase del bloque:**
>
> > *La silueta propone; el conocimiento del dominio dispone.*
>
> Es literalmente la misma estructura que en la EA1 con los atípicos: el criterio estadístico
> propone, el dominio decide. Vale la pena decirlo señalando esa continuidad.
>
> **Criterio de logro:** interpreta correctamente los dos criterios, reconoce que la inercia
> sola no decide, y **no cierra la decisión sin mirar el contenido de los grupos**.
"""
    ),
    # ======================================================================
    # BLOQUE 3 — AGRUPAR Y PERFILAR
    # ======================================================================
    md(
        """
---
# Bloque 3 · ⭐⭐ Agrupar, y después el trabajo de verdad

Ejecutar K-medias es una línea. Lo que sigue es la actividad.
"""
    ),
    md(
        """
### ✏️ TODO 5 — Agrupar
"""
    ),
    code(
        """
agrupada = grupos.agrupar(matriz, CONFIG)

print(f"k = {CONFIG['k']}")
print(agrupada["grupo"].value_counts().sort_index().to_string())
""",
        """
# TODO 5: agrupa con el k configurado.
agrupada = grupos.____(matriz, CONFIG)

print(f"k = {CONFIG['k']}")
print(agrupada["grupo"].value_counts().sort_index().to_string())
""",
    ),
    md(
        """
### ✏️ TODO 6 — Perfilar: ponerle nombre a cada grupo

Aquí está el trabajo. La tabla siguiente da la media de cada variable **en unidades de
desviación típica**: un `+2` significa "dos desviaciones por encima de la media general".
"""
    ),
    code(
        """
perfil = grupos.perfilar_grupos(agrupada, CONFIG)
perfil
""",
        """
# TODO 6: describe cada grupo por la media de sus variables.
perfil = grupos.____(agrupada, CONFIG)
perfil
""",
    ),
    md(
        """
### ✏️ TODO 7 — Los nombres

Mira el perfil y bautiza cada grupo. Un nombre que un colega entienda sin ver la tabla.

*Pista: fíjate especialmente en el grupo más pequeño. Sus valores no se parecen a nada.*
"""
    ),
    code(
        """
nombres = {
    0: "pequeños, lentos y algo altos",
    1: "medianos y en movimiento",
    2: "muy cerca del sensor (muchísimos puntos láser)",
    3: "enormes: largos, anchos y altos",
}
for numero, nombre in nombres.items():
    fila = perfil[perfil["grupo"] == numero].iloc[0]
    print(f"Grupo {numero}: {nombre:52s} {fila['n']:6,.0f} filas ({fila['pct']:5.2f} %)")
""",
        """
# TODO 7: ponle nombre a cada grupo, mirando su perfil.
nombres = {
    0: "____",
    1: "____",
    2: "____",
    3: "____",
}
for numero, nombre in nombres.items():
    fila = perfil[perfil["grupo"] == numero].iloc[0]
    print(f"Grupo {numero}: {nombre:52s} {fila['n']:6,.0f} filas ({fila['pct']:5.2f} %)")
""",
    ),
    code(
        """
# Autochequeo
assert all(n and not n.startswith("____") for n in nombres.values()), "faltan nombres"
mas_pequeno = perfil.loc[perfil["n"].idxmin()]
print(f"✅ El grupo más pequeño es el {int(mas_pequeno['grupo'])}: "
      f"{mas_pequeno['n']:,.0f} filas ({mas_pequeno['pct']:.2f} %)")
print(f"   box_length está a {mas_pequeno['box_length']:+.2f} desviaciones típicas.")
print("   ¿Qué objeto del tránsito es enorme, poco frecuente, y ya apareció en la EA1?")
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 3 ⭐⭐
>
> **Cifras medidas** (medias en desviaciones típicas):
>
> | Grupo | `box_length` | `box_width` | `box_height` | `speed_mps` | `num_lidar_points` | n | % |
> |---|---|---|---|---|---|---|---|
> | 0 | −1,03 | −1,19 | +0,30 | −0,93 | −0,16 | 13.787 | 36,4 % |
> | 1 | +0,53 | +0,71 | −0,31 | +0,59 | −0,31 | 19.927 | 52,6 % |
> | 2 | +0,21 | +0,32 | −0,20 | +0,19 | **+2,30** | 3.624 | 9,6 % |
> | **3** | **+4,90** | **+1,91** | **+4,68** | +0,57 | +0,05 | **573** | **1,51 %** |
>
> **El grupo 3 son los BUSES.** `box_length` a casi **cinco desviaciones típicas** y
> `box_height` otro tanto, en el 1,5 % de las filas.
>
> **Este es el momento de la sesión.** Son exactamente los mismos atípicos legítimos que en la
> EA1 aprendieron a **no eliminar** —los buses de 12 a 18 m—. Ahora aparecen solos, sin que
> nadie se lo pidiera, como un grupo con identidad propia.
>
> Merece decirse en voz alta:
>
> > *Si en la EA1 hubieran hecho caso al criterio IQR y eliminado los atípicos, este grupo no
> > existiría. Habrían borrado una categoría entera de objeto del dataset, y hoy el algoritmo no
> > tendría nada que encontrar.*
>
> **Y resuelve la discrepancia del bloque 2.** La silueta prefería k = 3 porque los buses son
> pocos y están pegados al resto de vehículos en el espacio euclídeo. Pero **para el negocio son
> una categoría distinta**: un bus se comporta distinto, ocupa distinto, frena distinto. Elegir
> k = 4 es una decisión de dominio que la métrica no respalda, y está bien tomada.
>
> **El grupo 2 merece su propia observación.** Se define casi solo por `num_lidar_points` a
> +2,3: son los objetos **cercanos al sensor**, de cualquier tipo. El algoritmo agrupó por una
> propiedad de la **medición**, no del objeto. Es un hallazgo real y una advertencia: no todo lo
> que un agrupamiento encuentra habla de la entidad; a veces habla del instrumento.
>
> **Criterio de logro:** nombra los cuatro grupos con lenguaje de dominio, identifica el grupo
> pequeño como objetos de gran tamaño y lo conecta con los atípicos legítimos de la EA1.
"""
    ),
    # ======================================================================
    # BLOQUE 4 — CONTRASTAR
    # ======================================================================
    md(
        """
---
# Bloque 4 · Contrastar con la etiqueta: **no es una evaluación**

Tenemos `object_type` guardado y sin usar. Ahora lo sacamos.

**Cuidado con lo que esto es y lo que no es.** El algoritmo nunca vio la etiqueta, así que no
puede acertar ni fallar. Esto es una **comprobación de sentido**: ¿la estructura que apareció
sola tiene alguna lectura conocida?

- Si un grupo concentra un tipo de objeto, encontraste algo con significado de dominio.
- Si todos los grupos tienen la misma mezcla, no encontraste nada útil, **por buena que sea la
  silueta**.
"""
    ),
    md(
        """
### ✏️ TODO 8 — El cruce
"""
    ),
    code(
        """
cruce = grupos.contrastar_con_etiqueta(agrupada, CONFIG)
print(cruce.to_string(index=False), "\\n")

sns.heatmap(cruce.set_index("grupo"), annot=True, fmt=".1f", cmap="Blues", cbar=False)
plt.title("% de cada tipo de objeto dentro de cada grupo")
plt.tight_layout()
plt.show()
""",
        """
# TODO 8: cruza los grupos descubiertos con la etiqueta conocida.
cruce = grupos.____(agrupada, CONFIG)
print(cruce.to_string(index=False), "\\n")

sns.heatmap(cruce.set_index("grupo"), annot=True, fmt=".1f", cmap="Blues", cbar=False)
plt.title("% de cada tipo de objeto dentro de cada grupo")
plt.tight_layout()
plt.show()
""",
    ),
    md(
        """
### ✏️ TODO 9 — Leerlo bien

**✍️ Tu respuesta:**

*(doble clic aquí y escribe)*

1. ¿Los grupos coinciden con los tipos de objeto? Responde con cifras.
2. Hay dos grupos que son casi 100 % `vehicle`. ¿Es un error del algoritmo? ¿Qué los distingue?
3. Un grupo mezcla peatones y señalética. ¿Por qué el algoritmo los junta, si para el negocio no
   tienen nada que ver?
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 4
>
> **Cifras medidas** (% de cada tipo dentro de cada grupo):
>
> | Grupo | `cyclist` | `pedestrian` | `sign` | `vehicle` |
> |---|---|---|---|---|
> | 0 | 4,87 | **73,10** | 21,85 | 0,18 |
> | 1 | 0,00 | 0,00 | 0,00 | **100,00** |
> | 2 | 1,71 | 16,83 | 2,07 | **79,39** |
> | 3 | 0,00 | 0,00 | 0,00 | **100,00** |
>
> **Respuesta 1:** parcialmente. El grupo 1 y el 3 son vehículos puros; el 0 es mayoritariamente
> peatones **pero con un 22 % de señalética**; el 2 está mezclado.
>
> **Respuesta 2 — la importante.** No es un error: los grupos 1 y 3 son vehículos **de tamaños
> muy distintos**. El 3 son los buses. El algoritmo encontró una subestructura real *dentro* de
> una categoría, que es algo que la etiqueta no capturaba.
>
> Vale la pena subrayarlo: **el agrupamiento encontró información que la etiqueta no tenía.** Ese
> es el argumento a favor del aprendizaje no supervisado, y aquí está demostrado en vez de
> afirmado.
>
> **Respuesta 3.** Peatones y señalética son **pequeños y están quietos**. En el espacio de
> variables que le dimos —geometría, velocidad, puntos— son casi indistinguibles. El algoritmo no
> se equivoca: **le dimos variables que no separan esas dos cosas.**
>
> La conclusión de método, que es la que hay que dejar:
>
> > *Un agrupamiento solo puede encontrar la estructura que las variables elegidas permiten ver.
> > Si dos categorías se confunden, la primera sospecha no es el algoritmo: son las variables.*
>
> **Pregunta para el curso:** ¿qué variable habría que agregar para separar peatones de
> señalética? Respuestas buenas: la variación de posición entre fotogramas (un peatón se mueve
> lentamente, una señal no se mueve nunca), o la altura del centro sobre el suelo. **Eso es
> ingeniería de características**, y es la respuesta correcta al problema.
>
> **Criterio de logro:** interpreta el cruce sin llamarlo "acierto", explica los dos grupos de
> vehículos y atribuye la confusión peatón/señalética a las variables, no al algoritmo.
"""
    ),
    # ======================================================================
    # BLOQUE 5 — PCA
    # ======================================================================
    md(
        """
---
# Bloque 5 · Reducción de dimensionalidad

Cinco variables no se pueden dibujar. **PCA** busca las direcciones de máxima varianza y
proyecta los datos sobre las primeras, conservando toda la información que pueda.

La cifra que importa no es cuánto explica la primera componente, sino **cuántas necesitas para
conservar el 90 %**. Si con pocas basta, tus variables eran redundantes entre sí — y eso también
es un hallazgo.
"""
    ),
    md(
        """
### ✏️ TODO 10 — ¿Cuánta redundancia hay?
"""
    ),
    code(
        """
varianza = grupos.resumir_pca(matriz, CONFIG)
print(varianza.to_string(index=False))

n90 = int((varianza["varianza_acumulada"] < 0.90).sum() + 1)
print(f"\\nCon {n90} de {len(varianza)} componentes se conserva el 90 % de la varianza.")
""",
        """
# TODO 10: ¿cuánta información conserva cada componente?
varianza = grupos.____(matriz, CONFIG)
print(varianza.to_string(index=False))

n90 = int((varianza["varianza_acumulada"] < 0.90).sum() + 1)
print(f"\\nCon {n90} de {len(varianza)} componentes se conserva el 90 % de la varianza.")
""",
    ),
    md(
        """
### ✏️ TODO 11 — Dibujar, sabiendo lo que se pierde
"""
    ),
    code(
        """
proyeccion = grupos.proyectar_2d(matriz, CONFIG)
proyeccion["grupo"] = agrupada["grupo"].to_numpy()

explicada = varianza.loc[1, "varianza_acumulada"]

fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
muestra = proyeccion.sample(4000, random_state=42)
sns.scatterplot(data=muestra, x="componente_1", y="componente_2", hue="grupo",
                palette="tab10", s=8, ax=ejes[0], legend="full")
ejes[0].set_title("Coloreado por GRUPO descubierto")
sns.scatterplot(data=muestra, x="componente_1", y="componente_2",
                hue=CONFIG["etiqueta_de_contraste"], palette="Set2", s=8, ax=ejes[1])
ejes[1].set_title("Coloreado por tipo de objeto REAL")
plt.suptitle(f"Proyección 2D — conserva el {100*explicada:.1f} % de la varianza")
plt.tight_layout()
plt.show()
""",
        """
# TODO 11: proyecta a dos dimensiones y dibuja.
proyeccion = grupos.____(matriz, CONFIG)
proyeccion["grupo"] = agrupada["grupo"].to_numpy()

explicada = varianza.loc[1, "varianza_acumulada"]

fig, ejes = plt.subplots(1, 2, figsize=(11, 4))
muestra = proyeccion.sample(4000, random_state=42)
sns.scatterplot(data=muestra, x="componente_1", y="componente_2", hue="____",
                palette="tab10", s=8, ax=ejes[0], legend="full")
ejes[0].set_title("Coloreado por GRUPO descubierto")
sns.scatterplot(data=muestra, x="componente_1", y="componente_2",
                hue=CONFIG["etiqueta_de_contraste"], palette="Set2", s=8, ax=ejes[1])
ejes[1].set_title("Coloreado por tipo de objeto REAL")
plt.suptitle(f"Proyección 2D — conserva el {100*explicada:.1f} % de la varianza")
plt.tight_layout()
plt.show()
""",
    ),
    code(
        """
# Autochequeo
assert abs(varianza["varianza_acumulada"].iloc[-1] - 1.0) < 1e-6, (
    "todas las componentes juntas deben explicar el 100 %"
)
print(f"✅ La proyección 2D conserva el {100*explicada:.1f} % de la varianza.")
print(f"   Es decir: el {100*(1-explicada):.1f} % de la información NO está en ese dibujo.")
print("   Dos puntos que se ven pegados pueden estar lejos en el espacio original.")
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Bloque 5
>
> **Cifras medidas:**
>
> | Componente | Varianza explicada | Acumulada |
> |---|---|---|
> | 1 | 0,4884 | 0,4884 |
> | 2 | 0,2136 | **0,7020** |
> | 3 | 0,1999 | 0,9019 |
> | 4 | 0,0767 | 0,9785 |
> | 5 | 0,0215 | 1,0000 |
>
> **Con 3 de 5 componentes se conserva el 90 %.** Hay redundancia: `box_length`, `box_width` y
> `box_height` miden aspectos del mismo tamaño y se mueven juntas.
>
> **La advertencia que hay que dar con el gráfico**, y darla siempre: la proyección conserva el
> **70,2 %** de la varianza, así que **casi el 30 % de la información no está en ese dibujo**.
> Dos puntos que se ven superpuestos pueden estar lejos en el espacio original.
>
> > *Un gráfico de PCA sin su porcentaje de varianza explicada es un gráfico bonito y engañoso.*
>
> Es la regla práctica que deben llevarse: siempre reportar ese número junto al dibujo.
>
> **Los dos paneles lado a lado son la mejor síntesis de la sesión.** A la izquierda, grupos
> nítidos y bien separados: K-medias hizo su trabajo. A la derecha, los tipos reales, que **no
> coinciden con esa partición**. Las dos cosas son ciertas a la vez, y entenderlo es entender el
> aprendizaje no supervisado.
>
> **Si sobra tiempo:** mencionar que existe t-SNE / UMAP, que separan visualmente mucho mejor,
> **y por eso mismo engañan más**: no preservan distancias globales, así que la separación que
> muestran no se puede interpretar como separación real. PCA es honesto y aburrido; conviene
> saber por qué eso es una virtud.
>
> **Criterio de logro:** reporta cuántas componentes hacen falta para el 90 %, y **acompaña el
> gráfico de su varianza explicada** al interpretarlo.
"""
    ),
    # ======================================================================
    # CIERRE
    # ======================================================================
    md(
        """
---
# Cierre · Informe de segmentación

Esta es la entrega de la Actividad 2.3. Máximo una página.

---

### El problema

**Qué buscábamos sin etiquetas:** `____`
**Variables usadas para agrupar:** `____`
**Por qué escalé antes de agrupar:** `____`

### Cuántos grupos, y por qué

| k | Inercia | Silueta |
|---|---|---|
| | | |

**k elegido:** `____`
**Criterio estadístico que lo respalda (o no):** `____`
**Argumento de dominio:** `____`

> Si tu k no es el de la silueta máxima, **eso está bien** y hay que defenderlo. Si sí lo es,
> también — pero di por qué el contenido de los grupos lo confirma.

### Los grupos, con nombre

| Grupo | Nombre | % de las filas | Qué lo caracteriza |
|---|---|---|---|
| 0 | `____` | | |
| 1 | `____` | | |
| 2 | `____` | | |
| 3 | `____` | | |

### Contraste con la etiqueta conocida

**¿Los grupos coinciden con los tipos de objeto?** `____`
**Un grupo que NO corresponde a un tipo, y qué significa:** `____`
**Dos tipos que el algoritmo confunde, y por qué:** `____`
**Qué variable habría que agregar para separarlos:** `____`

### Reducción de dimensionalidad

**Componentes para conservar el 90 %:** `____` de `____`
**Qué dice eso sobre mis variables:** `____`
**Varianza que conserva mi gráfico 2D:** `____` %

### La conclusión honesta

**¿Esta segmentación le sirve a alguien?** `____`

*(Responder "no del todo" está permitido y a veces es lo correcto. Lo que se evalúa es el
argumento: qué decisión podría tomar alguien con estos grupos que no pudiera tomar sin ellos.)*

**Qué haría distinto con más tiempo:** `____`
"""
    ),
    md_docente(
        """
> ### 🎓 Criterios de logro — Actividad 2.3 (IL 2.3)
>
> | Nivel | Descripción |
> |---|---|
> | **Destacado (4)** | Todo lo del nivel 3, y además: identifica el grupo de los buses y lo conecta con los atípicos legítimos de la EA1; atribuye la confusión peatón/señalética a las variables y **propone una variable nueva** que los separaría; acompaña el gráfico PCA de su varianza explicada sin que se lo pidan |
> | **Logrado (3)** | Los 11 TODO en verde; escala y explica por qué; justifica el k con dos criterios; nombra los grupos con lenguaje de dominio; interpreta el cruce **sin llamarlo acierto** |
> | **En desarrollo (2)** | Ejecuta el agrupamiento correctamente pero deja los grupos sin nombre, o elige k solo por la inercia; trata el cruce con la etiqueta como una evaluación de exactitud |
> | **Inicial (1)** | Agrupa sin escalar, o no interpreta ningún grupo |
>
> **Qué mirar al corregir, en este orden:**
>
> 1. **Los nombres de los grupos.** Si dicen "grupo 0, grupo 1", no hay hallazgo. Es lo primero
>    que revela si entendieron para qué sirve esto.
> 2. **Cómo describen el cruce.** *"El algoritmo acertó un 73 %"* es un error de concepto, no de
>    cálculo: no había nada que acertar. Baja a *En desarrollo* aunque todo lo demás esté bien.
> 3. **La conclusión final.** *"Los grupos están bien definidos"* no responde la pregunta. La
>    pregunta es qué decisión habilitan.
>
> **Errores frecuentes que conviene anticipar:**
>
> - Agrupar sin escalar. El resultado se ve razonable y está dominado por una sola variable.
> - Elegir k por la inercia mínima. Siempre gana el k más grande que se pruebe.
> - Interpretar el gráfico PCA como si fuera el espacio real, sin su varianza explicada.
>
> **Enlace con el resto de la asignatura:** las actividades 2.2 y 2.3 parten del **mismo**
> `detecciones_limpias.parquet` y hacen preguntas opuestas. Ambas pertenecen al **RA2** y se
> evalúan juntas en la **Parcial 2**. Vale la pena cerrar el ciclo señalando que el trabajo del
> RA1 —el que parecía el menos glamoroso— es el que hizo posibles los dos.
"""
    ),
]
