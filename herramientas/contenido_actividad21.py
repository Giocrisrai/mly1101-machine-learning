"""Fuente única del contenido de la Actividad 2.1 — Gestión de proyectos con CRISP-DM.

Indicador de logro **IL 2.1**: *implementa metodologías de trabajo (como CRISP-DM) para
estructurar el desarrollo del modelo.*

De este archivo salen dos notebooks:

- ``notebooks/12_alumno_crispdm.ipynb``   (versión con TODO)
- ``notebooks/12_docente_crispdm.ipynb``  (versión resuelta con pauta)

**6 horas pedagógicas** según el programa.

La idea de la sesión: el RA1 ya recorrió comprensión y preparación de datos **sin
nombrarlas**. Hoy se nombra el mapa, se cierra la fase que se saltaron (negocio) y se
planifica el resto del RA2 y del RA3. CRISP-DM no es una diapositiva: es la estructura
que el EFT va a pedir por escrito.

Las cifras de la pauta salen de ``detecciones_waymo_like.csv`` (semilla 42) y de
``src/crispdm.py``, cubierto por ``tests/test_crispdm.py``.

Regenerar tras editar:

    python herramientas/construir_notebooks.py
"""

from __future__ import annotations

from contenido_semana01 import URL_REPO, code, md, md_docente

CELDAS_ACT21: list[dict] = [
    md(
        """
# MLY1101 · Machine Learning — Actividad 2.1
## Gestión de proyectos con CRISP-DM

**Resultado de aprendizaje (RA2):** aplica modelos estadísticos al conjunto de datos
procesados para interpretarlos, **utilizando metodologías ágiles**, con la finalidad de
obtener conocimientos relevantes que permitan responder a las necesidades del contexto
de negocio, considerando aspectos éticos.

**Indicador de logro (IL 2.1):** implementa metodologías de trabajo (como **CRISP-DM**)
para estructurar el desarrollo del modelo.

---

### La idea central de hoy

El RA1 no empezó eligiendo un algoritmo. Empezó por los datos. Eso ya era CRISP-DM, solo
que nadie lo había nombrado.

Hoy hacemos tres cosas, y ninguna es memorizar siglas:

1. **Ponerle nombre** a lo que ya hicieron (comprensión y preparación de datos).
2. **Cerrar el hueco** que se saltaron: la comprensión del negocio. El problema de las
   detecciones LiDAR se les *entregó*; no lo formularon.
3. **Planificar** el resto del semestre sobre el mismo mapa, para que el informe del EFT
   no se escriba de memoria al final.

```
Comprensión del negocio → Datos → Preparación → Modelado → Evaluación → Despliegue
         └─ hueco de hoy ─┘      └── RA1 ──┘     └── RA2 / RA3 ──┘     └── EFT ──┘
```

> CRISP-DM no es una cascada. Evaluar puede devolverte al negocio ("esta métrica no
> responde la pregunta") o a los datos ("esta partición miente").

---

### Al final de la sesión debes entregar

Una **carta de proyecto CRISP-DM** del hilo Waymo (se valida en código) y la misma carta
rellenada para el **caso oficial** de tu equipo (Telco, House Prices o Spotify), que es
la que viaja a la Parcial 2 y al EFT.
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Actividad 2.1
>
> **La actividad son 6 horas pedagógicas.** El bloque guiado cubre ~4 h; las 2 h restantes
> son para que el equipo deje cerrada la carta del caso oficial. Sin esa carta, la
> Parcial 2 empieza eligiendo un algoritmo.
>
> | Bloque | Min | Foco |
> |---|---|---|
> | 0 · Encuadre | 15 | Empezaron en el medio, y eso tiene nombre |
> | 1 · Las seis fases ⭐ | 40 | Orden, ciclo, no cascada |
> | 2 · Retroceso sobre el RA1 ⭐⭐ | 45 | Nombrar lo que ya hicieron |
> | 3 · El negocio que no formularon ⭐⭐ | 50 | Pregunta + criterio medible |
> | 4 · Carta del proyecto Waymo | 40 | `validar_carta` en verde |
> | 5 · El mapa del curso | 25 | Supervisado y no supervisado son RA2 |
> | 6 · Caso oficial | 30 | Telco / Housing / Spotify |
> | Cierre | 15 | Qué pide el EFT y qué es "despliegue" aquí |
>
> **Los imprescindibles son el 2 y el 3.** Si falta tiempo, recorta el 5 (el mapa está
> en `crispdm.mapa_del_curso()` y se puede dejar de tarea).
>
> **Preguntar antes de mostrar.** El TODO 3 (¿esta pregunta empieza por el algoritmo?)
> y el TODO 5 (criterio sin cifra) se caen solos si dejas que contesten primero.
>
> **Regla de oro:** ninguna afirmación sin una cifra. Un "criterio de éxito" sin número
> es un deseo, y en esta pauta vale **Inicial**.
"""
    ),
    md(
        """
---
## Preparación del entorno
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
RUTA_DATOS = RAIZ / "datos" / "crudos" / "detecciones_waymo_like.csv"

print("Colab:", EN_COLAB, "| dataset:", RUTA_DATOS.exists())
"""
    ),
    code(
        """
import pandas as pd

import crispdm
import eda

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 140)

df = pd.read_csv(RUTA_DATOS)
print(f"{df.shape[0]:,} detecciones en {df['segment_id'].nunique()} segmentos")
print("fases CRISP-DM:", len(crispdm.FASES))
"""
    ),
    # ======================================================================
    # BLOQUE 1
    # ======================================================================
    md(
        """
---
# Bloque 1 · Las seis fases (y por qué no son una cascada)

CRISP-DM (Cross-Industry Standard Process for Data Mining) describe **seis fases**. El
orden importa, pero no es un edificio: es un ciclo. La evaluación puede devolver al
negocio o a los datos.

### ✏️ TODO 1 — El orden del ciclo

Asigna a `orden` las seis claves de `crispdm.FASES` **en el orden del ciclo**. No las
copies de memoria si puedes leer el módulo: parte del trabajo es usar la herramienta,
no recitarla.
"""
    ),
    code(
        """
orden = list(crispdm.FASES)
print(" → ".join(crispdm.NOMBRES[f] for f in orden))
assert orden == list(crispdm.FASES), "el orden canónico vive en crispdm.FASES"
""",
        """
orden = []  # TODO: las seis claves, en orden
print(" → ".join(crispdm.NOMBRES[f] for f in orden))
assert orden == list(crispdm.FASES), "el orden canónico vive en crispdm.FASES"
""",
    ),
    md(
        """
### ✏️ TODO 2 — ¿Cascada o ciclo?

`crispdm.RETORNOS` declara desde qué fase se puede volver, y a dónde. Imprímelo y
responde: si en la Actividad 2.2 la exactitud sale alta y el F1 de la clase minoritaria
sale bajo, **¿a qué fase vuelves y por qué?**
"""
    ),
    code(
        """
for origen, destinos in crispdm.RETORNOS.items():
    print(f"{crispdm.NOMBRES[origen]:<28} → {', '.join(crispdm.NOMBRES[d] for d in destinos)}")
"""
    ),
    md(
        """
**Si la exactitud es alta y el F1 de la minoría es bajo, vuelvo a:** `____`

**Por qué esa fase y no otra:** `____`
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — TODO 1 y 2
>
> **TODO 1.** Las seis, en este orden: comprensión del negocio → de los datos →
> preparación → modelado → evaluación → despliegue. Quien las recita en otro idioma
> (Business Understanding, etc.) está bien; las claves del módulo están en español.
>
> **TODO 2.** Se vuelve a **comprensión del negocio**. La exactitud respondió una
> pregunta que nadie hizo ("¿acierto filas?"). El F1 de la minoría revela que el
> criterio de éxito era otro: no perder las detecciones difíciles. Eso no se arregla
> cambiando el modelo primero; se arregla reescribiendo qué cuenta como éxito.
>
> También es aceptable volver a **comprensión de los datos** si argumentan el
> desbalance (~2 % de `CYCLIST`) o el patrón MNAR de `speed_mps`. Lo que no se
> acepta es "probar otro algoritmo" como primer movimiento: eso es quedarse en
> modelado.
>
> `RETORNOS['evaluacion']` contiene esas dos fases a propósito. El test lo fija.
"""
    ),
    # ======================================================================
    # BLOQUE 2
    # ======================================================================
    md(
        """
---
# Bloque 2 · ⭐⭐ Retroceso: el RA1 ya era CRISP-DM

Cuatro actividades, 23 horas, y nadie dijo "fase 2". Eso no fue un olvido: fue el
método. Hoy hay que **mapear** lo que ya está hecho para no volver a hacerlo y para
ver el hueco.

Estos hallazgos ya los midieron. Cada uno pertenece a **una** fase principal.

| Hallazgo | Fase (clave de `crispdm.FASES`) |
|---|---|
| El CSV tiene 10 defectos intencionales y 40.680 filas | `____` |
| `CYCLIST` es ~2 % de las filas | `____` |
| La tabla de decisiones de limpieza (qué se imputa, qué se tira) | `____` |
| El censo de Waymo: 793 de 798 segmentos son `sunny` | `____` |
| Parquet conserva tipos; CSV los pierde | `____` |
| Combinaciones de columnas que reidentifican | `____` |

### ✏️ TODO 3 — Completa el mapa en código
"""
    ),
    code(
        """
hallazgos = {
    "diez_defectos": "comprension_de_los_datos",
    "desbalance_cyclist": "comprension_de_los_datos",
    "tabla_de_decisiones": "preparacion_de_los_datos",
    "censo_sunny": "comprension_del_negocio",
    "parquet_vs_csv": "preparacion_de_los_datos",
    "reidentificacion": "comprension_del_negocio",
}

fases_validas = set(crispdm.FASES)
assert all(f in fases_validas for f in hallazgos.values())
print(pd.Series(hallazgos).rename("fase").to_string())
""",
        """
hallazgos = {
    "diez_defectos": "",            # TODO
    "desbalance_cyclist": "",       # TODO
    "tabla_de_decisiones": "",      # TODO
    "censo_sunny": "",              # TODO
    "parquet_vs_csv": "",           # TODO
    "reidentificacion": "",         # TODO
}

fases_validas = set(crispdm.FASES)
assert all(f in fases_validas for f in hallazgos.values()), "usa las claves de crispdm.FASES"
print(pd.Series(hallazgos).rename("fase").to_string())
""",
    ),
    md(
        """
**La fase que casi no aparece en esa tabla es:** `____`

**Eso significa, en una frase, que el RA1:** `____`
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — TODO 3
>
> | Hallazgo | Fase | Por qué |
> |---|---|---|
> | 10 defectos, 40.680 filas | `comprension_de_los_datos` | Diagnóstico. Actividad 1.3 |
> | `CYCLIST` ~2 % | `comprension_de_los_datos` | Desbalance medido; aún no es un KPI |
> | Tabla de decisiones | `preparacion_de_los_datos` | Imputar/tirar/conservar. 1.3 y pipeline |
> | 793/798 `sunny` | `comprension_del_negocio` | El sistema no va a operar solo con sol. 1.4 |
> | Parquet vs CSV | `preparacion_de_los_datos` | Actividad 1.2: el formato es una decisión |
> | Reidentificación | `comprension_del_negocio` | Privacidad es restricción de negocio, no de modelo. 1.4 |
>
> Se acepta poner el censo o la reidentificación en comprensión de los datos **si**
> argumentan que primero se miden y después se traducen a negocio. Lo que no se
> acepta es meter la tabla de decisiones en modelado.
>
> **La fase que casi no aparece es `modelado`.** El RA1 no entrenó nada. Tampoco
> aparece `despliegue`. El hueco de *formulación* (pregunta de negocio + criterio de
> éxito **del equipo de percepción**, no del censo) es lo que cierra el bloque 3:
> ética y sesgo ya tocaron negocio, pero la pregunta operativa sigue prestada.
>
> Cifras para no citar de memoria: 40.680 filas, 153 segmentos, `CYCLIST` 1,94 %
> en el CSV sucio (antes de normalizar `object_type`).
"""
    ),
    # ======================================================================
    # BLOQUE 3
    # ======================================================================
    md(
        """
---
# Bloque 3 · ⭐⭐ El negocio que no formularon

En el RA1 el problema llegó escrito: *"trabajas en el equipo de percepción…"*. Eso
ahorra tiempo y **esconde el trabajo**. Un cliente no entrega la pregunta limpia. La
primera fase de CRISP-DM consiste en escribirla de modo que se pueda fallar.

Hay dos maneras de fallar esta fase, y las dos tienen detector en `crispdm`:

1. Preguntar **qué algoritmo** usar. Eso no es una pregunta de negocio.
2. Declarar un criterio de éxito **sin cifra**. Eso es un deseo.

### ✏️ TODO 4 — ¿Esta pregunta empieza por el algoritmo?

Clasifica cada pregunta con `crispdm.empieza_por_el_algoritmo`.
"""
    ),
    code(
        """
preguntas = {
    "A": "¿Qué modelo usamos, random forest o red neuronal?",
    "B": "¿En qué condiciones el sensor deja de ser confiable?",
    "C": "Vamos a probar XGBoost porque gana las competencias",
    "D": "¿Se puede anticipar qué detecciones van a ser difíciles?",
}

for clave, texto in preguntas.items():
    marca = "algoritmo" if crispdm.empieza_por_el_algoritmo(texto) else "negocio"
    print(f"{clave}  {marca:10}  {texto}")
""",
        """
preguntas = {
    "A": "¿Qué modelo usamos, random forest o red neuronal?",
    "B": "¿En qué condiciones el sensor deja de ser confiable?",
    "C": "Vamos a probar XGBoost porque gana las competencias",
    "D": "¿Se puede anticipar qué detecciones van a ser difíciles?",
}

for clave, texto in preguntas.items():
    marca = "algoritmo" if crispdm.empieza_por_el_algoritmo(texto) else "negocio"
    print(f"{clave}  {marca:10}  {texto}")
""",
    ),
    md(
        """
**Las que empiezan por el algoritmo son:** `____`
**La pregunta de negocio del equipo de percepción, en una línea, es:** `____`
"""
    ),
    md(
        """
### ✏️ TODO 5 — Un criterio sin cifra no es un criterio

`crispdm.es_criterio_medible` exige un dígito: un umbral, un porcentaje, una cantidad.
Sin eso, nadie puede decir si el proyecto cumplió.
"""
    ),
    code(
        """
criterios = [
    "el modelo tiene que ser bueno",
    "mejorar la percepción del vehículo",
    "F1-macro ≥ 0,70 en detecciones difíciles",
    "recall de LEVEL_2 de al menos 0,60",
    "RMSE menor que 20.000 en el conjunto de prueba",
]
for texto in criterios:
    ok = "medible" if crispdm.es_criterio_medible(texto) else "deseo"
    print(f"{ok:8}  {texto}")
"""
    ),
    md(
        """
**Tu criterio de éxito para el hilo Waymo (una frase, con cifra):** `____`

No vale la exactitud global. Ya sabes por el RA1 que un modelo que siempre dice
`VEHICLE` acierta ~62 % y es inútil.
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — TODO 4 y 5
>
> **TODO 4.** A y C empiezan por el algoritmo. B y D son de negocio. La pregunta que
> queremos dejar escrita:
>
> *"¿Se puede anticipar cuándo una detección LiDAR no es confiable, para que el
> vehículo no actúe como si lo fuera?"*
>
> D es exactamente la pregunta de la Actividad 2.2. Si un equipo llega solo a eso,
> está en **Logrado**. Si además nombra la consecuencia operativa (reducir velocidad,
> pedir confirmación, no cambiar de carril), **Destacado**.
>
> **TODO 5.** Los dos primeros son deseos. Los tres últimos son medibles. Un criterio
> aceptable para Waymo, medido después en 2.2:
>
> - *F1 de `LEVEL_2` ≥ 0,60 en un split por `segment_id`.*
> - O: *recall de detecciones difíciles nocturnas ≥ 0,50.*
>
> **Inicial** si proponen "maximizar accuracy" o "el mejor modelo posible".
> En 2.2 van a ver 89,65 % de exactitud contra 88,96 % de un dummy: siete décimas.
> El F1-macro pasa de 0,47 a 0,70. Sembrar eso ahora evita que lleguen a 2.2
> celebrando la exactitud.
>
> El 62 % del dummy `VEHICLE` sale de la Actividad 1.3 (desbalance en el CSV sucio,
> antes de unificar `object_type`). No lo recites si puedes hacerles calcularlo.
"""
    ),
    # ======================================================================
    # BLOQUE 4
    # ======================================================================
    md(
        """
---
# Bloque 4 · La carta del proyecto (hilo Waymo)

Una carta de proyecto no es un informe. Es una página que otro equipo podría usar
para continuar sin preguntarte. `crispdm.validar_carta` comprueba cinco campos y dos
trampas: criterio vago y pregunta-algoritmo.

Los campos son: `pregunta_de_negocio`, `criterio_de_exito`, `fuentes`, `riesgos`,
`proxima_fase`.

### ✏️ TODO 6 — Rellena y valida
"""
    ),
    code(
        """
carta_waymo = {
    "pregunta_de_negocio": (
        "¿Se puede anticipar cuándo una detección LiDAR no es confiable?"
    ),
    "criterio_de_exito": "F1 de LEVEL_2 ≥ 0,60 en un split por segment_id",
    "fuentes": "detecciones_waymo_like.csv (sintético, semilla 42, 40.680 filas, 153 segmentos)",
    "riesgos": (
        "CYCLIST ~1,94 %; nulos MNAR de speed_mps de noche; "
        "censo real 793/798 sunny; duplicados lógicos"
    ),
    "proxima_fase": "modelado",
}

problemas = crispdm.validar_carta(carta_waymo)
print("problemas:", problemas if problemas else "ninguno — carta usable")
assert problemas == [], problemas
""",
        """
carta_waymo = {
    "pregunta_de_negocio": "",   # TODO: una pregunta, no un algoritmo
    "criterio_de_exito": "",     # TODO: con cifra
    "fuentes": "",
    "riesgos": "",               # TODO: con cifras del RA1, no "puede haber sesgo"
    "proxima_fase": "",          # TODO: una clave de crispdm.FASES
}

problemas = crispdm.validar_carta(carta_waymo)
print("problemas:", problemas if problemas else "ninguno — carta usable")
assert problemas == [], problemas
""",
    ),
    md(
        """
**Riesgo que más puede hundir el criterio de éxito, y con qué cifra:** `____`
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — TODO 6
>
> La carta resuelta de arriba **pasa** `validar_carta`. Se acepta cualquier formulación
> equivalente con cifra. `proxima_fase` tiene que ser `modelado`: preparación ya está.
>
> **El riesgo que hunde el criterio** es el desbalance / la clase `LEVEL_2` (~11 %) o
> el `CYCLIST` al 1,94 %. Un F1 de minoría de 0,60 es ambicioso: en 2.2 el bosque
> llega a **0,46** en `LEVEL_2`. Si un equipo pone 0,90 de F1, déjales la carta en
> verde (el validador no conoce el techo) y anota en la corrección que el umbral
> no está anclado a evidencia. Eso se discute en 2.2, no se resuelve hoy.
>
> Destacado: quien escribe el riesgo MNAR ("borrar `speed_mps` faltante deja al
> modelo más ciego de noche") está conectando 1.3 con el criterio de éxito.
"""
    ),
    # ======================================================================
    # BLOQUE 5
    # ======================================================================
    md(
        """
---
# Bloque 5 · El mapa del curso (para no repetir el error)

El programa pone el **aprendizaje supervisado y el no supervisado en el RA2**. El RA3
es hiperparámetros, ensamble y validación cruzada. Es el error que más cuesta deshacer
si se planta ahora.

`crispdm.mapa_del_curso()` es la correspondencia oficial de este repositorio.

### ✏️ TODO 7 — ¿Dónde vive el no supervisado?
"""
    ),
    code(
        """
mapa = pd.DataFrame(crispdm.mapa_del_curso())
print(mapa.to_string(index=False))
print()
print("RA de la Act. 2.3:", mapa.set_index("actividad").loc["2.3", "ra"])
print("RA de la Act. 3.1:", mapa.set_index("actividad").loc["3.1", "ra"])
"""
    ),
    md(
        """
**El no supervisado (Act. 2.3) pertenece al:** `____`
**El RA3 no es "la unidad de clustering". El RA3 es:** `____`

### ✏️ TODO 8 — Próximas fases, en una frase cada una

| Actividad | Fase CRISP-DM | Qué pregunta responde (una línea) |
|---|---|---|
| 2.2 Supervisado | `____` | `____` |
| 2.3 No supervisado | `____` | `____` |
| 2.4 Interpretación | `____` | `____` |
| 3.1–3.3 Optimización | `____` | `____` |
| EFT | las seis | `____` |
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — TODO 7 y 8
>
> 2.3 → **RA2**, fase `modelado`. 3.1 → **RA3**, fase `evaluacion`.
>
> | Actividad | Fase | Pregunta |
> |---|---|---|
> | 2.2 | modelado | ¿Se puede anticipar una detección difícil? |
> | 2.3 | modelado | Sin etiqueta, ¿aparecen grupos naturales? |
> | 2.4 | evaluación | ¿Qué le digo al negocio con estas métricas? |
> | 3.1–3.3 | evaluación | ¿El ajuste o el ensamble cambian algo que importe? |
> | EFT | las seis | El mismo ciclo, sobre Telco / Housing / Spotify |
>
> Si alguien escribe "RA3 = no supervisado", es **Inicial** en D1 aunque el resto
> esté perfecto. Es el error que el test `test_el_mapa_del_curso_pone_supervisado_y_no_supervisado_en_el_ra2`
> existe para impedir.
"""
    ),
    # ======================================================================
    # BLOQUE 6
    # ======================================================================
    md(
        """
---
# Bloque 6 · La misma carta, sobre el caso oficial

Las actividades usan detecciones LiDAR. Las **evaluaciones** (Parcial 2, EFT) se rinden
sobre *Telco Customer Churn*, *House Prices* o *Spotify Tracks*. El método tiene que
trasladarse; memorizar el hilo Waymo no basta.

Esta carta **no** la valida el código: el validador no conoce tu caso. La valida la
pauta. Mismas trampas: pregunta-algoritmo y criterio sin cifra.

---

## Carta CRISP-DM — caso oficial

**Equipo:** `____` · **Caso:** Telco / House Prices / Spotify · **Fecha:** `____`

| Campo | Contenido |
|---|---|
| **Pregunta de negocio** | `____` |
| **Criterio de éxito (con cifra)** | `____` |
| **Fuentes** | `____` |
| **Riesgos (con cifra o con columna)** | `____` |
| **Qué ya está hecho del RA1** | `____` |
| **Próxima fase** | `____` |
| **Qué sería "despliegue" en este curso** | `____` |

> Despliegue, aquí, no es un API en producción. El EFT pide informe en Markdown,
> notebook ejecutable, datos para reproducir y estructura de proyecto. Eso **es**
> despliegue a escala de asignatura: otra persona puede correrlo sin preguntarte.
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — caso oficial
>
> No hay una sola carta correcta. Hay trampas que sí son comunes:
>
> | Caso | Pregunta de negocio aceptable | Criterio medible | Trampa frecuente |
> |---|---|---|---|
> | **Telco Churn** | ¿A qué clientes conviene ofrecer retención **antes** de que se vayan? | Recall de churn ≥ X con precisión mínima Y (el costo de un falso positivo es un cupón; el de un falso negativo, un cliente) | Optimizar exactitud en un dataset desbalanceado |
> | **House Prices** | ¿Se puede estimar el precio con un error que el negocio acepte? | RMSE / MAE en unidad monetaria, o % de predicciones a ±10 % | Reportar solo R² |
> | **Spotify** | ¿Se puede anticipar popularidad **sin** usar fugas del propio ranking? | RMSE de popularidad, o F1 si binarizan; declarar qué columna sería fuga | Usar `popularity` derivada como feature |
>
> **Despliegue aceptable:** repo con `README`, notebook que corre de arriba abajo,
> datos o instrucciones de obtención, y la carta CRISP-DM dentro del informe.
> Pedir un microservicio es **fuera de alcance** y no se evalúa.
>
> Las 2 h no guiadas existen para que esta tabla salga del aula completa. Si sale
> a medias, la Parcial 2 empieza sin pregunta de negocio.
"""
    ),
    md(
        """
---
### ✅ Antes de cerrar

- [ ] TODO 1–7 en verde (`assert` incluidos).
- [ ] Carta Waymo con `validar_carta` vacío.
- [ ] Carta del caso oficial con pregunta, cifra y riesgos.
- [ ] El no supervisado quedó anotado en el **RA2**, no en el RA3.

### Lo que viene

- **Actividad 2.2:** modelado supervisado. La pregunta de negocio de hoy se vuelve
  un `y` (`detection_difficulty`) y un split por `segment_id`.
- **Actividad 2.3:** modelado no supervisado. Sigue siendo RA2.
- **Actividad 2.4:** evaluación traducida a lenguaje de negocio.
"""
    ),
    md_docente(
        """
> ### 🎓 Criterios de logro — Actividad 2.1 (IL 2.1)
>
> | Nivel | Descripción |
> |---|---|
> | **Destacado (4)** | Todo lo del nivel 3, y además: elige un umbral de éxito anclado a un riesgo medido en el RA1; en el caso oficial distingue el costo de los dos errores; declara un despliegue realista (reproducibilidad, no un API) |
> | **Logrado (3)** | Las seis fases en orden; el RA1 mapeado; carta Waymo que pasa `validar_carta`; caso oficial con pregunta de negocio y criterio con cifra; 2.3 anotado en el RA2 |
> | **En desarrollo (2)** | Recita las fases y rellena la carta, pero el criterio es vago o la pregunta huele a algoritmo; el caso oficial está en blanco |
> | **Inicial (1)** | Trata CRISP-DM como lista para memorizar; pone el no supervisado en el RA3; empieza por "vamos a usar un random forest" |
>
> **Qué mirar al corregir, en este orden:**
>
> 1. **`validar_carta` en verde** y que el contenido no sea un truco (un "F1 ≥ 0"
>    pasa el detector y no vale).
> 2. **El mapa 2.3 → RA2.** Si está mal, D1 no pasa de Inicial.
> 3. **La carta del caso oficial.** Es el puente a la Parcial 2. Sin ella, la
>    actividad está incompleta aunque el hilo Waymo esté perfecto.
"""
    ),
]
