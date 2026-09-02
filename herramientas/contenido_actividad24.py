"""Fuente única del contenido de la Actividad 2.4 — Interpretación y métricas.

Indicador de logro **IL 2.4**: *interpreta los resultados del desempeño del modelo,
traduciendo métricas técnicas a conocimientos para la organización.*

De este archivo salen dos notebooks:

- ``notebooks/13_alumno_interpretacion.ipynb``
- ``notebooks/13_docente_interpretacion.ipynb``

**5 horas pedagógicas** según el programa.

La idea de la sesión: en 2.2 midieron un F1 de 0,46 en la clase difícil. Nadie en
una reunión de operaciones sabe qué hacer con eso. Hoy se escribe la frase que
sí se puede usar, se pone costo a cada tipo de error, se traduce el hallazgo del
agrupamiento, y se cubre la interpretación de **regresión** (MAE/RMSE en unidades)
que el IL2.2 exige y que la Parcial 2 va a pedir sobre *House Prices*.

Las cifras de clasificación salen del mismo pipeline que la Act. 2.2
(``detecciones_waymo_like.csv``, semilla 42). Las funciones viven en
``src/interpretacion.py``, cubiertas por ``tests/test_interpretacion.py``.

Regenerar:  python herramientas/construir_notebooks.py
"""

from __future__ import annotations

from contenido_semana01 import URL_REPO, code, md, md_docente

CELDAS_ACT24: list[dict] = [
    md(
        """
# MLY1101 · Machine Learning — Actividad 2.4
## Interpretación y métricas de desempeño

**Resultado de aprendizaje (RA2):** aplica modelos estadísticos al conjunto de datos
procesados para **interpretarlos**, utilizando metodologías ágiles, con la finalidad de
obtener conocimientos relevantes que permitan responder a las necesidades del contexto
de negocio, considerando aspectos éticos.

**Indicador de logro (IL 2.4):** interpreta los resultados del desempeño del modelo,
traduciendo métricas técnicas a conocimientos para la organización.

---

### La idea central de hoy

En la Actividad 2.2 el bosque aleatorio sacó **89,65 %** de exactitud y **0,46** de F1
en las detecciones difíciles. La primera cifra entra fácil a una diapositiva. La
segunda no le dice nada a quien decide si el vehículo puede confiar en el sensor.

Hoy no se entrena un modelo nuevo. Se toma el que ya existe y se responde:

> **¿Qué le digo a la organización, en una frase que pueda usar?**

Tres traducciones, y las tres tienen detector en `interpretacion`:

1. De la matriz de confusión a **"de cada 100, encuentra X y pierde Y"**.
2. De esos conteos a un **costo**: no todos los errores pesan igual.
3. De un R² (regresión) a un error **en la unidad del negocio** (pesos, metros, puntos).

La tercera no es un extra. El IL2.2 pide clasificación **y** regresión; la 2.2 cubrió
la primera. La Parcial 2 sobre *House Prices* va a pedir la segunda.

---

### Al final de la sesión debes entregar

Un **memo de una página** para el equipo de percepción (hilo Waymo) y el mismo memo
para el **caso oficial** de tu equipo. Si el memo menciona "F1-macro" como única
conclusión, no está listo.
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — Actividad 2.4
>
> **5 horas pedagógicas.** ~3 h guiadas; las 2 h restantes son el memo del caso
> oficial, que es el puente a la Parcial 2.
>
> | Bloque | Min | Foco |
> |---|---|---|
> | 0 · Encuadre | 10 | El F1 no entra a la reunión |
> | 1 · De cada 100 ⭐⭐ | 45 | Matriz → frecuencia |
> | 2 · Costo asimétrico ⭐⭐ | 40 | FN vs FP en este dominio |
> | 3 · El agrupamiento no es una métrica | 25 | Los buses, en lenguaje de negocio |
> | 4 · Regresión: unidades, no R² ⭐ | 40 | Lo que pide House Prices |
> | 5 · Memo + caso oficial | 30 | Una página, sin jerga |
> | Cierre | 10 | Qué se evalúa en la Parcial 2 |
>
> **Imprescindibles: 1, 2 y 4.** El 3 se puede dejar en la pauta si falta tiempo.
>
> **No reentrenar a ciegas.** Una sola corrida del pipeline de 2.2; el resto es
> interpretación. Si alguien abre 2.2 y copia el informe, D5 queda en Inicial:
> hoy se exige la frase de negocio, no el `classification_report`.
>
> Cifras que no se recitan de memoria (semilla 42, pauta 2.2): exactitud **0,8965**,
> dummy **0,8896**, F1 `LEVEL_2` **0,4622**, recall **0,4028**, TP **456**, FN **676**,
> FP **385**. De cada 100 difíciles: **40** encontradas, **60** perdidas.
"""
    ),
    md(
        """
---
## Preparación del entorno

Reutilizamos los **mismos nodos** de la Actividad 2.2. No hay un modelo nuevo.
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
import yaml

import interpretacion
from kedro_mly1101.pipelines.preprocesamiento import nodes as limpieza
from kedro_mly1101.pipelines.supervisado import nodes as modelo

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 140)

PARAMETROS = yaml.safe_load(RUTA_PARAMETROS.read_text(encoding="utf-8"))

crudo = pd.read_csv(RUTA_DATOS)
paso = limpieza.normalizar_categorias(crudo, PARAMETROS["mapas_categorias"])
paso = limpieza.descubrir_faltantes(paso, PARAMETROS["centinelas"])
paso = limpieza.marcar_imposibles(paso, PARAMETROS["reglas_dominio"])
limpio = limpieza.quitar_duplicados_y_constantes(paso, PARAMETROS["columnas_a_descartar"])

tabla = modelo.preparar_variables(limpio, PARAMETROS["modelo"], PARAMETROS["fuga"])
marcada = modelo.particionar(tabla, PARAMETROS["modelo"])
clf = modelo.entrenar(marcada, PARAMETROS["modelo"])
metricas = modelo.evaluar_por_clase(clf, marcada, PARAMETROS["modelo"])
matriz = modelo.matriz_confusion(clf, marcada, PARAMETROS["modelo"])

print(metricas.to_string(index=False))
print()
print(matriz)
"""
    ),
    # ======================================================================
    # BLOQUE 1
    # ======================================================================
    md(
        """
---
# Bloque 1 · ⭐⭐ De la matriz a "de cada 100"

La matriz de confusión del pipeline tiene las **filas = lo real** y las
**columnas = lo predicho**. La clase que le importa al negocio es `LEVEL_2`:
detección difícil.

`interpretacion.leer_errores` lee TP, FN y FP. `por_cada_cien` los convierte en
una frecuencia. `frase_para_la_organizacion` escribe la oración **sin** nombres
de métrica.

### ✏️ TODO 1 — Cuenta los errores de la clase difícil
"""
    ),
    code(
        """
errores = interpretacion.leer_errores(matriz, positiva="LEVEL_2")
print(errores)

cien = interpretacion.por_cada_cien(encontrados=errores["tp"], perdidos=errores["fn"])
print(cien)

frase = interpretacion.frase_para_la_organizacion(
    encuentra=cien["encuentra"],
    pierde=cien["pierde"],
    sujeto="detecciones difíciles",
)
print(frase)
assert cien["encuentra"] + cien["pierde"] == 100
""",
        """
errores = interpretacion.leer_errores(matriz, positiva="")  # TODO: la clase difícil
print(errores)

cien = interpretacion.por_cada_cien(encontrados=0, perdidos=0)  # TODO: tp y fn
print(cien)

frase = interpretacion.frase_para_la_organizacion(
    encuentra=cien["encuentra"],
    pierde=cien["pierde"],
    sujeto="",  # TODO: en lenguaje de dominio, no "LEVEL_2"
)
print(frase)
assert cien["encuentra"] + cien["pierde"] == 100
""",
    ),
    md(
        """
**La exactitud del modelo es ~90 %. ¿Por qué esa cifra no puede ser la del memo?** `____`

**Tu frase para la organización (cópiala, o escríbela mejor):** `____`
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — TODO 1
>
> `LEVEL_2`: TP **456**, FN **676**, FP **385**. De cada 100 difíciles: **40** y **60**.
>
> Frase canónica: *"De cada 100 detecciones difíciles, el modelo encuentra 40 y
> pierde 60."*
>
> **Por qué la exactitud no entra al memo:** 88,96 % lo saca un modelo que **siempre
> responde LEVEL_1**, sin mirar los datos. Siete décimas de diferencia. El 90 %
> celebra acertar la mayoría; el negocio necesita no perder la minoría.
>
> Destacado: quien añade *"y produce 385 falsas alarmas en el conjunto de prueba"*
> está leyendo el FP, que el `por_cada_cien` no incluye a propósito (es sobre
> positivos reales).
"""
    ),
    # ======================================================================
    # BLOQUE 2
    # ======================================================================
    md(
        """
---
# Bloque 2 · ⭐⭐ No todos los errores cuestan igual

Dos errores, dos consecuencias:

| Error | Qué pasó | Qué hace el vehículo |
|---|---|---|
| **FN** | Era difícil y el modelo dijo "fácil" | Confía en una detección que no debía |
| **FP** | Era fácil y el modelo dijo "difícil" | Se pone prudente de más |

En percepción, el FN suele ser **más caro**: un peatón no detectado no se compara
con frenar sin necesidad. Eso no lo dice el F1. Lo dice el dominio.

`interpretacion.costo_esperado(n_fn, n_fp, costo_fn, costo_fp)` pondera.

### ✏️ TODO 2 — Ponle precio a cada error
"""
    ),
    code(
        """
# Unidad arbitraria: un FP (frenar de más) vale 1.
# Un FN (confiar en una detección mala) vale 10.
costo = interpretacion.costo_esperado(
    n_fn=errores["fn"],
    n_fp=errores["fp"],
    costo_fn=10,
    costo_fp=1,
)
print(f"Costo total en el conjunto de prueba: {costo:,.0f} unidades")
print(f"  de los FN: {errores['fn'] * 10:,.0f}")
print(f"  de los FP: {errores['fp'] * 1:,.0f}")
""",
        """
# TODO: usa errores["fn"] y errores["fp"] con un costo_fn mayor que costo_fp
costo = interpretacion.costo_esperado(n_fn=0, n_fp=0, costo_fn=1, costo_fp=1)
print(costo)
""",
    ),
    md(
        """
**Si el FN cuesta 10 y el FP cuesta 1, ¿qué porcentaje del costo total viene de
las detecciones difíciles que se perdieron?** `____`

**¿Pondrías este modelo a decidir si el vehículo confía en el sensor? Sí / No, y
en una frase de dominio (no de métrica):** `____`
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — TODO 2
>
> 676 × 10 + 385 × 1 = **7.145**. Los FN son 6.760, el **94,6 %** del costo.
>
> La razón 10:1 es un supuesto pedagógico, no un dato de Waymo. Si un equipo
> propone 5:1 o 20:1 y **recalcula**, está en Destacado: entendió que el número
> sale del dominio, no de sklearn.
>
> **¿Lo pondrías en producción?** La respuesta esperada es **no**, o "sí, pero
> solo como alerta, nunca como permiso para ignorar el objeto". Un "sí" sin
> condiciones, con F1 0,46 y 60 de cada 100 difíciles perdidas, es Inicial.
>
> Conexión con 2.1: el criterio de éxito era F1 de `LEVEL_2` ≥ 0,60. **No se
> cumple.** La carta de proyecto se actualiza; no se oculta.
"""
    ),
    # ======================================================================
    # BLOQUE 3
    # ======================================================================
    md(
        """
---
# Bloque 3 · El agrupamiento no es una métrica

En la Actividad 2.3 el modelo **no predijo nada**. Encontró un grupo pequeño
(~1,5 % de las filas) con `box_length` a casi cinco desviaciones típicas: los
**buses**, los mismos atípicos que en 1.3 se aprendió a *no* eliminar.

Eso no tiene F1. Tiene una frase de negocio o no tiene nada.

### ✏️ TODO 3 — Traduce el hallazgo, sin silueta

**Frase para el equipo de percepción:** `____`

*(Pista: no vale "k = 3 maximiza la silueta". Vale qué haría alguien con ese
grupo si diseña el siguiente modelo o la siguiente recolección de datos.)*
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — TODO 3
>
> Frase aceptable: *"Hay una población pequeña de objetos muy largos (buses) que
> el criterio IQR habría borrado; el agrupamiento los aisló solo. Cualquier modelo
> de percepción que se entrene tirándolos va a estar ciego ante ellos."*
>
> También vale: el grupo no recupera `object_type` (peatón y señalética se mezclan);
> el sensor no separa esas dos clases con las variables que tenemos.
>
> **Inicial:** recitar la tabla de siluetas. **En desarrollo:** describir el
> perfil numérico sin nombrarlo. **Logrado:** nombra buses y conecta con 1.3.
"""
    ),
    # ======================================================================
    # BLOQUE 4
    # ======================================================================
    md(
        """
---
# Bloque 4 · ⭐ Regresión: el error se reporta en unidades

El IL2.2 pide clasificación **y** regresión. La 2.2 cubrió clasificación. La
Parcial 2, sobre *House Prices*, va a pedir **estimar un precio**. Ahí un R² de
0,87 cabe en una diapositiva y no le dice a nadie si el error es $2.000.000 o
$20.000.

`interpretacion.metricas_minimas("regresion")` no incluye R². Incluye **MAE** y
**RMSE**, en la escala original.

### ✏️ TODO 4 — ¿Qué hay que reportar, según el tipo de problema?
"""
    ),
    code(
        """
print("clasificacion:", interpretacion.metricas_minimas("clasificacion"))
print("regresion    :", interpretacion.metricas_minimas("regresion"))
print()
print("¿Solo exactitud basta?", interpretacion.reporta_solo_promedio(["exactitud"]))
print("¿Solo R² basta?      ", interpretacion.reporta_solo_promedio(["r2"]))
print("¿MAE + RMSE basta?   ", interpretacion.reporta_solo_promedio(["mae", "rmse"]))
"""
    ),
    md(
        """
Un ejemplo mínimo, a escala de precio. El "modelo" predice mal de forma
controlada para que el error se pueda leer de un vistazo.
"""
    ),
    code(
        """
# Precios en la unidad del negocio (miles de dólares, como House Prices).
real = np.array([120.0, 180.0, 240.0, 400.0, 90.0])
# Un modelo que se equivoca poco en las casas baratas y mucho en la cara.
predicho = np.array([125.0, 175.0, 250.0, 310.0, 95.0])

errores_reg = interpretacion.error_en_unidades(real, predicho)
print(f"MAE  = {errores_reg['mae']:.1f} mil dólares")
print(f"RMSE = {errores_reg['rmse']:.1f} mil dólares")
print()
print("El MAE cabe en una frase de negocio:")
print(f"  'El modelo se equivoca, en promedio, {errores_reg['mae']:.0f} mil dólares por vivienda.'")
"""
    ),
    md(
        """
**¿Por qué el RMSE sale más alto que el MAE en este ejemplo?** `____`

**Frase de negocio para House Prices, usando MAE (no R²):** `____`
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — TODO 4
>
> Mínimas clasificación: precision, recall, f1. Regresión: mae, rmse. Exactitud
> sola y R² solo activan `reporta_solo_promedio`.
>
> MAE de este ejemplo: (|5|+|5|+|10|+|90|+|5|)/5 = **23**. RMSE se infla por
> la casa de 400 predicha 310 (residuo 90): el RMSE **penaliza los errores
> grandes**. Por eso se reportan los dos: el MAE es el "en promedio"; el RMSE
> avisa si hay colas.
>
> Frase House Prices: *"El modelo se equivoca, en promedio, X unidades
> monetarias por vivienda."* Si el equipo no tiene aún el dataset oficial, la
> plantilla basta. **Inicial** si la conclusión es "R² = 0,87, buen modelo".
>
> Spotify, si binarizan popularidad, vuelve a clasificación (F1). Si predicen
> el entero 0–100, es RMSE en puntos de popularidad.
"""
    ),
    # ======================================================================
    # BLOQUE 5
    # ======================================================================
    md(
        """
---
# Bloque 5 · Memo a la organización

Una página. Quien lo lee no tiene que saber qué es un F1. Si necesita una cifra
de sklearn, va en una nota al pie, no en el titular.

---

## Memo — hilo Waymo (equipo de percepción)

**Para:** `____` · **De:** `____` · **Fecha:** `____`

**Titular (una frase, con frecuencia o con dinero, sin nombre de métrica):**

`____`

**Qué se intentó responder:** `____`

**Qué no puede hacer este modelo:** `____`

**Costo del error que más duele, y por qué:** `____`

**Recomendación (usar / no usar / usar con condición):** `____`

**Qué haría falta para cambiar esa recomendación:** `____`

---

## Memo — caso oficial (Parcial 2 / EFT)

**Caso:** Telco / House Prices / Spotify · **Equipo:** `____`

| Campo | Contenido |
|---|---|
| Pregunta de negocio (de la carta 2.1) | `____` |
| Tipo de problema (clasificación / regresión) | `____` |
| Métricas mínimas que vamos a reportar | `____` |
| Titular en lenguaje de la organización | `____` |
| Error que más cuesta, y a quién | `____` |
| Qué sería "no usar el modelo" en este caso | `____` |

> Telco: un falso negativo es un cliente que se fue; un falso positivo es un cupón
> gastado. House Prices: el error se dice en dinero. Spotify: ojo con usar como
> *feature* algo derivado de la popularidad que se quiere predecir.
"""
    ),
    md_docente(
        """
> ### 🎓 Pauta docente — memos
>
> **Waymo, titular aceptable:** *"De cada 100 detecciones difíciles, el sistema
> pierde 60: no sirve como permiso para confiar en el sensor."*
>
> Recomendación esperada: no usar como decisión automática; como alerta, con
> supervisión, tal vez. Criterio de 2.1 (F1 ≥ 0,60) **no cumplido**.
>
> | Caso | Titular que vale | Titular que no vale |
> |---|---|---|
> | Telco | "De cada 100 clientes que se van a ir, detectamos X y perdemos Y; cada perdido cuesta más que un cupón" | "Accuracy 0,81" |
> | House Prices | "El error medio es X unidades monetarias por vivienda" | "R² = 0,87" |
> | Spotify | "RMSE de Y puntos de popularidad" / F1 si binarizan, declarando fugas | "El modelo predice bien" |
>
> Las 2 h no guiadas existen para que la tabla del caso oficial salga completa.
> Sin ella, la Parcial 2 vuelve a empezar por el algoritmo.
"""
    ),
    md(
        """
---
### ✅ Antes de cerrar

- [ ] TODO 1 en verde: frase "de cada 100" sin decir F1.
- [ ] TODO 2: costo calculado; recomendación de uso en lenguaje de dominio.
- [ ] TODO 4: MAE/RMSE distinguidos de R².
- [ ] Memo Waymo de una página.
- [ ] Memo del caso oficial, con métricas mínimas del tipo de problema.

### Lo que viene

- **RA3:** hiperparámetros, ensamble y validación cruzada. Van a descubrir que
  ajustar no arregla un problema de **sesgo** (la información no está en las
  variables). La interpretación de hoy es la que permite no celebrar una mejora
  de 0,0006 en el F1.
- **Parcial 2:** el mismo memo, sobre Telco / House Prices / Spotify.
"""
    ),
    md_docente(
        """
> ### 🎓 Criterios de logro — Actividad 2.4 (IL 2.4)
>
> | Nivel | Descripción |
> |---|---|
> | **Destacado (4)** | Todo lo del 3, y además: el costo FN/FP es un supuesto declarado y recalculable; el memo actualiza la carta 2.1 ("no se cumplió el umbral"); el caso oficial distingue el costo de los dos errores |
> | **Logrado (3)** | Frase "de cada 100" correcta; recomienda no usar el modelo como permiso; MAE/RMSE en unidades; memo del caso oficial con métricas mínimas del tipo de problema |
> | **En desarrollo (2)** | Traduce la matriz a conteos pero el memo sigue hablando de F1; la regresión se queda en "R² alto" |
> | **Inicial (1)** | Copia el `classification_report` y concluye "el modelo es bueno por la exactitud" |
>
> **Qué mirar al corregir, en este orden:**
>
> 1. **Que el titular no tenga nombre de métrica.** Es el discriminador de la
>    actividad.
> 2. **La recomendación de uso** anclada a 60 de cada 100 difíciles perdidas.
> 3. **El memo del caso oficial.** Sin él, D5 no pasa de En desarrollo.
"""
    ),
]
