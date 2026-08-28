"""Nodos de optimización y ensamble (RA3).

Tres preguntas, una por actividad:

- **3.1** ¿Cuánto mejora ajustar los hiperparámetros, y dónde se ajusta sin hacer trampa?
- **3.2** ¿Un ensamble más complejo gana lo suficiente para justificar su costo?
- **3.3** ¿La diferencia que mido entre dos modelos es real o es ruido del muestreo?

La tercera es la que gobierna el módulo. Comparar dos modelos por un número suelto es la
forma más común de engañarse: si la diferencia entre ellos es menor que la variabilidad
entre particiones, **no hay evidencia de que uno sea mejor**, por más decimales que tenga.

Todo usa ``GroupKFold`` sobre ``segment_id``, por el mismo motivo que la Actividad 2.2: las
detecciones de un segmento comparten contexto y no pueden repartirse entre pliegues.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def _catalogo_de_modelos(semilla: int) -> dict:
    """Los candidatos a comparar, del más simple al más complejo.

    El orden importa pedagógicamente: la comparación solo tiene sentido si el
    primero es deliberadamente tonto y el segundo, deliberadamente simple. Sin
    esos dos, "mi modelo saca 0,70" no significa nada.

    **Detalle que muerde y conviene conocer:** los modelos basados en árboles
    toleran valores faltantes de forma nativa desde scikit-learn 1.4 —el
    divisor aprende hacia qué lado mandar los ``NaN``—, pero la regresión
    logística **no**: falla con ``Input X contains NaN``. Por eso solo ese modelo
    lleva un ``SimpleImputer`` delante.

    Que unos modelos necesiten imputación y otros no es en sí un criterio de
    selección: imputar es inventar un valor, y el que puede evitarlo parte con
    ventaja de honestidad.
    """
    return {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "arbol": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=semilla
        ),
        "regresion_logistica": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=semilla
            ),
        ),
        "bosque_aleatorio": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=semilla,
            n_jobs=-1,
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_depth=6, class_weight="balanced", random_state=semilla
        ),
    }


def _partes(marcada: pd.DataFrame, config: dict):
    """Solo el conjunto de ENTRENAMIENTO: el de prueba no se toca hasta el final."""
    entrena = marcada[marcada["particion"] == "entrenamiento"]
    return (
        entrena[config["variables"]],
        entrena[config["objetivo"]],
        entrena[config["grupo"]],
    )


# ---------------------------------------------------------------------------
# 3.1 · Ajuste de hiperparámetros
# ---------------------------------------------------------------------------

def buscar_hiperparametros(marcada: pd.DataFrame, config: dict, ajuste: dict) -> pd.DataFrame:
    """Busca hiperparámetros con validación cruzada **por grupo**.

    Dos decisiones que no son opcionales:

    1. **La búsqueda ocurre solo dentro del entrenamiento.** El conjunto de
       prueba no participa. Si se ajusta mirando la prueba, la métrica final deja
       de estimar el desempeño futuro y pasa a describir el pasado.
    2. **Los pliegues respetan el segmento** (``GroupKFold``). Sin eso, el ajuste
       se optimizaría contra una estimación inflada por fuga.

    Se usa búsqueda aleatoria y no exhaustiva: con el mismo presupuesto de
    cómputo explora más regiones del espacio, y casi siempre encuentra algo
    equivalente o mejor que la rejilla.
    """
    X, y, grupos = _partes(marcada, config)

    espacio = {
        "n_estimators": ajuste["n_estimators"],
        "max_depth": ajuste["max_depth"],
        "min_samples_leaf": ajuste["min_samples_leaf"],
        "max_features": ajuste["max_features"],
    }
    busqueda = RandomizedSearchCV(
        RandomForestClassifier(
            class_weight="balanced", random_state=config["semilla"], n_jobs=-1
        ),
        param_distributions=espacio,
        n_iter=ajuste["n_combinaciones"],
        scoring=ajuste["metrica"],
        cv=GroupKFold(n_splits=ajuste["n_pliegues"]),
        random_state=config["semilla"],
        n_jobs=-1,
    )
    busqueda.fit(X, y, groups=grupos)

    resultados = pd.DataFrame(busqueda.cv_results_)
    columnas = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "param_n_estimators",
        "param_max_depth",
        "param_min_samples_leaf",
        "param_max_features",
    ]
    tabla = resultados[columnas].sort_values("rank_test_score").reset_index(drop=True)
    tabla[["mean_test_score", "std_test_score"]] = tabla[
        ["mean_test_score", "std_test_score"]
    ].round(4)
    return tabla


def comparar_ajuste_contra_defecto(
    marcada: pd.DataFrame, config: dict, ajuste: dict, busqueda: pd.DataFrame
) -> pd.DataFrame:
    """¿Cuánto ganó el ajuste sobre los valores por defecto?

    La respuesta suele decepcionar, y esa decepción es el aprendizaje: el ajuste
    de hiperparámetros da mejoras de segundo orden. Las de primer orden vienen de
    las variables, de la partición y de la definición del problema.

    Se compara además contra la **desviación típica entre pliegues**: si la
    ganancia es menor que ella, no hay evidencia de mejora.
    """
    X, y, grupos = _partes(marcada, config)
    cv = GroupKFold(n_splits=ajuste["n_pliegues"])

    por_defecto = RandomForestClassifier(
        class_weight="balanced", random_state=config["semilla"], n_jobs=-1
    )
    puntajes = cross_val_score(
        por_defecto, X, y, groups=grupos, cv=cv, scoring=ajuste["metrica"], n_jobs=-1
    )

    mejor = busqueda.iloc[0]
    filas = [
        {
            "configuracion": "valores por defecto",
            "media": round(float(puntajes.mean()), 4),
            "desv_entre_pliegues": round(float(puntajes.std()), 4),
        },
        {
            "configuracion": "mejor de la búsqueda",
            "media": float(mejor["mean_test_score"]),
            "desv_entre_pliegues": float(mejor["std_test_score"]),
        },
    ]
    tabla = pd.DataFrame(filas)
    ganancia = tabla.loc[1, "media"] - tabla.loc[0, "media"]
    tabla["ganancia"] = [0.0, round(ganancia, 4)]
    tabla["ganancia_supera_el_ruido"] = [
        pd.NA,
        bool(ganancia > tabla.loc[0, "desv_entre_pliegues"]),
    ]
    return tabla


def medir_fuga_por_ajustar_en_prueba(
    marcada: pd.DataFrame, config: dict, ajuste: dict
) -> pd.DataFrame:
    """La trampa del RA3: elegir hiperparámetros mirando el conjunto de prueba.

    Es más sutil que las fugas de la Actividad 2.2, porque nadie *usa* la prueba
    para entrenar: solo la mira para decidir. Pero al quedarse con la
    configuración que mejor puntúa **en la prueba**, esa puntuación deja de ser
    una estimación del desempeño futuro y se convierte en el máximo de una
    muestra, que siempre es optimista.

    Devuelve una fila por configuración y tres de resumen. Las tres miden cosas
    distintas, y conviene no confundirlas:

    - **El margen de la trampa**: cuánto separa a la mejor de la peor
      configuración *en la prueba*. Es lo que alguien se regalaría eligiendo con
      el criterio equivocado, y existe aunque en una corrida concreta los dos
      criterios coincidan.
    - **El optimismo del criterio tramposo**: cuánto gana de más frente a elegir
      por validación cruzada. Puede salir **cero** si ambos criterios eligen la
      misma configuración, y eso no significa que la trampa sea inofensiva:
      significa que esta vez la suerte no la premió.
    - **La brecha entre validación y prueba**: la validación cruzada puntúa
      sistemáticamente más bajo porque entrena con 4/5 de los datos. **No es
      fuga**, es un sesgo conocido y conservador, y por eso es un criterio de
      selección seguro.
    """
    from sklearn.metrics import f1_score

    X, y, grupos = _partes(marcada, config)
    prueba = marcada[marcada["particion"] == "prueba"]
    X_prueba, y_prueba = prueba[config["variables"]], prueba[config["objetivo"]]
    cv = GroupKFold(n_splits=ajuste["n_pliegues"])

    filas = []
    for profundidad in ajuste["max_depth"]:
        modelo = RandomForestClassifier(
            n_estimators=100,
            max_depth=profundidad,
            class_weight="balanced",
            random_state=config["semilla"],
            n_jobs=-1,
        )
        validacion = cross_val_score(
            modelo, X, y, groups=grupos, cv=cv, scoring=ajuste["metrica"], n_jobs=-1
        ).mean()
        modelo.fit(X, y)
        filas.append(
            {
                # ``None`` significa "sin límite de profundidad"; se guarda como
                # texto porque una columna con None y enteros mezclados se
                # convierte en float y rompe cualquier conversión posterior.
                "max_depth": "sin límite" if profundidad is None else str(profundidad),
                "validacion_cruzada": round(float(validacion), 4),
                "en_prueba": round(
                    float(f1_score(y_prueba, modelo.predict(X_prueba), average="macro")), 4
                ),
            }
        )

    tabla = pd.DataFrame(filas)
    honesto = tabla.loc[tabla["validacion_cruzada"].idxmax()]
    tramposo = tabla.loc[tabla["en_prueba"].idxmax()]

    resumen = pd.DataFrame(
        [
            {
                "max_depth": honesto["max_depth"],
                "criterio": "elegir por validación cruzada (correcto)",
                "f1_que_se_reportaria": honesto["en_prueba"],
            },
            {
                "max_depth": tramposo["max_depth"],
                "criterio": "elegir mirando la prueba (con fuga)",
                "f1_que_se_reportaria": tramposo["en_prueba"],
            },
            {
                "max_depth": "—",
                "criterio": "margen de la trampa (mejor − peor en prueba)",
                "f1_que_se_reportaria": round(
                    float(tabla["en_prueba"].max() - tabla["en_prueba"].min()), 4
                ),
            },
        ]
    )
    resumen["optimismo"] = [
        0.0,
        round(float(tramposo["en_prueba"] - honesto["en_prueba"]), 4),
        pd.NA,
    ]
    resumen["brecha_validacion_vs_prueba"] = [
        round(float((tabla["en_prueba"] - tabla["validacion_cruzada"]).mean()), 4)
    ] * 3

    tabla["criterio"] = "—"
    tabla["brecha_validacion_vs_prueba"] = (
        tabla["en_prueba"] - tabla["validacion_cruzada"]
    ).round(4)

    # Las columnas de resumen se crean como float con NaN, no con ``pd.NA``:
    # concatenar columnas enteramente NA está deprecado en pandas y cambiará de
    # comportamiento. Con NaN el dtype queda determinado y el aviso desaparece.
    tabla["f1_que_se_reportaria"] = np.nan
    tabla["optimismo"] = np.nan
    resumen["optimismo"] = pd.to_numeric(resumen["optimismo"], errors="coerce")

    # Las filas de resumen no tienen las columnas por configuración; se rellenan
    # con NaN para que el concat tenga el mismo esquema en ambos lados.
    for columna in ("validacion_cruzada", "en_prueba"):
        resumen[columna] = np.nan

    columnas = list(tabla.columns)
    return pd.concat([tabla, resumen[columnas]], ignore_index=True)


# ---------------------------------------------------------------------------
# 3.2 · Ensamble
# ---------------------------------------------------------------------------

def comparar_ensambles(marcada: pd.DataFrame, config: dict, ajuste: dict) -> pd.DataFrame:
    """Compara modelos individuales contra un ensamble por votación.

    El ensamble reduce **varianza**: promediar modelos que se equivocan en cosas
    distintas cancela parte del error. No arregla el **sesgo**: si todos los
    modelos comparten el mismo punto ciego, promediar no lo elimina.

    Se reporta el tiempo de entrenamiento junto a la métrica, porque la pregunta
    de esta actividad no es "¿mejora?" sino **"¿mejora lo suficiente para
    justificar el costo?"**.
    """
    X, y, grupos = _partes(marcada, config)
    cv = GroupKFold(n_splits=ajuste["n_pliegues"])

    modelos = _catalogo_de_modelos(config["semilla"])
    modelos["ensamble_votacion"] = VotingClassifier(
        estimators=[
            ("arbol", modelos["arbol"]),
            ("bosque", modelos["bosque_aleatorio"]),
            ("boosting", modelos["gradient_boosting"]),
        ],
        voting="soft",
        n_jobs=-1,
    )

    filas = []
    for nombre, modelo in modelos.items():
        inicio = time.perf_counter()
        puntajes = cross_val_score(
            modelo, X, y, groups=grupos, cv=cv, scoring=ajuste["metrica"], n_jobs=-1
        )
        segundos = time.perf_counter() - inicio
        filas.append(
            {
                "modelo": nombre,
                "media": round(float(puntajes.mean()), 4),
                "desv_entre_pliegues": round(float(puntajes.std()), 4),
                "peor_pliegue": round(float(puntajes.min()), 4),
                "mejor_pliegue": round(float(puntajes.max()), 4),
                "segundos": round(segundos, 1),
            }
        )
    return pd.DataFrame(filas).sort_values("media", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3.3 · Robustez y selección
# ---------------------------------------------------------------------------

def analizar_robustez(comparacion: pd.DataFrame) -> pd.DataFrame:
    """¿Las diferencias entre modelos superan la variabilidad entre pliegues?

    Es la pregunta que decide si la comparación significa algo. Se compara la
    diferencia de medias contra la desviación típica del mejor modelo: una regla
    práctica, no un test formal, pero suficiente para frenar la conclusión
    apresurada de que "mi modelo es mejor por 0,003".
    """
    mejor = comparacion.iloc[0]
    filas = []
    for _, fila in comparacion.iterrows():
        diferencia = mejor["media"] - fila["media"]
        filas.append(
            {
                "modelo": fila["modelo"],
                "media": fila["media"],
                "diferencia_vs_mejor": round(float(diferencia), 4),
                "ruido_del_mejor": mejor["desv_entre_pliegues"],
                "distinguible_del_mejor": bool(
                    diferencia > mejor["desv_entre_pliegues"]
                ),
            }
        )
    return pd.DataFrame(filas)


def tabla_de_seleccion(
    comparacion: pd.DataFrame, robustez: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """La decisión final: métrica, estabilidad, costo e interpretabilidad juntos.

    El IL3.4 pide **sustentar** la selección, no solo reportar el máximo. Un
    modelo que gana por un margen indistinguible del ruido pero tarda veinte
    veces más y no se puede explicar, no es la solución óptima: es la que sacó el
    número más alto una vez.
    """
    interpretabilidad = {
        "baseline": "trivial",
        "arbol": "alta (se puede dibujar)",
        "regresion_logistica": "alta (coeficientes)",
        "bosque_aleatorio": "media (importancias)",
        "gradient_boosting": "media (importancias)",
        "ensamble_votacion": "baja (tres modelos)",
    }
    tabla = comparacion.merge(
        robustez[["modelo", "distinguible_del_mejor"]], on="modelo", how="left"
    )
    tabla["interpretabilidad"] = tabla["modelo"].map(interpretabilidad)
    referencia = tabla["segundos"].min()
    tabla["veces_mas_lento"] = (tabla["segundos"] / referencia).round(1)
    return tabla[
        [
            "modelo",
            "media",
            "desv_entre_pliegues",
            "distinguible_del_mejor",
            "segundos",
            "veces_mas_lento",
            "interpretabilidad",
        ]
    ]
