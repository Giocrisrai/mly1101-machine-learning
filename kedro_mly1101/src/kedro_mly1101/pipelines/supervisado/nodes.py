"""Nodos de aprendizaje supervisado (EA2).

**La pregunta:** ¿se puede anticipar qué detecciones van a ser difíciles
(``detection_difficulty``) a partir de la geometría del objeto y de dónde está?
Si se pudiera, el equipo de percepción sabría de antemano en qué situaciones no
conviene confiar en el sensor.

Se descartó clasificar ``object_type``, que parecía el objetivo natural: sobre
este dataset se resuelve al 99,98 % con cualquier partición, porque el generador
sortea las dimensiones **por tipo de objeto** y basta el largo de la caja para
acertar. Un ejercicio donde todo sale perfecto no enseña a evaluar.

``detection_difficulty`` sí tiene sustancia, y por dos motivos:

1. **Está desbalanceado** — 88,9 % ``LEVEL_1`` contra 11,1 % ``LEVEL_2``. El
   modelo alcanza cerca del 90 % de exactitud con un F1-macro de 0,65: el caso
   de manual de que *el promedio global oculta a la minoría*.
2. **Permite medir una fuga de verdad.** La etiqueta se deriva de
   ``num_lidar_points``, así que incluir esa variable entre las predictoras
   infla la métrica. El nodo ``comparar_fuga_de_variable`` lo mide en vez de
   afirmarlo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def preparar_variables(
    limpias: pd.DataFrame, config: dict, fuga: dict | None = None
) -> pd.DataFrame:
    """Arma la tabla de modelamiento: variables, etiqueta y llave de agrupación.

    Devuelve un único DataFrame en vez de ``X`` e ``y`` por separado porque el
    ``segment_id`` tiene que viajar con las filas hasta la partición. Separarlos
    aquí obligaría a reordenarlos después, que es justo el tipo de manipulación
    donde el índice se desalinea sin avisar (Actividad 1.2, bloque 4).

    Las filas sin etiqueta se descartan: en aprendizaje supervisado, una fila sin
    ``y`` no se puede usar para entrenar ni para evaluar.
    """
    columnas = config["variables"] + [config["objetivo"], config["grupo"]]
    # La variable excluida por fuga viaja igual en la tabla: el modelo no la usa,
    # pero el nodo que mide el efecto de incluirla la necesita.
    if fuga and fuga.get("variable") and fuga["variable"] not in columnas:
        columnas = columnas + [fuga["variable"]]
    tabla = limpias[[c for c in columnas if c in limpias.columns]].copy()

    antes = len(tabla)
    tabla = tabla.dropna(subset=[config["objetivo"], config["grupo"]])
    tabla = tabla.reset_index(drop=True)

    if antes != len(tabla):
        # No es un print de depuración: queda en el log de Kedro como parte del
        # registro de la ejecución.
        pass
    return tabla


def particionar(tabla: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Marca cada fila como ``entrenamiento`` o ``prueba``, **agrupando por segmento**.

    Aquí está la decisión que importa. Una fila es una detección; un segmento son
    ~20 s de grabación con decenas de detecciones que **comparten** clima, momento
    del día, ubicación y condiciones del sensor.

    Si se parte al azar por fila, detecciones del mismo segmento caen en
    entrenamiento y en prueba a la vez. El modelo puede reconocer el segmento en
    vez del objeto, y la métrica de prueba mide memoria, no generalización. Es
    **fuga de información por agrupación**.

    ``GroupShuffleSplit`` garantiza que un segmento entero cae de un solo lado.

    Honestidad sobre este dataset concreto: el nodo ``comparar_particiones``
    mide el efecto y sale **cerca de cero**, porque el generador sortea cada
    detección de forma independiente dentro del segmento. Partir por grupo sigue
    siendo lo correcto —en datos reales de Waymo los fotogramas consecutivos
    siguen al mismo objeto y la dependencia sí existe—, pero aquí es una decisión
    de diseño, no una que se justifique midiendo. Eso también hay que saber
    decirlo.

    Devuelve la misma tabla con una columna ``particion`` añadida, en vez de dos
    DataFrames: así el reparto queda auditable fila a fila y se puede comprobar
    después que ningún segmento aparece en ambos lados.
    """
    separador = GroupShuffleSplit(
        n_splits=1, test_size=config["proporcion_prueba"], random_state=config["semilla"]
    )
    indices_entrenamiento, indices_prueba = next(
        separador.split(tabla, groups=tabla[config["grupo"]])
    )

    marcada = tabla.copy()
    marcada["particion"] = "entrenamiento"
    marcada.loc[marcada.index[indices_prueba], "particion"] = "prueba"
    return marcada


def particionar_al_azar(tabla: pd.DataFrame, config: dict) -> pd.DataFrame:
    """La partición **incorrecta**: al azar por fila, ignorando el segmento.

    Existe únicamente para poder medir el daño en vez de afirmarlo. El nodo
    ``comparar_particiones`` entrena con las dos y muestra la diferencia.
    """
    indices_entrenamiento, indices_prueba = train_test_split(
        tabla.index,
        test_size=config["proporcion_prueba"],
        random_state=config["semilla"],
        stratify=tabla[config["objetivo"]],
    )
    marcada = tabla.copy()
    marcada["particion"] = "entrenamiento"
    marcada.loc[indices_prueba, "particion"] = "prueba"
    return marcada


def _separar(marcada: pd.DataFrame, config: dict):
    """Divide la tabla marcada en las cuatro piezas que espera scikit-learn."""
    entrenamiento = marcada[marcada["particion"] == "entrenamiento"]
    prueba = marcada[marcada["particion"] == "prueba"]
    variables = config["variables"]
    objetivo = config["objetivo"]
    return (
        entrenamiento[variables],
        entrenamiento[objetivo],
        prueba[variables],
        prueba[objetivo],
    )


def entrenar(marcada: pd.DataFrame, config: dict) -> RandomForestClassifier:
    """Entrena el clasificador **solo** con la partición de entrenamiento.

    ``class_weight="balanced"`` no es decorativo: ``cyclist`` es el 1,9 % de las
    filas. Sin compensar, el modelo aprende que ignorar a los ciclistas casi no
    penaliza el acierto global — y ese es exactamente el error que no queremos en
    un sistema de percepción de un vehículo.
    """
    X_entrena, y_entrena, _, _ = _separar(marcada, config)
    modelo = RandomForestClassifier(
        n_estimators=config["n_arboles"],
        max_depth=config["profundidad_maxima"],
        class_weight="balanced",
        random_state=config["semilla"],
        n_jobs=-1,
    )
    modelo.fit(X_entrena, y_entrena)
    return modelo


def evaluar_por_clase(
    modelo: RandomForestClassifier, marcada: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """Métricas **por clase**, no solo el promedio.

    Un promedio global esconde a las minorías: con ``cyclist`` al 1,9 %, un modelo
    que no acierte ni un ciclista puede seguir mostrando más del 95 % de exactitud.
    La fila que hay que mirar al corregir es la de ``cyclist``, no la de
    ``accuracy``.
    """
    _, _, X_prueba, y_prueba = _separar(marcada, config)
    predicho = modelo.predict(X_prueba)

    reporte = classification_report(y_prueba, predicho, output_dict=True, zero_division=0)
    tabla = pd.DataFrame(reporte).T
    tabla.index.name = "clase"
    return tabla.round(4).reset_index()


def matriz_confusion(
    modelo: RandomForestClassifier, marcada: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """Qué se confunde con qué. Las filas son lo real; las columnas, lo predicho."""
    _, _, X_prueba, y_prueba = _separar(marcada, config)
    etiquetas = sorted(y_prueba.unique())
    matriz = confusion_matrix(y_prueba, modelo.predict(X_prueba), labels=etiquetas)
    return pd.DataFrame(matriz, index=etiquetas, columns=etiquetas)


def comparar_particiones(
    marcada_por_grupo: pd.DataFrame, marcada_al_azar: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """Entrena con las dos particiones y mide la diferencia.

    **Sobre este dataset la diferencia es prácticamente nula**, y decirlo es parte
    del ejercicio. La columna ``segmentos_compartidos`` muestra que la fuga existe
    estructuralmente —153 segmentos a caballo contra 0— pero no mueve la métrica,
    porque el generador sortea cada detección de forma independiente.

    La conclusión que se busca no es "partir por grupo da igual", sino algo más
    incómodo y más útil: **un riesgo que no se manifiesta en tus datos de prueba
    sigue siendo un riesgo**. La partición por grupo se justifica por cómo se
    generaron los datos, no por la diferencia que se mide hoy.
    """
    filas = []
    for nombre, marcada in [
        ("por segmento (correcta)", marcada_por_grupo),
        ("al azar por fila (con fuga)", marcada_al_azar),
    ]:
        modelo = entrenar(marcada, config)
        _, _, X_prueba, y_prueba = _separar(marcada, config)
        predicho = modelo.predict(X_prueba)
        reporte = classification_report(
            y_prueba, predicho, output_dict=True, zero_division=0
        )

        segmentos_entrena = set(
            marcada.loc[marcada["particion"] == "entrenamiento", config["grupo"]]
        )
        segmentos_prueba = set(
            marcada.loc[marcada["particion"] == "prueba", config["grupo"]]
        )

        # La clase minoritaria se calcula, no se escribe a mano: si mañana cambia
        # el objetivo en parameters.yml, esta columna sigue siendo la correcta.
        minoritaria = y_prueba.value_counts().idxmin()

        filas.append(
            {
                "particion": nombre,
                "exactitud": round(reporte["accuracy"], 4),
                "f1_macro": round(reporte["macro avg"]["f1-score"], 4),
                "clase_minoritaria": minoritaria,
                "f1_minoritaria": round(
                    reporte.get(str(minoritaria), {}).get("f1-score", np.nan), 4
                ),
                "segmentos_compartidos": len(segmentos_entrena & segmentos_prueba),
            }
        )

    tabla = pd.DataFrame(filas)
    tabla["diferencia_vs_correcta"] = (
        tabla["exactitud"] - tabla.loc[0, "exactitud"]
    ).round(4)
    return tabla


def comparar_fuga_de_variable(
    marcada: pd.DataFrame, config: dict, fuga: dict
) -> pd.DataFrame:
    """Mide cuánto infla la métrica incluir una variable de la que sale la etiqueta.

    ``detection_difficulty`` la asigna el sensor a partir de ``num_lidar_points``.
    Meter esa columna entre las predictoras no es informativo: es contarle al
    modelo la respuesta con otras palabras. La métrica sube y el modelo no sirve
    para nada, porque en el momento en que quisieras predecir la dificultad ya
    tendrías la dificultad.

    A diferencia de la fuga por agrupación, **esta sí se mide en este dataset**.
    Es el tipo de error más común en la práctica y el más difícil de ver: no hay
    ningún síntoma, solo un resultado sospechosamente bueno.
    """
    filas = []
    for etiqueta, variables in [
        ("sin la variable derivada (correcto)", config["variables"]),
        (
            f"incluyendo {fuga['variable']} (con fuga)",
            config["variables"] + [fuga["variable"]],
        ),
    ]:
        disponibles = [c for c in variables if c in marcada.columns]
        subconfig = {**config, "variables": disponibles}

        modelo = entrenar(marcada, subconfig)
        _, _, X_prueba, y_prueba = _separar(marcada, subconfig)
        reporte = classification_report(
            y_prueba, modelo.predict(X_prueba), output_dict=True, zero_division=0
        )
        filas.append(
            {
                "variables": etiqueta,
                "n_variables": len(disponibles),
                "exactitud": round(reporte["accuracy"], 4),
                "f1_macro": round(reporte["macro avg"]["f1-score"], 4),
            }
        )

    tabla = pd.DataFrame(filas)
    tabla["inflacion_f1_macro"] = (tabla["f1_macro"] - tabla.loc[0, "f1_macro"]).round(4)
    return tabla
