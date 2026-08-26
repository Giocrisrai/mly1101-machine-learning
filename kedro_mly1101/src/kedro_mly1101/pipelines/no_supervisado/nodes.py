"""Nodos de aprendizaje no supervisado (EA3).

**La pregunta:** sin decirle a nadie qué es cada objeto, ¿aparecen grupos naturales
en las detecciones? ¿Y coinciden con los tipos que el sensor etiquetó?

Es la contracara de la EA2. Allí había una etiqueta y se medía el acierto; aquí no
hay etiqueta y hay que **justificar** que la estructura encontrada significa algo.

Dos decisiones gobiernan estos nodos:

1. **Escalar antes de agrupar no es opcional.** K-medias mide distancias
   euclídeas. Sin escalar, ``num_lidar_points`` (que llega a miles) aplasta a
   ``box_height`` (que ronda 1,7): el resultado sería un agrupamiento por número de
   puntos disfrazado de agrupamiento por objeto.
2. **El escalador se ajusta solo con lo que el modelo puede ver.** Aquí no hay
   partición, así que se ajusta con todo; en un flujo con prueba, ajustarlo sobre
   el total sería fuga de información, igual que en la EA2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def preparar_matriz(limpias: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Selecciona las variables, descarta filas incompletas y **escala**.

    Devuelve las variables escaladas junto a la etiqueta real, que **no** se usa
    para agrupar: se guarda para poder contrastar después si los grupos que
    aparecieron solos se parecen a los tipos conocidos.
    """
    columnas = config["variables"]
    tabla = limpias[columnas + [config["etiqueta_de_contraste"]]].dropna()
    tabla = tabla.reset_index(drop=True)

    escalador = StandardScaler()
    escaladas = pd.DataFrame(
        escalador.fit_transform(tabla[columnas]), columns=columnas
    )
    escaladas[config["etiqueta_de_contraste"]] = tabla[
        config["etiqueta_de_contraste"]
    ].to_numpy()
    return escaladas


def elegir_k(matriz: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Prueba varios números de grupos y mide inercia y silueta.

    No existe "el k correcto". La inercia siempre baja al añadir grupos, así que
    por sí sola no decide nada; la silueta sí penaliza los grupos que se solapan.
    Aun así, la decisión final es de dominio: **cuántos grupos son útiles para
    quien va a usar el resultado.**
    """
    variables = config["variables"]
    X = matriz[variables].to_numpy()

    filas = []
    for k in config["k_a_probar"]:
        modelo = KMeans(n_clusters=k, random_state=config["semilla"], n_init=10)
        etiquetas = modelo.fit_predict(X)
        filas.append(
            {
                "k": k,
                "inercia": round(float(modelo.inertia_), 2),
                "silueta": round(
                    float(silhouette_score(X, etiquetas, sample_size=config["muestra_silueta"],
                                           random_state=config["semilla"])),
                    4,
                ),
            }
        )
    return pd.DataFrame(filas)


def agrupar(matriz: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Ajusta K-medias con el ``k`` elegido y devuelve la matriz con su grupo."""
    variables = config["variables"]
    modelo = KMeans(
        n_clusters=config["k"], random_state=config["semilla"], n_init=10
    )
    agrupada = matriz.copy()
    agrupada["grupo"] = modelo.fit_predict(matriz[variables].to_numpy())
    return agrupada


def perfilar_grupos(agrupada: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Describe cada grupo por la media de sus variables.

    Un agrupamiento sin interpretar no sirve de nada: "grupo 2" no es un hallazgo.
    Esta tabla es lo que permite ponerle nombre —*objetos grandes y rápidos*,
    *objetos pequeños y quietos*— y es lo que se evalúa.
    """
    perfil = agrupada.groupby("grupo")[config["variables"]].mean().round(3)
    perfil["n"] = agrupada.groupby("grupo").size()
    perfil["pct"] = (100 * perfil["n"] / len(agrupada)).round(2)
    return perfil.reset_index()


def contrastar_con_etiqueta(agrupada: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Cruza los grupos descubiertos con la etiqueta real, en porcentaje por grupo.

    **No es una evaluación**: el algoritmo nunca vio la etiqueta, así que no puede
    "acertar" ni "fallar". Es una comprobación de sentido. Si un grupo concentra el
    90 % de los peatones, la estructura que apareció sola tiene una lectura de
    dominio. Si todos los grupos tienen la misma mezcla, el agrupamiento no
    encontró nada útil, por buena que sea su silueta.
    """
    etiqueta = config["etiqueta_de_contraste"]
    cruce = pd.crosstab(agrupada["grupo"], agrupada[etiqueta], normalize="index")
    return (100 * cruce).round(2).reset_index()


def proyectar_2d(matriz: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Reduce a dos componentes principales para poder dibujar el resultado.

    PCA busca las direcciones de máxima varianza. ``varianza_explicada`` dice
    cuánta información conserva la proyección: si dos componentes explican poco,
    el dibujo es bonito y engañoso, y hay que decirlo.
    """
    variables = config["variables"]
    pca = PCA(n_components=2, random_state=config["semilla"])
    componentes = pca.fit_transform(matriz[variables].to_numpy())

    proyectada = pd.DataFrame(componentes, columns=["componente_1", "componente_2"])
    proyectada[config["etiqueta_de_contraste"]] = matriz[
        config["etiqueta_de_contraste"]
    ].to_numpy()
    # La varianza explicada NO viaja en ``df.attrs``: Parquet no serializa un
    # array de numpy ahí y la escritura revienta. Va como columna constante, que
    # además la deja visible al abrir el archivo.
    proyectada["varianza_explicada_2d"] = round(
        float(pca.explained_variance_ratio_.sum()), 4
    )
    return proyectada


def resumir_pca(matriz: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Cuánta varianza explica cada componente, y cuántas hacen falta.

    La cifra que importa no es la de la primera componente, sino **cuántas
    componentes necesitas para conservar el 90 %**. Si con dos de ocho basta, hay
    redundancia fuerte entre las variables y eso también es un hallazgo.
    """
    variables = config["variables"]
    pca = PCA(random_state=config["semilla"]).fit(matriz[variables].to_numpy())
    razon = pca.explained_variance_ratio_
    return pd.DataFrame(
        {
            "componente": np.arange(1, len(razon) + 1),
            "varianza_explicada": razon.round(4),
            "varianza_acumulada": razon.cumsum().round(4),
        }
    )
