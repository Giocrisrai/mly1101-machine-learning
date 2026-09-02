"""Utilidades de CRISP-DM para la Actividad 2.1 (MLY1101).

Funciones puras: reciben texto o estructuras y devuelven estructuras. No imprimen
ni grafican, para que sirvan en el notebook, en los tests y en la pauta.

Las seis fases son las de Chapman et al. (2000). El detalle pedagógico que este
módulo fija —y que un test protege— es el mapa del curso: supervisado y no
supervisado viven los dos en el RA2; el RA3 es evaluación/optimización.
"""

from __future__ import annotations

import re

FASES: tuple[str, ...] = (
    "comprension_del_negocio",
    "comprension_de_los_datos",
    "preparacion_de_los_datos",
    "modelado",
    "evaluacion",
    "despliegue",
)

NOMBRES: dict[str, str] = {
    "comprension_del_negocio": "Comprensión del negocio",
    "comprension_de_los_datos": "Comprensión de los datos",
    "preparacion_de_los_datos": "Preparación de los datos",
    "modelado": "Modelado",
    "evaluacion": "Evaluación",
    "despliegue": "Despliegue",
}

# Desde dónde se puede volver. CRISP-DM no es una cascada: evaluar puede
# devolver al problema de negocio ("esta métrica no responde la pregunta")
# o a los datos ("esta partición miente").
RETORNOS: dict[str, tuple[str, ...]] = {
    "evaluacion": ("comprension_del_negocio", "comprension_de_los_datos"),
    "modelado": ("preparacion_de_los_datos", "comprension_de_los_datos"),
}

CAMPOS_CARTA: tuple[str, ...] = (
    "pregunta_de_negocio",
    "criterio_de_exito",
    "fuentes",
    "riesgos",
    "proxima_fase",
)

# Palabras que delatan que se empezó eligiendo técnica, no preguntando.
_ALGORITMO = (
    "algoritmo",
    "qué modelo",
    "que modelo",
    "random forest",
    "xgboost",
    "red neuronal",
    "svm",
    "k-means",
    "kmedias",
    "vamos a probar",
)


def mapa_del_curso() -> list[dict[str, str]]:
    """Correspondencia actividad → RA → fase CRISP-DM.

    Una actividad puede tocar más de una fase; aquí se anota la **principal**,
    la que justifica que exista en el programa.
    """
    return [
        {"actividad": "1.1", "ra": "RA1", "fase": "comprension_de_los_datos"},
        {"actividad": "1.2", "ra": "RA1", "fase": "preparacion_de_los_datos"},
        {"actividad": "1.3", "ra": "RA1", "fase": "comprension_de_los_datos"},
        {"actividad": "1.4", "ra": "RA1", "fase": "comprension_del_negocio"},
        {"actividad": "2.1", "ra": "RA2", "fase": "comprension_del_negocio"},
        {"actividad": "2.2", "ra": "RA2", "fase": "modelado"},
        {"actividad": "2.3", "ra": "RA2", "fase": "modelado"},
        {"actividad": "2.4", "ra": "RA2", "fase": "evaluacion"},
        {"actividad": "3.1", "ra": "RA3", "fase": "evaluacion"},
        {"actividad": "3.2", "ra": "RA3", "fase": "evaluacion"},
        {"actividad": "3.3", "ra": "RA3", "fase": "evaluacion"},
    ]


def es_criterio_medible(texto: str) -> bool:
    """Un criterio de éxito sin cifra no es un criterio: es un deseo.

    Basta con que aparezca un dígito (umbral, porcentaje, cantidad). Las
    métricas sin umbral ("mejorar el F1") siguen sin ser medibles.
    """
    return bool(re.search(r"\d", texto or ""))


def empieza_por_el_algoritmo(pregunta: str) -> bool:
    """True si la pregunta de negocio es, en el fondo, elegir un modelo."""
    texto = (pregunta or "").casefold()
    return any(marca in texto for marca in _ALGORITMO)


def validar_carta(carta: dict) -> list[str]:
    """Devuelve los campos que impiden dar por cerrada la carta.

    Lista vacía = la carta es usable como insumo de las actividades 2.2 y 2.3.
    """
    problemas: list[str] = []
    for campo in CAMPOS_CARTA:
        valor = carta.get(campo)
        if valor is None or (isinstance(valor, str) and not valor.strip()):
            problemas.append(campo)

    if "criterio_de_exito" not in problemas:
        if not es_criterio_medible(str(carta.get("criterio_de_exito", ""))):
            problemas.append("criterio_de_exito")

    if "pregunta_de_negocio" not in problemas:
        if empieza_por_el_algoritmo(str(carta.get("pregunta_de_negocio", ""))):
            problemas.append("pregunta_de_negocio")

    if "proxima_fase" not in problemas:
        if carta.get("proxima_fase") not in FASES:
            problemas.append("proxima_fase")

    return problemas
