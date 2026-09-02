"""Utilidades de interpretación para la Actividad 2.4 (MLY1101).

Funciones puras: reciben conteos, matrices o vectores y devuelven estructuras.
No imprimen ni grafican. El punto pedagógico que fijan los tests es que una
métrica no es un conocimiento para la organización hasta que se traduce a
frecuencia, a costo y a una decisión.

Las cifras de la pauta de la Act. 2.2 (456 encontradas, 676 perdidas, 385
falsas alarmas sobre ``LEVEL_2``) son el caso de referencia de ``por_cada_cien``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METRICAS_CLASIFICACION: tuple[str, ...] = ("precision", "recall", "f1")
METRICAS_REGRESION: tuple[str, ...] = ("mae", "rmse")
_SOLO_PROMEDIO = {"r2", "r²", "exactitud", "accuracy"}


def por_cada_cien(encontrados: int, perdidos: int) -> dict[str, int]:
    """Traduce TP y FN a "de cada 100, encuentra X y pierde Y".

    Los dos enteros suman 100. El redondeo del segundo se ajusta para que no
    quede 99 ni 101 por el redondeo del primero.
    """
    total = encontrados + perdidos
    if total <= 0:
        raise ValueError("hace falta al menos un positivo real")
    encuentra = int(round(100 * encontrados / total))
    return {"encuentra": encuentra, "pierde": 100 - encuentra}


def leer_errores(matriz: pd.DataFrame, positiva: str) -> dict[str, int]:
    """Lee TP, FN y FP. Las filas son lo real; las columnas, lo predicho."""
    if positiva not in matriz.index or positiva not in matriz.columns:
        raise KeyError(f"la clase {positiva!r} no está en la matriz")
    tp = int(matriz.loc[positiva, positiva])
    fn = int(matriz.loc[positiva].sum() - tp)
    fp = int(matriz[positiva].sum() - tp)
    return {"tp": tp, "fn": fn, "fp": fp}


def costo_esperado(n_fn: int, n_fp: int, costo_fn: float, costo_fp: float) -> float:
    """Costo total = FN × costo de perder uno + FP × costo de una falsa alarma."""
    return n_fn * costo_fn + n_fp * costo_fp


def frase_para_la_organizacion(encuentra: int, pierde: int, sujeto: str) -> str:
    """Una oración sin nombres de métrica, lista para un informe de negocio."""
    return (
        f"De cada 100 {sujeto}, el modelo encuentra {encuentra} y pierde {pierde}."
    )


def metricas_minimas(problema: str) -> tuple[str, ...]:
    """Las métricas que hay que reportar; el promedio global no está en la lista."""
    if problema == "clasificacion":
        return METRICAS_CLASIFICACION
    if problema == "regresion":
        return METRICAS_REGRESION
    raise ValueError(f"problema desconocido: {problema!r}")


def reporta_solo_promedio(metricas: list[str]) -> bool:
    """True si lo reportado es solo exactitud o solo R²."""
    normalizadas = {m.strip().casefold() for m in metricas}
    return bool(normalizadas) and normalizadas <= _SOLO_PROMEDIO


def error_en_unidades(real: np.ndarray, predicho: np.ndarray) -> dict[str, float]:
    """MAE y RMSE en la escala original, no en un índice adimensional."""
    real = np.asarray(real, dtype=float)
    predicho = np.asarray(predicho, dtype=float)
    residuo = predicho - real
    return {
        "mae": float(np.mean(np.abs(residuo))),
        "rmse": float(np.sqrt(np.mean(residuo**2))),
    }
