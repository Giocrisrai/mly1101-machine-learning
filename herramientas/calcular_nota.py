"""Convierte puntajes de rúbrica en nota de 1,0 a 7,0.

Dos instrumentos:

* **Formativo (actividades).** Cinco dimensiones D1–D5, puntaje 0–4, pesos de
  ``docs/rubrica_ra1.md``. Siguen llamándose IL1…IL5 en la CLI.
* **Institucional (parciales y EFT).** Porcentaje de logro 100/80/60/30/0 sobre
  IE1–IE12, pesos de las pautas oficiales 2026.

Aplica la escala chilena con **exigencia configurable** (60 % por defecto): la
nota de aprobación (4,0) se alcanza al obtener ese porcentaje del puntaje
máximo, y la escala es lineal a ambos lados de ese punto.

    Sea P el puntaje ponderado (0 a 4), Pmax = 4 y e la exigencia.
    Sea U = e · Pmax el puntaje de aprobación.

        P < U :  nota = 1,0 + 3,0 · P / U
        P ≥ U :  nota = 4,0 + 3,0 · (P − U) / (Pmax − U)

    Con e = 0,60:  P = 0 → 1,0    P = 2,4 → 4,0    P = 4 → 7,0

El % de logro institucional se convierte con la misma escala:
``nota(logro / 100 · 4)``.

Uso:

    python herramientas/calcular_nota.py 3 4 3 2 4        # IL1 IL2 IL3 IL4 IL5
    python herramientas/calcular_nota.py 3 4 3 2 4 --exigencia 0.5
    python herramientas/calcular_nota.py --csv notas.csv  # curso completo
    python herramientas/calcular_nota.py --csv docs/ejemplo_notas_ep1.csv
    python herramientas/calcular_nota.py --csv docs/ejemplo_notas_ep2.csv
    python herramientas/calcular_nota.py --csv docs/ejemplo_notas_ep3.csv
    python herramientas/calcular_nota.py --csv docs/ejemplo_notas_eft.csv
    python herramientas/calcular_nota.py --instrumento ep1 --ie 80 60 100 60

El CSV formativo debe tener cabecera ``nombre,IL1,IL2,IL3,IL4,IL5``.
El CSV institucional, ``nombre,IE1,…`` (EP1 = IE1–IE4; EP2 = IE5–IE8;
EP3 = IE9–IE12; EFT = IE1–IE12).
Si no pasas ``--instrumento``, se infiere por las columnas.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# Pesos de docs/rubrica_ra1.md. Deben sumar 1.
INDICADORES: dict[str, float] = {
    "IL1": 0.20,  # Exploración e identificación de variables
    "IL2": 0.30,  # Identificación y cuantificación de problemas de calidad
    "IL3": 0.25,  # Fundamentación de las decisiones de preprocesamiento
    "IL4": 0.15,  # Tratamiento responsable de la información
    "IL5": 0.10,  # Comunicación y documentación
}

DESCRIPCIONES = {
    "IL1": "Exploración e identificación de variables",
    "IL2": "Problemas de calidad: hallazgo y cuantificación",
    "IL3": "Decisiones de preprocesamiento fundamentadas",
    "IL4": "Tratamiento responsable de la información",
    "IL5": "Comunicación y documentación",
}

PUNTAJE_MAXIMO = 4.0
EXIGENCIA_POR_DEFECTO = 0.60
NOTA_MINIMA = 1.0
NOTA_APROBACION = 4.0
NOTA_MAXIMA = 7.0

# Pautas oficiales 2026 (EP1/EP2/EP3/EFT). No son las D1–D5 formativas.
NIVELES_LOGRO: tuple[int, ...] = (0, 30, 60, 80, 100)
INSTRUMENTOS: dict[str, dict[str, float]] = {
    "ep1": {"IE1": 0.10, "IE2": 0.30, "IE3": 0.40, "IE4": 0.20},
    "ep2": {"IE5": 0.20, "IE6": 0.30, "IE7": 0.30, "IE8": 0.20},
    "ep3": {"IE9": 0.20, "IE10": 0.30, "IE11": 0.20, "IE12": 0.30},
    "eft": {
        **{f"IE{i}": 0.05 for i in range(1, 5)},
        **{f"IE{i}": 0.10 for i in range(5, 13)},
    },
}


def puntaje_ponderado(puntajes: dict[str, float]) -> float:
    """Combina los puntajes por indicador (0 a 4) según los pesos de la rúbrica.

    Raises:
        ValueError: si falta un indicador o un puntaje está fuera de 0–4.
    """
    faltan = set(INDICADORES) - set(puntajes)
    if faltan:
        raise ValueError(f"faltan indicadores: {sorted(faltan)}")
    for clave, valor in puntajes.items():
        if clave in INDICADORES and not 0 <= valor <= PUNTAJE_MAXIMO:
            raise ValueError(f"{clave} = {valor}: el puntaje debe estar entre 0 y 4")
    return sum(puntajes[clave] * peso for clave, peso in INDICADORES.items())


def nota(puntaje: float, exigencia: float = EXIGENCIA_POR_DEFECTO) -> float:
    """Convierte un puntaje ponderado (0 a 4) en nota de 1,0 a 7,0.

    Args:
        puntaje: puntaje ponderado.
        exigencia: fracción del puntaje máximo que corresponde a la nota 4,0.

    Returns:
        La nota redondeada a un decimal.
    """
    if not 0 < exigencia < 1:
        raise ValueError("la exigencia debe estar entre 0 y 1")
    umbral = exigencia * PUNTAJE_MAXIMO
    if puntaje < umbral:
        valor = NOTA_MINIMA + (NOTA_APROBACION - NOTA_MINIMA) * puntaje / umbral
    else:
        avance = (puntaje - umbral) / (PUNTAJE_MAXIMO - umbral)
        valor = NOTA_APROBACION + (NOTA_MAXIMA - NOTA_APROBACION) * avance
    return round(valor, 1)


def evaluar(puntajes: dict[str, float], exigencia: float = EXIGENCIA_POR_DEFECTO) -> dict:
    """Devuelve el detalle completo de la evaluación de un estudiante."""
    total = puntaje_ponderado(puntajes)
    calificacion = nota(total, exigencia)
    return {
        "puntaje_ponderado": round(total, 3),
        "porcentaje": round(100 * total / PUNTAJE_MAXIMO, 1),
        "nota": calificacion,
        "aprueba": calificacion >= NOTA_APROBACION,
    }


def logro_ponderado(instrumento: str, niveles: dict[str, int]) -> float:
    """Combina % de logro (100/80/60/30/0) según los pesos oficiales.

    Raises:
        KeyError: si el instrumento no existe.
        ValueError: si falta un IE o un nivel no está en la pauta.
    """
    if instrumento not in INSTRUMENTOS:
        raise KeyError(f"instrumento desconocido: {instrumento}")
    pesos = INSTRUMENTOS[instrumento]
    faltan = set(pesos) - set(niveles)
    if faltan:
        raise ValueError(f"faltan indicadores: {sorted(faltan)}")
    for clave, valor in niveles.items():
        if clave in pesos and valor not in NIVELES_LOGRO:
            raise ValueError(
                f"{clave} = {valor}: el nivel debe ser uno de {NIVELES_LOGRO}"
            )
    return sum(niveles[clave] * peso for clave, peso in pesos.items())


def evaluar_institucional(
    instrumento: str,
    niveles: dict[str, int],
    exigencia: float = EXIGENCIA_POR_DEFECTO,
) -> dict:
    """Nota chilena a partir de la rúbrica institucional (% de logro)."""
    logro = logro_ponderado(instrumento, niveles)
    calificacion = nota(logro / 100.0 * PUNTAJE_MAXIMO, exigencia)
    return {
        "instrumento": instrumento,
        "logro": round(logro, 1),
        "nota": calificacion,
        "aprueba": calificacion >= NOTA_APROBACION,
    }


def detectar_instrumento(columnas: set[str]) -> str | None:
    """Infiere el instrumento a partir de las cabeceras del CSV.

    Returns:
        ``None`` si es la rúbrica formativa (IL1–IL5). El nombre del
        instrumento institucional si aparecen sus IE. Si varios calzan
        (EP1 ⊂ EFT), gana el que más IE usa.
    """
    if set(INDICADORES) <= columnas:
        return None
    candidatos = [
        (len(pesos), nombre)
        for nombre, pesos in INSTRUMENTOS.items()
        if set(pesos) <= columnas
    ]
    if not candidatos:
        raise ValueError(
            "el CSV debe tener columnas IL1…IL5 o las IE de ep1/ep2/ep3/eft"
        )
    candidatos.sort(reverse=True)
    return candidatos[0][1]


def evaluar_csv(
    ruta: Path,
    exigencia: float = EXIGENCIA_POR_DEFECTO,
    instrumento: str | None = None,
) -> list[dict]:
    """Lee un CSV de curso y devuelve una fila de resultado por estudiante."""
    with ruta.open(encoding="utf-8") as archivo:
        filas = list(csv.DictReader(archivo))
    if not filas:
        raise ValueError(f"{ruta} no tiene filas de datos")
    columnas = set(filas[0])
    elegido = instrumento if instrumento is not None else detectar_instrumento(columnas)
    resultados: list[dict] = []
    for fila in filas:
        nombre = fila.get("nombre", "?")
        if elegido is None:
            puntajes = {clave: float(fila[clave]) for clave in INDICADORES}
            detalle = evaluar(puntajes, exigencia)
            detalle["nombre"] = nombre
            detalle["instrumento"] = None
        else:
            pesos = INSTRUMENTOS[elegido]
            niveles = {clave: int(fila[clave]) for clave in pesos}
            detalle = evaluar_institucional(elegido, niveles, exigencia)
            detalle["nombre"] = nombre
        resultados.append(detalle)
    return resultados


def _imprimir_detalle(puntajes: dict[str, float], exigencia: float) -> None:
    resultado = evaluar(puntajes, exigencia)
    print(f"{'Indicador':<52} {'Peso':>6} {'Punt.':>6} {'Aporte':>7}")
    print("-" * 74)
    for clave, peso in INDICADORES.items():
        aporte = puntajes[clave] * peso
        print(f"{clave} · {DESCRIPCIONES[clave][:46]:<46} {peso:>5.0%} {puntajes[clave]:>6.1f} {aporte:>7.2f}")
    print("-" * 74)
    print(f"{'Puntaje ponderado (de 4,0)':<52} {'':>6} {'':>6} {resultado['puntaje_ponderado']:>7.2f}")
    print(f"{'Logro':<52} {'':>6} {'':>6} {resultado['porcentaje']:>6.1f}%")
    print(f"\nExigencia: {exigencia:.0%}   →   NOTA: {resultado['nota']:.1f}   "
          f"({'aprobado' if resultado['aprueba'] else 'reprobado'})")


def _procesar_csv(
    ruta: Path, exigencia: float, instrumento: str | None = None
) -> None:
    filas = evaluar_csv(ruta, exigencia, instrumento)
    print(f"{'Estudiante':<34} {'Logro':>7} {'Nota':>6}")
    print("-" * 49)
    notas = []
    for fila in filas:
        notas.append(fila["nota"])
        logro = fila.get("porcentaje", fila.get("logro"))
        marca = "" if fila["aprueba"] else "  ⚠"
        print(f"{fila.get('nombre', '?'):<34} {logro:>6.1f}% {fila['nota']:>6.1f}{marca}")
    print("-" * 49)
    aprobados = sum(1 for n in notas if n >= NOTA_APROBACION)
    print(f"{len(notas)} estudiantes · promedio {sum(notas) / len(notas):.2f} · "
          f"aprobación {100 * aprobados / len(notas):.0f}%")


def _imprimir_institucional(
    instrumento: str, niveles: dict[str, int], exigencia: float
) -> None:
    resultado = evaluar_institucional(instrumento, niveles, exigencia)
    pesos = INSTRUMENTOS[instrumento]
    print(f"{'Indicador':<8} {'Peso':>6} {'Logro':>7} {'Aporte':>8}")
    print("-" * 32)
    for clave, peso in pesos.items():
        aporte = niveles[clave] * peso
        print(f"{clave:<8} {peso:>5.0%} {niveles[clave]:>6}% {aporte:>7.1f}")
    print("-" * 32)
    print(f"Logro ponderado {resultado['logro']:.1f}%")
    print(
        f"Exigencia: {exigencia:.0%}   →   NOTA: {resultado['nota']:.1f}   "
        f"({'aprobado' if resultado['aprueba'] else 'reprobado'})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("puntajes", nargs="*", type=float,
                        help="cinco puntajes de 0 a 4, en el orden IL1 IL2 IL3 IL4 IL5")
    parser.add_argument("--exigencia", type=float, default=EXIGENCIA_POR_DEFECTO,
                        help="fracción del puntaje máximo que da nota 4,0 (por defecto 0.6)")
    parser.add_argument("--csv", type=Path, help="archivo con cabecera nombre,IL1,…,IL5")
    parser.add_argument(
        "--instrumento",
        choices=sorted(INSTRUMENTOS),
        help="pauta institucional: ep1, ep2, ep3 o eft",
    )
    parser.add_argument(
        "--ie",
        nargs="*",
        type=int,
        help="niveles 100/80/60/30/0, uno por IE del instrumento",
    )
    args = parser.parse_args()

    if args.csv:
        _procesar_csv(args.csv, args.exigencia, args.instrumento)
        return

    if args.instrumento:
        pesos = INSTRUMENTOS[args.instrumento]
        if not args.ie or len(args.ie) != len(pesos):
            parser.error(
                f"{args.instrumento} espera {len(pesos)} niveles "
                f"({' '.join(pesos)}), se recibieron {len(args.ie or [])}"
            )
        niveles = dict(zip(pesos, args.ie))
        _imprimir_institucional(args.instrumento, niveles, args.exigencia)
        return

    if len(args.puntajes) != len(INDICADORES):
        parser.error(f"se esperaban {len(INDICADORES)} puntajes (IL1…IL5), "
                     f"se recibieron {len(args.puntajes)}")

    puntajes = dict(zip(INDICADORES, args.puntajes))
    _imprimir_detalle(puntajes, args.exigencia)


if __name__ == "__main__":
    main()
