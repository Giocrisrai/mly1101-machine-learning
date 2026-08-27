"""Convierte los puntajes de la rúbrica de la EA1 en nota de 1,0 a 7,0.

Aplica la escala chilena con **exigencia configurable** (60 % por defecto): la
nota de aprobación (4,0) se alcanza al obtener ese porcentaje del puntaje
máximo, y la escala es lineal a ambos lados de ese punto.

    Sea P el puntaje ponderado (0 a 4), Pmax = 4 y e la exigencia.
    Sea U = e · Pmax el puntaje de aprobación.

        P < U :  nota = 1,0 + 3,0 · P / U
        P ≥ U :  nota = 4,0 + 3,0 · (P − U) / (Pmax − U)

    Con e = 0,60:  P = 0 → 1,0    P = 2,4 → 4,0    P = 4 → 7,0

Uso:

    python herramientas/calcular_nota.py 3 4 3 2 4        # IL1 IL2 IL3 IL4 IL5
    python herramientas/calcular_nota.py 3 4 3 2 4 --exigencia 0.5
    python herramientas/calcular_nota.py --csv notas.csv  # curso completo

El CSV debe tener cabecera ``nombre,IL1,IL2,IL3,IL4,IL5``.
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


def _procesar_csv(ruta: Path, exigencia: float) -> None:
    with ruta.open(encoding="utf-8") as archivo:
        filas = list(csv.DictReader(archivo))
    if not filas:
        raise SystemExit(f"{ruta} no tiene filas de datos")
    faltan = set(INDICADORES) - set(filas[0])
    if faltan:
        raise SystemExit(f"el CSV debe tener las columnas {sorted(INDICADORES)}; faltan {sorted(faltan)}")

    print(f"{'Estudiante':<34} {'Logro':>7} {'Nota':>6}")
    print("-" * 49)
    notas = []
    for fila in filas:
        puntajes = {clave: float(fila[clave]) for clave in INDICADORES}
        resultado = evaluar(puntajes, exigencia)
        notas.append(resultado["nota"])
        marca = "" if resultado["aprueba"] else "  ⚠"
        print(f"{fila.get('nombre', '?'):<34} {resultado['porcentaje']:>6.1f}% {resultado['nota']:>6.1f}{marca}")
    print("-" * 49)
    aprobados = sum(1 for n in notas if n >= NOTA_APROBACION)
    print(f"{len(notas)} estudiantes · promedio {sum(notas) / len(notas):.2f} · "
          f"aprobación {100 * aprobados / len(notas):.0f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("puntajes", nargs="*", type=float,
                        help="cinco puntajes de 0 a 4, en el orden IL1 IL2 IL3 IL4 IL5")
    parser.add_argument("--exigencia", type=float, default=EXIGENCIA_POR_DEFECTO,
                        help="fracción del puntaje máximo que da nota 4,0 (por defecto 0.6)")
    parser.add_argument("--csv", type=Path, help="archivo con cabecera nombre,IL1,…,IL5")
    args = parser.parse_args()

    if args.csv:
        _procesar_csv(args.csv, args.exigencia)
        return

    if len(args.puntajes) != len(INDICADORES):
        parser.error(f"se esperaban {len(INDICADORES)} puntajes (IL1…IL5), "
                     f"se recibieron {len(args.puntajes)}")

    puntajes = dict(zip(INDICADORES, args.puntajes))
    _imprimir_detalle(puntajes, args.exigencia)


if __name__ == "__main__":
    main()
