"""Proyecto Kedro de MLY1101.

Los nodos de este proyecto **no reimplementan el análisis**: reutilizan las mismas
funciones puras de ``src/eda.py`` que los alumnos usan en los notebooks. Esa es la
razón de que ``src/eda.py`` se escribiera sin ``print``, sin gráficos y sin estado.

Para que ese ``import eda`` funcione hace falta que la carpeta ``src/`` del
repositorio esté en el ``sys.path``. Se hace aquí, en el ``__init__`` del paquete,
porque es lo primero que Kedro importa: cualquier otro sitio llegaría tarde.

Alternativa descartada: copiar ``eda.py`` dentro del proyecto Kedro. Tendríamos dos
copias que se desincronizan, que es exactamente lo que este repositorio evita en
todas partes.
"""

from __future__ import annotations

import sys
from pathlib import Path

__version__ = "2026.2.0"

# kedro_mly1101/src/kedro_mly1101/__init__.py -> subir cuatro niveles llega a la
# raíz del repositorio.
RAIZ_REPO = Path(__file__).resolve().parents[3]
RUTA_SRC = RAIZ_REPO / "src"

if str(RUTA_SRC) not in sys.path:
    sys.path.insert(0, str(RUTA_SRC))
