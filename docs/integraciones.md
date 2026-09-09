# Integraciones · dónde corre cada cosa (MLY1101)

Una sola tabla. Si un README, un notebook o un lab dice otra cosa, **gana este archivo**.

Hay **dos orígenes de datos**. No se mezclan y no son dos versiones del mismo
proyecto: Waymo es lo que diseñamos para aprender; Telco / Housing / Spotify es lo
que Duoc ya tenía como instrumento de las parciales y el EFT.

| Origen | Qué es | Dónde vive | Qué notebooks |
|---|---|---|---|
| **Curso (actividades + Kedro)** | Perception v2 (`lidar_box` + `stats`) | `datos/waymo_real/` gitignored | 14, 10, 00, 01–13, 04 |
| **Instrumento Duoc (parciales y EFT)** | Telco / Housing / Spotify | `datos/evaluaciones/` gitignored | **15** |

Sin el archivo en disco, el código **falla** y dice cómo obtenerlo. No hay CSV sintético.

---

## Dónde computar (medido 2026-09-08 / 09)

| Dónde | Para qué | Qué **no** es |
|---|---|---|
| **Google Colab** | Clase: actividades y lote de 8 segmentos (~8 MB). Auth Waymo. | Evaluación. Navegador embebido del IDE. |
| **Tu máquina** (`uv`) | Todo el repo, `kedro run` 34 nodos, pytest, notebook 15 | Obligatorio para el alumno (Colab alcanza el lote) |
| **GitHub Actions** | CI: ruff + pytest (salta Waymo y CSV oficiales) + notebooks al día | No baja parquet ni el zip Duoc |
| **AWS Academy** CloudShell | RAM/disco extra, **mismo** `git clone`. USD 50. `us-east-1` | EC2, EMR, Bedrock, Motion/v1. **No es evaluación.** |
| **Databricks Free Edition** | Git Folder del repo + Volume **vacío** hasta que **tú** subas el parquet. Un `count()` Spark | `kedro run` aquí. Community Edition (retirada). **No es evaluación.** |

Guías largas (no las dupliques): [Colab / Waymo](recorrido_waymo.md) · [Academy](aws_academy_laboratorio.md) · [Databricks](databricks_free.md) · [evaluaciones](evaluaciones.md) · [productos Waymo](productos_waymo.md).

---

## Colab (el camino de clase)

1. *Archivo → Guardar una copia en Drive* **antes** de autenticar con Google.
2. Corre las celdas en **Chrome, Safari o Brave**, en una ventana normal.
3. El navegador embebido del IDE (Cursor, VS Code Simple Browser, etc.) abre el *Allow*
   en otra pestaña: el kernel recibe `MessageError` y **no** queda autenticado
   (comprobado 2026-09-08).
4. Misma cuenta Google que aceptó [waymo.com/open/download](https://waymo.com/open/download/).
5. Notebook **15** (evaluaciones): **no** pide GCS. Sube el CSV del caso a
   `datos/evaluaciones/{telco\|housing\|spotify}/` después del clone.

---

## GitHub

- Repo público: [Giocrisrai/mly1101-machine-learning](https://github.com/Giocrisrai/mly1101-machine-learning).
- CI: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml). Verde = el clon **sin**
  parquet ni zip Duoc sigue siendo coherente.
- No subas `datos/waymo_real/` ni `datos/evaluaciones/` ni el notebook institucional de Telco.

---

## AWS Academy (Learner Lab 183052)

- Consola **N. Virginia (`us-east-1`)**. Usuario federado `voclabs/…`.
- **CloudShell** por defecto (`$HOME` ≈ 1 GB). Pega comandos en el navegador de verdad.
  *Actions → Run command* abre otro entorno y un modal que a veces no cierra.
- No bajes Motion ni Perception v1: un shard satura el disco.
- SageMaker `medium`/`large` solo si Colab/CloudShell no alcanzan para 530 k filas + RA3.
- `kedro run` **sí** cabe aquí. Spark de exploración: **Databricks**, no EMR.

```bash
git clone --depth 1 https://github.com/Giocrisrai/mly1101-machine-learning.git ~/mly1101
```

---

## Databricks Free Edition

Verificado 2026-09-08:

| Pieza | Resultado |
|---|---|
| Git Folder del repo público, rama `main`, **sin PAT** | OK |
| Volume managed `workspace.default.mly1101` → `/Volumes/workspace/default/mly1101` | **LIST 0 filas** (correcto: el parquet no va en git) |
| SQL warehouse *Serverless Starter Warehouse* | OK |
| `kedro run` dentro del workspace | **No.** El grafo de 34 nodos se queda en local / Colab / CloudShell |

Sube **tu** `detecciones_reales.parquet` al Volume si quieres un `spark.read.parquet(…).count()`.
No montes `gs://waymo_…`.

---

## Evaluaciones (instrumento Duoc, no el curso)

```bash
uv run python herramientas/preparar_casos_oficiales.py
uv run python herramientas/calcular_nota.py --instrumento ep1 --ie 80 60 100 60
```

Plantilla: `notebooks/15_alumno_evaluacion.ipynb`. Carga: `casos.cargar_caso` /
`casos.matriz_xy`. Detalle: [`evaluaciones.md`](evaluaciones.md).

---

## Comprobar que este mapa no se desactualizó

`tests/test_integraciones.py` falla si un notebook o un README vuelve a decir que el EFT
está pendiente, que Databricks es solo conceptual, o que faltan las evaluaciones oficiales.
