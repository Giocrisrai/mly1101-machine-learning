# Databricks Free Edition · Kedro y el hilo Waymo

**Para el alumno.** Cuenta **gratis** para sentir Spark. **No es evaluación.** No reemplaza
Colab (actividades) ni AWS Academy (más RAM del mismo repo) ni el `kedro run` local.

Community Edition se retiró en 2025. Usa
[Databricks Free Edition](https://www.databricks.com/learn/free-edition).

El contrato con este curso: **los nodos de Kedro no se reescriben.** Cambia el *dónde* viven
los archivos (un Volume). El *cómo* (pandas vs Spark) se prueba en **una** celda, no en el
grafo de 34 nodos.

Guía de tamaños y máquinas: [`recorrido_waymo.md`](recorrido_waymo.md).
Lab AWS: [`aws_academy_laboratorio.md`](aws_academy_laboratorio.md).
Notebook que lo enseña: `04_opcional_kedro_databricks.ipynb`.

---

## Qué encaja y qué no

| Sí | No |
|---|---|
| Importar los `.ipynb` del curso (14, 10, 04, 01) | Bajar el bucket Waymo (JPEG / nubes) |
| Subir **tu** parquet a un Volume **privado** | Hacer público el parquet (licencia Waymo) |
| Correr pandas en el driver (530 k filas caben) | Cluster EMR en Academy “porque es Spark” |
| Una celda PySpark con `count()` (evaluación perezosa) | Reescribir `src/eda.py` a Spark para la nota |
| Git Folder del repo (código) | `kedro run --pipeline waymo_real` **dentro** de Free Edition como entregable |
| Editar `catalog.yml` *en papel* (pandas → Spark/Delta) | Unity Catalog de producción, Jobs de pago, DLT |

`kedro run` se queda en **Colab / CloudShell / tu máquina**. Databricks es el sitio donde
ves que Spark no carga la tabla entera hasta el `count()`. Los nodos actuales hablan pandas
(`src/eda.py`); un DataFrame de Spark no los ejecuta tal cual.

---

## 1 · Crear la cuenta (gratis)

1. Abre [databricks.com/learn/free-edition](https://www.databricks.com/learn/free-edition).
2. Regístrate con correo personal o institucional (no hace falta tarjeta).
3. Entra al workspace. Si pide un compute: elige el **serverless / más chico**. Espera a que
   quede *Running* (1–3 min).
4. No conectes el workspace a la cuenta de AWS Academy. Son dos nubes distintas.

El cupo de horas es limitado. **Apaga** el compute al salir.

---

## 2 · Código: Git Folder (el mismo repo)

1. En el workspace: **Workspace → Create → Git Folder** (o *Repos*).
2. URL: `https://github.com/Giocrisrai/mly1101-machine-learning.git`
3. Rama `main`. Clone sparse no hace falta: el repo es liviano (el parquet Waymo **no** viene).

Quedas con notebooks, `src/`, `kedro_mly1101/` y el CSV de la pauta. Igual que un `git clone`.

**Probado 2026-09-08 en Free Edition:** el repo público clona sin PAT; queda la carpeta
`mly1101-machine-learning` en `main` (`datos/`, `docs/`, `kedro_mly1101/`, `notebooks/`,
`src/`). El parquet Waymo **no** viene en ese clone.

Si Git Folder falla (permisos Free Edition): **Import** del `.ipynb` suelto y sube el CSV a
mano. El análisis es el mismo.

---

## 3 · Datos: un Volume, no el bucket de Google

Waymo real **no** está en Git. Enlázalo como en Academy: el parquet que armaste en Colab
(notebook **14**).

1. **Catalog** (Unity Catalog) → el esquema `default` del workspace → **Create volume**
   `mly1101` (privado, tipo **Managed**; no External).
2. Upload: `detecciones_reales.parquet` (tu lote; ~20 MB con 40 segmentos).
3. Ruta en Free Edition (medida 2026-09-08; Volume vacío hasta que subas archivos):

```text
/Volumes/workspace/default/mly1101
```

En un notebook:

```python
from pathlib import Path
import pandas as pd

# 1) Volume (Databricks)
VOL = Path("/Volumes/workspace/default/mly1101")  # Free Edition; no lo hagas público
# 2) Git Folder (mismo layout que el clone)
REPO = Path.cwd()  # o Path("/Workspace/Repos/.../mly1101-machine-learning")

if (VOL / "detecciones_reales.parquet").exists():
    tabla = pd.read_parquet(VOL / "detecciones_reales.parquet")
    origen = "volume-real"
else:
    tabla = pd.read_parquet(REPO / "datos/waymo_real/detecciones_reales.parquet")
    origen = "repo-real"

print(origen, tabla.shape)
```

Sin parquet, **falla**: no hay CSV de pauta. `waymo.cargar_tabla_curso` es el mismo contrato.
**El Volume guarda la copia de trabajo, no un dataset distinto.** Colab, Kedro y Databricks
leen la misma tabla.

**No** montes `gs://waymo_open_dataset_v_2_0_1`. **No** ACL pública del Volume.

---

## 4 · Kedro: dónde corre cada pieza

```
Colab / CloudShell / local          Databricks Free Edition
──────────────────────────          ─────────────────────────
kedro run  (34 nodos, Perception v2) pandas en el driver (EDA)
catalog.yml  (rutas locales)        el YAML que *cambiarías* (no lo ejecutes aquí)
parameters.yml (decisiones)         las mismas reglas; no las dupliques
pickle del RF                       no lo sirvas como endpoint
```

El puente de ingeniería es **una edición de catálogo**, no un Dockerfile ni un Job:

```yaml
# Hoy (kedro_mly1101/conf/base/catalog.yml) — esto SÍ corre en CloudShell
detecciones_reales:
  type: pandas.ParquetDataset
  filepath: ../datos/waymo_real/detecciones_reales.parquet

# Lo que cambiarías si el equipo fuera a Spark de verdad
# detecciones_reales:
#   type: spark.SparkDataset
#   filepath: /Volumes/workspace/default/mly1101/detecciones
#   file_format: delta
```

Los nodos siguen siendo pandas. Pasar a Spark es reescribir el cuerpo (`filter` / `groupBy`),
no “activar Databricks” en Kedro. El notebook **04** lo dice con esa frase a propósito.

Para el proyecto de equipo: `kedro run --pipeline waymo_real` en Academy o local; Databricks
para el párrafo de escala del informe.

---

## 5 · Una celda Spark (para ver la pereza)

En un notebook del workspace, compute encendido:

```python
df = spark.read.parquet("/Volumes/workspace/default/mly1101/detecciones_reales.parquet")
# todavía no leyó las 530 k filas
print("plan:", df.filter("speed_mps > 1").count())  # aquí recorre el parquet
df.groupBy("object_type").count().show()
```

Si `spark` no existe (Colab, CloudShell): esa celda se salta. No instales un clúster Spark
en la EC2 del lab.

---

## 6 · Cerrar

1. Stop / disconnect del compute.
2. El parquet sigue en el Volume (privado). Copia a Drive si el workspace se caduca.
3. No lo bajes a un repo público.

Si Free Edition no deja crear Git Folder o Volumes, usa **Import + Upload** al FileStore y
avisa al docente. El lote de 8 sigue cabiendo en Colab.
