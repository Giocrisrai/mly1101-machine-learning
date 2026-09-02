# Programa oficial de MLY1101 — resumen operativo

Extraído del **Programa de Asignatura** de Duoc UC (Escuela de Informática y Telecomunicaciones)
y de las **Instrucciones y Pauta del EFT**, el 2026-08-27.

> **Para qué existe este documento.** Este repositorio se construyó primero y se contrastó con el
> programa después, y el contraste encontró desalineaciones reales. Esta es la referencia contra
> la que hay que verificar cualquier material nuevo **antes** de escribirlo.

---

## Antecedentes

| | |
|---|---|
| Sigla · Nombre | **MLY1101** · Machine Learning |
| Créditos | 4 SCT · 12 Duoc |
| Horas totales | **108** pedagógicas presenciales + 48 cronológicas de trabajo autónomo |
| Prerrequisito | MAT4152 |
| Ambiente | TAITE 9 (Taller de Alto Cómputo) |
| Competencia del perfil | C5 · Crear modelos predictivos complejos a partir del análisis de grandes volúmenes de datos |

---

## Los tres resultados de aprendizaje

### ⚠️ La confusión que hay que evitar

**El aprendizaje supervisado y el no supervisado están los DOS dentro del RA2.** El RA3 **no** es
"no supervisado": es optimización de hiperparámetros, ensamble y validación cruzada.

Es el error que ya se cometió una vez en este repositorio, y cuesta caro porque se propaga a los
notebooks, las pautas y los informes de los alumnos.

### RA1 · Ingeniería de Datos y Análisis Exploratorio

> *Recopila, a través de un trabajo colaborativo, sets de datos representativos y de calidad, a
> partir de distintas fuentes (texto plano, archivos CSV, otros) para responder a las necesidades
> del contexto de negocio, considerando aspectos éticos.*

| IL | Descripción | Actividad | Horas |
|---|---|---|---|
| **IL1.1** | Identifica diversas fuentes de datos y herramientas de trabajo colaborativo | Act 1.1 · Fuentes de Datos y Trabajo Colaborativo | 6 |
| **IL1.2** | Utiliza estructuras de datos en Python para el almacenamiento y manipulación eficiente de datasets | Act 1.2 · Estructuras de Datos y Almacenamiento en Python | 6 |
| **IL1.3** | Realiza un análisis exploratorio de datos (EDA) para detectar anomalías y asegurar la calidad de la información | Act 1.3 · Análisis Exploratorio de Datos (EDA) | 6 |
| **IL1.4** | Evalúa el impacto ético y los sesgos en los datos recopilados, garantizando el cumplimiento de estándares de privacidad | Act 1.4 · Impacto Ético, Sesgos y Privacidad | 5 |

Más **Eva Formativa 1** (cuestionario de calidad y ética, 1 h) y **Eva Parcial 1** (6 h).

### RA2 · Implementación y Análisis de Modelos de Machine Learning

> *Aplica modelos estadísticos al conjunto de datos procesados para interpretarlos, utilizando
> metodologías ágiles, con la finalidad de obtener conocimientos relevantes que permitan
> responder a las necesidades del contexto de negocio, considerando aspectos éticos.*

| IL | Descripción | Actividad | Horas |
|---|---|---|---|
| **IL2.1** | Implementa metodologías de trabajo (como **CRISP-DM**) para estructurar el desarrollo del modelo | Act 2.1 · Gestión de Proyectos con CRISP-DM | 6 |
| **IL2.2** | Construye modelos de aprendizaje supervisado para problemas de **regresión y clasificación** | Act 2.2 · Modelamiento Supervisado | 6 |
| **IL2.3** | Elabora algoritmos de aprendizaje no supervisado para descubrir patrones ocultos | Act 2.3 · Modelamiento No Supervisado | **12** |
| **IL2.4** | Interpreta los resultados del desempeño del modelo, traduciendo métricas técnicas a conocimientos para la organización | Act 2.4 · Interpretación y Métricas de Desempeño | 5 |

Más **Eva Formativa 2** (1 h) y **Eva Parcial 2** (6 h).

### RA3 · Optimización y Ensamble de Modelos Avanzados

> *Elabora soluciones avanzadas de aprendizaje automático mediante la optimización de
> hiperparámetros, técnicas de ensamble y validación cruzada, para garantizar la precisión y
> generalización del modelo frente a objetivos de negocio complejos.*

| IL | Descripción | Actividad | Horas |
|---|---|---|---|
| **IL3.1** | Aplica estrategias de ajuste de hiperparámetros | Act 3.1 · Ajuste de Hiperparámetros | 6 |
| **IL3.2** | Desarrolla modelos basados en técnicas de **ensamble** para mitigar sesgo y varianza | Act 3.2 · Modelos de Ensamble | 6 |
| **IL3.3** | Evalúa la generalización mediante **validación cruzada** y métricas avanzadas | Act 3.3 · Robustez y Selección de Modelos | 11 |
| **IL3.4** | Sustenta la selección de la solución óptima mediante comparación cuantitativa de modelos | *(misma actividad)* | |

Más **Eva Formativa 3** (1 h) y **Eva Parcial 3** (6 h).

### Evaluación Final Transversal — 12 h

---

## Evaluaciones y ponderación

| Evaluación | Situación | Ponderación parcial | Ponderación final |
|---|---|---|---|
| **Parcial 1** · Comprensión y preparación de los datos | Presentación | **30 %** | **60 %** en conjunto |
| **Parcial 2** · Construcción e interpretación de modelos | Presentación | **40 %** | |
| **Parcial 3** · Optimización y comparación de soluciones | Presentación | **30 %** | |
| **EFT** | Presentación grupal, **defensa individual** | — | **40 %** |

**La conversión a nota usa exigencia del 60 %**, según el Reglamento Académico. La calculadora
del repositorio (`herramientas/calcular_nota.py`) implementa esa escala.

### La rúbrica institucional no usa la escala 0–4

Usa **porcentaje de logro** sobre indicadores ponderados:

| Categoría | % logro |
|---|---|
| Muy buen desempeño | 100 % |
| Buen desempeño | 80 % |
| Desempeño aceptable | 60 % |
| Desempeño incipiente | 30 % |
| Desempeño no logrado | 0 % |

El EFT tiene **12 indicadores de evaluación (IE1–IE12)** con ponderaciones propias (IE1–IE4 al
5 %, IE5 al 10 %, y así). Las pautas de `docs/rubrica_*.md` de este repositorio son
**formativas** y usan otra escala; no sustituyen al instrumento institucional.

---

## Los casos oficiales

Las evaluaciones parciales y el EFT se rinden sobre **uno de estos tres casos**, no sobre el
dataset de las actividades:

| Caso | Problema | Tipo |
|---|---|---|
| **A** · Telco Customer Churn | Predicción de abandono de clientes | Clasificación |
| **B** · House Prices | Predicción de precios de viviendas | Regresión |
| **C** · Spotify Tracks | Predicción de popularidad de canciones | Regresión / clasificación |

Los datasets vienen en `EV PARCIALES MLY1101.zip` y en el anexo del EFT, junto con un **notebook
institucional resuelto** de Telco Churn para el docente.

> **Decisión de este repositorio (2026-08-27):** las **actividades** conservan el hilo único del
> dataset de detecciones LiDAR tipo Waymo; las **evaluaciones** usan los casos oficiales. Que el
> caso de aprendizaje y el de evaluación sean distintos es deliberado: obliga a demostrar que el
> método se traslada, en vez de reproducir un ejercicio memorizado.

---

## Qué exige el EFT

**Grupal con defensa individual.** 10 h de desarrollo, presentación de 10 min en la semana 18.
Preguntas cruzadas: cada estudiante responde por cualquier parte del trabajo, no solo la suya.

**Como mínimo debe incluir:**

- Dos modelos de **aprendizaje supervisado** comparados, acordes al problema.
- Una técnica de **aprendizaje no supervisado** para descubrir patrones o segmentos.
- Justificación del mejor modelo supervisado mediante comparación de métricas.
- Explicación del valor que aportó el análisis no supervisado al problema de negocio.
- **Optimización de hiperparámetros** y técnicas de validación.
- Análisis de **sesgos, limitaciones y consideraciones éticas**.

**Entregables:**

| Entregable | Detalle |
|---|---|
| **Informe técnico en Markdown** (`.md`) | Documento reproducible tipo README: problema, objetivos, **KPIs**, fuentes, EDA, **metodología CRISP-DM**, modelos, optimización, evaluación, conclusiones y limitaciones |
| **Notebook** (`.ipynb`) | Documentado, organizado y **completamente ejecutable** |
| **Datos y archivos complementarios** | De modo que la solución corra sin modificaciones |
| **Estructura de proyecto profesional** | `data/`, `notebooks/`, `models/`, `images/`, `README.md` |

> Ese último punto conecta directamente con lo que ya hace este repositorio: estructura
> versionada, pipeline reproducible y README que explica cómo ejecutarlo. El EFT pide exactamente
> el hábito que las actividades practican.

---

## Estado de cobertura del repositorio

| Elemento | Estado |
|---|---|
| Act 1.1 · 1.2 · 1.3 · 1.4 | ✅ notebooks alumno + docente, verificados |
| Act 2.1 · CRISP-DM | ✅ notebooks alumno + docente, pauta formativa |
| Act 2.2 · 2.3 | ✅ notebooks alumno + docente, verificados |
| Act 2.4 (interpretación) | ✅ notebooks alumno + docente, pauta formativa |
| Act 3.1 · 3.2 · 3.3 (todo el RA3) | ✅ notebooks alumno + docente, verificados |
| Evaluaciones formativas 1, 2 y 3 | ⏳ |
| Parciales 1, 2 y 3 · EFT | ⏳ — deben construirse sobre los casos oficiales |
| Pipeline reproducible (`kedro_mly1101/`) | ✅ RA1, RA2 y RA3 |
