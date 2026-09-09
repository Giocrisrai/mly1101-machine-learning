# Evaluaciones oficiales · MLY1101

**No es el hilo del curso.** Telco, House Prices (Ames) y Spotify Tracks son el
instrumento que entrega Duoc (zip de coordinación): fueron la forma inicial de tarea.
Las Act. 1.1–3.3 y `kedro run` se fueron a Perception v2 a propósito. Las tres
parciales y el EFT se rinden sobre **un** de esos casos oficiales, no sobre Waymo.

Los CSV y las pautas PDF los entrega la coordinación (`EV PARCIALES MLY1101.zip`,
anexo del EFT). **No están en este repositorio.**

## Preparar los datos (una vez, en tu máquina)

```bash
uv run python herramientas/preparar_casos_oficiales.py
# o:  --zip ~/Downloads/EV\ PARCIALES\ MLY1101.zip.zip
```

Quedan gitignored en `datos/evaluaciones/{telco,housing,spotify}/`.

Medido 2026-09-09: Telco **7.043 × 21** · Housing **2.930 × 82** · Spotify **114.000 × 21**.

## Calcular la nota institucional

Niveles de la pauta: **100 / 80 / 60 / 30 / 0**. Exigencia 60 % → nota 4,0.

```bash
uv run python herramientas/calcular_nota.py --instrumento ep1 --ie 80 60 100 60
uv run python herramientas/calcular_nota.py --csv docs/ejemplo_notas_ep1.csv
uv run python herramientas/calcular_nota.py --csv docs/ejemplo_notas_ep2.csv
uv run python herramientas/calcular_nota.py --csv docs/ejemplo_notas_ep3.csv
uv run python herramientas/calcular_nota.py --csv docs/ejemplo_notas_eft.csv
uv run python herramientas/calcular_nota.py --instrumento ep2 --ie 80 80 60 80
uv run python herramientas/calcular_nota.py --instrumento ep3 --ie 60 80 80 80
uv run python herramientas/calcular_nota.py --instrumento eft --ie \
  80 80 80 60  80 80 60 80  80 60 80 80
```

Orden de `--ie`: los IE de ese instrumento (EP1 = IE1–IE4, EP2 = IE5–IE8, EP3 = IE9–IE12,
EFT = IE1–IE12).

Las pautas D1–D5 de `docs/rubrica_*.md` siguen siendo **formativas**.

## Qué pide cada instancia (resumen)

| Instancia | Horas sala | Qué no puede faltar |
|---|---|---|
| **Formativa 1** | 1 h | [`docs/formativa_1.md`](formativa_1.md) · pauta [`formativa_1_pauta.md`](formativa_1_pauta.md) |
| **Formativa 2** | 1 h | [`docs/formativa_2.md`](formativa_2.md) · pauta [`formativa_2_pauta.md`](formativa_2_pauta.md) |
| **Formativa 3** | 1 h | [`docs/formativa_3.md`](formativa_3.md) · pauta [`formativa_3_pauta.md`](formativa_3_pauta.md) |
| **EP1** | 5 h | Fuentes, preparación, EDA, ética. Sin modelos. Guion: [`guion_ep1.md`](guion_ep1.md) |
| **EP2** | 6 h | CRISP-DM, **dos** supervisados, **un** no supervisado, interpretación. [`guion_ep2.md`](guion_ep2.md) |
| **EP3** | 6 h | Hiperparámetros, ensamble, CV, justificación. [`guion_ep3.md`](guion_ep3.md) |
| **EFT** | 12 h | Los doce IE; defensa individual. Guion: [`guion_eft.md`](guion_eft.md) |

Plantilla de notebook (un caso, falla si falta el CSV):
`notebooks/15_alumno_evaluacion.ipynb` / `15_docente_evaluacion.ipynb`.
Carga: `casos.cargar_caso` y `casos.matriz_xy` en `src/casos.py`.
Spotify con `muestra=` recorta **álbumes enteros** (no parte discos: el split de la EP2 lo prohíbe).

En Colab: el notebook 15 crea `datos/evaluaciones/{telco|housing|spotify}/`.
Sube el CSV del caso ahí. No pide cuenta Waymo. No uses el navegador embebido del IDE.

RAM (medido 2026-09-09): Telco X 7.043×30 (~2 MB) · Housing 2.930×276 (~6 MB) ·
Spotify 113.999×127 (~116 MB). **Colab free alcanza los tres.** El recorte a 8.000
filas en Spotify es para que EP2+EP3 terminen en segundos, no porque la matriz no quepa.

Entrega común: Markdown + notebook ejecutable + datos + carpeta profesional.
Presentación 10 min, preguntas cruzadas.

## Licencia

No copies el zip ni el notebook institucional Telco a GitHub. El repo público es CC BY-NC-SA 4.0;
el paquete Duoc no lo es.
