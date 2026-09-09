# Evaluaciones oficiales · MLY1101

**No es el hilo Waymo.** Las Act. 1.1–3.3 y `kedro run` usan Perception v2. Las tres
parciales y el EFT se rinden sobre **un** caso: Telco Churn, House Prices (Ames) o
Spotify Tracks.

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

En Colab: clone + **sube el CSV** a `datos/evaluaciones/{telco|housing|spotify}/`.
No pide cuenta Waymo. No uses el navegador embebido del IDE.

Entrega común: Markdown + notebook ejecutable + datos + carpeta profesional.
Presentación 10 min, preguntas cruzadas.

## Licencia

No copies el zip ni el notebook institucional Telco a GitHub. El repo público es CC BY-NC-SA 4.0;
el paquete Duoc no lo es.
