# Guion de clase — Actividad 2.1 · CRISP-DM (RA2)

**Asignatura:** MLY1101 Machine Learning · **Duración:** 6 h (4 h guiadas + 2 h carta oficial)
**Material:** `notebooks/12_alumno_crispdm.ipynb` y `12_docente_crispdm.ipynb`.

---

## Antes de entrar

- Tener en pantalla el solucionario. Los alumnos abren el de alumno desde el badge de Colab.
- No proyectar las seis fases al inicio. El TODO 1 existe para que las construyan.
- Recordar el error prohibido: **RA2 = supervisado y no supervisado; RA3 = optimización.**

---

## Coreografía

| Bloque | Min | Qué preguntas antes de mostrar |
|---|---|---|
| 0 · Encuadre | 15 | *"¿En qué fase del proyecto estábamos en la 1.3?"* Casi nadie dice "datos". Dicen "EDA". |
| 1 · Seis fases | 40 | TODO 1 en silencio 3 min. TODO 2: exactitud alta + F1 bajo → ¿qué hacen? Si dicen "otro modelo", anótalo: es el anti-ejemplo del cierre. |
| 2 · Retroceso RA1 ⭐⭐ | 45 | Un hallazgo por fila, a mano alzada, **antes** de correr el TODO 3. El censo `sunny` divide la sala: datos vs negocio. Las dos lecturas valen si argumentan. |
| 3 · Negocio ⭐⭐ | 50 | Proyecta "el modelo tiene que ser bueno" y pide un sí/no. Luego corre el detector. Deja que escriban **su** pregunta antes de mostrar la de 2.2. |
| 4 · Carta Waymo | 40 | Quien truca `F1 ≥ 0` para pasar el `assert`: no lo avergüences en público; en la rúbrica es D4 nivel 2. |
| 5 · Mapa del curso | 25 | *"El clustering, ¿en qué RA va?"* Si alguien dice RA3, esa es la clase. |
| 6 · Caso oficial | 30 + 2 h | No se cierra en el aula. La Parcial 2 no tiene pregunta de negocio si esta tabla sale vacía. |

Si recortas, recorta el bloque 5 (el mapa está en código) y **no** el 3.

---

## Cierre en una frase

> Un proyecto sin criterio medible no se puede fallar. Y un proyecto que no se puede fallar
> no se puede terminar: solo se puede abandonar.
