# Evaluación formativa 3 · RA3 (ajuste, ensamble y “¿ganó de verdad?”)

**MLY1101** · 1 hora · individual · **sin computador**.
Se rinde **después** de la Act. 3.3. Sin dataset extra.

Nombre: ________________________  Sección: ______  Fecha: __________

---

### 1. Mapa del RA3.

El RA3 **es**:

- [ ] A. Aprendizaje no supervisado (k-medias, PCA).
- [ ] B. Hiperparámetros, ensamble y validación cruzada.
- [ ] C. Recolectar más datos hasta que el F1 suba.
- [ ] D. Pasar de sklearn a una red neuronal.

### 2. Parámetro vs hiperparámetro.

En un árbol: ¿la partición de un nodo es parámetro o hiperparámetro? ¿Y `max_depth`?

### 3. Fuga en la búsqueda.

El equipo hace `GridSearchCV` **usando el conjunto de prueba** para elegir `C`. Qué está inflando y cómo lo harías bien (una frase).

### 4. Validación cruzada.

¿Qué pregunta responde un F1 de *test* único que la CV **no**, y al revés? No pidas la fórmula: la función de cada uno.

### 5. Ensamble.

Cien árboles **idénticos** (mismo seed, mismos datos, misma profundidad). ¿Es un ensamble que baja varianza? ¿Por qué?

### 6. “Ganó de verdad.”

La búsqueda sube el F1 de 0,51 a 0,52. El ruido estimado del experimento es 0,03. ¿Te cambias al modelo nuevo? Justifica.

### 7. EP3.

Une:

| IE | Qué pide |
|---|---|
| IE9 | A. Ensamble |
| IE10 | B. Selección y justificación |
| IE11 | C. Hiperparámetros |
| IE12 | D. Generalización (CV) |

### 8. Default vs ajustado.

Si el modelo por defecto ya cumple el KPI y el ajuste no supera el ruido, ¿qué entregas en la EP3? (No es “nada”.)

### 9. Métrica de selección.

El ranking interno usa exactitud; el KPI de negocio es recall de la clase cara. ¿Cuál usas para **elegir** el modelo del informe y por qué?

### 10. EFT.

Nombra **tres** cosas que el EFT exige y esta formativa **no** alcanza (defensa, ética, segundo supervisado, etc.). Una línea cada una.

---

*La calculadora institucional es* `python herramientas/calcular_nota.py --instrumento ep3 --ie …`
