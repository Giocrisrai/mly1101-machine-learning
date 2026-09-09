# Evaluación formativa 1 · RA1 (calidad y ética)

**MLY1101** · 1 hora · individual · **sin computador y sin dataset extra**.
Se rinde **después** de la Act. 1.4. No abre Telco / Housing / Spotify: eso es la EP1.

Escribe con lápiz. Si una cifra no la recuerdas, di *cómo la medirías*.

Nombre: ________________________  Sección: ______  Fecha: __________

---

### 1. (IL1.1) El CSV del equipo vive solo en el notebook de una persona.

¿Qué se rompe el día de la EP1? Marca **una**.

- [ ] A. Nada: Colab guarda una copia automática en el fork de todos.
- [ ] B. La trazabilidad de la fuente y el trabajo colaborativo: no hay un origen compartido.
- [ ] C. El tipo `object` de pandas, porque GitHub no acepta CSV.
- [ ] D. La ética, porque un CSV en un laptop viola siempre la ley de protección de datos.

### 2. (IL1.2) Una columna se llama `TotalCharges`, `df.dtypes` dice `object` y `isna().sum()` da 0.

¿Está lista para un modelo numérico? Justifica en **dos** frases. Qué comando usarías para ver el problema.

### 3. (IL1.3) Encuentras el valor `-999` en una columna de velocidad.

¿Es un dato válido, un nulo oculto, o no se puede saber sin el diccionario? ¿Qué hace `eda.resumen_calidad` con los centinelas?

### 4. (IL1.3) Outlier por IQR con \(k = 1,5\).

Escribe la regla (con \(Q_1\), \(Q_3\)). Si **todas** las velocidades de un sensor caen “fuera”, ¿borras la columna? ¿Por qué sí o no?

### 5. (IL1.3) Partes el dataset al azar **por fila** para “entrenamiento y prueba”.

En un hilo con muchas filas del mismo viaje / mismo cliente / misma casa, ¿qué fuga estás fabricando? Una frase.

### 6. (IL1.4) El 100 % de las grabaciones de un lote salieron con clima `sunny`.

Eso es sesgo de **muestreo** o de **procesamiento**? Qué consecuencia tiene si el sistema opera con lluvia.

### 7. (IL1.4) Un compañero propone **tirar** la clase rara (pocas filas) “para que el EDA se vea más limpio”.

Sobre quién recae el costo si el modelo llega a producción. Dos frases.

### 8. (IL1.4) El dataset “no tiene nombres de personas”, solo `customerID`, comuna, sexo y fecha de nacimiento.

¿Hay riesgo de reidentificación? Qué mirarías (cardinalidad / combinaciones), sin inventar una ley de memoria.

### 9. El equipo declara: *“0 % de nulos ⇒ ya podemos entrenar.”*

Refuta con **un** contraejemplo de esta unidad (calidad o ética). No hace falta una cifra exacta.

### 10. (mapa del curso) RA1, RA2 y RA3.

Une con una línea:

| Resultado | Lo que **sí** es |
|---|---|
| RA1 | A. Hiperparámetros, ensamble y validación cruzada |
| RA2 | B. Recopilar datos de calidad, con ética |
| RA3 | C. Supervisado **y** no supervisado |

---

*No hay notebook en esta hora. La EP1 sí usa el caso oficial del equipo.*
