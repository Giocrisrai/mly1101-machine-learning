# Sesgo de muestreo en el Waymo Open Dataset

Medido el 2026-08-16 con `herramientas/analizar_sesgo_waymo.py`.

**Qué contiene esto y qué no.** Son estadísticas agregadas calculadas sobre el Waymo Open
Dataset; **no** contiene datos de Waymo. Reproducirlo requiere aceptar los términos en
<https://waymo.com/open/terms/> y descargar los segmentos por cuenta propia:

```bash
python herramientas/descargar_waymo.py --muestra 250 --solo-stats   # ~6 MB
python herramientas/descargar_waymo.py --muestra 40                 # ~45 MB
python herramientas/analizar_sesgo_waymo.py --markdown docs/sesgo_waymo.md
```

**Advertencia sobre la muestra:** son los primeros 250 segmentos del listado del bucket, no una
muestra aleatoria. Los nombres son identificadores, así que el orden es arbitrario en la
práctica, pero las cifras son indicativas y no un censo del dataset completo.

```
Muestra de detecciones: 40 segmentos, 530.396 detecciones
Muestra de condiciones: 250 segmentos (solo stats)

Segmentos por momento del día
-----------------------------
           segmentos
momento             
Day              200
Dawn/Dusk         26
Night             24


Segmentos por clima
-------------------
       segmentos
clima           
sunny        249
rain           1


Segmentos por ubicación
-----------------------
                segmentos
lugar                    
location_sf           116
location_phx          109
location_other         25


Composición de objetos por momento del día (%)
----------------------------------------------
tipo         cyclist  pedestrian   sign  vehicle
time_of_day                                     
Dawn/Dusk       0.00        6.81  27.70    65.49
Day             0.47       26.57  25.80    47.15
Night           0.38       13.73  31.88    54.02


Composición de objetos por clima (%)
------------------------------------
tipo     cyclist  pedestrian   sign  vehicle
weather                                     
sunny       0.45       24.67  26.46    48.43


Calidad de la detección por momento del día (agregando TODAS las detecciones)
-----------------------------------------------------------------------------
             detecciones  puntos_medianos  pct_dificiles  pct_sin_puntos
time_of_day                                                             
Dawn/Dusk          17439             51.0           5.25            9.36
Day               461090             35.0          13.19            9.36
Night              51867             43.0           7.04            8.77


Lo mismo, pero calculado POR SEGMENTO (tasa de detecciones difíciles, %)
------------------------------------------------------------------------
             segmentos  mediana  minimo  maximo
time_of_day                                    
Dawn/Dusk            2     3.42    0.02    6.81
Day                 32     4.81    0.00   53.81
Night                6     4.25    0.00   13.17


Ojo con las dos tablas anteriores: si no coinciden, el promedio global está dominado
por unos pocos segmentos con muchas detecciones. La unidad de análisis es el segmento,
no la detección.


Peatones + ciclistas sobre el total (%)
---------------------------------------
               pct
time_of_day       
Dawn/Dusk     6.81
Day          27.05
Night        14.11

```
