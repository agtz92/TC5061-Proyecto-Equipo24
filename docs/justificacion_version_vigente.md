# Justificación de la versión vigente del modelo P1 (A/B testing)

Cierra el ciclo de monitoreo: el [reporte de drift](reporte_drift.md) detectó cambios en los datos de
2026, la [interpretación](interpretacion_drift.md) los atribuyó sobre todo a la expansión de la red, y la
[versión alternativa](version_alternativa.md) reentrenó el modelo con datos recientes. Aquí se comparan
ambas versiones sobre el mismo conjunto de evaluación y se decide cuál queda como versión vigente
(alias `@champion`, el que carga la API).

## 1. Versiones comparadas

Ambas son versiones del mismo modelo registrado, `mibici-equipo24-p1-demanda`, y solo difieren en la
ventana de entrenamiento: el estimador, los hiperparámetros, las variables, el tamaño de muestra y la
semilla son los mismos.

| | Champion (vigente) | Challenger (alternativa) |
|---|---|---|
| Versión y alias | v1, `@champion` | v2, `@challenger` |
| Corrida de origen | `p1-hgb-it400-lr0.1-d16-leaf50` | `p1-hgb-retrain-2026-06` |
| Estimador | HistGradientBoosting, pérdida Poisson | Igual |
| Hiperparámetros | 400 iteraciones, learning rate 0.1, profundidad 16, 50 muestras por hoja | Igual |
| Entrenamiento | 2022-01-01 a 2025-06-30 | 2022-01-01 a **2026-06-30** |
| Estaciones vistas en entrenamiento | 371 | 413 |

**Hipótesis.** Agregar el primer semestre de 2026 le da al modelo ejemplos de las estaciones abiertas en
2026, donde el champion tiene su mayor error, sin perjudicar a las estaciones existentes.

**Por qué hace falta este análisis.** Las cifras que ya existían no se pueden comparar entre sí: el
champion se midió de enero a septiembre de 2026 (WAPE de 61.4%), y el challenger de julio a septiembre
(60.5%). Son periodos con distinta mezcla de meses y estaciones, así que la diferencia no dice cuál es
mejor.

## 2. Diseño del A/B testing

### Conjunto de evaluación

- **Periodo:** 2026-07-01 00:00 a 2026-09-30 23:00 (`AB_EVAL_START` y `AB_EVAL_END` en
  [config.py](../mibici/config.py)), con todas las estación-horas, sin muestreo.
- **Mismas filas para ambas versiones:** se carga el conjunto una sola vez (`load_ab_eval_frame`) y cada
  versión predice sobre esas filas. La comparación es pareada: cada estación-hora tiene las dos
  predicciones y el valor real.
- **Sin fuga de información:** ninguna versión usó este periodo para entrenar ni para elegir
  hiperparámetros. El script lo verifica antes de evaluar (`validate_windows`) con el fin de
  entrenamiento que cada versión registró en MLflow, y se detiene si hay traslape.
- **Segmentos:** cada estación se clasifica como *existente* (ya tenía historia al cerrar el
  entrenamiento del champion, en junio de 2025) o *nueva*. Es el mismo corte de la interpretación del
  drift.

### Por qué una comparación fuera de línea y no tráfico real dividido

En un A/B en línea, la mitad de las peticiones iría a cada versión y las diferencias de demanda entre
grupos se mezclarían con las del modelo. Como en este problema sí tenemos el valor real de cada hora
(los viajes de los datos abiertos), es mejor que ambas versiones predigan **todas** las filas: se elimina
esa fuente de varianza y no se expone a la operación a un modelo sin validar. La desventaja es que no
mide efectos operativos (por ejemplo, cómo cambia el rebalanceo con cada modelo). Eso quedaría para un
despliegue gradual después de esta decisión.

### Métricas

| Métrica | Para qué sirve |
|---|---|
| **WAPE** (error absoluto total / viajes totales) | Métrica principal. Es la que se usó para elegir el champion (`val_wape`) y se interpreta en viajes, la unidad del operador. |
| WAPE por segmento | El drift se concentró en las estaciones nuevas; un promedio global podría ocultar que un segmento empeora. |
| Sesgo (predicción − real) | Una sobrepredicción sistemática manda bicicletas o camiones donde no hacen falta. |
| MAE, RMSE, devianza Poisson | Complementarias, para confirmar que la conclusión no depende de una sola métrica. |
| WAPE del baseline (misma hora de la semana anterior) | Mínimo que cualquier versión debe superar. |

### Incertidumbre

Los errores de una misma estación están correlacionados entre sí, y lo mismo pasa con los de un mismo
día (clima, eventos, días festivos). Tratar cada estación-hora como independiente daría intervalos
demasiado estrechos. Por eso la diferencia de WAPE se acompaña de dos intervalos bootstrap al 95%, con
2,000 remuestreos (`AB_BOOTSTRAP_REPS`):

- **Por estación:** se remuestrean estaciones completas.
- **Por día:** se remuestrean días completos.

La decisión usa el más conservador de los dos. Como apoyo se reporta la prueba de Diebold-Mariano sobre
el error absoluto diario, la misma que se usó en la entrega 2 para comparar contra el baseline.

## 3. Criterios de decisión

Se fijaron en [config.py](../mibici/config.py) **antes** de ejecutar la comparación, para que la regla no
se acomode a los resultados. El challenger reemplaza al champion solo si cumple **todos**. Ante un empate
o una mejora dudosa se conserva el champion: cambiar de versión tiene costo (redespliegue, riesgo de
regresión, invalidar comparaciones históricas) y no aporta nada si no hay mejora demostrada.

| # | Criterio | Regla | Por qué |
|---|---|---|---|
| 1 | Mejora práctica | WAPE champion − WAPE challenger ≥ 0.5 pp (`AB_MIN_MEJORA_WAPE`) | Con un WAPE cercano a 60% y un piso de ruido Poisson cercano a 50%, una mejora menor a medio punto no cambia ninguna decisión de rebalanceo. |
| 2 | Mejora significativa | Límite inferior del IC 95% (el más conservador) > 0 | Descarta que la mejora sea producto del azar del periodo evaluado. |
| 3 | Sin degradación por segmento | En estaciones existentes y en nuevas, el WAPE del challenger no puede superar al del champion por más de 0.5 pp (`AB_TOLERANCIA_SEGMENTO`) | Las estaciones existentes concentran cerca de 97% de los viajes; una mejora en las nuevas no justifica empeorar el núcleo de la red, ni al revés. |
| 4 | Sesgo controlado | El sesgo absoluto del challenger no puede crecer más de 0.02 viajes/h (`AB_TOLERANCIA_SESGO`) | El champion tiene un sesgo prácticamente nulo y esa propiedad no debe perderse. |
| 5 | Supera al baseline | WAPE challenger < WAPE de la regla de la semana anterior | Requisito mínimo para cualquier modelo en producción. |

## 4. Resultados y decisión

Esta sección la genera el script con las cifras reales de la ejecución; no se edita a mano. Al volver a
ejecutarlo, se reemplaza solo el contenido entre las marcas.

<!-- RESULTADOS_AB:INICIO (generado por mibici.modeling.p1_ab_test; no editar) -->

_Generado el 2026-11-15 11:23 con `python -m mibici.modeling.train ab-test` (corrida de MLflow `p1-ab-test-v1-vs-v2`, run `a8dab263aed2488b8b3f1215926f44af`)._

### Versiones comparadas

| | Champion | Challenger |
|---|---|---|
| Versión del registry | v1 (`@champion`) | v2 (`@challenger`) |
| Corrida de origen | `p1-hgb-it400-lr0.1-d16-leaf50` | `p1-hgb-retrain-2026-06` |
| Estimador | hgb | hgb |
| Fin del entrenamiento | 2025-06-30 23:00 | 2026-06-30 23:00 |

Conjunto de evaluación: 2026-07-01 00:00 a 2026-09-30 23:00, 1,001,510 estación-horas idénticas para ambas versiones (813,562 de estaciones existentes y 187,948 de estaciones nuevas).

### Métricas en el mismo conjunto

| Métrica | Champion | Challenger | Diferencia (challenger - champion) |
|---|---|---|---|
| WAPE total | 60.62% | 60.45% | -0.18 pp |
| WAPE estaciones existentes | 57.76% | 57.64% | -0.13 pp |
| WAPE estaciones nuevas | 100.51% | 99.66% | -0.85 pp |
| MAE (viajes/h) | 0.7378 | 0.7356 | -0.0022 |
| RMSE (viajes/h) | 1.3707 | 1.3643 | -0.0064 |
| Devianza Poisson | 0.9284 | 0.9231 | -0.0053 |
| Sesgo (viajes/h) | -0.0367 | -0.0397 | -0.0030 |
| Sesgo estaciones existentes | -0.0483 | -0.0505 | -0.0021 |
| Sesgo estaciones nuevas | +0.0136 | +0.0067 | -0.0069 |
| WAPE baseline (semana anterior) | 73.64% | 73.64% | (mismo baseline) |

### Diferencia pareada de WAPE (champion - challenger; positivo = el challenger mejora)

| Alcance | Diferencia | IC 95% | Estaciones donde mejora el challenger |
|---|---|---|---|
| Total, bootstrap por estación | +0.18 pp | [+0.10 pp, +0.27 pp] | |
| Total, bootstrap por día | +0.18 pp | [+0.14 pp, +0.22 pp] | |
| Estaciones existentes | +0.13 pp | [+0.05 pp, +0.23 pp] | 60% |
| Estaciones nuevas | +0.85 pp | [+0.37 pp, +1.43 pp] | 61% |

Prueba de Diebold-Mariano (H0: el challenger no reduce el error absoluto diario): p = 0.0000.

### Criterios de decisión

| Criterio | Regla | Resultado | ¿Cumple? |
|---|---|---|---|
| Mejora práctica del WAPE total | WAPE champion - WAPE challenger >= 0.5 pp | +0.18 pp | **No** |
| Mejora estadísticamente significativa | límite inferior del IC 95% (el más conservador) > 0 | [+0.10 pp, +0.27 pp] | Sí |
| Sin degradación en estaciones existentes | WAPE challenger - WAPE champion <= 0.5 pp | -0.13 pp | Sí |
| Sin degradación en estaciones nuevas | WAPE challenger - WAPE champion <= 0.5 pp | -0.85 pp | Sí |
| Sesgo controlado | sesgo absoluto: challenger - champion <= 0.02 viajes/h | +0.003 | Sí |
| Supera al baseline ingenuo | WAPE challenger < WAPE de la misma hora de la semana anterior | 60.45% vs 73.64% | Sí |

### Decisión

**Versión vigente: v1 (se conserva el champion).** El challenger no cumple: Mejora práctica del WAPE total. Sin evidencia suficiente de mejora, cambiar de modelo agrega riesgo y costo de despliegue sin beneficio.

Estado del registry: sin cambios: `@champion` sigue en v1.

<!-- RESULTADOS_AB:FIN -->

## 5. Justificación: la versión vigente sigue siendo la v1 (champion)

El challenger es **mejor, pero no lo suficiente** para justificar un cambio de versión. Cumple cinco de
los seis criterios y falla el de mejora práctica. Estas son las razones de la decisión.

### 5.1 La mejora es real, pero demasiado pequeña para importar

- **Es estadísticamente significativa.** Con un millón de estación-horas pareadas, cualquier diferencia
  sistemática lo es. Los dos intervalos bootstrap excluyen el cero (por estación [+0.10, +0.27] pp y por
  día [+0.14, +0.22] pp), y la prueba de Diebold-Mariano da p < 0.0001. Que la mejora sea
  significativa solo dice que no es azar; no dice que sea útil.
- **Su tamaño es marginal.** El WAPE baja de 60.62% a 60.45%: 0.18 puntos, es decir, 0.3% del error. En
  viajes, el MAE baja 0.0022 por estación-hora. Sobre todo el trimestre, son unos 2,160 viajes menos de
  error absoluto entre 1.22 millones de viajes, unos 23 viajes al día para toda la red: cerca de 0.05 por
  estación al día. Ninguna decisión de rebalanceo cambia con esa diferencia.
- **No es una mejora uniforme.** El challenger tiene menor error en 60% de las estaciones y mayor error en
  el 40% restante, tanto en las existentes como en las nuevas. Es un ajuste fino, no un modelo que
  generalice mejor.

El umbral de 0.5 pp se fijó antes de ejecutar la comparación justamente para separar "significativo" de
"relevante". Relajarlo después de ver que el challenger quedó en 0.18 pp sería ajustar la regla al
resultado.

### 5.2 La hipótesis del reentrenamiento no se confirmó

El objetivo del challenger era reducir el error en las **estaciones nuevas**, donde se concentró el drift.
Ese segmento mejora 0.85 pp (significativo, con IC [+0.37, +1.43] pp), pero el WAPE pasa de 100.5% a
99.7%: el error absoluto sigue siendo igual a la demanda. Seis meses más de datos no resuelven el
problema de fondo. Esas estaciones tienen pocas semanas de historia, así que sus rezagos y promedios
móviles son inestables, y eso no se corrige con más datos de entrenamiento del mismo tipo. Es la
confirmación empírica de lo que anticipó la [interpretación del drift](interpretacion_drift.md): las
estaciones nuevas necesitan una estrategia de arranque en frío, no un reentrenamiento.

### 5.3 El cambio tiene costo y casi ningún beneficio

- **El sesgo empeora ligeramente.** Pasa de −0.037 a −0.040 viajes/h (en estaciones existentes, de
  −0.048 a −0.051). Está dentro de la tolerancia, pero va en la dirección equivocada: el challenger
  subpredice un poco más justo en el segmento que concentra casi todos los viajes.
- **Promover implica riesgos.** Hay que reiniciar la API, la referencia del monitoreo de drift deja de
  ser comparable con la histórica y el comportamiento en producción del nuevo modelo es desconocido. A
  cambio, se ganarían 0.18 pp de WAPE.

### 5.4 Ambas versiones son aptas para producción

La decisión no se debe a que el challenger sea malo. Las dos versiones superan con amplitud al baseline
de la semana anterior (60.5% contra 73.6% de WAPE) y ninguna degrada a algún segmento. Simplemente, la
v1 sigue haciendo bien su trabajo con datos de 2026: el drift detectado no la deterioró al grado de
justificar un reemplazo. Esto coincide con la interpretación del drift: el modelo absorbe el cambio de
nivel porque sus entradas (rezagos y promedios móviles) se ajustan solas.

### 5.5 Conclusión

**Se conserva la v1 como versión vigente (`@champion`).** La v2 queda registrada con el alias
`@challenger` y la etiqueta `ab_resultado=no_vigente`, como evidencia del experimento. Las acciones que
siguen, en orden de impacto esperado:

1. **Arranque en frío para estaciones nuevas.** Es donde está el error (WAPE cercano a 100%) y el
   reentrenamiento no lo resolvió. Usar el promedio de estaciones vecinas o del grupo de P3 hasta que la
   estación acumule 8 semanas de historia.
2. **Repetir el A/B con el cuarto trimestre de 2026.** Un challenger entrenado hasta septiembre y
   evaluado de octubre a diciembre dirá si la ventaja crece con más datos posteriores a la expansión.
3. **Monitoreo por segmento.** Mantener las alertas de drift separadas para estaciones existentes y
   nuevas, como propone la interpretación.

## 6. Qué hacer con la decisión

**Si queda el challenger (v2):**

1. Promover moviendo alias, sin crear versiones nuevas:
   `python -m mibici.modeling.train ab-test --promote true`. El alias `@champion` pasa a la v2 y la v1
   queda en `@champion_anterior`.
2. Reiniciar la API o el contenedor: el modelo se carga al arrancar desde
   `models:/mibici-equipo24-p1-demanda@champion`, y `/health` debe reportar `"version": "2"`.
3. Para revertir, mover `@champion` de vuelta a la v1. No hace falta reentrenar ni reconstruir la
   imagen.
4. Usar el periodo de entrenamiento del challenger (hasta junio de 2026) como nueva referencia del
   monitoreo de drift.

**Si se conserva el champion (v1):**

1. No se mueve ningún alias; la API sigue sin cambios.
2. La v2 queda en el registry con la etiqueta `ab_resultado=no_vigente`, como evidencia del intento.
3. Si la mejora no es suficiente en las estaciones nuevas, la siguiente acción es la regla de arranque en
   frío propuesta en la [interpretación del drift](interpretacion_drift.md) (el promedio de estaciones
   vecinas o de su grupo de P3 hasta acumular 8 semanas de historia), no otro reentrenamiento igual.

En ambos casos, las dos versiones quedan etiquetadas en el registry con `ab_test_run_id` (la corrida
de la comparación) y `ab_resultado` (`vigente` o `no_vigente`).

## 7. Reproducción

Requiere el entorno del proyecto (`.venv`), el archivo `.env` con `MLFLOW_TRACKING_URI` y el panel de
datos construido (`python -m mibici.features`).

```bash
# Solo comparar y documentar (no mueve alias)
python -m mibici.modeling.train ab-test

# Comparar y, si el challenger cumple todos los criterios, promoverlo
python -m mibici.modeling.train ab-test --promote true

# Equivalentes
make ab-test                 # o: make ab-test PROMOTE=true
mlflow run . -e ab_test --env-manager local -P promote=false
```

Cada ejecución genera:

| Salida | Contenido |
|---|---|
| Corrida de MLflow `p1-ab-test-v1-vs-v2` | Métricas de ambas versiones con prefijo `champion_` y `challenger_`, diferencias e intervalos, criterios (`ab_criterios.csv`) y la etiqueta `ab_vigente` |
| `reports/ab_test/p1_ab_test.json` | Resultado completo en formato estructurado |
| Este documento, sección 4 | Tablas, criterios y decisión |

Código: [mibici/modeling/p1_ab_test.py](../mibici/modeling/p1_ab_test.py). Pruebas sin datos ni
servidor de MLflow: `python -m pytest tests/test_ab_test.py`.

## 8. Limitaciones

- **Un solo trimestre de evaluación.** Julio a septiembre incluye el regreso a clases de septiembre, el
  mes de mayor demanda, pero no el cuarto trimestre. La decisión debe revisarse cuando haya datos de
  octubre a diciembre.
- **Las estaciones nuevas pesan poco en el total.** Son cerca de 19% de las estación-horas evaluadas
  (187,948 de 1,001,510) y una fracción menor de los viajes (cerca de 3% de enero a septiembre, según la
  interpretación del drift), así que una mejora grande en ellas apenas mueve el WAPE total. Por eso se
  reportan por separado y tienen su propio criterio de no degradación.
- **Ventaja de información del challenger.** Vio estaciones nuevas durante su entrenamiento y el champion
  no. Esto no es una fuga: es justamente la hipótesis que se prueba. Ninguna de las dos versiones vio el
  periodo de evaluación.
- **Estaciones fuera de la nomenclatura vigente.** Las horas de estaciones que ya no aparecen en la
  nomenclatura de octubre de 2026 quedan fuera de la evaluación (ver
  [version_alternativa.md](version_alternativa.md)). Afecta por igual a ambas versiones.
- **Efecto operativo no medido.** La comparación mide el error de predicción, no su efecto en el
  rebalanceo. Ese efecto requiere un despliegue gradual después de la promoción.
- **Versiones reproducidas en un servidor local.** Esta comparación se ejecutó contra un servidor de
  MLflow local, con ambas versiones reentrenadas desde el código del repositorio (champion: corrida
  `ccb4c1e5`; challenger: corrida `85344bca`), no contra las corridas originales del servidor del equipo
  (`81731447` y `0f925ee4`). Los datos usan la nomenclatura de octubre de 2026, porque la de septiembre ya
  no está publicada (variable `MIBICI_NOMENCLATURA`). La reproducción es fiel: el challenger local obtuvo
  exactamente las métricas documentadas en [version_alternativa.md](version_alternativa.md) (WAPE 60.5%,
  57.7% en existentes y 99.9% en nuevas, sesgo −0.039, MAE 0.736, RMSE 1.364, 1,001,510 filas). Para
  dejar constancia en el registry del equipo, conviene repetir `ab-test` apuntando a ese servidor.
