# Interpretación del drift (P1 demanda)

Complementa el [reporte de Evidently](reporte_drift.md). Evidently detecta **qué** variables cambiaron;
aquí se analiza **por qué** cambiaron y **qué efecto tienen en el modelo vigente**
(`models:/mibici-equipo24-p1-demanda@champion`).

Se usó la misma distancia que Evidently para variables numéricas (Wasserstein normalizada por la desviación
estándar de la referencia, umbral 0.1). Referencia: enero 2022 a junio 2025 (periodo de entrenamiento).
Actual: enero a septiembre 2026. Muestra determinista de unas 200,000 estación-horas por periodo.

Reproducir:

```bash
python -m mibici.drift_interpretation
```

Resultados en `reports/evidently/drift_interpretacion.json`.

## 1. Variables afectadas y causa del drift

| Variable | Todas las estaciones | Solo estaciones existentes | Existentes vs referencia ene-sep |
|---|---|---|---|
| `month` | **0.251** | **0.317** | **0.115** |
| `dep_mean_28d` | **0.139** | **0.105** | 0.098 |
| `dep_mean_7d` | **0.125** | 0.092 | 0.090 |
| `dep_lag_168h` | **0.106** | 0.079 | 0.074 |
| `dep_lag_24h` | 0.099 | 0.073 | 0.072 |
| `lon` | 0.089 | 0.058 | 0.059 |
| `capacity` | 0.081 | 0.044 | 0.044 |
| Resto (`hour`, `dow`, `is_weekend`, `is_holiday`, `is_quincena`, `lat`) | < 0.03 | < 0.04 | < 0.04 |

En negritas, los valores que superan el umbral de 0.1.

**`month`: es un artefacto del periodo, no un cambio de comportamiento.** El periodo actual solo tiene
enero a septiembre, mientras que la referencia tiene los doce meses. Al comparar contra la referencia
restringida a enero-septiembre, la distancia baja de 0.317 a 0.115, prácticamente al umbral. El modelo no
necesita reentrenarse por esta variable. La alerta desaparecerá sola cuando el periodo actual cubra un año
completo.

**Variables de demanda (`dep_*`): hay dos causas, no una.**

- **Estaciones nuevas.** La red pasó de 371 a 469 estaciones, y las 100 nuevas tienen poca demanda: 0.43
  viajes por hora contra 1.30 de las existentes. Al quitarlas, tres de las cuatro variables de demanda
  quedan por debajo del umbral y `capacity` y `lon` bajan a la mitad.
- **Caída real en las estaciones existentes.** Aun sin estaciones nuevas, la demanda media de las
  estaciones que ya existían bajó de 1.47 a 1.30 viajes por hora (12% menos), y `dep_mean_28d` sigue
  arriba del umbral. La caída se repite de enero a agosto, con el punto más bajo en mayo (1.10 contra 1.50);
  en septiembre la demanda supera a la de referencia (1.63 contra 1.55). Por eso el reporte no debe atribuir
  todo el drift a la "dilución" por estaciones nuevas.

## 2. Implicaciones para el desempeño del modelo

| Segmento | Estaciones | Demanda media (viajes/h) | Predicción media | Sesgo | WAPE |
|---|---|---|---|---|---|
| Referencia 2022-2025 | 371 | 1.468 | 1.465 | -0.004 | 54.2% |
| Actual, todas | 469 | 1.228 | 1.225 | -0.003 | 61.4% |
| Actual, estaciones existentes | 369 | 1.299 | 1.295 | -0.004 | 60.3% |
| Actual, estaciones nuevas | 100 | 0.427 | 0.434 | +0.006 | **101.1%** |

- **El modelo absorbe el cambio de nivel.** Aunque la demanda bajó 16%, el sesgo sigue en cero (predice
  1.225 contra 1.228 reales). Esto ocurre porque las variables que derivan son precisamente sus entradas
  de rezago y promedios móviles: cuando la demanda baja, las entradas bajan y la predicción se ajusta sola.
  Este drift es el comportamiento esperado de una entrada adaptativa, no una señal de que el modelo esté
  roto.
- **El error sí crece, por dos razones.** El WAPE pasa de 54.2% a 61.4%. Una parte es mecánica: con menos
  viajes por hora, el ruido de conteo pesa más en el error relativo (el piso Poisson calculado en la
  entrega 2 subió de 50.3% en validación a 50.6% en prueba). La otra parte son las estaciones nuevas.
- **Las estaciones nuevas son el punto débil.** Ahí el WAPE es de 101%: el error absoluto es mayor que la
  demanda misma. Son estaciones con pocas semanas de historia, cuyos rezagos y promedios no son estables,
  y cuya ubicación (`lon`, `capacity`) cae en zonas poco representadas en el entrenamiento. Pesan poco en
  el total: 8% de las estación-horas y cerca de 3% de los viajes. Pero son justo las estaciones que
  planeación y rebalanceo necesitan entender al abrirlas.
- **Diferencia con el reporte de Evidently.** Ese reporte mide un sesgo de +0.149 y un R² de 0.30 con el
  modelo descargado en `models/p1`, que se describe como regresor Poisson. El champion del registry usado
  aquí es el HGB elegido en la Parte B (versión 1), y su sesgo es prácticamente cero. Conviene que el
  equipo confirme que todos usan la misma versión antes de comparar cifras en el reporte final.

## 3. Qué hacer

1. **No reentrenar por `month`**, y excluirla de la alerta de drift hasta que el periodo actual cubra un año
   completo, o comparar siempre contra los mismos meses de la referencia.
2. **Monitorear las variables de demanda por segmento.** La alerta útil es "drift en estaciones existentes",
   porque las estaciones nuevas siempre van a parecer drift durante sus primeras semanas.
3. **Tratar aparte a las estaciones nuevas.** Para ellas conviene un modelo o una regla de arranque en frío
   (por ejemplo, el promedio de estaciones vecinas o de su grupo de P3) hasta que acumulen 8 semanas de
   historia.
4. **El reentrenamiento es justificado, pero por la red, no por el nivel de demanda.** Incluir 2025-H2 y
   2026 en el entrenamiento le da al modelo ejemplos de las estaciones nuevas, que es donde más se equivoca.
   El A/B testing del punto 5 debe comparar ambas versiones **por segmento**, no solo en el total.
