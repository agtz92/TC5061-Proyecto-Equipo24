# Reporte de Monitoreo: Data Drift y Calidad de Datos con Evidently

Este documento resume los resultados del análisis de **Data Drift** (deriva de datos), **Data Quality** (calidad de datos) y **desempeño predictivo** generado con la librería [Evidently](https://www.evidentlyai.com/) para el modelo de predicción de demanda de estaciones MiBici (P1).

---

## 1. Resumen Ejecutivo

* **Artefacto interactivo:** [`reports/evidently/drift_report.html`](../reports/evidently/drift_report.html) (reporte autocontenido visualizable en cualquier navegador).
* **Artefacto estructurado:** [`reports/evidently/drift_report.json`](../reports/evidently/drift_report.json) (métricas serializadas para alertas CI/CD o pipelines automatizados).
* **Periodos comparados:**
  * **Referencia (*Training baseline*):** Enero 2022 a Junio 2025 (49,759 observaciones balanceadas en red de 371 estaciones).
  * **Actual (*Producción 2026*):** Enero 2026 a Septiembre 2026 (49,745 observaciones balanceadas en red ampliada a 469 estaciones).
* **Tasa de Drift global detectada:** **5 de 13 variables predictoras con deriva significativa (38.5%)**.
* **Estado de Calidad de Datos:** **100% íntegro**. Cero valores nulos (0.0%), cero columnas constantes y cero filas vacías.

---

## 2. Calidad de Datos (*Data Quality*)

El preset `DataSummaryPreset` de Evidently evaluó la completitud, unicidad y consistencia de los datos de producción en comparación con la referencia:

| Métrica de Calidad | Conjunto de Referencia | Conjunto Actual (Producción) | Diagnóstico |
|---|:---:|:---:|:---:|
| **Total de filas evaluadas** | 10,000 | 10,000 | Tamaño de muestra estadísticamente representativo |
| **Total de columnas** | 15 (13 features + 2 target/pred) | 15 | Esquema 100% compatible |
| **Valores nulos (*Missing Values*)** | 0 (0.00%) | 0 (0.00%) | Sin pérdida de datos en ingesta |
| **Filas vacías (*Empty Rows*)** | 0 | 0 | Sin registros corruptos |
| **Columnas vacías (*Empty Columns*)** | 0 | 0 | Todas las covariables tienen datos válidos |
| **Columnas constantes (*Constant*)** | 0 | 0 | Variabilidad conservada en todas las dimensiones |
| **Filas duplicadas** | 0 | 13 (0.13%) | Tasa insignificante esperada por muestreo |

---

## 3. Análisis de Data Drift por Variable

Evidently aplicó pruebas estadísticas acordes al tipo y distribución de cada covariable:
* **Distancia de Wasserstein normalizada** para variables numéricas continuas y discretas de rango amplio.
* **Distancia de Jensen-Shannon** para variables binarias y categóricas.
* **Umbral de decisión estándar:** $\text{Drift Score} > 0.10$.

### Tabla detallada de resultados

| Variable | Tipo / Dimensión | ¿Drift Detectado? | Drift Score | Umbral | Prueba Estadística | Interpretación del Fenómeno |
|---|---|:---:|:---:|:---:|---|---|
| `month` | Calendario | **SÍ ⚠** | **0.2810** | 0.10 | Distancia Wasserstein | La referencia contiene los 12 meses (2022–2025); producción sólo abarca enero a septiembre 2026 (faltan Q4). |
| `dep_mean_28d` | Demanda histórica | **SÍ ⚠** | **0.1348** | 0.10 | Distancia Wasserstein | La demanda base bajó de 1.47 a 1.22 viajes/hora debido a la dilución del tráfico por 98 nuevas estaciones. |
| `dep_mean_7d` | Demanda histórica | **SÍ ⚠** | **0.1312** | 0.10 | Distancia Wasserstein | Descenso generalizado del promedio móvil semanal por la expansión periférica del sistema. |
| `dep_lag_168h` | Demanda histórica | **SÍ ⚠** | **0.1117** | 0.10 | Distancia Wasserstein | Variación en la demanda a la misma hora de la semana anterior. |
| `dep_lag_24h` | Demanda histórica | **SÍ ⚠** | **0.1099** | 0.10 | Distancia Wasserstein | Variación en la demanda a la misma hora del día anterior. |
| `lon` | Geográfica | **no** | 0.0888 | 0.10 | Distancia Wasserstein | Cerca del umbral por nuevas estaciones periféricas en Zapopan y Tlaquepaque, pero aún estable. |
| `capacity` | Infraestructura | **no** | 0.0821 | 0.10 | Distancia Wasserstein | Las nuevas estaciones tienen capacidades similares (promedio ~16 anclajes) a las históricas. |
| `lat` | Geográfica | **no** | 0.0297 | 0.10 | Distancia Wasserstein | Distribución latitudinal conservada dentro de la Zona Metropolitana de Guadalajara. |
| `dow` | Calendario | **no** | 0.0192 | 0.10 | Distancia Wasserstein | Distribución uniforme por día de la semana inalterada. |
| `hour` | Calendario | **no** | 0.0136 | 0.10 | Distancia Wasserstein | Patrón horario (horas pico 8:00 y 18:00 hrs) se mantiene idéntico. |
| `is_weekend` | Calendario | **no** | 0.0073 | 0.10 | Distancia Jensen-Shannon | Proporción fija de fines de semana (~28.5%). |
| `is_quincena` | Calendario | **no** | 0.0057 | 0.10 | Distancia Jensen-Shannon | Distribución de días 15 y fin de mes invariante. |
| `is_holiday` | Calendario | **no** | 0.0038 | 0.10 | Distancia Jensen-Shannon | Proporción de días feriados oficiales consistente. |

---

## 4. Evaluación del Desempeño del Modelo en Producción

El preset `RegressionPreset` evaluó la capacidad predictiva del modelo champion (`mibici-equipo24-p1-demanda`, regresor Poisson) sobre los datos reales de 2026:

| Métrica de Regresión | Valor en Producción (2026) | Interpretación |
|---|:---:|---|
| **RMSE (*Root Mean Squared Error*)** | **1.8879** viajes/hora | Dispersión típica de los errores de predicción frente al conteo real por estación-hora. |
| **MAE (*Mean Absolute Error*)** | **1.1924** viajes/hora | En promedio, el modelo difiere en ~1 viaje del valor real. |
| **Sesgo medio (*Mean Error*)** | **+0.1494** viajes/hora | **Sobrepredicción sistemática**: el modelo asume una tasa promedio de 1.38 viajes/h cuando el valor real observado es 1.22 viajes/h. |
| **Coeficiente $R^2$** | **0.3047** | El modelo retiene capacidad explicativa del 30.5% de la varianza en producción a pesar del drift de red. |

---

## 5. Diagnóstico de Causa Raíz y Recomendaciones MLOps

1. **Causa del Data Drift en covariables de demanda:**
   * La red de MiBici creció un **26.4%** en 2026 (de 371 a 469 estaciones activas).
   * Al agregar estaciones en colonias periféricas con menor densidad inicial de viajes, el promedio global de viajes por estación cayó de 1.47 a 1.22.
   * Por consiguiente, las 4 variables de memoria (`dep_lag_24h`, `dep_lag_168h`, `dep_mean_7d`, `dep_mean_28d`) experimentaron deriva estructural respecto al entrenamiento.
2. **Recomendaciones de acción:**
   * **Reentrenamiento programado:** Incorporar los datos de 2025-H2 y 2026-H1 para recalibrar los coeficientes del regresor Poisson con la topología de 469 estaciones.
   * **Segmentación por clusters de estaciones:** Considerar features de densidad de red o modelos segmentados (estaciones céntricas consolidadas vs. estaciones periféricas de nueva creación).
   * **Política de monitoreo continuo:** Configurar un umbral de alerta en producción cuando más del 30% de las características experimenten drift, disparando una notificación automática al equipo de MLOps.

---

## 6. Reproducción del Reporte

Para ejecutar el monitoreo y regenerar los reportes HTML y JSON:

```bash
# Ejecutar con el entorno virtual de monitoreo:
.venv-monitoring\Scripts\python.exe mibici\drift_report.py
```

