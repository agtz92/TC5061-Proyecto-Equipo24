# Preparación de conjuntos de datos para monitoreo de producción

Este documento detalla la definición, justificación y preparación de los conjuntos de datos de **Referencia** (*Reference Dataset*) y **Actual** (*Current Dataset*) para el monitoreo de deriva (*Data Drift* y *Concept Drift*) y desempeño del modelo de demanda (P1) de MiBici en producción.

---

## 1. Definición y selección de periodos

| Característica | Conjunto de Referencia (*Reference*) | Conjunto Actual (*Current / Producción*) |
|---|---|---|
| **Periodo temporal** | **2022-01-01 00:00 — 2025-06-30 23:00** | **2026-01-01 00:00 — 2026-09-30 23:00** |
| **Duración** | 42 meses (3.5 años) | 9 meses (año en curso) |
| **Rol en el ciclo MLOps** | Periodo histórico de entrenamiento | Periodo de inferencia en producción (*post-entrenamiento*) |
| **Estaciones activas** | 371 estaciones | 469 estaciones (+98 nuevas estaciones) |
| **Demanda real promedio** | 1.47 viajes / estación-hora | 1.22 viajes / estación-hora |
| **Predicción promedio** | 1.50 viajes / estación-hora | 1.38 viajes / estación-hora |
| **Rutas generadas** | `data/monitoring/reference.parquet` (.csv) | `data/monitoring/current.parquet` (.csv) |

---

## 2. Justificación de los periodos elegidos

### A. ¿Por qué el periodo 2022-01 a 2025-06 como Referencia?
1. **Alineación con el entrenamiento:** Es el rango exacto de datos históricos con el que se ajustaron los parámetros del modelo champion (`mibici-equipo24-p1-demanda`).
2. **Distribución base estable:** Captura la estacionalidad post-pandemia en Guadalajara (patrones laborales de días hábiles, quincenas, días festivos y horas pico), excluyendo los años anómalos de COVID-19 (2014–2021).
3. **Punto de calibración:** Permite contrastar si las relaciones aprendidas entre las covariables (`P1_FEATURES`) y la demanda objetivo (`departures`) se mantienen estables en el tiempo.

### B. ¿Por qué el periodo 2026-01 a 2026-09 como Actual (Producción)?
1. **Fricción temporal estricta:** Es un periodo estrictamente posterior tanto al conjunto de entrenamiento como al de validación (julio-diciembre 2025). Simula de forma realista el comportamiento del modelo puesto en producción durante 2026.
2. **Presencia de Drift real en el sistema:**
   - **Expansión de la red:** En 2026, la red de MiBici pasó de 371 a **469 estaciones activas**, incorporando nuevas colonias y modificando las dinámicas de conectividad y capacidad de anclaje.
   - **Deriva estacional y de volumen:** Incluye el regreso a clases universitario de otoño (septiembre 2026, el mes de mayor demanda de la serie), lo que pone a prueba la capacidad de generalización del modelo frente a picos de demanda.
3. **Disponibilidad de Ground Truth verificado:** A diferencia de los registros efímeros o limitados de peticiones en pruebas locales de la API, los datos abiertos oficiales de MiBici de 2026 proveen el conteo exacto de viajes reales. Esto permite evaluar no solo *Data Drift* (cambio en las entradas), sino también *Target Drift* y métricas de desempeño del modelo en producción ($WAPE$, $MAE$, $R^2$).

---

## 3. Esquema de los datos preparados

Cada conjunto contiene exactamente las variables requeridas por el pipeline de inferencia y las herramientas de monitoreo (como Evidently o Deepchecks):

* **Identificadores temporales y espaciales:** `station_id`, `ts` (marca de tiempo en hora local de Guadalajara).
* **Variables predictoras (`P1_FEATURES`):**
  * Calendario: `hour`, `dow`, `month`, `is_weekend`, `is_holiday`, `is_quincena`.
  * Históricos de demanda: `dep_lag_24h`, `dep_lag_168h` (rezagos de 24 y 168 horas).
  * Promedios móviles: `dep_mean_7d`, `dep_mean_28d` (promedios a la misma hora en 7 y 28 días).
  * Infraestructura y geolocalización: `capacity`, `lat`, `lon`.
* **Etiqueta real (*Ground Truth*):** `departures` (salidas reales observadas en esa hora).
* **Predicción del modelo:** `prediction` (inferencia generada con el modelo champion `models/p1`).

---

## 4. Cómo regenerar o personalizar los conjuntos

El módulo [`mibici/monitoring.py`](../mibici/monitoring.py) permite extraer los conjuntos de forma determinista y reproducible:

```bash
# Extracción estándar (50,000 filas por conjunto para análisis balanceado):
python -m mibici.monitoring

# Personalizar fechas o tamaños de muestra:
python -m mibici.monitoring --reference-start 2024-01-01 --reference-end 2025-06-30 --current-start 2026-09-01 --current-end 2026-09-30 --sample-reference 20000 --sample-current 20000
```

