# Model Card · Demanda de viajes por estación (P1)

Ficha del modelo que sirve la API de producción del proyecto MiBici. Sigue la estructura de *Model Cards for
Model Reporting* (Mitchell et al., 2019). Las cifras salen de la verificación del Model Registry del
2026-11-26 ([evidencia](Evidencia%20del%20Model%20Registry/)), del A/B testing
([`reports/ab_test/p1_ab_test.json`](../reports/ab_test/p1_ab_test.json)), del monitoreo de drift
([`reporte_drift.md`](reporte_drift.md), [`interpretacion_drift.md`](interpretacion_drift.md)) y de la prueba
de carga y el análisis FinOps ([`reports/carga/reporte_carga.md`](../reports/carga/reporte_carga.md)).

| | |
|---|---|
| Nombre en el registry | `mibici-equipo24-p1-demanda` |
| Versión vigente | **v1**, alias `@champion` (la que carga la API) |
| Versión alternativa | v2, alias `@challenger` (evaluada en el A/B test; no se promovió) |
| Equipo | Equipo 24, TC5061 Operaciones de aprendizaje automático (MLOps), Tecnológico de Monterrey |
| Última actualización de esta ficha | 2026-11-26 |

## 1. Detalles del modelo

| | |
|---|---|
| Tarea | Regresión de conteos: viajes que inician en una estación en cada hora del día siguiente (horizonte de 24 h) |
| Estimador | `HistGradientBoostingRegressor` de scikit-learn con pérdida Poisson |
| Hiperparámetros | 400 iteraciones, tasa de aprendizaje 0.1, profundidad máxima 16, 50 muestras mínimas por hoja, semilla 42 |
| Criterio de selección | `val_wape` mínimo entre baseline, Poisson lineal, random forest y HGB (`SELECTION_METRIC` en `mibici/config.py`) |
| Entradas (13) | Calendario: `hour`, `dow`, `month`, `is_weekend`, `is_holiday`, `is_quincena`. Demanda pasada: `dep_lag_24h`, `dep_lag_168h`, `dep_mean_7d`, `dep_mean_28d`. Estación: `capacity`, `lat`, `lon` |
| Salida | Viajes esperados (número real no negativo) |
| Firma | 13 entradas `float` obligatorias; la firma de MLflow exige `float32` y rechaza otros tipos |
| Formato | MLflow 2.22.0, *flavors* `python_function` y `sklearn` |

**Linaje de la versión vigente.**

| Eslabón | Valor |
|---|---|
| Datos | `data_version = d9b669d05452` (`data_manifest.json` guardado como artefacto de la corrida) |
| Código | `git_commit = 85c61b1eeb00` (artefactos `config/`: `config.py`, `MLproject`, `requirements.txt`) |
| Corrida | `p1-hgb-it400-lr0.1-d16-leaf50`, `run_id = ccb4c1e5b8a04a59ab9500bff1eb9c6b`, experimento `mibici-equipo24` |
| Artefacto | `runs:/ccb4c1e5b8a04a59ab9500bff1eb9c6b/model` |
| Registry | `mibici-equipo24-p1-demanda` v1, tags `ab_resultado=vigente`, `ab_test_run_id=a8dab263...` |

## 2. Uso previsto

- **Usuario principal:** el operador del sistema MiBici, para planear la flota y el rebalanceo del día
  siguiente por estación, zona y franja horaria.
- **Uso dentro del alcance:** pronósticos agregados (por estación y hora, o sumados a zona y día) para
  estaciones con al menos 8 semanas de historia.
- **Fuera de alcance:**
  - Decisiones sobre personas: el modelo no usa ni produce datos de usuarios.
  - Pronósticos en tiempo real o de menos de 24 h: las entradas se cortan a las 23:59 del día anterior.
  - Estaciones con menos de 8 semanas de historia: sus rezagos y promedios no son estables (ver sección 6).
  - Eventos extraordinarios (cierres de vialidades, contingencias, eventos masivos), que no están en las
    entradas.

## 3. Factores

- **Antigüedad de la estación.** Es el factor que más cambia el error: las 100 estaciones abiertas en 2026
  tienen WAPE de 101% contra 60% de las existentes.
- **Nivel de demanda.** Con pocos viajes por hora el ruido de conteo domina el error relativo; el piso
  teórico (error de un pronóstico perfecto de la media Poisson) es de 50.3% en validación.
- **Temporada.** La demanda cambia por mes; en 2026 cayó 12% en las estaciones existentes de enero a agosto.

## 4. Métricas

- **WAPE** (error absoluto ponderado: suma de errores absolutos entre suma de viajes): métrica de decisión,
  porque se lee en viajes, la unidad del operador.
- **MAE** y **RMSE** en viajes por hora, **sesgo** (predicción media menos demanda media) y devianza Poisson.
- **Prueba de Diebold-Mariano** para comparar contra el baseline y entre versiones.
- **Baseline:** misma hora de la semana anterior (`dep_lag_168h`).

## 5. Datos

| | Entrenamiento | Validación | Prueba |
|---|---|---|---|
| Periodo | 2022-01-01 a 2025-06-30 | 2025-07-01 a 2025-12-31 | 2026-01-01 a 2026-09-30 |
| Uso | Ajuste (muestra de 2,001,674 estación-horas) | Selección de modelo e hiperparámetros | Estimación final del error |

- **Fuente:** datos abiertos de MiBici (Área Metropolitana de Guadalajara), 21 millones de viajes de 2022-01 a
  2026-09. Se excluye 2014-2021 porque la pandemia cambió los hábitos de viaje.
- **Preparación:** ingesta reproducible (`mibici/dataset.py`) que unifica codificaciones y formatos de
  fecha, elimina viajes inválidos y genera el manifiesto; panel estación por hora con features sin fuga
  de información (`mibici/features.py`).
- **Datos personales:** `usuario_id`, `genero` y `anio_nacimiento` se eliminan en la ingesta. El modelo solo
  ve conteos agregados por estación y hora.
- **Partición:** temporal, nunca aleatoria.

## 6. Análisis cuantitativo

**Contra el baseline** (versión vigente v1):

| Métrica | Baseline | v1 `@champion` | Mejora |
|---|---|---|---|
| WAPE validación | 71.19% | **56.99%** | 14.2 pp |
| WAPE prueba | 75.52% | **61.64%** | 13.9 pp |
| MAE validación | | 0.768 viajes/h | |
| RMSE validación | | 1.361 viajes/h | |
| Diebold-Mariano (valor p, validación) | | < 0.0001 | Diferencia significativa |

**En producción 2026, por segmento** (enero a septiembre de 2026):

| Segmento | Estaciones | Demanda media | Sesgo | WAPE |
|---|---|---|---|---|
| Referencia 2022-2025 | 371 | 1.468 viajes/h | -0.004 | 54.2% |
| Todas | 469 | 1.228 viajes/h | -0.003 | 61.4% |
| Estaciones existentes | 369 | 1.299 viajes/h | -0.004 | 60.3% |
| Estaciones nuevas | 100 | 0.427 viajes/h | +0.006 | **101.1%** |

**A/B testing v1 contra v2** (reentrenada con datos hasta junio de 2026; 1,001,510 estación-horas de julio a
septiembre de 2026):

| Criterio | Regla | Resultado | ¿Cumple? |
|---|---|---|---|
| Mejora práctica | WAPE v1 menos WAPE v2 de al menos 0.5 pp | +0.18 pp (60.62% contra 60.45%) | No |
| Mejora significativa | Límite inferior del IC 95% mayor que 0 | [+0.10, +0.27] pp | Sí |
| Estaciones existentes | No empeorar más de 0.5 pp | mejora 0.13 pp | Sí |
| Estaciones nuevas | No empeorar más de 0.5 pp | mejora 0.85 pp | Sí |
| Sesgo | No crecer más de 0.02 viajes/h | +0.003 | Sí |

**Decisión:** v1 sigue vigente porque la mejora de v2 no alcanza el umbral práctico
([`justificacion_version_vigente.md`](justificacion_version_vigente.md)).

## 7. Operación

| Aspecto | Cómo se cumple |
|---|---|
| Despliegue | API FastAPI en contenedor (`Containerfile`) que carga `models:/mibici-equipo24-p1-demanda@champion` al arrancar; endpoints `/predict`, `/predict/lote` (hasta 1,000 registros) y `/health` |
| Orquestación | `docker-compose.yml`: MLflow + API con *healthchecks*, red interna y límites de 1 CPU y 1 GB; perfil `ab` con la API del `@challenger` |
| Integración continua | GitHub Actions: Ruff, revisiones de seguridad, Pytest, construcción de la imagen y prueba de humo en cada push y PR |
| Capacidad | Unas 20 peticiones/s por contenedor de 1 CPU con p95 de 500 ms o menos; 0% de errores hasta 150 usuarios ([reporte de carga](../reports/carga/reporte_carga.md)) |
| Costo | USD 36.04 al mes por contenedor (AWS Fargate x86, 1 vCPU / 2 GB); USD 0.17 por millón de predicciones a capacidad |
| Monitoreo | Reporte de Evidently (calidad de datos, drift y desempeño) e interpretación por segmento; consumo de CPU y memoria con `docker stats` |
| Cambio de versión | Se mueve el alias `@champion` en el registry tras un A/B test que cumpla todos los criterios; la API no fija número de versión, así que no hace falta redesplegar código |

## 8. Consideraciones éticas

- **Privacidad:** no se guardan ni se usan datos de personas; se trabaja con conteos por estación.
- **Equidad territorial:** el error es mucho mayor en estaciones nuevas, que suelen estar en zonas
  periféricas. Si se usa para repartir bicicletas sin corregirlo, esas zonas podrían quedar desatendidas.
  Para estas estaciones conviene combinar el pronóstico con criterio del operador hasta que acumulen
  historia.
- **Datos abiertos:** el modelo hereda los posibles errores de registro del portal (anclajes fallidos,
  fechas corruptas); la ingesta los filtra y documenta.

## 9. Advertencias y recomendaciones

- **Drift:** 5 de 13 variables muestran drift en 2026. `month` es un artefacto del periodo incompleto; las
  variables de demanda reflejan la apertura de estaciones y una caída real de demanda. El modelo absorbe el
  cambio de nivel (sesgo cercano a cero), pero el error crece de 54% a 61%.
- **Qué vigilar y cuándo actuar** (de [`interpretacion_drift.md`](interpretacion_drift.md) y
  [`justificacion_version_vigente.md`](justificacion_version_vigente.md)):
  - No reentrenar por `month`, y excluirla de la alerta hasta que el periodo actual cubra un año completo.
  - Alertar por drift de las variables de demanda **en estaciones existentes**, porque las nuevas siempre
    parecen drift en sus primeras semanas.
  - Para estaciones nuevas, usar una regla de arranque en frío (promedio de estaciones vecinas o de su grupo
    de P3) hasta que acumulen 8 semanas de historia; el reentrenamiento de v2 no resolvió su error.
  - Cualquier nueva versión se promueve solo si cumple todos los criterios del A/B test, comparando por
    segmento y no solo en el total.
- **Pendientes de linaje** (de la verificación del registry): `git_dirty = true` en la corrida de origen, el
  modelo se entrenó como corrida local en vez de `mlflow run`, el `data_manifest.json` del repositorio no
  coincide con el `data_version` del modelo, y el alias heredado `@candidate` también apunta a v1. Ninguno
  cambia lo que sirve la API; se corrigen en el siguiente reentrenamiento.
- **Portabilidad:** el modelo vive en la base de MLflow del equipo (`mlflow.db` y `mlartifacts/`), no en
  Git. Para servirlo en otra máquina hay que copiar ambas (ver
  [`README_reproducibilidad.md`](README_reproducibilidad.md), sección 5).

## Referencias

- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D. y
  Gebru, T. (2019). Model cards for model reporting. *Proceedings of the Conference on Fairness,
  Accountability, and Transparency*, 220-229. https://doi.org/10.1145/3287560.3287596
