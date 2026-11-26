# Guía de despliegue, operación, monitoreo y gobernanza

Esta guía reúne en un solo lugar las piezas que llevan el modelo de demanda (P1) del registry a producción y
las que lo vigilan y documentan después. Es la continuación de
[`README_reproducibilidad.md`](README_reproducibilidad.md), que cubre la estructura, el feature engineering,
el MLflow Project, el Tracking y el Model Registry.

Los comandos son para **PowerShell en Windows**, desde la raíz del repositorio y con el `.venv` activado. En
macOS, Linux o WSL funcionan igual con `export VAR=valor` en lugar de `$env:VAR = "valor"`.

La evidencia de la última verificación está en
[Evidencia de Despliegue y Gobernanza/](Evidencia%20de%20Despliegue%20y%20Gobernanza/).

---

## 1. Mapa de componentes

| # | Componente | Archivos | Comando principal | Resultado o evidencia |
|---|---|---|---|---|
| 3.1 | API del modelo (FastAPI) | `mibici/api/main.py`, `mibici/api/schemas.py` | `uvicorn mibici.api.main:app` | `/predict`, `/predict/lote`, `/health`, `/docs` |
| 3.2 | Contenedor de la API | `Containerfile`, `.dockerignore`, `requirements-api.txt` | `docker build -f Containerfile -t mibici-api:1.0.0 .` | [`contenedor.md`](contenedor.md) |
| 3.3 | Orquestación | `docker-compose.yml` | `docker compose up -d --build` | MLflow y API en estado *healthy* |
| 3.4 | CI/CD | `.github/workflows/ci.yml` | Automático en cada push y PR | Pestaña **Actions** del repositorio; [`ci_cd.md`](ci_cd.md) |
| 3.5 | Pruebas de carga | `loadtest/locustfile.py`, `loadtest/generar_reporte.py` | `locust -f loadtest/locustfile.py ...` | [`reports/carga/reporte_carga.md`](../reports/carga/reporte_carga.md) |
| 4.1 | Drift con Evidently | `mibici/monitoring.py`, `mibici/drift_report.py`, `mibici/drift_interpretation.py`, `requirements-monitoring.txt` | `.venv-monitoring\Scripts\python.exe mibici\drift_report.py` | [`reporte_drift.md`](reporte_drift.md), [`interpretacion_drift.md`](interpretacion_drift.md), `reports/evidently/` |
| 4.2 | A/B testing | `mibici/modeling/p1_retrain.py`, `mibici/modeling/p1_ab_test.py` | `python -m mibici.modeling.train ab-test` | [`justificacion_version_vigente.md`](justificacion_version_vigente.md), `reports/ab_test/p1_ab_test.json` |
| 4.3 | FinOps | `loadtest/monitor_recursos.py`, `mibici/finops.py` | `python loadtest/generar_reporte.py` | Secciones de recursos y FinOps de [`reporte_carga.md`](../reports/carga/reporte_carga.md) |
| 4.4 | Model Card | `docs/MODEL_CARD.md` | No aplica | [`MODEL_CARD.md`](MODEL_CARD.md) |

Todo gira alrededor de un solo modelo: `models:/mibici-equipo24-p1-demanda@champion` (versión 1). La API lo
carga por alias, el monitoreo lo evalúa, el A/B test lo compara con `@challenger` y el Model Card lo
documenta.

```
Model Registry ──@champion──► API (contenedor) ──► Locust + docker stats ──► FinOps
      ▲                           │
      │                           └──► datos de producción ──► Evidently (drift)
      └── mover alias ◄── A/B test (@champion vs @challenger) ◄────────────┘
```

---

## 2. Despliegue y operación

### 2.1 Levantar el sistema (Docker Compose)

```powershell
# Carpeta con mlflow.db y mlartifacts/ del equipo (ver README_reproducibilidad.md, sección 5)
$env:MLFLOW_STORE = "C:\ruta\a\la\base"
docker compose up -d --build
docker compose ps                          # mlflow y api en "healthy" (la API tarda ~1 min)
Invoke-RestMethod http://localhost:8000/health
# estado=ok, modelo=mibici-equipo24-p1-demanda, version=1, uri=models:/...@champion
```

| Servicio | Puerto en el anfitrión | Qué hace |
|---|---|---|
| `mlflow` | `127.0.0.1:5001` | Tracking y Model Registry con la base de `MLFLOW_STORE` |
| `api` | `127.0.0.1:8000` | Sirve `@champion`; espera a que MLflow esté sano antes de arrancar |
| `api-challenger` (perfil `ab`) | `127.0.0.1:8001` | Sirve `@challenger`: `docker compose --profile ab up -d` |

La API corre con un límite de 1 CPU y 1 GB (`API_CPUS`, `API_MEMORY`) para que la prueba de carga y el costo
se midan sobre un tamaño de máquina conocido. Para detener todo: `docker compose --profile ab down`.

### 2.2 Probar la API

Documentación interactiva en http://localhost:8000/docs. Desde PowerShell:

```powershell
$body = @{ hour=8; dow=1; month=9; is_weekend=0; is_holiday=0; is_quincena=0; dep_lag_24h=6;
           dep_lag_168h=7; dep_mean_7d=5.4; dep_mean_28d=5.9; capacity=19; lat=20.6767; lon=-103.3475 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -Body $body -ContentType "application/json"
# viajes_estimados = ~7
```

### 2.3 CI/CD (GitHub Actions)

El workflow corre solo en cada push a `main` y a las ramas `feat/**`, `feature/**`, `docs/**` y `fix/**`, y
en cada PR hacia `main`:

1. **Pruebas unitarias y de calidad:** `ruff check .`, `python security/run_checks.py --skip-audit`, `pytest`.
2. **Construcción y validación de la imagen** (si el paso 1 pasa): `docker build` con el `Containerfile` y una
   prueba de humo que importa las dependencias y la API dentro del contenedor.

Para correr lo mismo en local antes de abrir un PR:

```powershell
ruff check .
python security/run_checks.py --skip-audit
python -m pytest -q tests
docker build -f Containerfile -t mibici-api:local .
```

### 2.4 Prueba de carga y recursos

Con el sistema arriba, en dos terminales:

```powershell
# Terminal 1: CPU y memoria de los contenedores cada ~5 s
python loadtest/monitor_recursos.py --duracion 320

# Terminal 2: escenario escalonado de 10 a 150 usuarios (5 min)
pip install -r requirements-carga.txt
locust -f loadtest/locustfile.py --host http://localhost:8000 --headless `
  --csv reports/carga/locust --csv-full-history --html reports/carga/locust.html
```

Después, `python loadtest/generar_reporte.py` regenera las gráficas y
[`reports/carga/reporte_carga.md`](../reports/carga/reporte_carga.md), que incluye las secciones de
recursos y FinOps.

---

## 3. Monitoreo y gobernanza

### 3.1 Drift con Evidently

Evidently va en un entorno aparte para no mezclarse con las dependencias de MLflow. Solo hay que crearlo una
vez:

```powershell
py -3.12 -m venv .venv-monitoring
.venv-monitoring\Scripts\python.exe -m pip install -r requirements-monitoring.txt
```

Después, con el `.venv` del proyecto activado y el servidor de MLflow arriba:

```powershell
python -m mibici.monitoring                          # data/monitoring/reference y current
.venv-monitoring\Scripts\python.exe mibici\drift_report.py
python -m mibici.drift_interpretation                # análisis por segmento
```

| Resultado | Valor |
|---|---|
| Calidad de datos | 0% nulos, 0 filas vacías, 0 columnas constantes |
| Variables con drift | 5 de 13: `month`, `dep_mean_28d`, `dep_mean_7d`, `dep_lag_168h`, `dep_lag_24h` |
| Causa | `month`: periodo actual incompleto (artefacto). Demanda: 100 estaciones nuevas con poca demanda y una caída real de 12% en las existentes |
| Efecto en el modelo | Sesgo cercano a cero; WAPE de 54.2% a 61.4%; 101% en estaciones nuevas |

### 3.2 A/B testing entre versiones

```powershell
python -m mibici.modeling.train retrain      # registra v2 como @challenger (no toca @champion)
python -m mibici.modeling.train ab-test      # compara y escribe la decisión; --promote true para promover
```

Resultado del 2026-11-15 (corrida `p1-ab-test-v1-vs-v2`): v2 mejora el WAPE 0.18 pp, por debajo del umbral
práctico de 0.5 pp, así que **`@champion` sigue en v1**. Para revertir una promoción basta con mover el alias
de vuelta; la API no fija número de versión.

### 3.3 FinOps

| Resultado | Valor |
|---|---|
| Cuello de botella | CPU (91-96% desde 50 usuarios); memoria al 22% |
| Capacidad con p95 ≤ 500 ms | ~20 peticiones/s por contenedor de 1 CPU |
| Costo por contenedor | USD 36.04 al mes (AWS Fargate x86, 1 vCPU / 2 GB) |
| Costo por inferencia | USD 0.70 por millón de peticiones; USD 0.17 por millón de predicciones |
| Mayores ahorros | Predicción en lote o precalculada con caché, ARM (~20%), Spot solo para réplicas extra (~69%), sacar MLflow del camino de la inferencia |

### 3.4 Model Card

[`MODEL_CARD.md`](MODEL_CARD.md) documenta el modelo vigente: detalles y linaje, uso previsto y fuera de
alcance, factores, métricas, datos, análisis cuantitativo (baseline, segmentos y A/B), operación,
consideraciones éticas y advertencias. Se actualiza cada vez que cambia `@champion`.

---

## 4. Fallas conocidas

### 4.1 La API se queda en "Waiting for application startup" y termina con `MlflowException`

El registry apunta a una corrida cuyos artefactos no están en `mlartifacts/` de esa carpeta (por ejemplo, si
el modelo se entrenó en otra computadora). El log dice `failed while downloading one or more artifacts`.
Copia `mlflow.db` **y** `mlartifacts/` juntos desde la misma base del equipo
([`README_reproducibilidad.md`](README_reproducibilidad.md), sección 5), o levanta temporalmente la versión
que sí tenga artefactos con `docker compose --profile ab up -d api-challenger`.

### 4.2 MLflow levanta vacío

`MLFLOW_STORE` apunta a una carpeta sin `mlflow.db`, y MLflow crea una base nueva. Revisa la ruta, corre
`docker compose up -d --force-recreate` y confirma en http://localhost:5001 → **Models** que aparece
`mibici-equipo24-p1-demanda`.

### 4.3 `error: externally-managed-environment` al instalar Locust en WSL o Linux

Crea un entorno virtual: `python3 -m venv ~/venv-carga`, `source ~/venv-carga/bin/activate` y luego
`pip install -r requirements-carga.txt`.

### 4.4 `ImportError: cannot import name 'infer_dtype_from_object'` al correr el reporte de drift

El entorno de monitoreo tiene pandas 3, que Evidently 0.7 no soporta. Reinstala con las versiones fijas:
`.venv-monitoring\Scripts\python.exe -m pip install -r requirements-monitoring.txt`.

### 4.5 Archivos "modificados" sin cambios reales en Windows

Git marca como modificados archivos que solo cambian de fin de línea (CRLF contra LF), sobre todo si se
alterna entre Git de Windows y de WSL. Comprueba con `git diff --ignore-cr-at-eol --stat` (sin salida = sin
cambios reales) y límpialos con `git restore .`.

---

## 5. Lista de verificación final

- [ ] `docker compose ps` muestra `mibici-mlflow` y `mibici-api` en *healthy*.
- [ ] `Invoke-RestMethod http://localhost:8000/health` devuelve `version = 1` y `uri = ...@champion`.
- [ ] `POST /predict` responde 200 con `viajes_estimados`; una entrada inválida responde 422.
- [ ] La última ejecución de **Actions** en `main` está en verde (pruebas e imagen).
- [ ] `reports/carga/reporte_carga.md` tiene 0% de errores y las secciones de recursos y FinOps.
- [ ] `reports/evidently/drift_report.html` abre en el navegador y muestra 5 de 13 variables con drift.
- [ ] `reports/ab_test/p1_ab_test.json` registra la decisión "vigente = champion".
- [ ] En **Models**, `@champion` → v1 y `@challenger` → v2, con el tag `ab_resultado`.
- [ ] [`MODEL_CARD.md`](MODEL_CARD.md) coincide con la versión que sirve la API.
