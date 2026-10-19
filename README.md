# TC5061-Proyecto-Equipo24 · MiBici

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto integrador de **TC5061 Operaciones de aprendizaje automático (MLOps)**. Usamos los datos abiertos de
[MiBici](https://www.mibici.net/es/datos-abiertos/), el sistema de bicicleta pública del Área Metropolitana de
Guadalajara, para responder tres preguntas de operación con Machine Learning.

## Preguntas del proyecto

| # | Pregunta de negocio | Tarea | Planteamiento | Quién lo usa |
|---|---|---|---|---|
| P1 | ¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro? | Regresión | Salidas por estación en cada hora del día siguiente (horizonte 24 h), agregables a zona y día | Operador del sistema (flota) |
| P2 | ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes? | Clasificación | Clase {normal, vaciado, saturación} por estación y hora del día siguiente, con etiqueta proxy de flujo neto y huellas de censura | Equipo de rebalanceo |
| P3 | ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican? | Clustering | Grupos de estaciones por perfil horario de salidas y llegadas, proyectados en el mapa | Planeación urbana |

El ML Canvas, la formulación completa y la arquitectura están en [`references/`](references/).

## Datos

| | |
|---|---|
| Fuente | Portal de datos abiertos de MiBici (`datos_abiertos_AAAA_MM.csv` y `nomenclatura_2026_09.csv`) |
| Rango | **2022-01 a 2026-09** (57 meses). Se excluye 2014-2021 porque la pandemia cambió los hábitos de viaje |
| Tamaño | 21,036,075 viajes, 1.8 GB en CSV, 473 estaciones con viajes, 484 en la nomenclatura |
| Retos de formato | 36 meses en Latin-1 y 21 en UTF-8; encabezado corrupto en 2024-02 y 2024-05; fechas `DD/MM/AAAA h:mm` en 2023-11, 2024-09 y 2024-10 |
| Datos personales | `Usuario_Id`, `Genero` y `Año_de_nacimiento` se eliminan en la ingesta; los modelos solo usan agregados por estación y hora |
| Partición | Temporal: entrenamiento hasta 2025-06, validación 2025-07 a 2025-12, prueba 2026-01 a 2026-09 |

Los datos **no** se suben a Git. Se reconstruyen con un comando y se versionan con
[`data_manifest.json`](data_manifest.json): hash SHA-256 y número de registros de cada archivo. Su huella corta
(`data_version`) se registra como parámetro y etiqueta en cada run de MLflow.

## Estructura del repositorio

Generada con [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)
(`pipx install cookiecutter-data-science` y `ccds` con `include_code_scaffold = Yes`); ese resultado es el
primer commit del repositorio.

```
├── MLproject              <- Entry points de MLflow Projects (ingest, features, p1, p2, p3, main)
├── python_env.yaml        <- Entorno para `mlflow run`
├── Makefile               <- Atajos: make data, make features, make experiment, make smoke
├── data_manifest.json     <- Versión de los datos (se genera con la ingesta; sí se versiona)
├── requirements.txt       <- Dependencias con versión fija
├── data/                  <- Ignorado por Git
│   ├── raw/               <- CSV originales descargados del portal
│   ├── interim/           <- Parquet por mes, ya normalizado y sin datos personales
│   └── processed/         <- Panel estación × hora con features y perfiles de estaciones
├── models/                <- Modelos serializados (ignorado por Git)
├── notebooks/             <- Solo exploración: 1.0-jagg-exploracion-viajes.ipynb
├── references/            <- ML Canvas, propuesta y diagrama de arquitectura
├── reports/figures/       <- Figuras para reportes
├── tests/                 <- Pruebas y generador de datos sintéticos
└── mibici/                <- Módulo del proyecto
    ├── config.py          <- Rutas, fechas, partición, semillas, umbrales y MLflow (único lugar)
    ├── dataset.py         <- Ingesta reproducible y manifiesto de datos
    ├── features.py        <- Panel estación × hora, rezagos, etiquetas P2 y perfiles P3
    ├── modeling/train.py  <- Experimento base en MLflow (baselines de P1, P2 y P3)
    ├── modeling/predict.py<- Predicción batch (siguiente entrega)
    └── plots.py           <- Gráficas compartidas (siguiente entrega)
```

## Instalación

Requiere Python 3.12 (también funciona con 3.11 y 3.13) y un servidor de MLflow.

```bash
git clone https://github.com/<usuario>/TC5061-Proyecto-Equipo24.git
cd TC5061-Proyecto-Equipo24
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # MIBICI_TEAM=equipo24 y la URI de MLflow
```

Servidor de MLflow (en otra terminal). Se usa el puerto 5001 porque en macOS el 5000 lo ocupa el receptor de AirPlay:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts \
  --serve-artifacts --host 127.0.0.1 --port 5001
```

## Uso

```bash
# 1. Ingesta: descarga 2022-01 a 2026-09, normaliza, limpia y escribe data_manifest.json
python -m mibici.dataset
python -m mibici.dataset verify          # comprueba los CSV contra el manifiesto

# 2. Features: panel estación × hora y perfiles de estaciones
python -m mibici.features

# 3. Experimento base: un run por pregunta (P1, P2, P3) más K-Means para P3
python -m mibici.modeling.train all
```

Lo mismo con MLflow Projects (usa el entorno actual; el nombre del experimento debe coincidir con el de `.env`):

```bash
export MLFLOW_TRACKING_URI=http://localhost:5001
mlflow run . -e ingest   --env-manager local
mlflow run . -e features --env-manager local
mlflow run . -e p1 --env-manager local --experiment-name mibici-equipo24
mlflow run . -e p2 --env-manager local --experiment-name mibici-equipo24
mlflow run . -e p3 --env-manager local --experiment-name mibici-equipo24 -P model=kmeans -P k=5
```

Prueba de punta a punta **sin internet** con datos sintéticos que imitan los formatos del portal:

```bash
make smoke
```

## Experimento base

Experimento `mibici-equipo24` en MLflow. Runs:

| Run | Pregunta | Baseline | Métricas principales | Artefactos |
|---|---|---|---|---|
| `p1-baseline-estacional-lag168h` | P1 | Misma estación y hora de la semana anterior | `val_/test_` mae, rmse, wape, poisson_deviance, wape_station_day, wape_poisson_floor | WAPE vs. piso Poisson por nivel de demanda |
| `p2-baseline-regla-q3` | P2 | Alerta si las salidas o llegadas de la semana anterior superan el Q3 propio de la estación | `val_/test_` f1_macro, precision/recall por clase, expected_cost_per_1000, alert_rate | Matriz de confusión |
| `p3-baseline-municipio` | P3 | Agrupación por municipio (GDL, ZPN, TLQ) | `train_` silhouette, davies_bouldin, calinski_harabasz | Mapa y perfiles por grupo |
| `p3-kmeans-k5` | P3 | K-Means sobre PCA de los perfiles | Las anteriores más jaccard_bootstrap_mean y ari_vs_municipio | Mapa y perfiles por grupo |

Los criterios para aprobar un modelo (brecha cerrada contra el piso Poisson en P1, costo esperado en P2,
silueta y estabilidad bootstrap en P3) están justificados en la propuesta (`references/`).

## Buenas prácticas y dónde se aplican

| Práctica | Cómo se aplica |
|---|---|
| Un experimento por proyecto y runs con nombre descriptivo | `MLFLOW_EXPERIMENT = mibici-<equipo>` en `config.py`; cada run tiene `run_name` explícito |
| Registrar lo necesario para reproducir | Params: `start_month`, `end_month`, partición, `data_version`, `seed`, `model` e hiperparámetros |
| Ligar el run al código y a la pregunta | Tags: `git_commit`, `git_dirty`, `question`, `team`, `data_version` |
| Métricas con el conjunto en el nombre | `val_mae`, `test_wape`, `train_silhouette`, etc. |
| Sin credenciales ni datos personales en MLflow | Solo métricas agregadas, gráficas y tablas por estación; columnas personales eliminadas en la ingesta |
| Notebooks solo para exploración | `notebooks/1.0-jagg-exploracion-viajes.ipynb`; la lógica vive en `mibici/` |
| Configuración en un solo lugar | Rutas, fechas, semillas, umbrales y MLflow en `mibici/config.py` |
| Dependencias fijas y entorno reproducible | `requirements.txt` con versiones exactas; `python_env.yaml` para `mlflow run` |
| Datos, modelos, MLflow y credenciales fuera de Git | `.gitignore` excluye `data/`, `mlruns/`, `mlartifacts/`, `*.db`, modelos y `.env` |
| Sin fuga de información | Features con corte a las 23:59 del día anterior; umbrales y perfiles solo con entrenamiento |

## Flujo de trabajo con Git

- **GitHub Flow.** `main` está protegida: nadie hace push directo; todo cambio entra por Pull Request con al
  menos una revisión aprobada.
- **Ramas:** `tipo/descripcion-corta`, con tipo `feat`, `fix`, `docs`, `exp` (experimentos) o `chore`.
  Ejemplo: `feat/ingesta-mibici`, `exp/p1-lightgbm`.
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/es/v1.0.0/), en español:
  `feat(dataset): detecta codificación por archivo`, `fix(features): rezago sin fuga`.
- **Pull Requests:** se usa la plantilla `.github/pull_request_template.md` (qué cambia, pregunta que atiende,
  cómo se probó y enlace al run de MLflow si aplica). Cada integrante abre al menos un PR y revisa al menos uno
  de otra persona; el autor no aprueba su propio PR.
- **Protección de `main`** (Settings → Branches → Add branch ruleset): requerir Pull Request antes de
  integrar, al menos 1 aprobación, descartar aprobaciones al subir nuevos commits y bloquear force push.

## Integrantes y roles

| Integrante | Rol principal |
|---|---|
| José Alfredo Gutiérrez Guerra | _por definir_ |
| Antonio Abraham Díaz Arvizu | Integrante |
| _Integrante 3_ | _por definir_ |
| José Antonio Padilla Pereira | _por definir_ |
