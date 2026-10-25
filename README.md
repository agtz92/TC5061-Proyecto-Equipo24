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
├── MLproject              <- Entry points de MLflow Projects (ingest, features, train, baselines)
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
    ├── modeling/train.py  <- CLI de entrenamiento (entry point `train` de MLproject)
    ├── modeling/tracking.py <- Registro en MLflow: tags, parámetros, figuras y modelo
    ├── modeling/p1_demand.py, p2_risk.py, p3_clusters.py <- Modelos y evaluación por pregunta
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

### Servidor de tracking con base de datos

El Model Registry de MLflow requiere un backend con base de datos, así que el servidor usa SQLite
(`mlflow.db`, excluido de Git junto con `mlartifacts/` y `mlruns/`). En otra terminal, desde la raíz
del repo:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts \
  --serve-artifacts --host 127.0.0.1 --port 5001
```

Usamos el puerto **5001** porque en macOS el 5000 lo ocupa el receptor de AirPlay (responde 403). En Linux
o Windows puede usarse `--port 5000`, ajustando la URI. Después, en la terminal de trabajo:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5001
export MLFLOW_EXPERIMENT_NAME=mibici-equipo24
```

## Reproducir desde un clon limpio (MLflow Project)

Todas las corridas se lanzan desde el MLflow Project (`MLproject`), nunca desde un notebook.
`--env-manager local` usa el entorno virtual ya instalado con `requirements.txt`; sin esa bandera MLflow
crea un entorno nuevo a partir de `python_env.yaml` (mismas versiones fijas; requiere `pyenv`).

```bash
# 1. Ingesta: descarga 2022-01 a 2026-09, normaliza, limpia y escribe data_manifest.json
mlflow run . -e ingest --env-manager local

# 2. Entrenamiento: construye las features si faltan, entrena, evalúa y registra el run
mlflow run . -e train --env-manager local -P question=p1 -P model=rf -P n_estimators=300
mlflow run . -e train --env-manager local -P question=p2 -P model=hgb -P learning_rate=0.05
mlflow run . -e train --env-manager local -P question=p3 -P model=kmeans -P k=6

# Búsqueda de hiperparámetros con nested runs (una corrida padre y una hija por combinación)
mlflow run . -e search --env-manager local -P question=p1

# Baselines de la propuesta (P1, P2 y P3 en un solo comando)
mlflow run . -e baselines --env-manager local
```

### Puntos de entrada

| Entry point | Qué ejecuta | Parámetros |
|---|---|---|
| `ingest` | `mibici/dataset.py` | `start`, `end` (AAAA-MM) |
| `features` | `mibici/features.py` (opcional; `train` lo corre si faltan las features) | |
| `train` | `mibici/modeling/train.py` | `question`, `model` y los hiperparámetros de la tabla siguiente |
| `search` | Búsqueda de hiperparámetros con nested runs (`SEARCH_GRIDS` en `config.py`) | `question`, `sample_rows` |
| `baselines` | Los tres baselines de la propuesta | |
| `register` | Registra un run en el Model Registry con nombre canónico y linaje completo | `question`, `run_id`, `alias` (por defecto `champion`) |

Todo parámetro que no se indique vale `config` y se toma de `mibici/config.py` (`START_MONTH`, `END_MONTH`,
`DEFAULT_MODEL`, `HYPERPARAMS`, `TRAIN_SAMPLE_ROWS`, `SEED`). Así `MLproject` no repite rutas, fechas ni
semillas. En cada run, los parámetros con el nombre del `MLproject` muestran lo que se pasó en la línea de
comandos y los parámetros `hp_*` y `estimator` muestran el valor efectivo usado.

### Modelos e hiperparámetros por pregunta

| Pregunta | `model` | Hiperparámetros que usa | Por defecto (`config.py`) |
|---|---|---|---|
| P1 demanda | `baseline`, `poisson`, `rf`, `hgb` | poisson: `alpha` · rf: `n_estimators`, `max_depth`, `min_samples_leaf` · hgb: `max_iter`, `learning_rate`, `max_depth`, `min_samples_leaf` | `hgb` (pérdida Poisson) |
| P2 riesgo | `baseline`, `logreg`, `rf`, `hgb` | logreg: `C` · rf y hgb: como en P1 | `hgb` con pesos de clase, calibrado |
| P3 tipologías | `municipio`, `kmeans`, `ward`, `gmm` | `k` | `kmeans`, `k=5` |

- **P1:** entrena con una muestra determinista de 2 millones de filas estación-hora (hash de estación,
  hora y semilla). Métricas `val_`/`test_`: mae, rmse, wape, poisson_deviance, wape_station_day,
  wape_baseline, wape_poisson_floor, `gap_closed` (fracción de la brecha baseline → piso cerrada) y
  `dm_pvalue` (Diebold-Mariano contra el baseline).
- **P2:** entrena con todas las horas de riesgo y una muestra de las normales; calibra las probabilidades
  (isotónica) con validación y alerta si P(riesgo) ≥ 1/(1+c). Métricas: expected_cost_per_1000 (y el del
  baseline en las mismas filas), cost_reduction_vs_baseline, precision/recall/f1 por clase,
  avg_precision por clase de riesgo, brier_risk, alerts_per_day.
- **P3:** silhouette, davies_bouldin, calinski_harabasz, Jaccard bootstrap (media y mínimo por cluster) y
  ARI contra el municipio.
- P1 y P2 registran el modelo con firma y ejemplo de entrada (`model/`), listo para el Model Registry.
  P3 registra un asignador (estandarización + PCA + centroide más cercano) que ubica cualquier estación,
  nueva o existente, en el grupo cuyo centroide está más cerca.

### Experimentación: búsquedas, etiquetas y artefactos

**Búsquedas con nested runs.** `mlflow run . -e search -P question=p1|p2|p3` crea una corrida padre
(`p1-search-hgb`, `p2-search-hgb`, `p3-search-kmeans`) y una corrida hija por cada combinación de la rejilla
definida en `SEARCH_GRIDS` (`config.py`). Al terminar, la corrida padre registra la tabla de resultados
(`data/<pregunta>_search_<modelo>_resultados.csv`), la etiqueta `best_child_run_id`, la etiqueta
`best_hyperparams` y la métrica `best_<métrica>`. La mejor hija se elige con `SELECTION_METRIC`, calculada en
**validación** (nunca en prueba): `val_wape` en P1, `val_expected_cost_per_1000` en P2 y `train_silhouette`
en P3.

**Etiquetas para filtrar en la interfaz** (además de `team`, `question`, `git_commit`, `git_dirty` y
`data_version`):

| Etiqueta | Valores | Filtro de ejemplo en MLflow |
|---|---|---|
| `stage_of_work` | `baseline`, `search`, `candidate` | `tags.stage_of_work = 'candidate'` |
| `model_family` | `baseline`, `glm`, `bagging_trees`, `boosting_trees`, `partitional`, `hierarchical`, `probabilistic` | `tags.model_family = 'boosting_trees'` |
| `model` | nombre del estimador (`hgb`, `rf`, `kmeans`, ...) | `tags.question = 'P1' and tags.model = 'hgb'` |

**Artefactos de cada corrida**, organizados en subcarpetas y sin datos personales:

| Carpeta | Contenido |
|---|---|
| `model/` | Modelo con firma (`signature`) y ejemplo de entrada |
| `plots/` | P1: WAPE por nivel de demanda, real contra predicho y residuales · P2: matriz de confusión y curva precisión-recall · P3: mapa y perfiles horarios por grupo |
| `data/` | `data_manifest.json` (versión de los datos) y tablas agregadas por estación o por nivel de demanda |
| `config/` | `run_config.json` (parámetros y etiquetas efectivos), `config.py`, `MLproject`, `python_env.yaml` y `requirements.txt` |

El conjunto de entrenamiento se registra además con `mlflow.log_input` (pestaña *Datasets* del run): MLflow
guarda su esquema, perfil y digest, no las filas. Las figuras ya no se copian a `reports/figures/` en cada
corrida, porque esa carpeta está en Git y sobrescribirla marcaría el árbol como modificado
(`git_dirty=true`). Para copiarlas, exporta `MIBICI_SAVE_FIGURES=1`.

### Modelo elegido y registro en el Model Registry

Cada corrida de `train` (no-baseline, no-búsqueda) registra automáticamente su modelo
en el MLflow Model Registry con nombres canónicos:

| Pregunta | Nombre en el registry |
|---|---|
| P1 demanda | `mibici-equipo24-p1-demanda` |
| P2 riesgo | `mibici-equipo24-p2-riesgo` |
| P3 tipologías | `mibici-equipo24-p3-grupos` |

Cada versión conserva su **linaje completo** en las etiquetas de la versión:
`question`, `estimator`, `experiment_name`, `source_run_id`, `source_run_name`,
`git_commit`, `git_dirty`, `data_version`, `team`, `model_family` y `stage_of_work`.

Para **promover el modelo elegido** tras comparar corridas en la UI, usa el entry point
`register` con el `run_id` de la corrida ganadora y el alias `champion`:

```bash
# Desde la UI de MLflow, copia el run_id de la corrida elegida
mlflow run . -e register --env-manager local \
  -P question=p1 -P run_id=<run_id> -P alias=champion

# O directamente como módulo
python -m mibici.modeling.train register \
  --question p1 --run-id <run_id> --alias champion
```

El entry point también funciona para P2 y P3.

| Pregunta | Modelo elegido | Comando para reproducirlo |
|---|---|---|
| P1 | _por definir tras comparar runs_ | `mlflow run . -e train --env-manager local -P question=p1 -P model=...` |
| P2 | _por definir tras comparar runs_ | `mlflow run . -e train --env-manager local -P question=p2 -P model=...` |
| P3 | _por definir tras comparar runs_ | `mlflow run . -e train --env-manager local -P question=p3 -P model=...` |

### Sin MLflow Project

Los mismos pasos como módulos de Python (útil para depurar):

```bash
python -m mibici.dataset
python -m mibici.dataset verify          # comprueba los CSV contra el manifiesto
python -m mibici.features
python -m mibici.modeling.train run --question p1 --model rf --n-estimators 300
python -m mibici.modeling.train baselines
```

Prueba de punta a punta **sin internet** con datos sintéticos que imitan los formatos del portal:

```bash
make smoke
```

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
| Damián López Virgen | Analista de Datos y Visualización |
| José Antonio Padilla Pereira | Integrante |
