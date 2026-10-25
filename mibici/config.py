"""Configuración central del proyecto MiBici.

Todas las rutas, fechas, semillas, umbrales y nombres de MLflow viven aquí.
Ningún otro archivo debe repetir estos valores: se importan desde este módulo.
Los valores que dependen de la máquina (equipo, URI de MLflow) se pueden
sobrescribir con variables de entorno en el archivo .env (no versionado).
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# --------------------------------------------------------------------------
# Rutas (estructura de Cookiecutter Data Science)
# --------------------------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Archivos intermedios y procesados
TRIPS_INTERIM_DIR = INTERIM_DATA_DIR / "trips"            # un Parquet por mes
STATIONS_PARQUET = INTERIM_DATA_DIR / "stations.parquet"
PANEL_PARQUET = PROCESSED_DATA_DIR / "panel_station_hour.parquet"
PROFILES_PARQUET = PROCESSED_DATA_DIR / "station_profiles.parquet"

# Manifiesto de datos: SÍ se versiona en Git (vive fuera de data/)
DATA_MANIFEST = PROJ_ROOT / "data_manifest.json"

# --------------------------------------------------------------------------
# Fuente de datos
# --------------------------------------------------------------------------
PORTAL_URL = "https://www.mibici.net/es/datos-abiertos/"
NOMENCLATURA_FILE = "nomenclatura_2026_09.csv"
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (TC5061 MLOps; proyecto academico MiBici)"}
HTTP_TIMEOUT_S = 120

# Rango de fechas del proyecto (post COVID). Formato AAAA-MM, inclusivo.
START_MONTH = "2022-01"
END_MONTH = "2026-09"

# Partición temporal (inclusiva). Nunca aleatoria.
TRAIN_END = "2025-06-30 23:00"
VAL_START, VAL_END = "2025-07-01 00:00", "2025-12-31 23:00"
TEST_START, TEST_END = "2026-01-01 00:00", "2026-09-30 23:00"

# Zona horaria de los registros (hora local de Guadalajara)
TIMEZONE = "America/Mexico_City"

# --------------------------------------------------------------------------
# Esquema de los CSV de viajes
# --------------------------------------------------------------------------
# Los encabezados cambian de codificación entre meses (p. ej. "Año_de_nacimiento"
# llega corrupto en 2024-02 y 2024-05), así que las columnas se mapean por POSICIÓN.
RAW_COLUMNS = [
    "viaje_id",
    "usuario_id",
    "genero",
    "anio_nacimiento",
    "inicio",
    "fin",
    "origen_id",
    "destino_id",
]
# Columnas con datos personales: se eliminan al convertir a Parquet.
PERSONAL_COLUMNS = ["usuario_id", "genero", "anio_nacimiento"]

# Formatos de fecha observados en los archivos 2022-2026.
DATE_FORMATS = ["%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S"]
MAX_UNPARSED_DATES_PCT = 1.0  # falla la ingesta si más del 1% de fechas no se reconoce

# Reglas de limpieza
MIN_TRIP_SECONDS_SAME_STATION = 60  # < 1 min con mismo origen y destino = anclaje fallido
MAX_TRIP_HOURS = 3

# --------------------------------------------------------------------------
# Features y etiquetas
# --------------------------------------------------------------------------
LAG_HOURS = [24, 168]
ROLLING_DAYS = [7, 28]

# P2: etiqueta proxy de riesgo
P2_NET_FLOW_THRESHOLD = 0.5      # |flujo neto| / capacidad
P2_CENSOR_MIN_EXPECTED = 3.0     # salidas (o llegadas) esperadas para buscar huella de censura
P2_CENSOR_WEEKS = 8              # semanas previas para la expectativa
P2_BASELINE_QUANTILE = 0.75      # regla baseline: tercer cuartil propio de la estación
P2_COST_FN_OVER_FP = 3.0         # c = costo FN / costo FP (supuesto inicial; se estima con datos)

# P3: clustering
P3_K_RANGE = list(range(3, 11))
P3_DEFAULT_K = 5

# --------------------------------------------------------------------------
# Modelado (valores por defecto del entry point `train` de MLproject)
# --------------------------------------------------------------------------
# Variables de entrada de P1 y P2. Todas se conocen a las 00:00 del día anterior
# a la predicción (ver docstrings de mibici.features).
P1_FEATURES = [
    "hour", "dow", "month", "is_weekend", "is_holiday", "is_quincena",
    "dep_lag_24h", "dep_lag_168h", "dep_mean_7d", "dep_mean_28d",
    "capacity", "lat", "lon",
]
P2_FEATURES = P1_FEATURES + [
    "arr_lag_24h", "arr_lag_168h", "netrel_mean_28d", "exp_dep_8w", "exp_arr_8w",
]

# Modelos disponibles por pregunta y el que se usa si no se indica otro.
MODEL_CHOICES = {
    "p1": ["baseline", "poisson", "rf", "hgb"],
    "p2": ["baseline", "logreg", "rf", "hgb"],
    "p3": ["municipio", "kmeans", "ward", "gmm"],
}
DEFAULT_MODEL = {"p1": "hgb", "p2": "hgb", "p3": "kmeans"}

# Hiperparámetros por defecto. Cada modelo usa solo los que le aplican:
# rf: n_estimators, max_depth, min_samples_leaf | hgb: max_iter, learning_rate,
# max_depth, min_samples_leaf | poisson: alpha | logreg: C | kmeans/ward/gmm: k
HYPERPARAMS = {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_leaf": 50,
    "max_iter": 300,
    "learning_rate": 0.1,
    "alpha": 1e-4,
    "C": 1.0,
    "k": P3_DEFAULT_K,
}

# Tamaño de la muestra de entrenamiento (filas estación-hora). La muestra es
# determinista: se elige por hash de (estación, hora, SEED).
TRAIN_SAMPLE_ROWS = 2_000_000
P2_CALIBRATION_ROWS = 500_000   # filas de validación para calibrar probabilidades (P2)

# --------------------------------------------------------------------------
# Experimentación (Parte B): familias de modelos, búsquedas y criterio de selección
# --------------------------------------------------------------------------
# Etiqueta model_family de cada run, para filtrar y comparar en la interfaz de MLflow.
MODEL_FAMILY = {
    "baseline": "baseline", "municipio": "baseline",
    "poisson": "glm", "logreg": "glm",
    "rf": "bagging_trees", "hgb": "boosting_trees",
    "kmeans": "partitional", "ward": "hierarchical", "gmm": "probabilistic",
}

# Búsquedas de hiperparámetros del entry point `search`: una corrida padre por
# búsqueda y una corrida hija (nested run) por combinación de la rejilla.
SEARCH_GRIDS = {
    "p1": {"model": "hgb", "grid": {"learning_rate": [0.05, 0.1], "max_iter": [200, 400], "max_depth": [8, 16]}},
    "p2": {"model": "hgb", "grid": {"learning_rate": [0.05, 0.1], "max_depth": [6, 12], "min_samples_leaf": [50, 200]}},
    "p3": {"model": "kmeans", "grid": {"k": P3_K_RANGE}},
}

# Métrica con la que la corrida padre elige a su mejor hija. Se usa validación
# (nunca prueba) para no seleccionar con el conjunto que reporta el desempeño final.
SELECTION_METRIC = {
    "p1": ("val_wape", "min"),                       # error en viajes, unidad de decisión del operador
    "p2": ("val_expected_cost_per_1000", "min"),     # costo de la decisión de enviar camión
    "p3": ("train_silhouette", "max"),               # cohesión y separación (sin etiquetas)
}

# --------------------------------------------------------------------------
# Reproducibilidad y MLflow
# --------------------------------------------------------------------------
SEED = 42

TEAM = os.getenv("MIBICI_TEAM", "equipo24")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", f"mibici-{TEAM}")

# Configura loguru para no romper las barras de progreso de tqdm
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except (ModuleNotFoundError, ValueError):
    pass
