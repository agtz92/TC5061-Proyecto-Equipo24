"""Genera reporte de Data Drift y Data Quality con Evidently 0.7.x.

Compara el conjunto de referencia (entrenamiento MiBici 2022-01 a 2025-06)
con el conjunto actual (producción 2026-01 a 2026-09) y produce reportes HTML y JSON.

Uso:
    .venv-monitoring\\Scripts\\python.exe mibici/drift_report.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import evidently
from evidently import Report
from evidently.core.datasets import DataDefinition, Dataset, Regression
from evidently.presets import DataDriftPreset, DataSummaryPreset, RegressionPreset

print(
    f"Python {sys.version_info.major}.{sys.version_info.minor} | "
    f"evidently {evidently.__version__} | pandas {pd.__version__}",
    flush=True,
)

# ---------------------------------------------------------------------------
# Rutas y configuración de variables
# ---------------------------------------------------------------------------
MONITORING_DIR = Path("data/monitoring")
REPORTS_DIR = Path("reports/evidently")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "hour",
    "dow",
    "month",
    "is_weekend",
    "is_holiday",
    "is_quincena",
    "dep_lag_24h",
    "dep_lag_168h",
    "dep_mean_7d",
    "dep_mean_28d",
    "capacity",
    "lat",
    "lon",
]
TARGET_COL = "departures"
PRED_COL = "prediction"

# ---------------------------------------------------------------------------
# Cargar datasets
# ---------------------------------------------------------------------------
print("\n[1/4] Cargando datasets...", flush=True)
ref = pd.read_parquet(MONITORING_DIR / "reference.parquet")
cur = pd.read_parquet(MONITORING_DIR / "current.parquet")
print(f"  Referencia : {len(ref):,} filas  ({ref['ts'].min()} → {ref['ts'].max()})")
print(f"  Actual     : {len(cur):,} filas  ({cur['ts'].min()} → {cur['ts'].max()})")

cols = FEATURE_COLS + [TARGET_COL, PRED_COL]
ref_df = ref[[c for c in cols if c in ref.columns]].copy()
cur_df = cur[[c for c in cols if c in cur.columns]].copy()

# Muestra representativa para pruebas estadísticas de drift
N_SAMPLE = min(10000, len(ref_df), len(cur_df))
ref_sample = ref_df.sample(N_SAMPLE, random_state=42).reset_index(drop=True)
cur_sample = cur_df.sample(N_SAMPLE, random_state=42).reset_index(drop=True)
print(f"  Muestra para reporte: {N_SAMPLE:,} filas por partición", flush=True)

# ---------------------------------------------------------------------------
# Definición de esquema para Evidently
# ---------------------------------------------------------------------------
print("\n[2/4] Configurando definición de datos y presets...", flush=True)
data_def = DataDefinition(
    numerical_columns=FEATURE_COLS,
    regression=[Regression(name="default", target=TARGET_COL, prediction=PRED_COL)],
)

ref_dataset = Dataset.from_pandas(ref_sample, data_definition=data_def)
cur_dataset = Dataset.from_pandas(cur_sample, data_definition=data_def)

report = Report(
    metrics=[
        DataSummaryPreset(),
        DataDriftPreset(),
        RegressionPreset(),
    ]
)

# ---------------------------------------------------------------------------
# Ejecutar análisis
# ---------------------------------------------------------------------------
print("[3/4] Ejecutando análisis Evidently (cálculo de métricas y drift)...", flush=True)
snapshot = report.run(current_data=cur_dataset, reference_data=ref_dataset)

# ---------------------------------------------------------------------------
# Guardar resultados
# ---------------------------------------------------------------------------
print("[4/4] Guardando reportes...", flush=True)

html_path = REPORTS_DIR / "drift_report.html"
snapshot.save_html(str(html_path))
print(f"  ✓ Reporte HTML interactivo : {html_path}", flush=True)

json_path = REPORTS_DIR / "drift_report.json"
snapshot.save_json(str(json_path))
print(f"  ✓ Resumen JSON             : {json_path}", flush=True)

# ---------------------------------------------------------------------------
# Resumen en consola de Data Quality y Data Drift
# ---------------------------------------------------------------------------
try:
    data_dict = json.loads(snapshot.json())
    metrics_list = data_dict.get("metrics", [])

    print("\n" + "=" * 76)
    print("RESUMEN DE CALIDAD DE DATOS (Data Quality)")
    print("=" * 76)
    row_count = None
    missing_total = None
    for m in metrics_list:
        name = m.get("metric_name", "")
        if name == "RowCount()":
            row_count = int(m.get("value", 0))
        elif name == "DatasetMissingValueCount()":
            missing_total = m.get("value", {})
        elif "EmptyRowsCount" in name:
            print(f"  Filas vacías               : {m.get('value')}")
        elif "DuplicatedRowCount" in name:
            print(f"  Filas duplicadas           : {m.get('value')}")

    if row_count:
        print(f"  Total filas analizadas     : {row_count:,} por partición")
    if missing_total:
        print(f"  Valores nulos en dataset   : {missing_total.get('count', 0)} ({missing_total.get('share', 0):.2%})")

    print("\n" + "=" * 76)
    print("RESUMEN DE DATA DRIFT POR VARIABLE (MiBici P1 Demanda)")
    print("=" * 76)
    print(f"{'Variable':<18} {'¿Drift?':>8}  {'Score':>8}  {'Umbral':>7}  Método de Prueba")
    print("-" * 76)

    drift_count = 0
    total_vars = 0

    # Parsear ValueDrift de cada variable
    for m in metrics_list:
        name = m.get("metric_name", "")
        if "ValueDrift" in name:
            total_vars += 1
            # Extraer columna, método y umbral del string metric_name
            # Formato: ValueDrift(column=...,method=...,threshold=0.1)
            col_match = re.search(r"column=([^,)]+)", name)
            method_match = re.search(r"method=([^,)]+)", name)
            thresh_match = re.search(r"threshold=([^,)]+)", name)

            col = col_match.group(1) if col_match else name
            method = method_match.group(1) if method_match else "N/A"
            threshold = float(thresh_match.group(1)) if thresh_match else 0.1
            score = float(m.get("value", float("nan")))

            drift_detected = score > threshold
            if drift_detected:
                drift_count += 1
            detected_str = "SÍ ⚠" if drift_detected else "no "

            print(f"  {col:<16} {detected_str:>8}  {score:>8.4f}  {threshold:>7.2f}  {method}")

    print("-" * 76)
    share = (drift_count / total_vars) * 100 if total_vars else 0
    print(f"  Variables con drift: {drift_count}/{total_vars} ({share:.1f}%)")

    print("\n" + "=" * 76)
    print("DESEMPEÑO DEL MODELO EN PRODUCCIÓN (Regression Metrics)")
    print("=" * 76)
    for m in metrics_list:
        name = m.get("metric_name", "")
        if "RMSE" in name:
            print(f"  RMSE                       : {float(m.get('value', 0)):.4f} viajes/hora")
        elif "MAE" in name and "AbsMax" not in name:
            val = m.get("value", {})
            mae_mean = val.get("mean", 0) if isinstance(val, dict) else val
            print(f"  MAE                        : {float(mae_mean):.4f} viajes/hora")
        elif "R2Score" in name:
            print(f"  R² Score                   : {float(m.get('value', 0)):.4f}")
        elif "MeanError" in name:
            val = m.get("value", {})
            me_mean = val.get("mean", 0) if isinstance(val, dict) else val
            print(f"  Sesgo medio (Mean Error)   : {float(me_mean):+.4f} viajes/hora")

except Exception as e:
    print(f"Nota al procesar resumen en consola: {e}")

print("\n✓ Proceso completado exitosamente.")
