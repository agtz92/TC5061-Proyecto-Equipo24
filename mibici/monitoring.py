"""Preparación de conjuntos de datos para monitoreo de drift y desempeño en producción.

Define y genera:
- Conjunto de Referencia (Reference Dataset): datos históricos con los que se entrenó el modelo.
- Conjunto Actual (Current Dataset): datos de producción en un periodo posterior (datos abiertos 2026).

Las predicciones salen del modelo vigente del Model Registry (`models:/mibici-equipo24-p1-demanda@champion`,
el mismo que sirve la API), y la versión usada queda registrada en `data/monitoring/modelo_monitoreo.json`.

Ejemplos de uso:
    python -m mibici.monitoring
    python -m mibici.monitoring --model-uri models:/mibici-equipo24-p1-demanda@challenger
    python -m mibici.monitoring --reference-start 2025-01-01 --reference-end 2025-06-30 --current-start 2026-09-01 --current-end 2026-09-30
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
import typer

from mibici.config import (
    CHAMPION_ALIAS,
    MLFLOW_TRACKING_URI,
    P1_FEATURES,
    PROJ_ROOT,
    REGISTRY_NAME,
    START_MONTH,
    TEST_END,
    TEST_START,
    TRAIN_END,
)
from mibici.features import load_supervised_frame

app = typer.Typer(help="Preparación de datasets de monitoreo (Reference vs Current).")

MONITORING_DIR = PROJ_ROOT / "data" / "monitoring"
# Mismo modelo que sirve la API y que evalúa mibici/drift_interpretation.py.
DEFAULT_MODEL_URI = f"models:/{REGISTRY_NAME['p1']}@{CHAMPION_ALIAS}"
TARGET_COL = "departures"


def cargar_modelo(model_uri: str = DEFAULT_MODEL_URI):
    """Carga el modelo y devuelve (modelo, info) con la versión del registry que se usó.

    `model_uri` puede ser `models:/<nombre>@<alias>`, `models:/<nombre>/<versión>`, `runs:/...` o una
    carpeta local con un modelo de MLflow.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"Cargando modelo desde '{model_uri}'...")
    model = mlflow.pyfunc.load_model(str(model_uri))
    info = {"model_uri": str(model_uri), "run_id": model.metadata.run_id}
    if str(model_uri).startswith("models:/") and "@" in str(model_uri):
        nombre, alias = str(model_uri).removeprefix("models:/").split("@")
        mv = mlflow.MlflowClient().get_model_version_by_alias(nombre, alias)
        info.update({"model_name": nombre, "alias": alias, "version": str(mv.version), "run_id": mv.run_id})
    logger.info(f"Modelo cargado: {info}")
    return model, info


def predecir_con_modelo(df: pd.DataFrame, model) -> np.ndarray:
    """Calcula predicciones para un DataFrame con un modelo pyfunc ya cargado."""
    X = df[P1_FEATURES].astype("float32")
    preds = model.predict(X)
    return np.clip(np.asarray(preds, dtype=float), 0, None).round(3)


def preparar_datasets(
    reference_start: str = f"{START_MONTH}-01",
    reference_end: str = TRAIN_END,
    current_start: str = TEST_START,
    current_end: str = TEST_END,
    sample_reference: int | None = 50_000,
    sample_current: int | None = 50_000,
    output_dir: Path = MONITORING_DIR,
    export_csv: bool = True,
    model_uri: str = DEFAULT_MODEL_URI,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extrae, procesa y guarda los conjuntos de datos de referencia y actual."""
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["station_id", TARGET_COL, *P1_FEATURES]

    logger.info(f"Extrayendo conjunto de REFERENCIA ({reference_start} a {reference_end})...")
    df_ref = load_supervised_frame(
        start=reference_start,
        end=reference_end,
        columns=columns,
        sample_rows=sample_reference,
    )

    logger.info(f"Extrayendo conjunto ACTUAL ({current_start} a {current_end})...")
    df_cur = load_supervised_frame(
        start=current_start,
        end=current_end,
        columns=columns,
        sample_rows=sample_current,
    )

    # Predicciones del modelo vigente. Si no se puede cargar, se detiene: un reporte de desempeño
    # sin predicciones, o con las de otro modelo, no sirve para monitorear el que está en producción.
    model, info = cargar_modelo(model_uri)
    logger.info("Generando predicciones para referencia...")
    df_ref["prediction"] = predecir_con_modelo(df_ref, model)
    logger.info("Generando predicciones para actual...")
    df_cur["prediction"] = predecir_con_modelo(df_cur, model)
    info.update({"reference": [reference_start, reference_end], "current": [current_start, current_end],
                 "reference_rows": len(df_ref), "current_rows": len(df_cur)})
    (output_dir / "modelo_monitoreo.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    # Guardar en Parquet
    ref_parquet = output_dir / "reference.parquet"
    cur_parquet = output_dir / "current.parquet"
    df_ref.to_parquet(ref_parquet, index=False)
    df_cur.to_parquet(cur_parquet, index=False)
    logger.success(f"Referencia guardada: {ref_parquet} ({len(df_ref):,} filas)")
    logger.success(f"Actual guardado:     {cur_parquet} ({len(df_cur):,} filas)")

    if export_csv:
        ref_csv = output_dir / "reference.csv"
        cur_csv = output_dir / "current.csv"
        df_ref.to_csv(ref_csv, index=False)
        df_cur.to_csv(cur_csv, index=False)
        logger.info(f"Archivos CSV exportados en {output_dir}")

    return df_ref, df_cur


@app.callback(invoke_without_command=True)
def main(
    reference_start: str = typer.Option(
        f"{START_MONTH}-01", help="Inicio del periodo de referencia (AAAA-MM-DD)"
    ),
    reference_end: str = typer.Option(
        TRAIN_END, help="Fin del periodo de referencia (AAAA-MM-DD HH:MM)"
    ),
    current_start: str = typer.Option(
        TEST_START, help="Inicio del periodo actual en producción (AAAA-MM-DD)"
    ),
    current_end: str = typer.Option(
        TEST_END, help="Fin del periodo actual en producción (AAAA-MM-DD HH:MM)"
    ),
    sample_reference: int = typer.Option(
        50_000, help="Número de filas muestreadas para referencia (0 para todas)"
    ),
    sample_current: int = typer.Option(
        50_000, help="Número de filas muestreadas para actual (0 para todas)"
    ),
    export_csv: bool = typer.Option(
        True, help="Exportar también copias en formato CSV"
    ),
    model_uri: str = typer.Option(
        DEFAULT_MODEL_URI, help="Modelo para las predicciones (alias, versión, runs:/ o carpeta local)"
    ),
):
    """Prepara y almacena los conjuntos de datos para análisis de drift de MiBici."""
    s_ref = None if sample_reference <= 0 else sample_reference
    s_cur = None if sample_current <= 0 else sample_current

    ref, cur = preparar_datasets(
        reference_start=reference_start,
        reference_end=reference_end,
        current_start=current_start,
        current_end=current_end,
        sample_reference=s_ref,
        sample_current=s_cur,
        export_csv=export_csv,
        model_uri=model_uri,
    )

    print("\n" + "=" * 60)
    print("RESUMEN DE CONJUNTOS DE DATOS PREPARADOS PARA MONITOREO")
    print("=" * 60)
    print(f"Referencia (Entrenamiento): {reference_start} -> {reference_end}")
    print(f"  Filas: {len(ref):,}")
    print(f"  Estaciones: {ref['station_id'].nunique()}")
    print(f"  Demanda real promedio (departures): {ref[TARGET_COL].mean():.2f}")
    if "prediction" in ref.columns:
        print(f"  Prediccion promedio (prediction):   {ref['prediction'].mean():.2f}")

    print(f"\nActual (Produccion):        {current_start} -> {current_end}")
    print(f"  Filas: {len(cur):,}")
    print(f"  Estaciones: {cur['station_id'].nunique()}")
    print(f"  Demanda real promedio (departures): {cur[TARGET_COL].mean():.2f}")
    if "prediction" in cur.columns:
        print(f"  Prediccion promedio (prediction):   {cur['prediction'].mean():.2f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    app()

