"""Verificación de carga y predicción de modelos registrados en producción (@champion).

Comprueba que los tres modelos (P1 demanda, P2 riesgo, P3 tipologías) existen
en el MLflow Model Registry, cuentan con el alias @champion y pueden cargarse
para generar inferencias.

Uso:
    python tests/test_model_registry.py
"""

from __future__ import annotations

import os
import cloudpickle
import mlflow
import pandas as pd

from mibici.config import (
    MLFLOW_TRACKING_URI,
    P1_FEATURES,
    P2_FEATURES,
    PROFILES_PARQUET,
    REGISTRY_NAME,
)


def load_champion(client: mlflow.MlflowClient, name: str):
    """Resuelve y descarga el modelo en producción via models:/{name}@champion."""
    mv = client.get_model_version_by_alias(name, "champion")
    uri = f"models:/{name}@champion"
    print(f"      Alias @champion -> v{mv.version}  (run: {mv.run_id[:8]})")
    print(f"      URI            : {uri}")

    local_dir = mlflow.artifacts.download_artifacts(uri)
    model_dir = os.path.join(local_dir, "model") if os.path.isdir(os.path.join(local_dir, "model")) else local_dir
    pkl_path = os.path.join(model_dir, "model.pkl")

    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            model = cloudpickle.load(f)
    else:
        try:
            model = mlflow.sklearn.load_model(model_dir)
        except Exception:
            model = mlflow.pyfunc.load_model(model_dir)

    return model, mv


def verify_models():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    results = {}

    # ------------------------------------------------------------------
    # P1: Demanda (regresión - salidas por estación-hora)
    # ------------------------------------------------------------------
    name_p1 = REGISTRY_NAME["p1"]
    print(f"\n[1/3] P1 Demanda  models:/{name_p1}@champion")
    model_p1, mv_p1 = load_champion(client, name_p1)
    sample_p1 = pd.DataFrame([{col: 1.0 for col in P1_FEATURES}])
    pred_p1 = float(model_p1.predict(sample_p1)[0])
    print(f"      Predicción (salidas estimadas): {pred_p1:.2f} viajes")
    print(f"      Estado: OK")
    results["p1"] = {"name": name_p1, "version": mv_p1.version, "run_id": mv_p1.run_id}

    # ------------------------------------------------------------------
    # P2: Riesgo operativo (clasificación - vaciado/saturación/normal)
    # ------------------------------------------------------------------
    name_p2 = REGISTRY_NAME["p2"]
    print(f"\n[2/3] P2 Riesgo  models:/{name_p2}@champion")
    model_p2, mv_p2 = load_champion(client, name_p2)
    sample_p2 = pd.DataFrame([{col: 1.0 for col in P2_FEATURES}])
    pred_p2 = model_p2.predict(sample_p2)[0]
    print(f"      Predicción (clase operativa): {pred_p2}")
    print(f"      Estado: OK")
    results["p2"] = {"name": name_p2, "version": mv_p2.version, "run_id": mv_p2.run_id}

    # ------------------------------------------------------------------
    # P3: Tipologías de estación (clustering - grupo asignado)
    # ------------------------------------------------------------------
    name_p3 = REGISTRY_NAME["p3"]
    print(f"\n[3/3] P3 Grupos  models:/{name_p3}@champion")
    model_p3, mv_p3 = load_champion(client, name_p3)
    prof = pd.read_parquet(PROFILES_PARQUET)
    feat_cols = [c for c in prof.columns if c.startswith(("dep_", "arr_"))] + [
        "ratio_arr_dep",
        "dur_median",
        "pct_circular",
    ]
    sample_p3 = prof[feat_cols].head(1).fillna(0)
    pred_p3 = int(model_p3.predict(sample_p3)[0])
    print(f"      Predicción (cluster asignado): Grupo {pred_p3}")
    print(f"      Estado: OK")
    results["p3"] = {"name": name_p3, "version": mv_p3.version, "run_id": mv_p3.run_id}

    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("TODOS LOS MODELOS @champion SE CARGARON Y EVALUARON CON ÉXITO")
    print("=" * 65)
    for q, info in results.items():
        print(f"  {q.upper()}  {info['name']}  v{info['version']}  run={info['run_id'][:8]}  @champion")
    return results


if __name__ == "__main__":
    verify_models()
