"""Interpretación del drift detectado por Evidently (punto 3 de la entrega de monitoreo).

Responde tres preguntas sobre el reporte de `mibici/drift_report.py`:
1. ¿El drift de las variables de demanda viene de las estaciones nuevas o también de las existentes?
2. ¿El drift de `month` es un cambio real o un artefacto de que el periodo actual no tiene octubre-diciembre?
3. ¿Cómo afecta el drift al error del modelo vigente (@champion), por segmento?

Usa la misma distancia que Evidently para numéricas: Wasserstein normalizada por la desviación estándar
de la referencia, con umbral 0.1.

    python -m mibici.drift_interpretation
"""

from __future__ import annotations

import json

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from mibici.config import (
    MLFLOW_TRACKING_URI,
    P1_FEATURES,
    REGISTRY_NAME,
    REPORTS_DIR,
    START_MONTH,
    TEST_END,
    TEST_START,
    TRAIN_END,
)
from mibici.features import load_supervised_frame

UMBRAL = 0.1
DEMANDA = ["dep_lag_24h", "dep_lag_168h", "dep_mean_7d", "dep_mean_28d"]
OUT_DIR = REPORTS_DIR / "evidently"


def drift_score(ref: pd.Series, cur: pd.Series) -> float:
    """Wasserstein normalizada (como el stattest por defecto de Evidently para numéricas)."""
    sd = ref.std()
    return float(wasserstein_distance(ref, cur) / sd) if sd > 0 else 0.0


def wape(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.abs(y - p).sum() / max(y.sum(), 1))


def main(sample_rows: int = 200_000) -> dict:
    cols = ["station_id", "departures", *P1_FEATURES]
    ref = load_supervised_frame(f"{START_MONTH}-01", TRAIN_END, cols, sample_rows=sample_rows)
    cur = load_supervised_frame(TEST_START, TEST_END, cols, sample_rows=sample_rows)
    viejas = set(ref["station_id"].unique())
    cur["segmento"] = np.where(cur["station_id"].isin(viejas), "existentes", "nuevas")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    uri = f"models:/{REGISTRY_NAME['p1']}@champion"
    modelo = mlflow.pyfunc.load_model(uri)
    for df in (ref, cur):
        df["prediction"] = np.clip(modelo.predict(df[P1_FEATURES].astype("float32")), 0, None)

    existentes = cur[cur["segmento"] == "existentes"]
    ref_sin_q4 = ref[ref["month"] <= 9]

    # 1 y 2: drift por variable en distintos cortes del periodo actual
    filas = []
    for f in P1_FEATURES:
        filas.append({
            "variable": f,
            "todas_las_estaciones": drift_score(ref[f], cur[f]),
            "solo_existentes": drift_score(ref[f], existentes[f]),
            "existentes_vs_ref_ene_sep": drift_score(ref_sin_q4[f], existentes[f]),
        })
    drift = pd.DataFrame(filas).set_index("variable").round(4)

    # 3: error del modelo por segmento
    def resumen(df: pd.DataFrame) -> dict:
        y, p = df["departures"].to_numpy(float), df["prediction"].to_numpy(float)
        return {"filas": len(df), "estaciones": int(df["station_id"].nunique()),
                "demanda_media": round(float(y.mean()), 3), "prediccion_media": round(float(p.mean()), 3),
                "sesgo": round(float((p - y).mean()), 3), "wape": round(wape(y, p), 4)}

    errores = {"referencia_2022_2025": resumen(ref), "actual_todas": resumen(cur),
               "actual_existentes": resumen(existentes), "actual_nuevas": resumen(cur[cur["segmento"] == "nuevas"])}

    # Demanda media por mes en estaciones existentes: separa estacionalidad de cambio de nivel
    por_mes = pd.DataFrame({
        "referencia": ref_sin_q4.groupby("month")["departures"].mean(),
        "actual_existentes": existentes.groupby("month")["departures"].mean(),
    }).round(3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    salida = {"modelo": uri, "umbral": UMBRAL, "drift_por_variable": drift.reset_index().to_dict("records"),
              "error_por_segmento": errores, "demanda_por_mes": por_mes.reset_index().to_dict("records")}
    (OUT_DIR / "drift_interpretacion.json").write_text(json.dumps(salida, indent=1, ensure_ascii=False), encoding="utf-8")

    pd.set_option("display.width", 140)
    print("\nDrift por variable (Wasserstein normalizada, umbral 0.1):\n", drift)
    print("\nError del modelo @champion por segmento:\n", pd.DataFrame(errores).T)
    print("\nDemanda media por mes (enero-septiembre):\n", por_mes)
    logger.success(f"Resultados en {OUT_DIR / 'drift_interpretacion.json'}")
    return salida


if __name__ == "__main__":
    main()
