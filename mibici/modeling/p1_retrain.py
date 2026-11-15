"""P1, versión alternativa del modelo de demanda (challenger) para el ciclo de monitoreo.

Contexto
--------
El reporte de drift (docs/interpretacion_drift.md) mostró que la red pasó de 371 a 469
estaciones en 2026 y que el error del champion se concentra en las estaciones nuevas
(WAPE cercano a 100%), cuya historia no aparece en el entrenamiento (2022-01 a 2025-06).
La hipótesis que se prueba aquí es que un reentrenamiento con datos recientes reduce
ese error sin perjudicar a las estaciones existentes.

Diseño experimental
-------------------
Se cambia un solo factor respecto al champion: la ventana de entrenamiento.

    champion   : entrena de 2022-01 a TRAIN_END         (2025-06-30)
    challenger : entrena de 2022-01 a RETRAIN_TRAIN_END (2026-06-30)

Todo lo demás se hereda del champion leyendo su corrida en MLflow: familia de modelo,
hiperparámetros, variables de entrada (P1_FEATURES), tamaño de la muestra de
entrenamiento y semilla. Si además se cambiaran los hiperparámetros, una diferencia en
el A/B testing no podría atribuirse a los datos recientes.

El conjunto de evaluación del A/B testing (AB_EVAL_START a AB_EVAL_END, julio a
septiembre de 2026) queda fuera del entrenamiento de ambas versiones. Los hiperparámetros
del champion se eligieron con la validación de 2025-H2, así que el challenger tampoco
usa el conjunto de evaluación para seleccionar nada.

Registro
--------
El modelo se registra como una versión nueva de `mibici-equipo24-p1-demanda` con el
alias `challenger`. El alias `champion` no se modifica: la promoción depende del
resultado del A/B testing (punto 5), no del entrenamiento.

Uso:
    python -m mibici.modeling.train retrain
    mlflow run . -e retrain --env-manager local
"""

from __future__ import annotations

import duckdb
from loguru import logger
import mlflow
import numpy as np
import pandas as pd

from mibici.config import (
    AB_EVAL_END,
    AB_EVAL_START,
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    HYPERPARAMS,
    MLFLOW_TRACKING_URI,
    P1_FEATURES,
    PANEL_PARQUET,
    REGISTRY_NAME,
    RETRAIN_TRAIN_END,
    START_MONTH,
    TRAIN_END,
    TRAIN_SAMPLE_ROWS,
)
from mibici.features import load_supervised_frame, sql_timestamp
from mibici.modeling.p1_demand import (
    HP_BY_MODEL,
    TARGET,
    build_model,
    plot_real_vs_pred,
    regression_metrics,
)
from mibici.modeling.tracking import (
    log_sklearn_model,
    log_table,
    log_training_input,
    register_model_version,
    start_run,
)

# Columnas que se leen del panel: identificador, objetivo, baseline y variables de entrada.
COLUMNS = ["station_id", TARGET, "dep_lag_168h", *P1_FEATURES]


# ---------------------------------------------------------------------------
# Especificación heredada del champion
# ---------------------------------------------------------------------------
def cast_hyperparam(raw: str, default):
    """Convierte un parámetro de MLflow (siempre texto) al tipo del valor en HYPERPARAMS.

    MLflow guarda los parámetros como cadenas, por lo que "12", "12.0" y "None" deben
    traducirse de vuelta al tipo que espera el estimador de scikit-learn.
    """
    if raw is None or str(raw).strip().lower() == "none":
        return None
    if isinstance(default, int) and not isinstance(default, bool):
        return int(float(raw))
    if isinstance(default, float):
        return float(raw)
    return raw


def champion_spec(client: mlflow.MlflowClient, name: str = REGISTRY_NAME["p1"]) -> dict:
    """Lee del Model Registry la versión @champion y la corrida que la produjo.

    Devuelve lo necesario para replicar su entrenamiento con otra ventana de datos:
    estimador, hiperparámetros efectivos, tamaño de muestra y fin de su entrenamiento.
    Los hiperparámetros se registraron con prefijo hp_ (ver tracking.start_run); se
    acepta también el nombre sin prefijo por compatibilidad con corridas anteriores.
    """
    mv = client.get_model_version_by_alias(name, CHAMPION_ALIAS)
    run = client.get_run(mv.run_id)
    params, tags = run.data.params, run.data.tags
    model = tags.get("model") or params.get("estimator")
    if model not in HP_BY_MODEL or model == "baseline":
        raise ValueError(f"El champion usa un estimador que no se puede reentrenar: {model!r}")

    hp = dict(HYPERPARAMS)
    for k in HP_BY_MODEL[model]:
        raw = params.get(f"hp_{k}", params.get(k))
        if raw is None:
            raise ValueError(f"La corrida del champion ({mv.run_id}) no registró el hiperparámetro {k}")
        hp[k] = cast_hyperparam(raw, HYPERPARAMS[k])

    return {
        "version": str(mv.version),
        "run_id": mv.run_id,
        "model": model,
        "hp": hp,
        "train_end": params.get("train_end", TRAIN_END),
        "train_sample_rows": int(float(params.get("train_sample_rows", TRAIN_SAMPLE_ROWS))),
        "data_version": tags.get("data_version", "desconocida"),
    }


def validate_windows(champion_train_end: str, train_end: str,
                     eval_start: str = AB_EVAL_START, eval_end: str = AB_EVAL_END) -> None:
    """Verifica que las ventanas temporales permitan una comparación sin fuga de información.

    1. El challenger debe ver datos posteriores al champion (si no, no hay reentrenamiento).
    2. El entrenamiento del challenger debe terminar antes del conjunto de evaluación.
    3. El conjunto de evaluación debe tener un inicio anterior a su fin.
    """
    ch_end, new_end = pd.Timestamp(champion_train_end), pd.Timestamp(train_end)
    ev_start, ev_end = pd.Timestamp(eval_start), pd.Timestamp(eval_end)
    if new_end <= ch_end:
        raise ValueError(f"train_end ({train_end}) debe ser posterior al del champion ({champion_train_end})")
    if new_end >= ev_start:
        raise ValueError(f"train_end ({train_end}) se traslapa con la evaluación A/B ({eval_start})")
    if ev_start >= ev_end:
        raise ValueError(f"Ventana de evaluación inválida: {eval_start} a {eval_end}")


# ---------------------------------------------------------------------------
# Conjunto de evaluación compartido con el A/B testing
# ---------------------------------------------------------------------------
def stations_seen_until(end: str) -> set[int]:
    """Estaciones con al menos una hora utilizable (rezago de 168 h) hasta la fecha dada.

    Sirve para separar la evaluación en estaciones existentes y nuevas respecto al
    periodo de entrenamiento del champion, el mismo corte usado en la interpretación
    del drift.
    """
    query = f"""
        SELECT DISTINCT station_id FROM read_parquet('{PANEL_PARQUET}')
        WHERE ts <= TIMESTAMP '{sql_timestamp(end)}' AND dep_lag_168h IS NOT NULL"""  # nosec B608
    return set(duckdb.sql(query).df()["station_id"].astype(int))


def load_ab_eval_frame(champion_train_end: str = TRAIN_END) -> pd.DataFrame:
    """Conjunto de evaluación del A/B testing: todas las estación-horas de AB_EVAL_START a AB_EVAL_END.

    No se muestrea, para que ambas versiones se evalúen sobre exactamente las mismas filas.
    La columna `segmento` indica si la estación ya existía en el entrenamiento del
    champion ("existentes") o apareció después ("nuevas").
    """
    df = load_supervised_frame(AB_EVAL_START, AB_EVAL_END, COLUMNS)
    known = stations_seen_until(champion_train_end)
    df["segmento"] = np.where(df["station_id"].astype(int).isin(known), "existentes", "nuevas")
    return df


def evaluate(y: np.ndarray, pred: np.ndarray, base: np.ndarray, segment: np.ndarray,
             prefix: str = "ab_eval") -> dict[str, float]:
    """Métricas de P1 en el total y por segmento de estación.

    - Métricas generales: MAE, RMSE, WAPE y devianza Poisson (mismas que en el entrenamiento).
    - bias: media de (predicción - real); positivo indica sobrepredicción.
    - wape_baseline: WAPE de la regla "misma hora de la semana anterior", como referencia.
    - Por segmento: WAPE, sesgo y número de filas, porque el drift se concentró en las
      estaciones nuevas y un promedio global podría ocultar ese efecto.
    """
    m = regression_metrics(y, pred, prefix)
    m[f"{prefix}_bias"] = float(np.mean(pred - y))
    m[f"{prefix}_wape_baseline"] = float(np.abs(y - base).sum() / max(y.sum(), 1))
    m[f"{prefix}_rows"] = float(len(y))
    for seg in ("existentes", "nuevas"):
        mask = segment == seg
        if not mask.any():
            continue
        ys, ps = y[mask], pred[mask]
        m[f"{prefix}_wape_{seg}"] = float(np.abs(ys - ps).sum() / max(ys.sum(), 1))
        m[f"{prefix}_bias_{seg}"] = float(np.mean(ps - ys))
        m[f"{prefix}_rows_{seg}"] = float(mask.sum())
    return m


# ---------------------------------------------------------------------------
# Entrenamiento y registro del challenger
# ---------------------------------------------------------------------------
def run(train_end: str = RETRAIN_TRAIN_END, sample_rows: int | None = None,
        alias: str = CHALLENGER_ALIAS) -> dict:
    """Entrena el challenger con la ventana extendida, lo evalúa y lo registra en el Model Registry.

    Parámetros
    ----------
    train_end : fin (inclusivo) de la ventana de entrenamiento del challenger.
    sample_rows : filas de la muestra de entrenamiento. None usa las mismas del champion,
        de modo que ambas versiones tengan el mismo presupuesto de datos y cómputo.
    alias : alias que recibe la nueva versión en el registry.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    champ = champion_spec(client)
    validate_windows(champ["train_end"], train_end)

    model, hp = champ["model"], champ["hp"]
    rows = sample_rows or champ["train_sample_rows"]
    used = {k: hp[k] for k in HP_BY_MODEL[model]}
    logger.info(f"Champion v{champ['version']} ({model}, entrenado hasta {champ['train_end']}): {used}")

    # Los parámetros train_end, val_* y test_* sobrescriben los valores de config.py que
    # start_run registra por defecto, para que la corrida describa sus ventanas reales.
    params = {
        "target": "salidas por estacion-hora", "horizon_h": 24,
        "features": ",".join(P1_FEATURES), "train_sample_rows": rows,
        "train_end": train_end,
        "val_start": "no_aplica", "val_end": "no_aplica",
        "test_start": AB_EVAL_START, "test_end": AB_EVAL_END,
        "retrain_strategy": "ventana_extendida",
        "champion_version": champ["version"], "champion_run_id": champ["run_id"],
        "champion_train_end": champ["train_end"],
        **used,
    }
    run_name = f"p1-{model}-retrain-{pd.Timestamp(train_end):%Y-%m}"

    with start_run("P1", run_name, model, params, stage="reentrenamiento", stage_of_work=alias):
        mlflow.set_tags({"retrain_of_version": champ["version"], "retrain_of_run_id": champ["run_id"]})

        # 1. Entrenamiento con la ventana extendida (misma muestra determinista por hash)
        train = load_supervised_frame(f"{START_MONTH}-01", train_end, COLUMNS, sample_rows=rows)
        recent = train["ts"] > pd.Timestamp(champ["train_end"])
        logger.info(f"Challenger: {len(train):,} filas, {int(recent.sum()):,} posteriores al champion")
        est = build_model(model, hp).fit(train[P1_FEATURES], train[TARGET])
        mlflow.log_metrics({
            "train_rows": len(train),
            "train_rows_after_champion": int(recent.sum()),
            "train_share_after_champion": float(recent.mean()),
            "train_stations": train["station_id"].nunique(),
        })
        log_training_input(train[["station_id", "ts", TARGET, *P1_FEATURES]], "p1_retrain_sample")
        if hasattr(est, "feature_importances_"):
            imp = pd.DataFrame({"feature": P1_FEATURES, "importance": est.feature_importances_})
            log_table(imp.sort_values("importance", ascending=False), "p1_importancia_variables.csv")
        log_sklearn_model(est, train[P1_FEATURES])
        del train

        # 2. Evaluación en el conjunto reservado para el A/B testing (sin muestreo)
        ev = load_ab_eval_frame(champ["train_end"])
        y = ev[TARGET].to_numpy(float)
        pred = np.clip(est.predict(ev[P1_FEATURES]), 0, None)
        metrics = evaluate(y, pred, ev["dep_lag_168h"].to_numpy(float), ev["segmento"].to_numpy())
        mlflow.log_metrics(metrics)
        plot_real_vs_pred(y, pred, f"{model}-challenger")
        logger.info(f"Challenger en evaluación A/B: WAPE {metrics['ab_eval_wape']:.3f} "
                    f"(existentes {metrics.get('ab_eval_wape_existentes', float('nan')):.3f}, "
                    f"nuevas {metrics.get('ab_eval_wape_nuevas', float('nan')):.3f})")

        # 3. Especificación del reentrenamiento, adjunta a la corrida para su trazabilidad
        spec = {
            "estrategia": "ventana_extendida",
            "factor_modificado": "fin de la ventana de entrenamiento",
            "champion": {k: champ[k] for k in ("version", "run_id", "model", "train_end", "data_version")},
            "challenger": {"train_start": f"{START_MONTH}-01 00:00", "train_end": train_end,
                           "train_sample_rows": rows},
            "hiperparametros_heredados": used,
            "evaluacion_ab": {"start": AB_EVAL_START, "end": AB_EVAL_END, "muestreo": "ninguno"},
        }
        mlflow.log_dict(spec, "config/retrain_spec.json")

        # 4. Registro como versión nueva del mismo modelo, con alias challenger
        version = register_model_version("p1", alias=alias)
        name = REGISTRY_NAME["p1"]
        for k, v in {"role": alias, "retrain_of_version": champ["version"],
                     "train_window": f"{START_MONTH}-01 a {train_end}",
                     "ab_eval_window": f"{AB_EVAL_START} a {AB_EVAL_END}"}.items():
            client.set_model_version_tag(name, version, k, v)
        current = client.get_model_version(name, version).description or ""
        client.update_model_version(name, version, description=(
            f"{current} · Reentrenamiento de v{champ['version']} con ventana extendida hasta "
            f"{train_end} (mismos hiperparámetros). Evaluación A/B: {AB_EVAL_START} a {AB_EVAL_END}."
        ))
        logger.success(f"Challenger registrado: {name} v{version} @{alias} (champion v{champ['version']} sin cambios)")

    return {"version": version, "champion_version": champ["version"], **metrics}
