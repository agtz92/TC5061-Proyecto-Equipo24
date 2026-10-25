"""P2: riesgo de vaciado o saturación por estación y hora del día siguiente.

Modelos: baseline (regla del tercer cuartil), logreg, rf, hgb.
Decisión: con probabilidades calibradas, se alerta si P(riesgo) >= 1/(1+c), donde
c = costo de un falso negativo / costo de un falso positivo (P2_COST_FN_OVER_FP).
Criterio de aprobación (propuesta): costo esperado menor que la regla baseline.
"""

from __future__ import annotations

import duckdb
from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mibici.config import (
    P2_BASELINE_QUANTILE,
    P2_CALIBRATION_ROWS,
    P2_COST_FN_OVER_FP,
    P2_FEATURES,
    P2_NET_FLOW_THRESHOLD,
    PANEL_PARQUET,
    SEED,
    START_MONTH,
    TEST_END,
    TEST_START,
    TRAIN_END,
    VAL_END,
    VAL_START,
)
from mibici.features import LABELS, load_supervised_frame
from mibici.modeling.tracking import (
    ABBR,
    log_figure,
    log_sklearn_model,
    log_table,
    log_training_input,
    register_model_version,
    start_run,
)

TARGET = "p2_label"
HP_BY_MODEL = {
    "baseline": [],
    "logreg": ["C"],
    "rf": ["n_estimators", "max_depth", "min_samples_leaf"],
    "hgb": ["max_iter", "learning_rate", "max_depth", "min_samples_leaf"],
}


def build_model(model: str, hp: dict):
    """Crea el clasificador de P2 con pesos de clase (las clases de riesgo son raras)."""
    if model == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=hp["C"], class_weight="balanced",
                                                                  max_iter=1000, random_state=SEED))
    if model == "rf":
        return RandomForestClassifier(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
                                      min_samples_leaf=hp["min_samples_leaf"], class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=SEED)
    if model == "hgb":
        return HistGradientBoostingClassifier(max_iter=hp["max_iter"], learning_rate=hp["learning_rate"],
                                              max_depth=hp["max_depth"], min_samples_leaf=hp["min_samples_leaf"],
                                              class_weight="balanced", random_state=SEED)
    raise ValueError(f"Modelo no soportado para P2: {model}")


def station_quantiles() -> pd.DataFrame:
    """Umbral Q3 de salidas y llegadas por estación, solo con entrenamiento."""
    train = f"read_parquet('{PANEL_PARQUET}') WHERE ts <= TIMESTAMP '{TRAIN_END}'"
    return duckdb.sql(f"""SELECT station_id,
                              quantile_cont(departures, {P2_BASELINE_QUANTILE}) AS q_dep,
                              quantile_cont(arrivals, {P2_BASELINE_QUANTILE}) AS q_arr
                          FROM {train} GROUP BY 1""").df().set_index("station_id")


def baseline_predict(df: pd.DataFrame, q: pd.DataFrame) -> np.ndarray:
    """Regla Q3: alerta si las salidas (llegadas) de hace una semana superan el Q3 propio.
    Estaciones sin historia en entrenamiento no tienen umbral y no generan alerta."""
    t = df[["station_id", "dep_lag_168h", "arr_lag_168h"]].join(q, on="station_id").astype("float64")
    t[["q_dep", "q_arr"]] = t[["q_dep", "q_arr"]].fillna(np.inf)
    dep_hi = (t["dep_lag_168h"] > t["q_dep"]).to_numpy()
    arr_hi = (t["arr_lag_168h"] > t["q_arr"]).to_numpy()
    dep_ge = (t["dep_lag_168h"] >= t["arr_lag_168h"]).to_numpy()
    return np.select([dep_hi & dep_ge, arr_hi], ["vaciado", "saturacion"], "normal")


def aligned_proba(clf, X: pd.DataFrame) -> pd.DataFrame:
    """Probabilidades con una columna por cada clase de LABELS (0 si el modelo no la vio)."""
    proba = pd.DataFrame(clf.predict_proba(X), columns=clf.classes_)
    return proba.reindex(columns=LABELS, fill_value=0.0)


def decide(proba: pd.DataFrame, c: float) -> np.ndarray:
    """Alerta si P(riesgo) >= 1/(1+c); el tipo de alerta es la clase de riesgo más probable."""
    p_risk = 1 - proba["normal"].to_numpy()
    risk_type = np.where(proba["vaciado"] >= proba["saturacion"], "vaciado", "saturacion")
    return np.where(p_risk >= 1 / (1 + c), risk_type, "normal")


def expected_cost(y_true, y_pred, c: float) -> float:
    """Costo por 1,000 estación-horas: c por cada riesgo no avisado, 1 por cada falsa alerta."""
    risk_true, risk_pred = np.asarray(y_true) != "normal", np.asarray(y_pred) != "normal"
    fn = (risk_true & ~risk_pred).sum()
    fp = (~risk_true & risk_pred).sum()
    return float(1000 * (c * fn + fp) / len(risk_true))


def plot_precision_recall(y: np.ndarray, proba: pd.DataFrame, model: str) -> None:
    """Curva precisión-recall de cada clase de riesgo en prueba; la línea punteada es su prevalencia."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for lab, color in [("vaciado", "#dd8452"), ("saturacion", "#55a868")]:
        if (y == lab).any():
            PrecisionRecallDisplay.from_predictions(y == lab, proba[lab], ax=ax, name=lab, color=color)
            ax.axhline((y == lab).mean(), color=color, ls=":", lw=1)
    ax.set_title(f"P2 {model}: precisión-recall (prueba)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    log_figure(fig, f"p2_{model}_precision_recall_test.png")


def classification_metrics(y, pred, prefix: str) -> dict[str, float]:
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    m = {f"{prefix}_f1_macro": f1_score(y, pred, labels=LABELS, average="macro", zero_division=0),
         f"{prefix}_expected_cost_per_1000": expected_cost(y, pred, P2_COST_FN_OVER_FP),
         f"{prefix}_alert_rate": float((np.asarray(pred) != "normal").mean())}
    for lab, pi, ri, fi in zip(LABELS, p, r, f):
        m.update({f"{prefix}_precision_{lab}": pi, f"{prefix}_recall_{lab}": ri, f"{prefix}_f1_{lab}": fi})
    return m


def run(model: str, hp: dict, sample_rows: int, stage_of_work: str | None = None, nested: bool = False) -> dict:
    """Entrena (si aplica), calibra con validación y evalúa en prueba un modelo de P2; devuelve sus métricas."""
    used = {k: hp[k] for k in HP_BY_MODEL[model]}
    params = {"label_netflow_threshold": P2_NET_FLOW_THRESHOLD, "cost_fn_over_fp": P2_COST_FN_OVER_FP,
              "decision_threshold": round(1 / (1 + P2_COST_FN_OVER_FP), 4), "horizon_h": 24,
              "baseline_quantile": P2_BASELINE_QUANTILE, **used}
    if model != "baseline":
        params.update({"features": ",".join(P2_FEATURES), "train_sample_rows": sample_rows,
                       "train_sampling": "todas las horas con riesgo + muestra de normales",
                       "calibration": f"isotonica con {P2_CALIBRATION_ROWS} filas de validacion"})
    suffix = "-".join(f"{ABBR[k]}{v}" for k, v in used.items())
    run_name = f"p2-{model}" + (f"-{suffix}" if suffix else "")
    cols = ["station_id", TARGET, "dep_lag_168h", "arr_lag_168h", *P2_FEATURES]

    metrics = {}
    with start_run("P2", run_name, model, params, stage_of_work=stage_of_work, nested=nested):
        q = station_quantiles()
        clf = None
        if model != "baseline":
            train = load_supervised_frame(f"{START_MONTH}-01", TRAIN_END, cols,
                                          sample_rows=sample_rows, keep_all_risk=True)
            logger.info(f"P2 {model}: entrenando con {len(train):,} filas")
            mlflow.log_metric("train_rows", len(train))
            log_training_input(train[["station_id", "ts", TARGET, *P2_FEATURES]], "p2_train_sample")
            raw = build_model(model, hp).fit(train[P2_FEATURES], train[TARGET])
            del train
            # Calibración con validación (prevalencia real): corrige el sobremuestreo de riesgo.
            cal = load_supervised_frame(VAL_START, VAL_END, cols, sample_rows=P2_CALIBRATION_ROWS, seed=SEED + 1)
            clf = CalibratedClassifierCV(FrozenEstimator(raw), method="isotonic").fit(cal[P2_FEATURES], cal[TARGET])
            del cal
            log_sklearn_model(clf, load_supervised_frame(VAL_START, VAL_END, cols, sample_rows=1000)[P2_FEATURES])

        for split, (a, b) in {"val": (VAL_START, VAL_END), "test": (TEST_START, TEST_END)}.items():
            df = load_supervised_frame(a, b, cols)
            y = df[TARGET].to_numpy()
            base = baseline_predict(df, q)
            m = {f"{split}_expected_cost_per_1000_baseline": expected_cost(y, base, P2_COST_FN_OVER_FP),
                 # segunda referencia: no alertar nunca (cada hora con riesgo es un falso negativo)
                 f"{split}_expected_cost_per_1000_no_alert": expected_cost(y, np.full(len(y), "normal"), P2_COST_FN_OVER_FP)}
            if clf is None:
                pred = base
            else:
                proba = aligned_proba(clf, df[P2_FEATURES])
                pred = decide(proba, P2_COST_FN_OVER_FP)
                risk_true = (y != "normal").astype(int)
                m[f"{split}_brier_risk"] = brier_score_loss(risk_true, 1 - proba["normal"])
                for lab in ["vaciado", "saturacion"]:
                    if (y == lab).any():
                        m[f"{split}_avg_precision_{lab}"] = average_precision_score(y == lab, proba[lab])
            m.update(classification_metrics(y, pred, split))
            cb = m[f"{split}_expected_cost_per_1000_baseline"]
            m[f"{split}_cost_reduction_vs_baseline"] = (cb - m[f"{split}_expected_cost_per_1000"]) / cb if cb else float("nan")
            m[f"{split}_alerts_per_day"] = float((pred != "normal").sum() / max(df["ts"].dt.date.nunique(), 1))
            ca = m[f"{split}_expected_cost_per_1000_no_alert"]
            m[f"{split}_cost_reduction_vs_no_alert"] = (ca - m[f"{split}_expected_cost_per_1000"]) / ca if ca else float("nan")
            mlflow.log_metrics(m)
            metrics.update(m)
            if split == "test" and clf is not None:
                plot_precision_recall(y, proba, model)
            logger.info(f"P2 {model} {split}: costo {m[f'{split}_expected_cost_per_1000']:.2f} (baseline {cb:.2f})")
            if split == "test":
                cm = confusion_matrix(y, pred, labels=LABELS, normalize="true")
                fig, ax = plt.subplots(figsize=(5, 4))
                im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
                ax.set_xticks(range(3), LABELS)
                ax.set_yticks(range(3), LABELS)
                for i in range(3):
                    for j in range(3):
                        ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center")
                ax.set_xlabel(f"Predicción ({model})")
                ax.set_ylabel("Etiqueta proxy")
                ax.set_title(f"P2 {model}: matriz de confusión (prueba)")
                fig.colorbar(im)
                log_figure(fig, f"p2_{model}_matriz_confusion_test.png")
                log_table(pd.DataFrame(cm, index=LABELS, columns=LABELS).reset_index(names="real"),
                          f"p2_{model}_matriz_confusion_test.csv")
        # Registrar en el Model Registry (solo corridas que no son hijas de búsqueda)
        if not nested and model != "baseline":
            register_model_version("p2")
        logger.success(f"Run P2 ({run_name}) registrado")
    return metrics
