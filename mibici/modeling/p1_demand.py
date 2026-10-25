"""P1: viajes que inician por estación en cada hora del día siguiente (regresión).

Modelos: baseline (misma hora de la semana anterior), poisson, rf, hgb.
Criterio de aprobación (propuesta): cerrar al menos 50% de la brecha entre el
WAPE del baseline y el piso Poisson, con mejora significativa (Diebold-Mariano).
"""

from __future__ import annotations

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mibici.config import (
    P1_FEATURES,
    SEED,
    START_MONTH,
    TEST_END,
    TEST_START,
    TRAIN_END,
    VAL_END,
    VAL_START,
)
from mibici.features import load_supervised_frame
from mibici.modeling.tracking import (
    ABBR,
    log_figure,
    log_sklearn_model,
    log_table,
    log_training_input,
    register_model_version,
    start_run,
)

TARGET = "departures"
SPLITS = {"val": (VAL_START, VAL_END), "test": (TEST_START, TEST_END)}
HP_BY_MODEL = {
    "baseline": [],
    "poisson": ["alpha"],
    "rf": ["n_estimators", "max_depth", "min_samples_leaf"],
    "hgb": ["max_iter", "learning_rate", "max_depth", "min_samples_leaf"],
}


def build_model(model: str, hp: dict):
    """Crea el estimador de scikit-learn para P1 según el nombre y los hiperparámetros."""
    if model == "poisson":
        return make_pipeline(StandardScaler(), PoissonRegressor(alpha=hp["alpha"], max_iter=500))
    if model == "rf":
        return RandomForestRegressor(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
                                     min_samples_leaf=hp["min_samples_leaf"], n_jobs=-1, random_state=SEED)
    if model == "hgb":
        return HistGradientBoostingRegressor(loss="poisson", max_iter=hp["max_iter"],
                                             learning_rate=hp["learning_rate"], max_depth=hp["max_depth"],
                                             min_samples_leaf=hp["min_samples_leaf"], random_state=SEED)
    raise ValueError(f"Modelo no soportado para P1: {model}")


def poisson_floor_mae(lam: np.ndarray) -> np.ndarray:
    """Error absoluto esperado del mejor pronóstico posible si y ~ Poisson(lam).

    Es el piso irreducible: ni un modelo perfecto de la media baja de aquí.
    Se calcula con la mediana (el mejor estimador bajo error absoluto).
    """
    uniq, inv = np.unique(np.round(np.clip(lam, 1e-3, None), 2), return_inverse=True)
    maes = np.empty(len(uniq))
    for i, L in enumerate(uniq):
        k = np.arange(0, int(L + 10 * np.sqrt(L) + 20))
        maes[i] = float((poisson.pmf(k, L) * np.abs(k - poisson.median(L))).sum())
    return maes[inv]


def plot_real_vs_pred(y: np.ndarray, pred: np.ndarray, model: str) -> None:
    """Real contra predicho (promedio por nivel real) y distribución de residuales en prueba."""
    df = pd.DataFrame({"y": np.minimum(y, 30), "p": pred, "r": y - pred})
    g = df.groupby("y")["p"].agg(mean="mean", q10=lambda v: v.quantile(0.1), q90=lambda v: v.quantile(0.9))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].fill_between(g.index, g["q10"], g["q90"], alpha=0.25, label="P10-P90 de la predicción")
    axes[0].plot(g.index, g["mean"], marker="o", ms=3, label="Predicción media")
    axes[0].plot([0, 30], [0, 30], "k--", lw=1, label="Ideal")
    axes[0].set_xlabel("Viajes reales por estación-hora (30 = 30 o más)")
    axes[0].set_ylabel("Viajes predichos")
    axes[0].set_title(f"P1 {model}: real contra predicho (prueba)")
    axes[0].legend(fontsize=8)
    axes[1].hist(df["r"].clip(-15, 15), bins=61, color="steelblue")
    axes[1].set_xlabel("Residual = real - predicho (recortado a +/-15)")
    axes[1].set_ylabel("Estación-horas")
    axes[1].set_title(f"Residuales (media {df['r'].mean():.3f})")
    fig.tight_layout()
    log_figure(fig, f"p1_{model}_real_vs_predicho.png")


def regression_metrics(y: np.ndarray, pred: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mae": mean_absolute_error(y, pred),
        f"{prefix}_rmse": root_mean_squared_error(y, pred),
        f"{prefix}_wape": float(np.abs(y - pred).sum() / max(y.sum(), 1)),
        f"{prefix}_poisson_deviance": mean_poisson_deviance(y, np.clip(pred, 0.01, None)),
    }


def diebold_mariano_pvalue(err_base: np.ndarray, err_model: np.ndarray, days: pd.Series) -> float:
    """Prueba de Diebold-Mariano (unilateral) sobre el error absoluto diario.

    H0: el modelo no es mejor que el baseline. Se agregan los errores por día para
    reducir la autocorrelación horaria.
    """
    d = pd.Series(err_base - err_model).groupby(days.to_numpy()).mean()
    if len(d) < 3 or d.std(ddof=1) == 0:
        return float("nan")
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return float(1 - stats.norm.cdf(t))


def run(model: str, hp: dict, sample_rows: int, stage_of_work: str | None = None, nested: bool = False) -> dict:
    """Entrena (si aplica) y evalúa un modelo de P1; registra todo en MLflow y devuelve sus métricas."""
    used = {k: hp[k] for k in HP_BY_MODEL[model]}
    params = {"target": "salidas por estacion-hora", "horizon_h": 24,
              "features": ",".join(P1_FEATURES) if model != "baseline" else "dep_lag_168h",
              "train_sample_rows": sample_rows if model != "baseline" else 0, **used}
    suffix = "-".join(f"{ABBR[k]}{v}" for k, v in used.items())
    run_name = f"p1-{model}" + (f"-{suffix}" if suffix else "")
    cols = ["station_id", TARGET, "dep_lag_168h", "dep_mean_28d", *P1_FEATURES]

    metrics = {}
    with start_run("P1", run_name, model, params, stage_of_work=stage_of_work, nested=nested):
        est = None
        if model != "baseline":
            train = load_supervised_frame(f"{START_MONTH}-01", TRAIN_END, cols, sample_rows=sample_rows)
            logger.info(f"P1 {model}: entrenando con {len(train):,} filas")
            est = build_model(model, hp).fit(train[P1_FEATURES], train[TARGET])
            mlflow.log_metric("train_rows", len(train))
            log_training_input(train[["station_id", "ts", TARGET, *P1_FEATURES]], "p1_train_sample")
            if hasattr(est, "feature_importances_"):
                imp = pd.DataFrame({"feature": P1_FEATURES, "importance": est.feature_importances_})
                log_table(imp.sort_values("importance", ascending=False), "p1_importancia_variables.csv")
            log_sklearn_model(est, train[P1_FEATURES])
            del train

        rows = []
        for split, (a, b) in SPLITS.items():
            df = load_supervised_frame(a, b, cols)
            y = df[TARGET].to_numpy(float)
            base = df["dep_lag_168h"].to_numpy(float)
            pred = base if est is None else np.clip(est.predict(df[P1_FEATURES]), 0, None)
            m = regression_metrics(y, pred, split)
            lam = df["dep_mean_28d"].fillna(df["dep_lag_168h"]).to_numpy(float)
            floor = poisson_floor_mae(lam)
            wape_base = float(np.abs(y - base).sum() / max(y.sum(), 1))
            wape_floor = float(floor.sum() / max(y.sum(), 1))
            gap = wape_base - wape_floor
            m.update({
                f"{split}_wape_baseline": wape_base,
                f"{split}_wape_poisson_floor": wape_floor,
                f"{split}_gap_closed": float((wape_base - m[f"{split}_wape"]) / gap) if gap > 0 else float("nan"),
            })
            if est is not None:
                m[f"{split}_dm_pvalue"] = diebold_mariano_pvalue(np.abs(y - base), np.abs(y - pred), df["ts"].dt.date)
            day = pd.DataFrame({"s": df["station_id"], "d": df["ts"].dt.date, "y": y, "p": pred}).groupby(["s", "d"]).sum()
            m[f"{split}_wape_station_day"] = float((day["y"] - day["p"]).abs().sum() / max(day["y"].sum(), 1))
            mlflow.log_metrics(m)
            metrics.update(m)
            if split == "test":
                plot_real_vs_pred(y, pred, model)
            bins = pd.cut(lam, [-0.01, 0.5, 1, 3, 10, 30, np.inf], labels=["<0.5", "0.5-1", "1-3", "3-10", "10-30", ">30"])
            g = pd.DataFrame({"bin": bins, "err": np.abs(y - pred), "err_base": np.abs(y - base),
                              "floor": floor, "y": y}).groupby("bin", observed=True).sum()
            g = g.assign(wape=g["err"] / g["y"], wape_baseline=g["err_base"] / g["y"],
                         wape_floor=g["floor"] / g["y"], split=split).reset_index()
            rows.append(g)
            logger.info(f"P1 {model} {split}: WAPE {m[f'{split}_wape']:.3f} (baseline {wape_base:.3f}, piso {wape_floor:.3f})")

        tab = pd.concat(rows)
        log_table(tab, f"p1_{model}_wape_por_nivel_de_demanda.csv")
        t = tab[tab["split"] == "test"]
        fig, ax = plt.subplots(figsize=(7.5, 4))
        x = np.arange(len(t))
        if model != "baseline":
            ax.bar(x - 0.27, t["wape"], 0.27, label=model)
        ax.bar(x, t["wape_baseline"], 0.27, label="Baseline lag 168 h")
        ax.bar(x + 0.27, t["wape_floor"], 0.27, label="Piso Poisson")
        ax.set_xticks(x, t["bin"].astype(str))
        ax.set_xlabel("Viajes promedio por estación-hora")
        ax.set_ylabel("WAPE (prueba)")
        ax.set_title(f"P1 {model}: WAPE por nivel de demanda")
        ax.legend()
        log_figure(fig, f"p1_{model}_wape_por_nivel.png")
        # Registrar en el Model Registry (solo corridas que no son hijas de búsqueda)
        if not nested and model != "baseline":
            register_model_version("p1")
        logger.success(f"Run P1 ({run_name}) registrado")
    return metrics
