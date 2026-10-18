"""Experimento base del proyecto MiBici en MLflow (un run por pregunta).

    python -m mibici.modeling.train p1                 # baseline estacional de demanda
    python -m mibici.modeling.train p2                 # baseline regla Q3 de riesgo
    python -m mibici.modeling.train p3                 # baseline por municipio
    python -m mibici.modeling.train p3 --model kmeans --k 5

O, empaquetado con MLflow Projects:  mlflow run . -e p1 --env-manager local

Cada run registra:
- tags: team, question, git_commit, git_dirty, data_version
- params: rango de fechas, partición, versión de datos, semilla, modelo e hiperparámetros
- métricas con prefijo del conjunto (val_*, test_*)
- artefactos: gráficas en PNG y tablas CSV de soporte
Nunca se registran datos a nivel de viaje ni de usuario.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile

import duckdb
from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_poisson_deviance,
    precision_recall_fscore_support,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
import typer

from mibici.config import (
    DATA_MANIFEST,
    END_MONTH,
    FIGURES_DIR,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    P2_BASELINE_QUANTILE,
    P2_COST_FN_OVER_FP,
    P2_NET_FLOW_THRESHOLD,
    P3_DEFAULT_K,
    PANEL_PARQUET,
    PROFILES_PARQUET,
    PROJ_ROOT,
    SEED,
    START_MONTH,
    TEAM,
    TEST_END,
    TEST_START,
    TRAIN_END,
    VAL_END,
    VAL_START,
)
from mibici.dataset import load_data_version
from mibici.features import LABELS, load_panel

app = typer.Typer(help="Experimento base MiBici (P1, P2, P3).")


# ---------------------------------------------------------------------------
# Utilidades de MLflow
# ---------------------------------------------------------------------------
def git_info() -> dict[str, str]:
    """Commit actual y si hay cambios sin commit (para ligar el run al código)."""
    def run(*args):
        return subprocess.run(["git", *args], cwd=PROJ_ROOT, capture_output=True, text=True).stdout.strip()
    commit = run("rev-parse", "HEAD") or "desconocido"
    dirty = "true" if run("status", "--porcelain", "--untracked-files=no") else "false"
    return {"git_commit": commit, "git_dirty": dirty}


def start_run(question: str, run_name: str, model: str, extra_params: dict) -> mlflow.ActiveRun:
    """Abre un run con las etiquetas y parámetros comunes de reproducibilidad."""
    if os.environ.get("MLFLOW_RUN_ID"):
        # Lanzado con `mlflow run`: MLflow ya creó el run en el experimento indicado
        # con --experiment-name; solo se le pone nombre descriptivo.
        run = mlflow.start_run()
        mlflow.set_tag("mlflow.runName", run_name)
    else:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        run = mlflow.start_run(run_name=run_name)
    data_version = load_data_version()
    start_m, end_m = START_MONTH, END_MONTH
    if DATA_MANIFEST.exists():  # rango realmente ingerido
        man = json.loads(DATA_MANIFEST.read_text())
        start_m, end_m = man["start_month"], man["end_month"]
    mlflow.set_tags({"team": TEAM, "question": question, "data_version": data_version,
                     "stage": "experimento_base", **git_info()})
    mlflow.log_params({
        "start_month": start_m, "end_month": end_m, "train_end": TRAIN_END,
        "val_start": VAL_START, "val_end": VAL_END, "test_start": TEST_START, "test_end": TEST_END,
        "data_version": data_version, "seed": SEED, "model": model, **extra_params,
    })
    return run


def log_figure(fig, name: str) -> None:
    """Registra la figura en MLflow y la guarda también en reports/figures/."""
    mlflow.log_figure(fig, f"figures/{name}")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def log_table(df: pd.DataFrame, name: str) -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / name
        df.to_csv(p, index=False)
        mlflow.log_artifact(str(p), artifact_path="tables")


# ---------------------------------------------------------------------------
# P1: demanda (regresión de conteos)
# ---------------------------------------------------------------------------
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


def p1_metrics(y: np.ndarray, pred: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mae": mean_absolute_error(y, pred),
        f"{prefix}_rmse": root_mean_squared_error(y, pred),
        f"{prefix}_wape": float(np.abs(y - pred).sum() / max(y.sum(), 1)),
        f"{prefix}_poisson_deviance": mean_poisson_deviance(y, np.clip(pred, 0.01, None)),
    }


def run_p1() -> None:
    """Baseline estacional ingenuo: misma estación y hora de la semana anterior."""
    cols = ["station_id", "ts", "departures", "dep_lag_168h", "dep_mean_28d"]
    with start_run("P1", "p1-baseline-estacional-lag168h", "seasonal_naive_lag168h",
                   {"target": "salidas por estacion-hora", "horizon_h": 24, "lag_hours": 168}):
        rows = []
        for split, (a, b) in {"val": (VAL_START, VAL_END), "test": (TEST_START, TEST_END)}.items():
            df = load_panel(a, b, cols).dropna(subset=["dep_lag_168h"])
            y, pred = df["departures"].to_numpy(float), df["dep_lag_168h"].to_numpy(float)
            mlflow.log_metrics(p1_metrics(y, pred, split))
            # Piso Poisson, con la media reciente de la estación como estimación de lambda
            lam = df["dep_mean_28d"].fillna(df["dep_lag_168h"]).to_numpy(float)
            floor = poisson_floor_mae(lam)
            mlflow.log_metric(f"{split}_wape_poisson_floor", float(floor.sum() / max(y.sum(), 1)))
            # Nivel estación-día (agregado)
            day = df.assign(d=df["ts"].dt.date).groupby(["station_id", "d"])[["departures", "dep_lag_168h"]].sum()
            mlflow.log_metric(f"{split}_wape_station_day",
                              float((day["departures"] - day["dep_lag_168h"]).abs().sum() / day["departures"].sum()))
            # WAPE por nivel de demanda
            bins = pd.cut(lam, [-0.01, 0.5, 1, 3, 10, 30, np.inf], labels=["<0.5", "0.5-1", "1-3", "3-10", "10-30", ">30"])
            g = pd.DataFrame({"bin": bins, "abs_err": np.abs(y - pred), "floor": floor, "y": y}).groupby("bin", observed=True).sum()
            g = g.assign(wape=g["abs_err"] / g["y"], wape_floor=g["floor"] / g["y"], split=split).reset_index()
            rows.append(g)
        tab = pd.concat(rows)
        log_table(tab, "p1_wape_por_nivel_de_demanda.csv")
        t = tab[tab["split"] == "test"]
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(t))
        ax.bar(x - 0.2, t["wape"], 0.4, label="Baseline lag 168 h")
        ax.bar(x + 0.2, t["wape_floor"], 0.4, label="Piso Poisson (irreducible)")
        ax.set_xticks(x, t["bin"].astype(str))
        ax.set_xlabel("Viajes promedio por estación-hora")
        ax.set_ylabel("WAPE (prueba)")
        ax.set_title("P1: error del baseline vs. error mínimo posible")
        ax.legend()
        log_figure(fig, "p1_wape_vs_piso_poisson.png")
        logger.success("Run P1 registrado")


# ---------------------------------------------------------------------------
# P2: riesgo de vaciado o saturación (clasificación)
# ---------------------------------------------------------------------------
def expected_cost(y_true: pd.Series, y_pred: pd.Series, c: float) -> float:
    """Costo por 1,000 estación-horas: c por cada riesgo no avisado, 1 por cada falsa alerta."""
    risk_true, risk_pred = y_true != "normal", y_pred != "normal"
    fn = (risk_true & ~risk_pred).sum()
    fp = (~risk_true & risk_pred).sum()
    return float(1000 * (c * fn + fp) / len(y_true))


def run_p2() -> None:
    """Baseline regla Q3: alerta si las salidas (llegadas) de la misma hora de la
    semana anterior superan el tercer cuartil propio de la estación en entrenamiento."""
    cols = ["station_id", "ts", "departures", "arrivals", "dep_lag_168h", "arr_lag_168h", "p2_label"]
    params = {"label_netflow_threshold": P2_NET_FLOW_THRESHOLD, "baseline_quantile": P2_BASELINE_QUANTILE,
              "cost_fn_over_fp": P2_COST_FN_OVER_FP, "horizon_h": 24}
    with start_run("P2", "p2-baseline-regla-q3", "rule_station_q3", params):
        # Umbrales por estación solo con entrenamiento (sin mirar validación ni prueba)
        train = f"read_parquet('{PANEL_PARQUET}') WHERE ts <= TIMESTAMP '{TRAIN_END}'"
        q = duckdb.sql(f"""SELECT station_id,
                               quantile_cont(departures, {P2_BASELINE_QUANTILE}) AS q_dep,
                               quantile_cont(arrivals, {P2_BASELINE_QUANTILE}) AS q_arr
                           FROM {train} GROUP BY 1""").df().set_index("station_id")
        prev = duckdb.sql(f"SELECT p2_label, count(*) AS n FROM {train} GROUP BY 1").df()
        prev["p"] = prev["n"] / prev["n"].sum()
        mlflow.log_metrics({f"train_prevalence_{r.p2_label}": float(r.p) for r in prev.itertuples()})
        for split, (a, b) in {"val": (VAL_START, VAL_END), "test": (TEST_START, TEST_END)}.items():
            df = load_panel(a, b, cols).dropna(subset=["dep_lag_168h", "arr_lag_168h"]).join(q, on="station_id")
            dep_hi = df["dep_lag_168h"] > df["q_dep"]
            arr_hi = df["arr_lag_168h"] > df["q_arr"]
            pred = np.select([dep_hi & (df["dep_lag_168h"] >= df["arr_lag_168h"]), arr_hi], ["vaciado", "saturacion"], "normal")
            y, pred = df["p2_label"], pd.Series(pred, index=df.index)
            p, r, f, _ = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
            m = {f"{split}_f1_macro": f1_score(y, pred, labels=LABELS, average="macro", zero_division=0),
                 f"{split}_expected_cost_per_1000": expected_cost(y, pred, P2_COST_FN_OVER_FP),
                 f"{split}_alert_rate": float((pred != "normal").mean())}
            for lab, pi, ri, fi in zip(LABELS, p, r, f):
                m.update({f"{split}_precision_{lab}": pi, f"{split}_recall_{lab}": ri, f"{split}_f1_{lab}": fi})
            mlflow.log_metrics(m)
            if split == "test":
                cm = confusion_matrix(y, pred, labels=LABELS, normalize="true")
                fig, ax = plt.subplots(figsize=(5, 4))
                im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
                ax.set_xticks(range(3), LABELS)
                ax.set_yticks(range(3), LABELS)
                for i in range(3):
                    for j in range(3):
                        ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center")
                ax.set_xlabel("Predicción (regla Q3)")
                ax.set_ylabel("Etiqueta proxy")
                ax.set_title("P2: matriz de confusión normalizada (prueba)")
                fig.colorbar(im)
                log_figure(fig, "p2_matriz_confusion_test.png")
        logger.success("Run P2 registrado")


# ---------------------------------------------------------------------------
# P3: tipologías de estaciones (clustering)
# ---------------------------------------------------------------------------
def jaccard_stability(X: np.ndarray, labels: np.ndarray, k: int, n_boot: int = 50) -> float:
    """Estabilidad bootstrap de Hennig (2007): Jaccard promedio de cada cluster original
    con su mejor coincidencia al re-agrupar remuestras. >= 0.75 se considera estable."""
    rng = np.random.default_rng(SEED)
    scores = np.zeros((n_boot, k))
    for b in range(n_boot):
        idx = rng.choice(len(X), len(X), replace=True)
        lb = KMeans(k, n_init=10, random_state=SEED + b).fit_predict(X[idx])
        for c in range(k):
            orig = set(np.where(labels[idx] == c)[0])
            best = 0.0
            for c2 in range(k):
                new = set(np.where(lb == c2)[0])
                if orig or new:
                    best = max(best, len(orig & new) / len(orig | new))
            scores[b, c] = best
    return float(scores.mean())


def run_p3(model: str, k: int) -> None:
    """Baseline por municipio (geografía administrativa) o K-Means sobre los perfiles."""
    prof = pd.read_parquet(PROFILES_PARQUET)
    feat_cols = [c for c in prof.columns if c.startswith(("dep_", "arr_"))] + ["ratio_arr_dep", "dur_median", "pct_circular"]
    X = StandardScaler().fit_transform(prof[feat_cols].fillna(0))
    pca = PCA(n_components=0.9, random_state=SEED).fit(X)
    Xp = pca.transform(X)
    if model == "municipio":
        labels = pd.factorize(prof["municipio"])[0]
        run_name, params = "p3-baseline-municipio", {"grouping": "prefijo GDL/ZPN/TLQ de la nomenclatura"}
    else:
        labels = KMeans(k, n_init=20, random_state=SEED).fit_predict(Xp)
        run_name, params = f"p3-kmeans-k{k}", {"k": k, "n_init": 20}
    params.update({"n_features": len(feat_cols), "pca_variance": 0.9, "pca_components": int(pca.n_components_),
                   "profile_period_end": TRAIN_END, "n_stations": len(prof)})
    with start_run("P3", run_name, model, params):
        n_lab = len(np.unique(labels))
        m = {"train_n_clusters": n_lab}
        if n_lab > 1:
            m.update({"train_silhouette": silhouette_score(Xp, labels),
                      "train_davies_bouldin": davies_bouldin_score(Xp, labels),
                      "train_calinski_harabasz": calinski_harabasz_score(Xp, labels)})
        if model == "kmeans":
            m["train_jaccard_bootstrap_mean"] = jaccard_stability(Xp, labels, k)
            mun = pd.factorize(prof["municipio"])[0]
            m["train_ari_vs_municipio"] = adjusted_rand_score(mun, labels)
        mlflow.log_metrics(m)
        out = prof[["station_id", "lat", "lon", "municipio"]].assign(cluster=labels)
        log_table(out, f"p3_{model}_asignacion_estaciones.csv")
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(out["lon"], out["lat"], c=labels, cmap="tab10", s=14)
        ax.set_title(f"P3: grupos de estaciones ({run_name})")
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        ax.legend(*sc.legend_elements(), title="Grupo", fontsize=7, loc="lower left")
        log_figure(fig, f"p3_mapa_{model}.png")
        prof_mean = pd.DataFrame(X, columns=feat_cols).assign(c=labels).groupby("c").mean()
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
        for ax, kind in zip(axes, ["dep_wd", "arr_wd"]):
            cols = [f"{kind}_h{h:02d}" for h in range(24)]
            for c, row in prof_mean[cols].iterrows():
                ax.plot(range(24), row.values, label=f"Grupo {c}")
            ax.set_title("Salidas" if kind.startswith("dep") else "Llegadas")
            ax.set_xlabel("Hora (día hábil)")
        axes[0].set_ylabel("Proporción estandarizada")
        axes[1].legend(fontsize=7)
        fig.suptitle("P3: perfil horario promedio por grupo")
        log_figure(fig, f"p3_perfiles_{model}.png")
        logger.success(f"Run P3 ({model}) registrado")


@app.command()
def p1():
    """Experimento base de P1."""
    run_p1()


@app.command()
def p2():
    """Experimento base de P2."""
    run_p2()


@app.command()
def p3(model: str = typer.Option("municipio", help="municipio | kmeans"), k: int = P3_DEFAULT_K):
    """Experimento base de P3."""
    run_p3(model, k)


@app.command(name="all")
def run_all():
    """Corre los cuatro runs del experimento base."""
    run_p1()
    run_p2()
    run_p3("municipio", P3_DEFAULT_K)
    run_p3("kmeans", P3_DEFAULT_K)


if __name__ == "__main__":
    app()
