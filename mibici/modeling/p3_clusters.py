"""P3: grupos de estaciones con patrones de uso similares (clustering).

Modelos: municipio (baseline: prefijo GDL/ZPN/TLQ), kmeans, ward, gmm.
Criterios de aprobación (propuesta): silueta > 0.25 (Kaufman y Rousseeuw, 1990)
y Jaccard bootstrap >= 0.75 por cluster (Hennig, 2007).
"""

from __future__ import annotations

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mibici.config import PROFILES_PARQUET, SEED, TRAIN_END
from mibici.modeling.tracking import log_figure, log_sklearn_model, log_table, log_training_input, start_run

HP_BY_MODEL = {"municipio": [], "kmeans": ["k"], "ward": ["k"], "gmm": ["k"]}
PCA_VARIANCE = 0.9
N_BOOT = 50


def fit_predict(model: str, X: np.ndarray, k: int, seed: int = SEED) -> np.ndarray:
    """Asigna un grupo a cada estación con el algoritmo pedido."""
    if model == "kmeans":
        return KMeans(k, n_init=20, random_state=seed).fit_predict(X)
    if model == "ward":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    if model == "gmm":
        return GaussianMixture(k, covariance_type="full", n_init=5, random_state=seed).fit_predict(X)
    raise ValueError(f"Modelo no soportado para P3: {model}")


def jaccard_stability(model: str, X: np.ndarray, labels: np.ndarray, k: int, n_boot: int = N_BOOT) -> np.ndarray:
    """Estabilidad bootstrap de Hennig (2007) por cluster.

    Para cada remuestra se vuelve a agrupar y, para cada cluster original, se toma
    el Jaccard con su mejor coincidencia. Devuelve el promedio por cluster.
    """
    rng = np.random.default_rng(SEED)
    scores = np.zeros((n_boot, k))
    for b in range(n_boot):
        idx = rng.choice(len(X), len(X), replace=True)
        lb = fit_predict(model, X[idx], k, seed=SEED + b)
        for c in range(k):
            orig = set(np.where(labels[idx] == c)[0])
            best = 0.0
            for c2 in np.unique(lb):
                new = set(np.where(lb == c2)[0])
                if orig or new:
                    best = max(best, len(orig & new) / len(orig | new))
            scores[b, c] = best
    return scores.mean(axis=0)


def run(model: str, hp: dict, stage_of_work: str | None = None, nested: bool = False) -> dict:
    """Agrupa las estaciones por su perfil de uso; registra calidad, estabilidad, mapa y modelo."""
    prof = pd.read_parquet(PROFILES_PARQUET)
    feat_cols = [c for c in prof.columns if c.startswith(("dep_", "arr_"))] + ["ratio_arr_dep", "dur_median", "pct_circular"]
    X = StandardScaler().fit_transform(prof[feat_cols].fillna(0))
    pca = PCA(n_components=PCA_VARIANCE, random_state=SEED).fit(X)
    Xp = pca.transform(X)
    used = {k: hp[k] for k in HP_BY_MODEL[model]}
    params = {"n_features": len(feat_cols), "pca_variance": PCA_VARIANCE, "pca_components": int(pca.n_components_),
              "profile_period_end": TRAIN_END, "n_stations": len(prof), **used}
    if model == "municipio":
        labels = pd.factorize(prof["municipio"])[0]
        params["grouping"] = "prefijo GDL/ZPN/TLQ de la nomenclatura"
        run_name = "p3-baseline-municipio"
    else:
        labels = fit_predict(model, Xp, hp["k"])
        params["bootstrap_samples"] = N_BOOT
        run_name = f"p3-{model}-k{hp['k']}"

    with start_run("P3", run_name, model, params, stage_of_work=stage_of_work, nested=nested):
        log_training_input(prof[["station_id", *feat_cols]], "p3_station_profiles", source=PROFILES_PARQUET)
        n_lab = len(np.unique(labels))
        m = {"train_n_clusters": n_lab}
        if n_lab > 1:
            m.update({"train_silhouette": silhouette_score(Xp, labels),
                      "train_davies_bouldin": davies_bouldin_score(Xp, labels),
                      "train_calinski_harabasz": calinski_harabasz_score(Xp, labels)})
        sizes = pd.Series(labels).value_counts().sort_index()
        out = prof[["station_id", "lat", "lon", "municipio"]].assign(cluster=labels)
        if model != "municipio":
            jac = jaccard_stability(model, Xp, labels, hp["k"])
            m.update({"train_jaccard_bootstrap_mean": float(jac.mean()),
                      "train_jaccard_bootstrap_min": float(jac.min()),
                      "train_ari_vs_municipio": adjusted_rand_score(pd.factorize(prof["municipio"])[0], labels)})
            log_table(pd.DataFrame({"cluster": range(hp["k"]), "jaccard": jac, "n_stations": sizes.reindex(range(hp["k"])).values}),
                      f"p3_{model}_estabilidad_por_cluster.csv")
        mlflow.log_metrics(m)
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
        fig.suptitle(f"P3 {model}: perfil horario promedio por grupo")
        log_figure(fig, f"p3_perfiles_{model}.png")
        # Modelo registrable: asigna una estación (perfil de rasgos) al grupo con el centroide
        # más cercano en el mismo espacio estandarizado y reducido con PCA. En K-Means equivale
        # a su predict; en Ward y GMM permite asignar estaciones nuevas con la misma interfaz.
        assigner = make_pipeline(StandardScaler(), PCA(n_components=PCA_VARIANCE, random_state=SEED),
                                 NearestCentroid()).fit(prof[feat_cols].fillna(0), labels)
        log_sklearn_model(assigner, prof[feat_cols].fillna(0))
        logger.success(f"Run P3 ({run_name}) registrado")
    return m
