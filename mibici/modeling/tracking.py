"""Utilidades de MLflow compartidas por los tres problemas (P1, P2, P3).

Concentra en un solo lugar las buenas prácticas de registro del proyecto:
nombre de run descriptivo, etiquetas de equipo/pregunta/commit/versión de datos,
parámetros de reproducción y artefactos guardados también en reports/figures/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from mibici.config import (
    DATA_MANIFEST,
    END_MONTH,
    FIGURES_DIR,
    HYPERPARAMS,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
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

_PROJECT_RUN: str | None = None  # run creado por `mlflow run`, si existe

# Abreviaturas de hiperparámetros para nombres de run cortos (p. ej. p1-rf-n300-d12-leaf50)
ABBR = {"n_estimators": "n", "max_depth": "d", "min_samples_leaf": "leaf", "max_iter": "it",
        "learning_rate": "lr", "alpha": "a", "C": "C", "k": "k"}


def git_info() -> dict[str, str]:
    """Commit actual y si hay cambios sin commit (para ligar el run al código)."""

    def run(*args):
        return subprocess.run(["git", *args], cwd=PROJ_ROOT, capture_output=True, text=True).stdout.strip()

    commit = run("rev-parse", "HEAD") or "desconocido"
    dirty = "true" if run("status", "--porcelain", "--untracked-files=no") else "false"
    return {"git_commit": commit, "git_dirty": dirty}


def data_range() -> tuple[str, str]:
    """Rango de meses realmente ingerido (del manifiesto) o el de config.py."""
    if DATA_MANIFEST.exists():
        man = json.loads(DATA_MANIFEST.read_text())
        return man["start_month"], man["end_month"]
    return START_MONTH, END_MONTH


def start_run(question: str, run_name: str, model: str, params: dict, stage: str = "modelado"):
    """Abre un run con etiquetas y parámetros comunes de reproducibilidad.

    Si el proceso lo lanzó `mlflow run`, MLflow ya creó el run (en el experimento
    de --experiment-name o MLFLOW_EXPERIMENT_NAME); aquí solo se le da nombre.
    """
    global _PROJECT_RUN
    env_run = os.environ.get("MLFLOW_RUN_ID")
    if env_run and _PROJECT_RUN is None:
        # Primer run del proceso: se usa el que creó `mlflow run`.
        _PROJECT_RUN = env_run
        run = mlflow.start_run(run_id=env_run)
        mlflow.set_tag("mlflow.runName", run_name)
    elif _PROJECT_RUN is not None:
        # Más de un run en el mismo `mlflow run` (p. ej. baselines): runs hijos
        # en el mismo experimento, ligados al run del proyecto.
        parent = mlflow.get_run(_PROJECT_RUN)
        child = mlflow.MlflowClient().create_run(
            parent.info.experiment_id, run_name=run_name,
            tags={"mlflow.parentRunId": _PROJECT_RUN, **{k: v for k, v in parent.data.tags.items()
                                                       if k.startswith("mlflow.source") or k.startswith("mlflow.project")}})
        run = mlflow.start_run(run_id=child.info.run_id)
    else:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        run = mlflow.start_run(run_name=run_name)
    data_version = load_data_version()
    start_m, end_m = data_range()
    # Los hiperparámetros se registran con prefijo hp_ (valor efectivo, ya resuelto
    # desde config.py). Así no chocan con los parámetros que `mlflow run` registra
    # tal como se pasaron en la línea de comandos (p. ej. "config").
    params = {(f"hp_{k}" if k in HYPERPARAMS else k): v for k, v in params.items()}
    mlflow.set_tags({"team": TEAM, "question": question.upper(), "data_version": data_version,
                     "stage": stage, "model": model, **git_info()})
    mlflow.log_params({
        "start_month": start_m, "end_month": end_m, "train_end": TRAIN_END,
        "val_start": VAL_START, "val_end": VAL_END, "test_start": TEST_START, "test_end": TEST_END,
        "data_version": data_version, "seed": SEED, "estimator": model, **params,
    })
    return run


def log_figure(fig, name: str) -> None:
    """Registra la figura en MLflow y la guarda también en reports/figures/."""
    mlflow.log_figure(fig, f"figures/{name}")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def log_table(df: pd.DataFrame, name: str) -> None:
    """Registra una tabla agregada (nunca datos por viaje o por usuario) como CSV."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / name
        df.to_csv(p, index=False)
        mlflow.log_artifact(str(p), artifact_path="tables")


def log_sklearn_model(model, X_example: pd.DataFrame) -> None:
    """Registra el modelo con firma y ejemplo de entrada (base para el Model Registry)."""
    from mlflow.models import infer_signature

    example = X_example.head(5)
    pred = model.predict(example)
    mlflow.sklearn.log_model(model, artifact_path="model",
                             signature=infer_signature(example, pred), input_example=example)
