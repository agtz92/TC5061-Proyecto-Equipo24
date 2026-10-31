"""Utilidades de MLflow compartidas por los tres problemas (P1, P2, P3).

Concentra en un solo lugar las buenas prácticas de registro del proyecto:
nombre de run descriptivo, etiquetas de equipo/pregunta/commit/versión de datos,
parámetros de reproducción y artefactos organizados en subcarpetas:

    model/   modelo con firma y ejemplo de entrada (P1 y P2)
    plots/   gráficas de desempeño (copia en reports/figures/ con MIBICI_SAVE_FIGURES=1)
    data/    manifiesto de datos y tablas agregadas (nunca datos por viaje o usuario)
    config/  configuración efectiva de la corrida, config.py, MLproject y entorno
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess  # nosec B404 (solo git rev-parse/status con argumentos fijos)
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
    MODEL_FAMILY,
    PANEL_PARQUET,
    PROJ_ROOT,
    REGISTRY_DESCRIPTION,
    REGISTRY_NAME,
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

# Archivos de configuración que se adjuntan a cada corrida (config/)
CONFIG_FILES = ["mibici/config.py", "MLproject", "python_env.yaml", "requirements.txt"]


def git_info() -> dict[str, str]:
    """Commit actual y si hay cambios sin commit (para ligar el run al código)."""

    def run(*args):
        # Comando fijo "git" con argumentos definidos en este módulo (sin entrada externa).
        return subprocess.run(  # nosec B603 B607
            ["git", *args], cwd=PROJ_ROOT, capture_output=True, text=True
        ).stdout.strip()

    commit = run("rev-parse", "HEAD") or "desconocido"
    dirty = "true" if run("status", "--porcelain", "--untracked-files=no") else "false"
    return {"git_commit": commit, "git_dirty": dirty}


def data_range() -> tuple[str, str]:
    """Rango de meses realmente ingerido (del manifiesto) o el de config.py."""
    if DATA_MANIFEST.exists():
        man = json.loads(DATA_MANIFEST.read_text())
        return man["start_month"], man["end_month"]
    return START_MONTH, END_MONTH


def default_stage_of_work(model: str) -> str:
    """baseline para los modelos de referencia, candidate para el resto."""
    return "baseline" if MODEL_FAMILY.get(model) == "baseline" else "candidate"


def start_run(question: str, run_name: str, model: str, params: dict, stage: str = "modelado",
              stage_of_work: str | None = None, nested: bool = False):
    """Abre un run con etiquetas, parámetros y artefactos comunes de reproducibilidad.

    - nested=True: corrida hija dentro de la corrida activa (búsquedas de hiperparámetros).
    - Si el proceso lo lanzó `mlflow run`, MLflow ya creó el run (en el experimento
      de --experiment-name o MLFLOW_EXPERIMENT_NAME); aquí solo se le da nombre.
    """
    global _PROJECT_RUN
    env_run = os.environ.get("MLFLOW_RUN_ID")
    if nested:
        run = mlflow.start_run(run_name=run_name, nested=True)
    elif env_run and _PROJECT_RUN is None:
        # Primer run del proceso: se usa el que creó `mlflow run`.
        _PROJECT_RUN = env_run
        run = mlflow.start_run(run_id=env_run)
        # Se retira la variable: si quedara definida, mlflow.start_run(nested=True) la
        # reutilizaría y la primera corrida hija se registraría dentro de la corrida padre.
        os.environ.pop("MLFLOW_RUN_ID", None)
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
    tags = {"team": TEAM, "question": question.upper(), "data_version": data_version,
            "stage": stage, "model": model, "model_family": MODEL_FAMILY.get(model, model),
            "stage_of_work": stage_of_work or default_stage_of_work(model), **git_info()}
    mlflow.set_tags(tags)
    all_params = {
        "start_month": start_m, "end_month": end_m, "train_end": TRAIN_END,
        "val_start": VAL_START, "val_end": VAL_END, "test_start": TEST_START, "test_end": TEST_END,
        "data_version": data_version, "seed": SEED, "estimator": model, **params,
    }
    mlflow.log_params(all_params)
    log_run_files({"run_name": run_name, "question": question.upper(), "tags": tags, "params": all_params})
    return run


def log_run_files(run_config: dict) -> None:
    """Adjunta la versión de los datos (data/) y la configuración efectiva (config/)."""
    if DATA_MANIFEST.exists():
        mlflow.log_artifact(str(DATA_MANIFEST), artifact_path="data")
    mlflow.log_dict(run_config, "config/run_config.json")
    for rel in CONFIG_FILES:
        p = PROJ_ROOT / rel
        if p.exists():
            mlflow.log_artifact(str(p), artifact_path="config")


def log_training_input(df: pd.DataFrame, name: str, source: Path = PANEL_PARQUET) -> None:
    """Registra el conjunto de entrenamiento como input del run (esquema, perfil y digest).

    MLflow no copia las filas: guarda su huella, así que dos corridas con la misma
    muestra comparten digest. Solo contiene agregados por estación y hora.
    """
    ds = mlflow.data.from_pandas(df, source=str(source), name=name)
    mlflow.log_input(ds, context="training")


def log_figure(fig, name: str) -> None:
    """Registra la figura en plots/ del run.

    Solo se copia a reports/figures/ si MIBICI_SAVE_FIGURES=1: esa carpeta está en Git
    y sobrescribirla en cada corrida dejaría el árbol modificado (git_dirty=true) en
    las corridas siguientes. La fuente de verdad de cada figura es su run en MLflow.
    """
    mlflow.log_figure(fig, f"plots/{name}")
    if os.environ.get("MIBICI_SAVE_FIGURES") == "1":
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def log_table(df: pd.DataFrame, name: str) -> None:
    """Registra una tabla agregada (nunca datos por viaje o por usuario) como CSV en data/."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / name
        df.to_csv(p, index=False)
        mlflow.log_artifact(str(p), artifact_path="data")


def log_sklearn_model(model, X_example: pd.DataFrame) -> None:
    """Registra el modelo en model/ con firma y ejemplo de entrada (base para el Model Registry)."""
    from mlflow.models import infer_signature

    example = X_example.head(5)
    pred = model.predict(example)
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=infer_signature(example, pred),
        input_example=example,
        serialization_format="cloudpickle",
    )


def register_model_version(
    question: str,
    run_id: str | None = None,
    *,
    alias: str = "candidate",
) -> str:
    """Registra el artefacto model/ del run en el Model Registry y devuelve la versión.

    - Crea el modelo registrado con descripción si todavía no existe.
    - Actualiza la descripción del modelo si ha cambiado (no sobreescribe anotaciones
      manuales: sólo actualiza cuando REGISTRY_DESCRIPTION difiere).
    - Etiqueta la versión con el linaje completo: experimento, run de origen,
      commit de git y versión de datos.
    - Asigna el alias indicado (por defecto 'candidate') a la nueva versión.

    Parámetros
    ----------
    question : str
        'p1', 'p2' o 'p3'.  Determina el nombre del modelo en el registry.
    run_id : str | None
        Run de MLflow del que se toma el artefacto model/. Si es None se usa
        el run activo en este momento.
    alias : str
        Alias que se asigna a la nueva versión (por defecto 'candidate').
    """
    import mlflow.sklearn  # noqa: F401  asegura que el flavour está importado

    q = question.lower()
    model_name = REGISTRY_NAME[q]
    description = REGISTRY_DESCRIPTION[q]

    client = mlflow.MlflowClient()

    # --- Obtener metadatos del run -------------------------------------------
    if run_id is None:
        active = mlflow.active_run()
        if active is None:
            raise RuntimeError("register_model_version: no hay run activo ni se proporcionó run_id.")
        run_id = active.info.run_id

    run = client.get_run(run_id)
    run_tags = run.data.tags
    experiment_id = run.info.experiment_id
    experiment = client.get_experiment(experiment_id)
    experiment_name = experiment.name if experiment else experiment_id
    data_version = run_tags.get("data_version", "desconocida")
    git_commit = run_tags.get("git_commit", "desconocido")
    model_tag = run_tags.get("model", q)
    run_name = run.info.run_name or run_id

    # --- Crear el modelo registrado si no existe (con descripción) --------------
    try:
        registered = client.get_registered_model(model_name)
        # Actualiza descripción solo si cambió (preserva anotaciones manuales)
        if registered.description != description:
            client.update_registered_model(model_name, description=description)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(model_name, description=description)

    # --- Registrar la versión desde el artefacto model/ del run ----------------
    artifact_uri = f"runs:/{run_id}/model"
    mv = client.create_model_version(
        name=model_name,
        source=artifact_uri,
        run_id=run_id,
        description=(
            f"Pregunta: {q.upper()} · Estimador: {model_tag} · "
            f"Experimento: {experiment_name} · Run: {run_name} ({run_id[:8]}) · "
            f"Commit: {git_commit[:12]} · Versión datos: {data_version}"
        ),
    )

    # --- Etiquetar la versión con el linaje completo ---------------------------
    lineage_tags = {
        "question": q.upper(),
        "estimator": model_tag,
        "experiment_name": experiment_name,
        "source_run_id": run_id,
        "source_run_name": run_name,
        "git_commit": git_commit,
        "git_dirty": run_tags.get("git_dirty", "unknown"),
        "data_version": data_version,
        "team": run_tags.get("team", TEAM),
        "model_family": run_tags.get("model_family", ""),
        "stage_of_work": run_tags.get("stage_of_work", ""),
    }
    for k, v in lineage_tags.items():
        client.set_model_version_tag(model_name, mv.version, k, v)

    # --- Asignar alias (MLflow >= 2.8) ----------------------------------------
    try:
        client.set_registered_model_alias(model_name, alias, mv.version)
    except AttributeError:
        # MLflow < 2.8: los alias no existen; solo se loguea una advertencia.
        import warnings
        warnings.warn(
            f"Tu versión de MLflow no soporta aliases. Actualiza a >= 2.8 para usar '{alias}'.",
            stacklevel=2,
        )

    from loguru import logger
    logger.success(
        f"Modelo registrado: {model_name} v{mv.version} "
        f"[run={run_id[:8]}, commit={git_commit[:12]}, datos={data_version}]"
    )
    return mv.version
