"""Entrenamiento y evaluación de P1, P2 y P3 registrados en MLflow.

Se ejecuta desde el MLflow Project (punto de entrada `train`):

    mlflow run . -e train -P question=p1 -P model=rf -P n_estimators=300
    mlflow run . -e train -P question=p2 -P model=hgb -P learning_rate=0.05
    mlflow run . -e train -P question=p3 -P model=kmeans -P k=6

Cualquier parámetro que se deje en "config" toma su valor de mibici/config.py
(DEFAULT_MODEL, HYPERPARAMS, TRAIN_SAMPLE_ROWS); así MLproject no repite valores.

Cada run registra:
- tags: team, question, model, stage, git_commit, git_dirty, data_version
- params: rango de fechas, partición, versión de datos, semilla, modelo e hiperparámetros
- métricas con prefijo del conjunto (train_*, val_*, test_*)
- artefactos: gráficas, tablas agregadas y el modelo con firma (P1 y P2)
Nunca se registran datos a nivel de viaje ni de usuario.
"""

from __future__ import annotations

from loguru import logger
import typer

from mibici.config import (
    DEFAULT_MODEL,
    HYPERPARAMS,
    MODEL_CHOICES,
    PANEL_PARQUET,
    PROFILES_PARQUET,
    TRAIN_SAMPLE_ROWS,
)
from mibici.modeling import p1_demand, p2_risk, p3_clusters

app = typer.Typer(help="Entrenamiento de modelos MiBici (P1, P2, P3) con MLflow.")

DEFAULT = "config"  # valor que significa "usar el de config.py"


def resolve(value: str, default, cast):
    """Convierte el valor recibido del CLI; "config" o vacío usa el default de config.py."""
    if value is None or str(value).strip().lower() in ("", DEFAULT):
        return default
    return cast(value)


def ensure_features() -> None:
    """Construye el panel y los perfiles si aún no existen (clon limpio + ingesta)."""
    if not PANEL_PARQUET.exists() or not PROFILES_PARQUET.exists():
        from mibici.features import main as build_features

        logger.info("No hay features procesadas; ejecutando mibici.features")
        build_features()


def train_one(question: str, model: str, hp: dict, sample_rows: int) -> None:
    question = question.lower()
    if question not in MODEL_CHOICES:
        raise typer.BadParameter(f"question debe ser una de {list(MODEL_CHOICES)}")
    if model not in MODEL_CHOICES[question]:
        raise typer.BadParameter(f"model para {question} debe ser uno de {MODEL_CHOICES[question]}")
    ensure_features()
    if question == "p1":
        p1_demand.run(model, hp, sample_rows)
    elif question == "p2":
        p2_risk.run(model, hp, sample_rows)
    else:
        p3_clusters.run(model, hp)


@app.command()
def run(
    question: str = typer.Option(..., help="p1 | p2 | p3"),
    model: str = typer.Option(DEFAULT, help="Ver MODEL_CHOICES en config.py"),
    n_estimators: str = typer.Option(DEFAULT),
    max_depth: str = typer.Option(DEFAULT),
    min_samples_leaf: str = typer.Option(DEFAULT),
    max_iter: str = typer.Option(DEFAULT),
    learning_rate: str = typer.Option(DEFAULT),
    alpha: str = typer.Option(DEFAULT),
    c: str = typer.Option(DEFAULT, "--C", help="Regularización inversa de logreg"),
    k: str = typer.Option(DEFAULT, help="Número de grupos (P3)"),
    sample_rows: str = typer.Option(DEFAULT, help="Filas de la muestra de entrenamiento"),
):
    """Entrena y evalúa un modelo para una pregunta y lo registra en MLflow."""
    hp = {
        "n_estimators": resolve(n_estimators, HYPERPARAMS["n_estimators"], int),
        "max_depth": resolve(max_depth, HYPERPARAMS["max_depth"], int),
        "min_samples_leaf": resolve(min_samples_leaf, HYPERPARAMS["min_samples_leaf"], int),
        "max_iter": resolve(max_iter, HYPERPARAMS["max_iter"], int),
        "learning_rate": resolve(learning_rate, HYPERPARAMS["learning_rate"], float),
        "alpha": resolve(alpha, HYPERPARAMS["alpha"], float),
        "C": resolve(c, HYPERPARAMS["C"], float),
        "k": resolve(k, HYPERPARAMS["k"], int),
    }
    q = question.lower()
    train_one(q, resolve(model, DEFAULT_MODEL.get(q), str), hp, resolve(sample_rows, TRAIN_SAMPLE_ROWS, int))


@app.command()
def baselines():
    """Corre los baselines de las tres preguntas (experimento base de la propuesta)."""
    for q, m in [("p1", "baseline"), ("p2", "baseline"), ("p3", "municipio")]:
        train_one(q, m, HYPERPARAMS, TRAIN_SAMPLE_ROWS)


if __name__ == "__main__":
    app()
