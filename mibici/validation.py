"""Validacion de integridad y drift de los datasets tabulares de MiBici.

Ejemplos:
    python -m mibici.validation --question all
    python -m mibici.validation --question p1 --sample-rows 10000

Los reportes HTML se escriben en reports/deepchecks/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer

# Deepchecks 0.19.1 uses this NumPy alias, removed in NumPy 2.
if not hasattr(np, "Inf"):
    setattr(np, "Inf", np.inf)

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity, train_test_validation

from mibici.config import (
    P1_FEATURES,
    P2_FEATURES,
    PANEL_PARQUET,
    PROFILES_PARQUET,
    START_MONTH,
    STATIONS_PARQUET,
    TEST_END,
    TEST_START,
    TRAIN_END,
)
from mibici.features import load_supervised_frame

app = typer.Typer(help="Validacion de datasets con Deepchecks.")
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports" / "deepchecks"
DEFAULT_SAMPLE_ROWS = 10_000


def _supervised_dataset(
    frame: pd.DataFrame,
    label: str,
    features: list[str],
) -> Dataset:
    """Construye un Dataset con las columnas objetivo, de fecha y features explicitas."""
    required = {"ts", label, *features}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas para Deepchecks: {sorted(missing)}")
    if frame.empty:
        raise ValueError("El conjunto de datos esta vacio.")
    return Dataset(
        frame,
        label=label,
        datetime_name="ts",
        features=features,
        cat_features=[],
    )


def _save_integrity_report(dataset: Dataset, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    resultado=data_integrity().run(dataset)
    resultado.save_as_html(str(path),as_widget=False,connected=False)
    return path


def _save_split_report(train: Dataset, test: Dataset, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    train_test_validation().run(train, test).save_as_html(str(path))
    return path


def _validate_supervised_question(question: str, sample_rows: int) -> list[Path]:
    if question == "p1":
        label, features = "departures", P1_FEATURES
    else:
        label, features = "p2_label", P2_FEATURES

    columns = ["station_id", label, *features]
    train = load_supervised_frame(
        f"{START_MONTH}-01", TRAIN_END, columns, sample_rows=sample_rows
    )
    test = load_supervised_frame(TEST_START, TEST_END, columns, sample_rows=sample_rows)
    train_dataset = _supervised_dataset(train, label, features)
    test_dataset = _supervised_dataset(test, label, features)

    outputs = [
        _save_integrity_report(train_dataset, REPORTS_DIR / f"{question}_integridad_train.html"),
        _save_integrity_report(test_dataset, REPORTS_DIR / f"{question}_integridad_test.html"),
        _save_split_report(train_dataset, test_dataset, REPORTS_DIR / f"{question}_train_test.html"),
    ]
    return outputs


def _validate_profiles() -> Path:
    profiles = pd.read_parquet(PROFILES_PARQUET)
    required = {"station_id", "municipio"}
    missing = required.difference(profiles.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en perfiles P3: {sorted(missing)}")
    if profiles.empty:
        raise ValueError("El conjunto de perfiles P3 esta vacio.")

    features = [column for column in profiles.columns if column != "station_id"]
    dataset = Dataset(
        profiles,
        index_name="station_id",
        cat_features=["municipio"],
        features=features,
    )
    return _save_integrity_report(dataset, REPORTS_DIR / "p3_integridad.html")


def validate(question: str = "all", sample_rows: int = DEFAULT_SAMPLE_ROWS) -> list[Path]:
    """Ejecuta validaciones P1/P2/P3 y devuelve las rutas de los reportes creados."""
    question = question.lower()
    valid_questions = {"all", "p1", "p2", "p3"}
    if question not in valid_questions:
        raise ValueError(f"question debe ser una de {sorted(valid_questions)}")
    if sample_rows < 1:
        raise ValueError("sample_rows debe ser mayor que cero.")

    if question in {"all", "p1", "p2"}:
        required_files = [PANEL_PARQUET, STATIONS_PARQUET]
        missing = [path for path in required_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "No se encontraron los datos procesados requeridos: "
                f"{missing}. Ejecuta primero `make data` y `make features`."
            )
    if question in {"all", "p3"} and not PROFILES_PARQUET.is_file():
        raise FileNotFoundError(
            f"No se encontro {PROFILES_PARQUET}. Ejecuta primero `make data` y `make features`."
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = ["p1", "p2"] if question == "all" else ([question] if question in {"p1", "p2"} else [])
    outputs = [
        path
        for current_question in questions
        for path in _validate_supervised_question(current_question, sample_rows)
    ]
    if question in {"all", "p3"}:
        outputs.append(_validate_profiles())
    return outputs


@app.command()
def run(
    question: str = typer.Option("all", help="all | p1 | p2 | p3"),
    sample_rows: int = typer.Option(DEFAULT_SAMPLE_ROWS, min=1, help="Maximo de filas por split"),
) -> None:
    """Ejecuta Deepchecks y guarda reportes HTML bajo reports/deepchecks/."""
    try:
        outputs = validate(question, sample_rows)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    for path in outputs:
        typer.echo(f"Reporte: {path}")


if __name__ == "__main__":
    app()