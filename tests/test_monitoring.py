"""El monitoreo debe predecir con el modelo vigente del registry (@champion) y dejar constancia de cuál usó."""

from __future__ import annotations

import json

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from mibici import monitoring
from mibici.config import P1_FEATURES


def _frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.random(n) for c in P1_FEATURES})
    df["station_id"] = rng.integers(1, 20, n)
    df["departures"] = rng.poisson(1.5, n).astype(float)
    return df


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    """Registry SQLite temporal con dos versiones: v1 predice 1.0 (champion) y v2 predice 5.0."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(monitoring, "MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("prueba-monitoreo")
    X = _frame(10, 0)[P1_FEATURES].astype("float32")
    for valor in (1.0, 5.0):
        with mlflow.start_run():
            modelo = DummyRegressor(strategy="constant", constant=valor).fit(X, np.zeros(len(X)))
            mlflow.sklearn.log_model(modelo, "model", registered_model_name="p1-prueba")
    cliente = mlflow.MlflowClient()
    cliente.set_registered_model_alias("p1-prueba", "champion", "1")
    cliente.set_registered_model_alias("p1-prueba", "challenger", "2")
    yield "models:/p1-prueba@champion"
    mlflow.set_tracking_uri(None)


def test_monitoreo_usa_el_champion_y_registra_la_version(registry, tmp_path, monkeypatch):
    monkeypatch.setattr(monitoring, "load_supervised_frame",
                        lambda start, end, columns, sample_rows: _frame(50, 1)[columns])
    salida = tmp_path / "monitoring"
    ref, cur = monitoring.preparar_datasets(output_dir=salida, export_csv=False, model_uri=registry)

    assert np.allclose(ref["prediction"], 1.0) and np.allclose(cur["prediction"], 1.0)
    info = json.loads((salida / "modelo_monitoreo.json").read_text(encoding="utf-8"))
    assert info["alias"] == "champion" and info["version"] == "1"
    assert info["reference_rows"] == 50 and info["current_rows"] == 50
    assert (salida / "reference.parquet").exists() and (salida / "current.parquet").exists()


def test_monitoreo_falla_si_no_puede_cargar_el_modelo(registry, tmp_path, monkeypatch):
    """Antes se omitían las predicciones en silencio; ahora un modelo inexistente detiene el proceso."""
    monkeypatch.setattr(monitoring, "load_supervised_frame",
                        lambda start, end, columns, sample_rows: _frame(5, 2)[columns])
    with pytest.raises(Exception):
        monitoring.preparar_datasets(output_dir=tmp_path / "m", export_csv=False,
                                     model_uri="models:/p1-prueba@no-existe")


def test_default_apunta_al_champion_de_p1():
    assert monitoring.DEFAULT_MODEL_URI == "models:/mibici-equipo24-p1-demanda@champion"
