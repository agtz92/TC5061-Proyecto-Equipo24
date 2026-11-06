"""Pruebas de la API de demanda (mibici.api) con un modelo falso: no requieren servidor de MLflow."""

import pytest
from fastapi.testclient import TestClient

import mibici.api.main as api
from mibici.api.schemas import DemandaEntrada
from mibici.config import P1_FEATURES

EJEMPLO = DemandaEntrada.model_config["json_schema_extra"]["example"]


class ModeloFalso:
    """Devuelve dep_mean_7d y verifica que reciba las columnas y el tipo de la firma."""

    def predict(self, X):
        assert list(X.columns) == P1_FEATURES
        assert all(str(t) == "float32" for t in X.dtypes)
        return X["dep_mean_7d"].to_numpy() - 1.0


@pytest.fixture
def cliente(monkeypatch):
    monkeypatch.setattr(api, "cargar_modelo", lambda uri=api.MODEL_URI: {
        "modelo": ModeloFalso(), "version": "7", "uri": uri})
    with TestClient(api.app) as c:
        yield c


def test_health_reporta_modelo_y_version(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200
    assert r.json() == {"estado": "ok", "modelo": "mibici-equipo24-p1-demanda", "version": "7",
                        "uri": "models:/mibici-equipo24-p1-demanda@champion"}


def test_predict_devuelve_json_con_la_prediccion(cliente):
    r = cliente.post("/predict", json=EJEMPLO)
    assert r.status_code == 200
    assert r.json() == {"viajes_estimados": pytest.approx(EJEMPLO["dep_mean_7d"] - 1.0)}


def test_prediccion_negativa_se_recorta_a_cero(cliente):
    r = cliente.post("/predict", json=dict(EJEMPLO, dep_mean_7d=0.2))
    assert r.json()["viajes_estimados"] == 0.0


def test_lote_respeta_el_orden(cliente):
    regs = [dict(EJEMPLO, dep_mean_7d=v) for v in (3.0, 9.0, 5.0)]
    r = cliente.post("/predict/lote", json={"registros": regs})
    assert r.status_code == 200
    assert [p["viajes_estimados"] for p in r.json()["predicciones"]] == [2.0, 8.0, 4.0]


@pytest.mark.parametrize("cambio, campo", [
    ({"hour": 24}, "hour"),
    ({"dow": -1}, "dow"),
    ({"lat": 19.43}, "lat"),
    ({"dep_lag_24h": -3}, "dep_lag_24h"),
    ({"capacity": 0}, "capacity"),
    ({"otra_variable": 1}, "otra_variable"),
])
def test_entradas_invalidas_devuelven_422(cliente, cambio, campo):
    r = cliente.post("/predict", json=dict(EJEMPLO, **cambio))
    assert r.status_code == 422
    assert campo in {str(e["loc"][-1]) for e in r.json()["detail"]}


def test_falta_una_variable_devuelve_422(cliente):
    incompleto = {k: v for k, v in EJEMPLO.items() if k != "capacity"}
    assert cliente.post("/predict", json=incompleto).status_code == 422


def test_fin_de_semana_inconsistente_con_dow(cliente):
    r = cliente.post("/predict", json=dict(EJEMPLO, dow=6, is_weekend=0))
    assert r.status_code == 422


def test_lote_vacio_o_demasiado_grande(cliente):
    assert cliente.post("/predict/lote", json={"registros": []}).status_code == 422
    assert cliente.post("/predict/lote", json={"registros": [EJEMPLO] * 1001}).status_code == 422


def test_sin_modelo_cargado_devuelve_503(monkeypatch):
    monkeypatch.setattr(api, "estado", {})
    entrada = DemandaEntrada(**EJEMPLO)
    with pytest.raises(api.HTTPException) as err:
        api.predecir([entrada])
    assert err.value.status_code == 503
