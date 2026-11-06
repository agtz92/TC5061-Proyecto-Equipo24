"""API REST con FastAPI para el modelo de demanda de MiBici (P1).

El modelo se recupera del MLflow Model Registry por nombre y alias al arrancar:

    MODEL_URI = models:/mibici-equipo24-p1-demanda@champion   (valor por defecto)
    MLFLOW_TRACKING_URI = http://127.0.0.1:5001                 (o http://host.containers.internal:5001
                                                                  dentro de un contenedor)

MODEL_URI también acepta una carpeta local con el modelo ya descargado del registry
(por ejemplo, empaquetada en la imagen), en cuyo caso no se necesita el servidor.

Ejecutar en local:
    uvicorn mibici.api.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException
import mlflow
import numpy as np
import pandas as pd

from mibici.api.schemas import DemandaEntrada, DemandaSalida, LoteEntrada, LoteSalida, Salud
from mibici.config import MLFLOW_TRACKING_URI, P1_FEATURES, REGISTRY_NAME

MODEL_NAME = REGISTRY_NAME["p1"]
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MODEL_URI = os.getenv("MODEL_URI", f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

estado: dict = {}


def cargar_modelo(uri: str = MODEL_URI) -> dict:
    """Carga el modelo (pyfunc) y resuelve su versión en el registry si la URI es models:/."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    modelo = mlflow.pyfunc.load_model(uri)
    version = "local"
    if uri.startswith("models:/") and "@" in uri:
        nombre, alias = uri.removeprefix("models:/").split("@")
        version = mlflow.MlflowClient().get_model_version_by_alias(nombre, alias).version
    return {"modelo": modelo, "version": str(version), "uri": uri}


@asynccontextmanager
async def lifespan(app: FastAPI):
    estado.update(cargar_modelo())
    yield
    estado.clear()


app = FastAPI(
    title="MiBici · API de demanda (P1)",
    description="Estima los viajes que inician en una estación de MiBici en una hora del día siguiente.",
    version="1.0.0",
    lifespan=lifespan,
)


def predecir(registros: list[DemandaEntrada]) -> list[float]:
    if "modelo" not in estado:
        raise HTTPException(status_code=503, detail="El modelo no está cargado")
    # Mismo orden y tipo (float32) que la firma registrada del modelo
    X = pd.DataFrame([r.model_dump() for r in registros])[P1_FEATURES].astype("float32")
    pred = np.clip(np.asarray(estado["modelo"].predict(X), dtype=float), 0, None)
    if not np.all(np.isfinite(pred)):
        raise HTTPException(status_code=500, detail="El modelo devolvió valores no finitos")
    return [round(float(p), 3) for p in pred]


@app.get("/health", response_model=Salud)
def health() -> Salud:
    """Estado del servicio y del modelo cargado."""
    if "modelo" not in estado:
        raise HTTPException(status_code=503, detail="El modelo no está cargado")
    return Salud(estado="ok", modelo=MODEL_NAME, version=estado["version"], uri=estado["uri"])


@app.post("/predict", response_model=DemandaSalida)
def predict(entrada: DemandaEntrada) -> DemandaSalida:
    """Predicción para una estación en una hora."""
    return DemandaSalida(viajes_estimados=predecir([entrada])[0])


@app.post("/predict/lote", response_model=LoteSalida)
def predict_lote(lote: LoteEntrada) -> LoteSalida:
    """Predicción para varias estación-horas en una sola petición."""
    preds = predecir(lote.registros)
    return LoteSalida(modelo=MODEL_NAME, version=estado["version"],
                      predicciones=[DemandaSalida(viajes_estimados=p) for p in preds])
