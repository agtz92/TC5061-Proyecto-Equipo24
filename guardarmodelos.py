from pathlib import Path
import mlflow

from mibici.config import MLFLOW_TRACKING_URI, REGISTRY_NAME

# La URI se toma de MLFLOW_TRACKING_URI (archivo .env o variable de entorno),
# no se escribe en el código. Ver .env.example.
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Mismos nombres que usan la API, el monitoreo y el A/B test (mibici/config.py).
# Antes P1 apuntaba a "Mibici-equipo24-P1-demanda", un registro viejo distinto del @champion vigente.
modelos = dict(REGISTRY_NAME)
 
for pregunta, nombre in modelos.items():
    destino = Path("models") / pregunta / "champion"
    destino.mkdir(parents=True, exist_ok=True)
 
    ruta = mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{nombre}@champion",
        dst_path=str(destino),
    )
 
    print(f"{pregunta} descargado en: {ruta}")
