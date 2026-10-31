from pathlib import Path
import mlflow

from mibici.config import MLFLOW_TRACKING_URI

# La URI se toma de MLFLOW_TRACKING_URI (archivo .env o variable de entorno),
# no se escribe en el código. Ver .env.example.
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

modelos = {
    "p1": "Mibici-equipo24-P1-demanda",  # así quedó registrado (con mayúsculas)
    "p2": "mibici-equipo24-p2-riesgo",
    "p3": "mibici-equipo24-p3-grupos",
}
 
for pregunta, nombre in modelos.items():
    destino = Path("models") / pregunta / "champion"
    destino.mkdir(parents=True, exist_ok=True)
 
    ruta = mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{nombre}@champion",
        dst_path=str(destino),
    )
 
    print(f"{pregunta} descargado en: {ruta}")
