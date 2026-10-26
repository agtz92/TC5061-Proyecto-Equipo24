from pathlib import Path
import mlflow
 
# Cambia la dirección si tu servidor usa otra.
mlflow.set_tracking_uri("http://localhost:5001")

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
