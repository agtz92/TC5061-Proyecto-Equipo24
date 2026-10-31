import mlflow

from mibici.config import MLFLOW_TRACKING_URI

# La URI se toma de MLFLOW_TRACKING_URI (archivo .env o variable de entorno),
# no se escribe en el código. Ver .env.example.
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
 
nombres = [
    "Mibici-equipo24-P1-demanda",
    "mibici-equipo24-p2-riesgo",
    "mibici-equipo24-p3-grupos",
]
 
for nombre in nombres:
    uri = f"models:/{nombre}@champion"
 
    try:
        modelo = mlflow.pyfunc.load_model(uri)
        print(f"OK: {uri} se cargó correctamente")
    except Exception as error:
        print(f"ERROR: {uri}\n{error}")