import mlflow
 
# Usa la dirección de tu servidor de MLflow.
mlflow.set_tracking_uri("http://127.0.0.1:5001")
 
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