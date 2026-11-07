# Imagen OCI de la API de demanda de MiBici (P1).
# Compatible con Podman y Docker (formato Dockerfile):
#   podman build -t mibici-api:1.0.0 .
#   docker build -f Containerfile -t mibici-api:1.0.0 .
# Ver la sección "Contenedor de la API" del README.

FROM docker.io/library/python:3.12-slim-bookworm

LABEL org.opencontainers.image.title="mibici-api" \
      org.opencontainers.image.description="API REST (FastAPI) del modelo de demanda de MiBici (P1), Equipo 24 TC5061" \
      org.opencontainers.image.version="1.0.0"

# MLFLOW_TRACKING_URI apunta al servidor de MLflow del anfitrión visto desde el contenedor (Podman).
# Con Docker: -e MLFLOW_TRACKING_URI=http://host.docker.internal:5001
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    MLFLOW_TRACKING_URI=http://host.containers.internal:5001 \
    MODEL_ALIAS=champion

WORKDIR /app

# 1. Dependencias primero: esta capa se reutiliza mientras no cambie requirements-api.txt
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# 2. Código del proyecto (solo el paquete; .dockerignore excluye datos, .env y entornos)
COPY mibici/ ./mibici/

# 3. Usuario sin privilegios
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Estado del servicio: GET /health (no se usa HEALTHCHECK porque el formato OCI no lo admite)
CMD ["uvicorn", "mibici.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
