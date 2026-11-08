# Evidencia de construcción, ejecución y pruebas del contenedor de la API

Este documento contiene la memoria técnica y evidencia de ejecución de la imagen OCI de la API de demanda de MiBici (P1), empaquetada mediante [`Containerfile`](../Containerfile) y probada con **Podman**.

---

## 1. Especificaciones de la imagen

* **Archivo de definición:** [`Containerfile`](../Containerfile) (cumple estándar OCI, compatible con Podman y Docker).
* **Imagen base:** `docker.io/library/python:3.12-slim-bookworm`.
* **Usuario de ejecución:** `appuser` (UID `10001`, sin privilegios de root).
* **Puerto expuesto:** `8000/tcp`.
* **Servicio:** FastAPI + Uvicorn (`mibici.api.main:app`).
* **Variables de entorno por defecto:**
  * `PYTHONDONTWRITEBYTECODE=1`
  * `PYTHONUNBUFFERED=1`
  * `PIP_NO_CACHE_DIR=1`
  * `PYTHONPATH=/app`
  * `MLFLOW_TRACKING_URI=http://host.containers.internal:5001`
  * `MODEL_ALIAS=champion`

---

## 2. Construcción de la imagen (`podman build`)

Comando ejecutado desde la raíz del proyecto:

```bash
podman build -t mibici-api:1.0.0 .
```

### Salida de la construcción:
```text
STEP 1/11: FROM docker.io/library/python:3.12-slim-bookworm
STEP 2/11: LABEL org.opencontainers.image.title="mibici-api" ...
STEP 3/11: ENV PYTHONDONTWRITEBYTECODE=1 ...
STEP 4/11: WORKDIR /app
STEP 5/11: COPY requirements-api.txt .
STEP 6/11: RUN pip install --no-cache-dir -r requirements-api.txt
--> Successfully installed fastapi uvicorn mlflow scikit-learn pandas numpy pyarrow requests ...
STEP 7/11: COPY mibici/ ./mibici/
STEP 8/11: RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
STEP 9/11: USER appuser
STEP 10/11: EXPOSE 8000
STEP 11/11: CMD ["uvicorn", "mibici.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
COMMIT mibici-api:1.0.0
--> 3253fad1939e
Successfully tagged localhost/mibici-api:1.0.0
```

Verificación en el catálogo local (`podman images`):
```text
REPOSITORY            TAG     IMAGE ID      CREATED         SIZE
localhost/mibici-api  1.0.0   3253fad1939e  20 minutes ago  929 MB
```

---

## 3. Ejecución del contenedor (`podman run`)

Se arrancó el contenedor montando el modelo campeón (`mibici-equipo24-p1-demanda@champion`) almacenado en `models/p1` en modo solo lectura (`:ro`) y mapeando el puerto `8000`:

```bash
podman run --rm --name mibici-api -d -p 8000:8000 \
  -v "${PWD}/models/p1:/modelo:ro" \
  -e MODEL_URI=/modelo \
  localhost/mibici-api:1.0.0
```

### Registro de arranque (`podman logs mibici-api`):
```text
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## 4. Pruebas de los endpoints y evidencia de respuesta

### Prueba 1: Diagnóstico de salud (`GET /health`)

#### Petición:
```powershell
Invoke-RestMethod -Uri "http://<IP_CONTENEDOR>:8000/health" -Method Get | ConvertTo-Json
```
*(o con cURL: `curl -i http://localhost:8000/health`)*

#### Respuesta (HTTP 200 OK):
```json
{
  "estado": "ok",
  "modelo": "mibici-equipo24-p1-demanda",
  "version": "local",
  "uri": "/modelo"
}
```

---

### Prueba 2: Predicción de demanda individual (`POST /predict`)

Estima los viajes que iniciarán en una estación en una hora determinada usando las variables predictoras del modelo de demanda (`P1_FEATURES`).

#### Petición:
```powershell
$body = @{
  hour = 8
  dow = 1
  month = 9
  is_weekend = 0
  is_holiday = 0
  is_quincena = 0
  dep_lag_24h = 6.0
  dep_lag_168h = 7.0
  dep_mean_7d = 5.4
  dep_mean_28d = 5.9
  capacity = 19
  lat = 20.6767
  lon = -103.3475
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://<IP_CONTENEDOR>:8000/predict" -Method Post -Body $body -ContentType "application/json" | ConvertTo-Json
```

#### Respuesta (HTTP 200 OK):
```json
{
  "viajes_estimados": 2.425
}
```

---

### Prueba 3: Predicción por lote (`POST /predict/lote`)

Permite realizar inferencia sobre múltiples estaciones y horas en una sola llamada HTTP.

#### Petición:
```powershell
$lote = @{
  registros = @(
    @{
      hour = 8; dow = 1; month = 9; is_weekend = 0; is_holiday = 0; is_quincena = 0
      dep_lag_24h = 6.0; dep_lag_168h = 7.0; dep_mean_7d = 5.4; dep_mean_28d = 5.9
      capacity = 19; lat = 20.6767; lon = -103.3475
    },
    @{
      hour = 18; dow = 4; month = 9; is_weekend = 0; is_holiday = 0; is_quincena = 0
      dep_lag_24h = 12.0; dep_lag_168h = 10.0; dep_mean_7d = 11.2; dep_mean_28d = 10.5
      capacity = 24; lat = 20.6767; lon = -103.3475
    }
  )
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri "http://<IP_CONTENEDOR>:8000/predict/lote" -Method Post -Body $lote -ContentType "application/json" | ConvertTo-Json -Depth 5
```

#### Respuesta (HTTP 200 OK):
```json
{
  "modelo": "mibici-equipo24-p1-demanda",
  "version": "local",
  "predicciones": [
    {
      "viajes_estimados": 2.425
    },
    {
      "viajes_estimados": 8.03
    }
  ]
}
```

---

### Prueba 4: Validación de esquema y control de errores

Se verificó el comportamiento ante peticiones con entradas inválidas (por ejemplo, una hora fuera de rango `hour: 25` o inconsistencia entre `dow` e `is_weekend`).

#### Respuesta (HTTP 422 Unprocessable Entity):
El servidor rechaza la petición antes de pasarla al modelo gracias a las validaciones de Pydantic, garantizando la integridad de los datos de entrada:
```json
{
  "detail": [
    {
      "type": "less_than_equal",
      "loc": ["body", "hour"],
      "msg": "Input should be less than or equal to 23",
      "input": 25,
      "ctx": { "le": 23 }
    }
  ]
}
```

---

## 5. Registro de accesos del servidor

Consola de Uvicorn dentro del contenedor evidenciando la recepción y respuesta de las peticiones de prueba:

```text
INFO:     172.28.0.1:58412 - "GET /health HTTP/1.1" 200 OK
INFO:     172.28.0.1:58419 - "POST /predict HTTP/1.1" 200 OK
INFO:     172.28.0.1:57733 - "POST /predict/lote HTTP/1.1" 200 OK
INFO:     172.28.0.1:65082 - "POST /predict HTTP/1.1" 422 Unprocessable Entity
INFO:     172.28.0.1:62805 - "POST /predict HTTP/1.1" 200 OK
```

