"""Prueba de carga de la API de demanda de MiBici (P1) con Locust.

Escenario: usuarios de una app/tablero que piden pronósticos de demanda para el día siguiente.
Cada usuario virtual repite, con una pausa de 0.5 a 1.5 s entre acciones (tiempo de "lectura"):

| Tarea                 | Peso | Qué simula                                                       |
|-----------------------|------|------------------------------------------------------------------|
| POST /predict         | 10   | Consulta de una estación en una hora (el caso principal)         |
| POST /predict/lote    | 2    | Pronóstico de las 24 horas de una estación (CARGA_LOTE registros) |
| GET  /health          | 1    | Sondeo de disponibilidad (balanceador / monitoreo)               |
| POST /predict inválido| 1    | Entrada fuera de rango: se espera HTTP 422 y cuenta como éxito   |

Las entradas se generan con una red sintética de 300 estaciones dentro del Área Metropolitana
de Guadalajara (semilla fija), con rezagos coherentes con el perfil horario de demanda y con
is_weekend consistente con dow, para que pasen la validación de Pydantic igual que en producción.

Perfil de carga (por defecto, CARGA_ESCENARIO=escalonado): escalones de 60 s con 10, 25, 50,
100 y 150 usuarios (5 min en total). Así se ve en qué punto la latencia se dispara o aparecen
errores con los límites del compose (1 CPU y 1 GB para la API). Con CARGA_ESCENARIO=constante,
la carga se controla con -u, -r y -t de Locust.

Ejecutar (con el sistema levantado con `docker compose up -d --build`):

    locust -f loadtest/locustfile.py --host http://localhost:8000 --headless \
        --csv reports/carga/locust --csv-full-history --html reports/carga/locust.html
    python loadtest/generar_reporte.py

Variables de entorno opcionales:
    CARGA_ESCENARIO   escalonado (por defecto) | constante
    CARGA_USUARIOS    usuarios por escalón, separados por coma (por defecto 10,25,50,100,150)
    CARGA_ETAPA_SEG   duración de cada escalón en segundos (por defecto 60)
    CARGA_LOTE        registros por petición a /predict/lote (por defecto 24)
"""

from __future__ import annotations

import math
import os
import random

from locust import HttpUser, LoadTestShape, between, task

# Límites de la API (mibici/api/schemas.py)
LAT_MIN, LAT_MAX = 20.60, 20.74  # núcleo de la red de MiBici, dentro de los límites de la API
LON_MIN, LON_MAX = -103.43, -103.30
TAMANO_LOTE = int(os.getenv("CARGA_LOTE", "24"))

# Perfil horario típico de salidas (relativo): picos de la mañana y de la tarde
PERFIL_HORA = [0.05, 0.03, 0.02, 0.02, 0.05, 0.25, 0.70, 1.30, 1.60, 1.10, 0.80, 0.80,
               0.90, 1.00, 1.00, 0.95, 1.05, 1.30, 1.50, 1.20, 0.85, 0.60, 0.35, 0.15]

_rng = random.Random(24)
ESTACIONES = [
    {
        "lat": round(_rng.uniform(LAT_MIN, LAT_MAX), 6),
        "lon": round(_rng.uniform(LON_MIN, LON_MAX), 6),
        "capacity": _rng.randint(12, 36),
        "base": _rng.uniform(0.5, 8.0),  # salidas por hora en una hora "promedio"
    }
    for _ in range(300)
]


def registro(rng: random.Random, estacion: dict | None = None, hour: int | None = None) -> dict:
    """Una estación-hora válida para /predict, con rezagos coherentes con la hora."""
    est = estacion or rng.choice(ESTACIONES)
    if hour is None:
        hour = rng.choices(range(24), weights=PERFIL_HORA)[0]
    dow = rng.randint(0, 6)
    fin_semana = int(dow >= 5)
    nivel = est["base"] * PERFIL_HORA[hour] * (0.6 if fin_semana else 1.0)

    def ruido(x: float, sd: float) -> float:
        return round(min(500.0, max(0.0, rng.gauss(x, sd * max(x, 0.5)))), 2)

    return {
        "hour": hour,
        "dow": dow,
        "month": rng.randint(1, 12),
        "is_weekend": fin_semana,
        "is_holiday": int(rng.random() < 0.03),
        "is_quincena": int(rng.random() < 2 / 30),
        "dep_lag_24h": float(round(ruido(nivel, 0.5))),
        "dep_lag_168h": float(round(ruido(nivel, 0.5))),
        "dep_mean_7d": ruido(nivel, 0.2),
        "dep_mean_28d": ruido(nivel, 0.1),
        "capacity": est["capacity"],
        "lat": est["lat"],
        "lon": est["lon"],
    }


class UsuarioPronostico(HttpUser):
    """Cliente de la API que consulta pronósticos de demanda."""

    wait_time = between(0.5, 1.5)

    def on_start(self):
        self.rng = random.Random()

    @task(10)
    def predecir(self):
        with self.client.post("/predict", json=registro(self.rng), name="POST /predict",
                              catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
                return
            valor = r.json().get("viajes_estimados")
            if not isinstance(valor, (int, float)) or not math.isfinite(valor) or valor < 0:
                r.failure(f"Predicción inválida: {valor!r}")

    @task(2)
    def predecir_lote(self):
        est = self.rng.choice(ESTACIONES)
        horas = [h % 24 for h in range(TAMANO_LOTE)]
        cuerpo = {"registros": [registro(self.rng, est, h) for h in horas]}
        with self.client.post("/predict/lote", json=cuerpo, name="POST /predict/lote",
                              catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
            elif len(r.json().get("predicciones", [])) != TAMANO_LOTE:
                r.failure("El lote no devolvió una predicción por registro")

    @task(1)
    def salud(self):
        with self.client.get("/health", name="GET /health", catch_response=True) as r:
            if r.status_code != 200 or r.json().get("estado") != "ok":
                r.failure(f"HTTP {r.status_code}")

    @task(1)
    def entrada_invalida(self):
        """Hora 25: la API debe rechazarla con 422 sin llegar al modelo."""
        cuerpo = dict(registro(self.rng), hour=25)
        with self.client.post("/predict", json=cuerpo, name="POST /predict (422 esperado)",
                              catch_response=True) as r:
            if r.status_code == 422:
                r.success()
            else:
                r.failure(f"Se esperaba 422 y llegó HTTP {r.status_code}")


if os.getenv("CARGA_ESCENARIO", "escalonado") == "escalonado":

    class CargaEscalonada(LoadTestShape):
        """Escalones de usuarios de duración fija; termina al acabar el último."""

        usuarios = [int(u) for u in os.getenv("CARGA_USUARIOS", "10,25,50,100,150").split(",")]
        duracion = int(os.getenv("CARGA_ETAPA_SEG", "60"))

        def tick(self):
            etapa = int(self.get_run_time() // self.duracion)
            if etapa >= len(self.usuarios):
                return None
            return self.usuarios[etapa], 10
