"""Esquemas de entrada y salida de la API (Pydantic).

Las variables son las mismas `P1_FEATURES` con las que se entrenó el modelo de demanda.
Todas se conocen a las 00:00 del día anterior a la hora que se predice.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Área Metropolitana de Guadalajara, con margen: evita coordenadas fuera de la red de MiBici.
LAT_MIN, LAT_MAX = 20.55, 20.80
LON_MIN, LON_MAX = -103.50, -103.25


class DemandaEntrada(BaseModel):
    """Una estación en una hora del día siguiente."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "hour": 8, "dow": 1, "month": 9, "is_weekend": 0, "is_holiday": 0, "is_quincena": 0,
        "dep_lag_24h": 6, "dep_lag_168h": 7, "dep_mean_7d": 5.4, "dep_mean_28d": 5.9,
        "capacity": 19, "lat": 20.6767, "lon": -103.3475,
    }})

    hour: int = Field(..., ge=0, le=23, description="Hora del día (0-23)")
    dow: int = Field(..., ge=0, le=6, description="Día de la semana (0 = lunes)")
    month: int = Field(..., ge=1, le=12, description="Mes (1-12)")
    is_weekend: int = Field(..., ge=0, le=1, description="1 si es sábado o domingo")
    is_holiday: int = Field(..., ge=0, le=1, description="1 si es día festivo oficial")
    is_quincena: int = Field(..., ge=0, le=1, description="1 si es día 15 o fin de mes")
    dep_lag_24h: float = Field(..., ge=0, le=500, description="Salidas de la misma hora hace 24 h")
    dep_lag_168h: float = Field(..., ge=0, le=500, description="Salidas de la misma hora hace 7 días")
    dep_mean_7d: float = Field(..., ge=0, le=500, description="Promedio de salidas a esa hora, últimos 7 días")
    dep_mean_28d: float = Field(..., ge=0, le=500, description="Promedio de salidas a esa hora, últimos 28 días")
    capacity: int = Field(..., ge=1, le=100, description="Anclajes de la estación")
    lat: float = Field(..., ge=LAT_MIN, le=LAT_MAX, description="Latitud de la estación")
    lon: float = Field(..., ge=LON_MIN, le=LON_MAX, description="Longitud de la estación")

    @model_validator(mode="after")
    def fin_de_semana_consistente(self):
        """is_weekend debe coincidir con dow (5 = sábado, 6 = domingo), como en el entrenamiento."""
        if self.is_weekend != int(self.dow >= 5):
            raise ValueError("is_weekend no coincide con dow (sábado = 5, domingo = 6)")
        return self


class LoteEntrada(BaseModel):
    """Varias estación-horas en una sola petición (hasta 1,000)."""

    model_config = ConfigDict(extra="forbid")
    registros: list[DemandaEntrada] = Field(..., min_length=1, max_length=1000)


class DemandaSalida(BaseModel):
    viajes_estimados: float = Field(..., ge=0, description="Viajes que se estima inicien en esa hora")


class LoteSalida(BaseModel):
    modelo: str
    version: str
    predicciones: list[DemandaSalida]


class Salud(BaseModel):
    estado: str
    modelo: str
    version: str
    uri: str
