"""Feature engineering para las tres preguntas del proyecto.

    python -m mibici.features            # construye panel estación × hora y perfiles

Regla general: toda variable usada para predecir la hora h del día D solo usa
información disponible a las 23:59 del día D-1 (la predicción se hace a las
00:00 para el día siguiente). Por eso los rezagos son de 24 h o más y los
promedios móviles terminan en el día anterior.

Las agregaciones pesadas se hacen con DuckDB directamente sobre los Parquet
mensuales (≈21 millones de viajes) para no cargarlos completos en memoria.
"""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici.config import (
    END_MONTH,
    P2_CENSOR_MIN_EXPECTED,
    P2_CENSOR_WEEKS,
    P2_NET_FLOW_THRESHOLD,
    PANEL_PARQUET,
    PROFILES_PARQUET,
    START_MONTH,
    STATIONS_PARQUET,
    TRAIN_END,
    TRIPS_INTERIM_DIR,
)

app = typer.Typer(help="Construcción de features de MiBici.")

LABELS = ["normal", "vaciado", "saturacion"]


# ---------------------------------------------------------------------------
# Calendario
# ---------------------------------------------------------------------------
def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def mexico_holidays(years: range) -> set[date]:
    """Días de descanso obligatorio en México (Ley Federal del Trabajo, art. 74).

    Pertinencia (P1, P2): en festivos desaparece el patrón de traslado laboral y
    la demanda por hora cambia de forma; sin esta variable el modelo los trata
    como un lunes o un jueves normales.
    """
    out = set()
    for y in years:
        out |= {date(y, 1, 1), date(y, 5, 1), date(y, 9, 16), date(y, 12, 25),
                _nth_weekday(y, 2, 0, 1),   # primer lunes de febrero
                _nth_weekday(y, 3, 0, 3),   # tercer lunes de marzo
                _nth_weekday(y, 11, 0, 3)}  # tercer lunes de noviembre
        if (y - 2024) % 6 == 0:
            out.add(date(y, 10, 1))         # transmisión del Poder Ejecutivo
    return out


def add_calendar_features(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    """Agrega hora, día de la semana, mes, fin de semana, festivo y quincena.

    Pertinencia (P1, P2): la demanda de bicicleta compartida sigue los ciclos de
    traslado (picos de mañana y tarde en días hábiles); quincena captura los días
    de pago (15 y último del mes). Todas se conocen de antemano: no hay fuga.
    """
    ts = df[ts_col]
    hol = mexico_holidays(range(ts.dt.year.min(), ts.dt.year.max() + 1))
    out = df.copy()
    out["hour"] = ts.dt.hour.astype("int8")
    out["dow"] = ts.dt.dayofweek.astype("int8")
    out["month"] = ts.dt.month.astype("int8")
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")
    out["is_holiday"] = ts.dt.date.isin(hol).astype("int8")
    out["is_quincena"] = ((ts.dt.day == 15) | ts.dt.is_month_end).astype("int8")
    return out


# ---------------------------------------------------------------------------
# Panel estación × hora (P1 y P2)
# ---------------------------------------------------------------------------
def build_station_hour_panel(con: duckdb.DuckDBPyConnection) -> None:
    """Crea la tabla `panel` con salidas, llegadas y flujo neto por estación y hora.

    - Se rellena con ceros cada hora en que la estación estaba activa (entre su
      primer y último viaje), para distinguir "sin demanda" de "sin dato" (P1).
    - flujo_neto = llegadas - salidas; normalizado por capacidad define el riesgo (P2).
    """
    trips = str(TRIPS_INTERIM_DIR / "trips_*.parquet")
    con.execute(f"""
        CREATE OR REPLACE TABLE dep AS
        SELECT origen_id AS station_id, date_trunc('hour', inicio) AS ts, count(*)::INT AS departures
        FROM read_parquet('{trips}') GROUP BY 1, 2;
        CREATE OR REPLACE TABLE arr AS
        SELECT destino_id AS station_id, date_trunc('hour', fin) AS ts, count(*)::INT AS arrivals
        FROM read_parquet('{trips}') GROUP BY 1, 2;
        CREATE OR REPLACE TABLE active AS
        SELECT station_id, min(ts) AS first_ts, max(ts) AS last_ts
        FROM (SELECT station_id, ts FROM dep UNION ALL SELECT station_id, ts FROM arr) GROUP BY 1;
        CREATE OR REPLACE TABLE hours AS
        SELECT unnest(generate_series(TIMESTAMP '{START_MONTH}-01',
               (DATE '{END_MONTH}-01' + INTERVAL 1 MONTH - INTERVAL 1 HOUR)::TIMESTAMP,
               INTERVAL 1 HOUR)) AS ts;
        CREATE OR REPLACE TABLE panel AS
        SELECT a.station_id, h.ts,
               coalesce(d.departures, 0) AS departures,
               coalesce(r.arrivals, 0) AS arrivals,
               coalesce(r.arrivals, 0) - coalesce(d.departures, 0) AS net_flow,
               s.capacity
        FROM active a
        JOIN hours h ON h.ts BETWEEN a.first_ts AND a.last_ts
        JOIN read_parquet('{STATIONS_PARQUET}') s ON s.station_id = a.station_id
        LEFT JOIN dep d ON d.station_id = a.station_id AND d.ts = h.ts
        LEFT JOIN arr r ON r.station_id = a.station_id AND r.ts = h.ts;
    """)


def add_lag_and_rolling_features(con: duckdb.DuckDBPyConnection) -> None:
    """Agrega rezagos y promedios móviles sin fuga de información.

    - lag_24h, lag_168h: misma estación y hora hace 1 día y 1 semana. Capturan la
      inercia y la estacionalidad semanal; lag_168h es además el baseline de P1.
    - dep_mean_7d, dep_mean_28d: promedio de salidas a esa misma hora en los 7 y
      28 días previos (termina en D-1). Separa el nivel reciente del ruido horario.
    - netrel_mean_28d: flujo neto / capacidad promedio a esa hora en 28 días (P2).
    - exp_dep_8w, exp_arr_8w: salidas y llegadas esperadas para esa estación, día
      de la semana y hora en las 8 semanas previas; base de la huella de censura.
    """
    con.execute(f"""
        CREATE OR REPLACE TABLE features AS
        SELECT *,
          lag(departures, 24)  OVER w_st AS dep_lag_24h,
          lag(departures, 168) OVER w_st AS dep_lag_168h,
          lag(arrivals, 24)    OVER w_st AS arr_lag_24h,
          lag(arrivals, 168)   OVER w_st AS arr_lag_168h,
          avg(departures) OVER (w_sh ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING)  AS dep_mean_7d,
          avg(departures) OVER (w_sh ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING) AS dep_mean_28d,
          avg(net_flow::DOUBLE / nullif(capacity, 0))
              OVER (w_sh ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING) AS netrel_mean_28d,
          avg(departures) OVER (w_sdh ROWS BETWEEN {P2_CENSOR_WEEKS} PRECEDING AND 1 PRECEDING) AS exp_dep_8w,
          avg(arrivals)   OVER (w_sdh ROWS BETWEEN {P2_CENSOR_WEEKS} PRECEDING AND 1 PRECEDING) AS exp_arr_8w
        FROM panel
        WINDOW w_st  AS (PARTITION BY station_id ORDER BY ts),
               w_sh  AS (PARTITION BY station_id, hour(ts) ORDER BY ts),
               w_sdh AS (PARTITION BY station_id, dayofweek(ts), hour(ts) ORDER BY ts);
    """)


def add_p2_labels(con: duckdb.DuckDBPyConnection) -> None:
    """Etiqueta proxy de riesgo para P2 (normal / vaciado / saturacion).

    Los datos no traen inventario de bicicletas, así que el riesgo se infiere:
    - vaciado: flujo neto / capacidad <= -umbral, o huella de censura de salidas
      (se esperaban >= P2_CENSOR_MIN_EXPECTED salidas y hubo 0: probablemente no
      había bicicletas).
    - saturacion: flujo neto / capacidad >= +umbral, o huella de censura de llegadas.
    Las expectativas solo usan semanas anteriores, así que la etiqueta no mira al futuro.
    """
    t, m = P2_NET_FLOW_THRESHOLD, P2_CENSOR_MIN_EXPECTED
    con.execute(f"""
        CREATE OR REPLACE TABLE features AS
        SELECT *,
          CASE
            WHEN (capacity > 0 AND net_flow::DOUBLE / capacity <= -{t})
                 OR (exp_dep_8w >= {m} AND departures = 0 AND net_flow <= 0) THEN 'vaciado'
            WHEN (capacity > 0 AND net_flow::DOUBLE / capacity >= {t})
                 OR (exp_arr_8w >= {m} AND arrivals = 0 AND net_flow >= 0) THEN 'saturacion'
            ELSE 'normal'
          END AS p2_label,
          (exp_dep_8w >= {m} AND departures = 0)::INT AS censor_empty,
          (exp_arr_8w >= {m} AND arrivals = 0)::INT AS censor_full
        FROM features;
    """)


# ---------------------------------------------------------------------------
# Perfiles por estación (P3)
# ---------------------------------------------------------------------------
def build_station_profiles(con: duckdb.DuckDBPyConnection, until: str = TRAIN_END) -> pd.DataFrame:
    """Vector de uso por estación para clustering, calculado solo con entrenamiento.

    96 proporciones (salidas y llegadas por hora, en día hábil y en fin de semana;
    cada bloque de 24 suma 1) más razón llegadas/salidas, duración mediana y % de
    viajes circulares. Distingue estaciones residenciales (salidas de mañana),
    laborales (llegadas de mañana), mixtas y recreativas (fin de semana, circulares).
    """
    trips = str(TRIPS_INTERIM_DIR / "trips_*.parquet")
    long = con.execute(f"""
        WITH t AS (SELECT * FROM read_parquet('{trips}') WHERE inicio <= TIMESTAMP '{until}'),
        ev AS (
          SELECT origen_id AS station_id, 'dep' AS kind, hour(inicio) AS h,
                 CASE WHEN dayofweek(inicio) IN (0, 6) THEN 'we' ELSE 'wd' END AS dt FROM t
          UNION ALL
          SELECT destino_id, 'arr', hour(fin),
                 CASE WHEN dayofweek(fin) IN (0, 6) THEN 'we' ELSE 'wd' END FROM t)
        SELECT station_id, kind || '_' || dt || '_h' || lpad(h::VARCHAR, 2, '0') AS col, count(*) AS n
        FROM ev GROUP BY ALL
    """).df()
    wide = long.pivot_table(index="station_id", columns="col", values="n", fill_value=0)
    for block in ["dep_wd", "dep_we", "arr_wd", "arr_we"]:
        cols = [f"{block}_h{h:02d}" for h in range(24)]
        for c in cols:
            if c not in wide:
                wide[c] = 0
        tot = wide[cols].sum(axis=1).replace(0, np.nan)
        wide[cols] = wide[cols].div(tot, axis=0).fillna(0)
    extra = con.execute(f"""
        WITH t AS (SELECT * FROM read_parquet('{trips}') WHERE inicio <= TIMESTAMP '{until}')
        SELECT s.station_id,
               coalesce(a.n, 0)::DOUBLE / nullif(d.n, 0) AS ratio_arr_dep,
               d.dur_median, d.pct_circular, d.n AS trips_out
        FROM (SELECT DISTINCT origen_id AS station_id FROM t) s
        JOIN (SELECT origen_id AS station_id, count(*) AS n, median(duracion_s) AS dur_median,
                     avg((origen_id = destino_id)::INT) AS pct_circular
              FROM t GROUP BY 1) d USING (station_id)
        LEFT JOIN (SELECT destino_id AS station_id, count(*) AS n FROM t GROUP BY 1) a USING (station_id)
    """).df().set_index("station_id")
    prof = wide[sorted(wide.columns)].join(extra, how="inner")
    st = pd.read_parquet(STATIONS_PARQUET).set_index("station_id")[["lat", "lon", "capacity", "municipio"]]
    return prof.join(st, how="inner").reset_index()


# ---------------------------------------------------------------------------
# Lectura de conjuntos para modelado
# ---------------------------------------------------------------------------
def load_panel(start: str, end: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Lee del panel procesado solo el periodo y las columnas necesarias."""
    cols = ", ".join(columns) if columns else "*"
    q = f"""SELECT {cols} FROM read_parquet('{PANEL_PARQUET}')
            WHERE ts BETWEEN TIMESTAMP '{start}' AND TIMESTAMP '{end}'"""
    return duckdb.sql(q).df()


@app.callback(invoke_without_command=True)
def main():
    """Construye data/processed/panel_station_hour.parquet y station_profiles.parquet."""
    PANEL_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    logger.info("Agregando viajes por estación y hora...")
    build_station_hour_panel(con)
    logger.info("Calculando rezagos, promedios móviles y etiquetas de P2...")
    add_lag_and_rolling_features(con)
    add_p2_labels(con)
    con.execute(f"COPY (SELECT * FROM features ORDER BY station_id, ts) TO '{PANEL_PARQUET}' (FORMAT parquet)")
    n, n_st = con.execute("SELECT count(*), count(DISTINCT station_id) FROM features").fetchone()
    logger.info(f"Panel: {n:,} filas estación-hora, {n_st} estaciones")
    prof = build_station_profiles(con)
    prof.to_parquet(PROFILES_PARQUET, index=False)
    logger.success(f"Perfiles P3: {len(prof)} estaciones -> {PROFILES_PARQUET.name}")


if __name__ == "__main__":
    app()
