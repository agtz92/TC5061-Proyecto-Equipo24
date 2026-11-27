"""Genera las peticiones de la demo a partir de datos reales del panel (periodo de prueba, 2026).

Toma un día que el modelo no vio en el entrenamiento y arma, con las mismas entradas que usa la API:
  - 01_hora_pico.json        una estación concurrida en su hora pico de la mañana
  - 02_madrugada.json        la misma estación de madrugada
  - 03_estacion_nueva.json   una estación abierta en 2026, en su hora pico
  - 04_invalido.json         una petición inválida (hour = 25), que la API debe rechazar con 422
  - 05_lote_24h.json         las 24 horas de la estación concurrida, para /predict/lote
  - real_<fecha>.csv         los viajes reales de esas estación-horas, para comparar

Uso (desde la raíz del repo, con data/processed e data/interim generados):
    python demo/generar_ejemplos.py
    python demo/generar_ejemplos.py --fecha 2026-09-23 --estacion 51 --nueva 428
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from mibici.config import P1_FEATURES, STATIONS_PARQUET
from mibici.features import load_supervised_frame

SALIDA = Path(__file__).parent / "ejemplos"
ENTEROS = {"hour", "dow", "month", "is_weekend", "is_holiday", "is_quincena", "capacity"}


def registro(fila) -> dict:
    """Una fila del panel en el formato de DemandaEntrada de la API."""
    out = {}
    for c in P1_FEATURES:
        v = float(fila[c])
        out[c] = int(round(v)) if c in ENTEROS else round(v, 4)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--fecha", default="2026-09-23", help="Día del periodo de prueba (no visto al entrenar)")
    p.add_argument("--estacion", type=int, default=51, help="Estación concurrida (la de más viajes en sep-2026)")
    p.add_argument("--nueva", type=int, default=428, help="Estación abierta en 2026")
    args = p.parse_args()

    cols = ["station_id", "departures", *P1_FEATURES]
    df = load_supervised_frame(f"{args.fecha} 00:00", f"{args.fecha} 23:00", cols)
    nombres = duckdb.sql(f"SELECT station_id, name FROM read_parquet('{STATIONS_PARQUET}')").df()  # nosec B608
    nombre = dict(zip(nombres.station_id, nombres.name))

    est = df[df.station_id == args.estacion].sort_values("hour")
    nueva = df[df.station_id == args.nueva].sort_values("hour")
    if len(est) != 24 or nueva.empty:
        raise SystemExit(f"Faltan horas para la estación {args.estacion} o {args.nueva} el {args.fecha}")

    pico = est.loc[est[est.hour.between(6, 10)].departures.idxmax()]
    madrugada = est[est.hour == 3].iloc[0]
    pico_nueva = nueva.loc[nueva.departures.idxmax()]
    invalido = {**registro(pico), "hour": 25}

    SALIDA.mkdir(exist_ok=True)
    archivos = {
        "01_hora_pico.json": registro(pico),
        "02_madrugada.json": registro(madrugada),
        "03_estacion_nueva.json": registro(pico_nueva),
        "04_invalido.json": invalido,
        "05_lote_24h.json": {"registros": [registro(f) for _, f in est.iterrows()]},
    }
    for nombre_archivo, contenido in archivos.items():
        (SALIDA / nombre_archivo).write_text(json.dumps(contenido, indent=2, ensure_ascii=False), encoding="utf-8")

    filas = [("01_hora_pico", pico), ("02_madrugada", madrugada), ("03_estacion_nueva", pico_nueva)]
    filas += [(f"05_lote_24h_h{int(f.hour):02d}", f) for _, f in est.iterrows()]
    lineas = ["ejemplo,fecha,station_id,estacion,hour,viajes_reales"]
    for clave, f in filas:
        sid = int(f.station_id)
        lineas.append(f'{clave},{args.fecha},{sid},"{nombre.get(sid, "")}",{int(f.hour)},{int(f.departures)}')
    (SALIDA / f"real_{args.fecha}.csv").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"Ejemplos escritos en {SALIDA} ({args.fecha}; estación {args.estacion}: {nombre.get(args.estacion)};"
          f" nueva {args.nueva}: {nombre.get(args.nueva)})")


if __name__ == "__main__":
    main()
