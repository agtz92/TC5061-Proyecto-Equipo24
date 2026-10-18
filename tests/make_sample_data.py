"""Genera CSV sintéticos con los mismos formatos que publica MiBici.

Sirve para probar la ingesta y el experimento base sin descargar 1.8 GB:

    python tests/make_sample_data.py data/sample
    python -m mibici.dataset --source-dir data/sample --start 2025-01 --end 2026-02

Reproduce las variantes reales observadas en 2022-2026:
- Latin-1 con comillas y fechas ISO (la mayoría de 2022-2024)
- UTF-8 con comillas y fechas ISO (la mayoría de 2025-2026)
- encabezado 'Año_de_nacimiento' corrupto (2024-02, 2024-05)
- sin comillas, fin de línea CRLF y fechas 'DD/MM/AAAA h:mm' (2023-11, 2024-09, 2024-10)
Los viajes son aleatorios: no contienen datos de ninguna persona real.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HEADER = ["Viaje_Id", "Usuario_Id", "Genero", "Año_de_nacimiento",
          "Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"]


def make_month(month: str, n_st: int, rng: np.random.Generator, start_id: int) -> pd.DataFrame:
    start = pd.Timestamp(month + "-01")
    end = start + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23)
    hours = pd.date_range(start, end, freq="h")
    # Perfil horario con picos de mañana y tarde en días hábiles
    base = np.where(hours.dayofweek < 5, 1.0, 0.5) * (
        0.3 + np.exp(-((hours.hour - 8) ** 2) / 4) + 0.8 * np.exp(-((hours.hour - 18) ** 2) / 5))
    rows = []
    tid = start_id
    st_kind = rng.integers(0, 3, n_st)  # 0 residencial, 1 laboral, 2 mixta
    for s in range(n_st):
        morning = np.exp(-((hours.hour - 8) ** 2) / 4)
        tilt = {0: 1 + morning, 1: 1 - 0.6 * morning, 2: np.ones(len(hours))}[int(st_kind[s])]
        lam = base * tilt * rng.uniform(0.3, 2.0)
        counts = rng.poisson(lam)
        for h, c in zip(hours[counts > 0], counts[counts > 0]):
            for _ in range(c):
                ini = h + pd.Timedelta(seconds=int(rng.integers(0, 3600)))
                dur = pd.Timedelta(seconds=int(rng.gamma(3, 250)) + 30)
                dest = int(rng.integers(0, n_st)) + 2
                rows.append((tid, int(rng.integers(1, 99999)), rng.choice(["M", "F"]),
                             int(rng.integers(1960, 2006)), ini, ini + dur, s + 2, dest))
                tid += 1
    return pd.DataFrame(rows, columns=HEADER)


def write(df: pd.DataFrame, path: Path, variant: str) -> None:
    df = df.sort_values("Inicio_del_viaje")
    if variant == "ddmm":
        f = lambda t: f"{t.day:02d}/{t.month:02d}/{t.year} {t.hour}:{t.minute:02d}"
        df = df.assign(Inicio_del_viaje=df["Inicio_del_viaje"].map(f), Fin_del_viaje=df["Fin_del_viaje"].map(f))
        text = df.to_csv(index=False, lineterminator="\r\n")
        path.write_bytes(text.encode("cp1252"))
        return
    df = df.assign(Inicio_del_viaje=df["Inicio_del_viaje"].dt.strftime("%Y-%m-%d %H:%M:%S"),
                   Fin_del_viaje=df["Fin_del_viaje"].dt.strftime("%Y-%m-%d %H:%M:%S"))
    text = df.to_csv(index=False, quoting=1)
    if variant == "corrupt":
        text = text.replace("Año_de_nacimiento", "A}\xe4e_nacimiento", 1)
    path.write_bytes(text.encode("utf-8" if variant == "utf8" else "cp1252"))


def main(out: str = "data/sample", start: str = "2025-01", end: str = "2026-02", n_st: int = 25) -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    variants = ["latin1", "utf8", "corrupt", "ddmm"]
    tid = 1
    for i, p in enumerate(pd.period_range(start, end, freq="M")):
        m = p.strftime("%Y-%m")
        df = make_month(m, n_st, rng, tid)
        tid += len(df) + 10
        write(df, out_dir / f"datos_abiertos_{m.replace('-', '_')}.csv", variants[i % 4])
    st = pd.DataFrame({
        "id": range(2, n_st + 2),
        "name": [f"({'GDL' if i % 3 else 'ZPN'}-{i:03d}) Estación sintética {i}" for i in range(n_st)],
        "latitude": 20.67 + rng.normal(0, 0.02, n_st),
        "longitude": -103.36 + rng.normal(0, 0.02, n_st),
        "dpcapacity": rng.integers(12, 30, n_st),
    })
    (out_dir / "nomenclatura_2026_09.csv").write_text(st.to_csv(index=False, quoting=1), encoding="utf-8")
    print(f"Datos sintéticos en {out_dir}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
