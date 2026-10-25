"""Ingesta reproducible de los datos abiertos de MiBici.

Un solo comando, sin pasos manuales:

    python -m mibici.dataset            # descarga, normaliza, limpia y versiona
    python -m mibici.dataset verify     # comprueba los CSV locales contra el manifiesto

Qué hace:
1. Lee la página del portal y obtiene el enlace de cada archivo mensual
   (cada mes vive en una carpeta distinta del servidor, así que no se adivina la URL).
2. Descarga los meses de START_MONTH a END_MONTH y la nomenclatura a data/raw/.
3. Normaliza: detecta la codificación (UTF-8 o Latin-1), asigna nombres de
   columna por posición (algunos encabezados llegan corruptos) y reconoce los
   dos formatos de fecha que usa el portal.
4. Limpia, elimina las columnas con datos personales y guarda un Parquet por mes
   en data/interim/trips/.
5. Escribe data_manifest.json (versionado en Git) con el hash SHA-256 y el número
   de registros de cada archivo; su huella corta es la "versión de los datos" que
   se registra en cada run de MLflow.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
from urllib.parse import urljoin

from loguru import logger
import pandas as pd
import requests
from tqdm import tqdm
import typer

from mibici.config import (
    DATA_MANIFEST,
    DATE_FORMATS,
    END_MONTH,
    HTTP_HEADERS,
    HTTP_TIMEOUT_S,
    MAX_TRIP_HOURS,
    MAX_UNPARSED_DATES_PCT,
    MIN_TRIP_SECONDS_SAME_STATION,
    NOMENCLATURA_FILE,
    PERSONAL_COLUMNS,
    PORTAL_URL,
    RAW_COLUMNS,
    RAW_DATA_DIR,
    START_MONTH,
    STATIONS_PARQUET,
    TRIPS_INTERIM_DIR,
)

app = typer.Typer(help="Ingesta de datos abiertos de MiBici.")

TRIPS_RE = re.compile(r"""href=["']([^"']*datos_abiertos_(\d{4})_(\d{2})\.csv)["']""")


# ---------------------------------------------------------------------------
# Descubrimiento y descarga
# ---------------------------------------------------------------------------
def month_range(start: str, end: str) -> list[str]:
    """Lista de meses 'AAAA-MM' entre start y end, inclusivo."""
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def list_portal_files(portal_url: str = PORTAL_URL) -> tuple[dict[str, str], str]:
    """Obtiene del portal la URL de cada CSV mensual y la de la nomenclatura.

    Returns:
        (urls_por_mes, url_nomenclatura). Las llaves de urls_por_mes son 'AAAA-MM'.
    """
    html = requests.get(portal_url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT_S).text
    months = {f"{y}-{m}": urljoin(portal_url, href) for href, y, m in TRIPS_RE.findall(html)}
    nom = re.search(rf"""href=["']([^"']*{re.escape(NOMENCLATURA_FILE)})["']""", html)
    if not nom:
        raise RuntimeError(f"No se encontró {NOMENCLATURA_FILE} en {portal_url}")
    return months, urljoin(portal_url, nom.group(1))


def sha256_file(path: Path) -> str:
    """Hash SHA-256 de un archivo, leído en bloques."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, force: bool = False) -> Path:
    """Descarga url a dest (escritura atómica). No vuelve a bajar si ya existe."""
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT_S, stream=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(dest)
    return dest


# ---------------------------------------------------------------------------
# Normalización
# ---------------------------------------------------------------------------
def decode_bytes(raw: bytes) -> tuple[str, str]:
    """Decodifica un CSV probando UTF-8 y, si falla, Latin-1 (cp1252).

    En 2022-2026 conviven ambos: 36 meses en Latin-1 y 21 en UTF-8.
    """
    try:
        return raw.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace"), "latin-1"


def parse_dates(s: pd.Series) -> tuple[pd.Series, str]:
    """Convierte texto a datetime probando cada formato de DATE_FORMATS.

    El portal usa 'AAAA-MM-DD hh:mm:ss' en la mayoría de los meses y
    'DD/MM/AAAA h:mm' en 2023-11, 2024-09 y 2024-10. Se prueba un formato a la vez
    sobre los valores que aún no se reconocen, para no confundir día y mes.
    """
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    used = []
    for fmt in DATE_FORMATS:
        missing = out.isna() & s.notna()
        if not missing.any():
            break
        parsed = pd.to_datetime(s[missing], format=fmt, errors="coerce")
        if parsed.notna().any():
            used.append(fmt)
        out[missing] = parsed
    return out, " | ".join(used)


def read_trips_csv(path: Path) -> tuple[pd.DataFrame, dict]:
    """Lee un CSV mensual de viajes y devuelve un DataFrame con nombres normalizados.

    Las columnas se asignan por posición (RAW_COLUMNS) porque el encabezado de
    'Año_de_nacimiento' llega con caracteres corruptos en algunos meses.
    """
    text, encoding = decode_bytes(path.read_bytes())
    df = pd.read_csv(io.StringIO(text), dtype=str, header=0, skipinitialspace=True)
    if df.shape[1] != len(RAW_COLUMNS):
        raise ValueError(f"{path.name}: se esperaban {len(RAW_COLUMNS)} columnas, hay {df.shape[1]}")
    df.columns = RAW_COLUMNS
    df["inicio"], fmt_ini = parse_dates(df["inicio"])
    df["fin"], _ = parse_dates(df["fin"])
    meta = {"encoding": encoding, "date_format": fmt_ini, "rows_raw": int(len(df))}
    return df, meta


def clean_trips(df: pd.DataFrame, valid_stations: set[int]) -> tuple[pd.DataFrame, dict]:
    """Aplica las reglas de limpieza y elimina los datos personales.

    Reglas: sin duplicados de viaje_id; fechas reconocidas; duración entre 0 y
    MAX_TRIP_HOURS; se quitan viajes de menos de 1 minuto con mismo origen y
    destino (anclajes fallidos) y viajes con estaciones fuera de la nomenclatura.
    """
    stats = {}
    n0 = len(df)
    bad_dates = df["inicio"].isna() | df["fin"].isna()
    pct_bad = 100 * bad_dates.mean() if n0 else 0.0
    if pct_bad > MAX_UNPARSED_DATES_PCT:
        raise ValueError(f"{pct_bad:.2f}% de fechas no reconocidas; revisar DATE_FORMATS")
    df = df.loc[~bad_dates].drop(columns=PERSONAL_COLUMNS)

    for c in ["viaje_id", "origen_id", "destino_id"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df = df.dropna(subset=["viaje_id", "origen_id", "destino_id"])
    df = df.drop_duplicates("viaje_id")

    dur = (df["fin"] - df["inicio"]).dt.total_seconds()
    ok_dur = (dur > 0) & (dur <= MAX_TRIP_HOURS * 3600)
    failed_dock = (df["origen_id"] == df["destino_id"]) & (dur < MIN_TRIP_SECONDS_SAME_STATION)
    ok_st = df["origen_id"].isin(valid_stations) & df["destino_id"].isin(valid_stations)
    stats.update(
        dropped_bad_dates=int(bad_dates.sum()),
        dropped_duration=int((~ok_dur).sum()),
        dropped_failed_dock=int((failed_dock & ok_dur).sum()),
        dropped_unknown_station=int((~ok_st & ok_dur & ~failed_dock).sum()),
    )
    df = df.loc[ok_dur & ~failed_dock & ok_st].copy()
    df["duracion_s"] = dur.loc[df.index].astype("int32")
    df = df.astype({"viaje_id": "int64", "origen_id": "int32", "destino_id": "int32"})
    stats["rows_clean"] = int(len(df))
    stats["dropped_total"] = int(n0 - len(df))
    return df.reset_index(drop=True), stats


def read_stations(path: Path) -> pd.DataFrame:
    """Lee la nomenclatura y la normaliza (id, nombre, lat, lon, capacidad, municipio)."""
    text, _ = decode_bytes(path.read_bytes())
    st = pd.read_csv(io.StringIO(text))
    st.columns = ["station_id", "name", "lat", "lon", "capacity"]
    st["municipio"] = st["name"].str.extract(r"^\((\w+)-")[0].fillna("NA")
    return st.astype({"station_id": "int32", "capacity": "int16"})


# ---------------------------------------------------------------------------
# Manifiesto (versión de los datos)
# ---------------------------------------------------------------------------
def manifest_version(files: list[dict], nomenclatura: dict) -> str:
    """Huella corta y determinista del contenido de los datos de origen."""
    key = json.dumps(
        {"files": [(f["file"], f["sha256"]) for f in files],
         "nomenclatura": (nomenclatura["file"], nomenclatura["sha256"])},
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def load_data_version(path: Path = DATA_MANIFEST) -> str:
    """Versión de los datos registrada en el manifiesto (para etiquetar runs)."""
    if not path.exists():
        return "sin-manifiesto"
    return json.loads(path.read_text())["data_version"]


# ---------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------
@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    start: str = START_MONTH,
    end: str = END_MONTH,
    source_dir: Path = typer.Option(None, help="Usar CSV ya descargados en esta carpeta"),
    force: bool = typer.Option(False, help="Volver a descargar aunque ya existan"),
):
    """Descarga, normaliza, limpia y versiona los datos de MiBici."""
    if ctx.invoked_subcommand is not None:
        return
    # MLproject pasa "config" para usar el rango definido en config.py
    start = START_MONTH if start.strip().lower() in ("", "config") else start
    end = END_MONTH if end.strip().lower() in ("", "config") else end
    months = month_range(start, end)
    raw_dir = source_dir or RAW_DATA_DIR

    if source_dir is None:
        logger.info(f"Leyendo enlaces del portal {PORTAL_URL}")
        urls, nom_url = list_portal_files()
        missing = [m for m in months if m not in urls]
        if missing:
            raise RuntimeError(f"Meses sin archivo en el portal: {missing}")
        download(nom_url, raw_dir / NOMENCLATURA_FILE, force)
        for m in tqdm(months, desc="Descargando"):
            download(urls[m], raw_dir / f"datos_abiertos_{m.replace('-', '_')}.csv", force)
    else:
        urls, nom_url = {}, None

    nom_path = raw_dir / NOMENCLATURA_FILE
    stations = read_stations(nom_path)
    STATIONS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    stations.to_parquet(STATIONS_PARQUET, index=False)
    valid = set(stations["station_id"].tolist())
    nomenclatura = {"file": NOMENCLATURA_FILE, "url": nom_url, "sha256": sha256_file(nom_path),
                    "stations": int(len(stations)), "capacity_total": int(stations["capacity"].sum())}

    TRIPS_INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for m in tqdm(months, desc="Normalizando"):
        path = raw_dir / f"datos_abiertos_{m.replace('-', '_')}.csv"
        df, meta = read_trips_csv(path)
        df, stats = clean_trips(df, valid)
        df.to_parquet(TRIPS_INTERIM_DIR / f"trips_{m.replace('-', '_')}.parquet", index=False)
        files.append({"month": m, "file": path.name, "url": urls.get(m),
                      "sha256": sha256_file(path), "bytes": path.stat().st_size, **meta, **stats})
        logger.info(f"{m}: {meta['rows_raw']:,} filas, {stats['rows_clean']:,} limpias, {meta['encoding']}")

    manifest = {
        "source": PORTAL_URL,
        "start_month": start,
        "end_month": end,
        "data_version": manifest_version(files, nomenclatura),
        "totals": {"months": len(files), "rows_raw": sum(f["rows_raw"] for f in files),
                   "rows_clean": sum(f["rows_clean"] for f in files)},
        "nomenclatura": nomenclatura,
        "files": files,
    }
    DATA_MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    logger.success(f"Manifiesto escrito: data_version={manifest['data_version']} "
                   f"({manifest['totals']['rows_clean']:,} viajes limpios)")


@app.command()
def verify(raw_dir: Path = RAW_DATA_DIR):
    """Comprueba que los CSV locales coinciden con los hashes del manifiesto."""
    manifest = json.loads(DATA_MANIFEST.read_text())
    bad = [f["file"] for f in manifest["files"]
           if not (raw_dir / f["file"]).exists() or sha256_file(raw_dir / f["file"]) != f["sha256"]]
    if bad:
        logger.error(f"{len(bad)} archivos no coinciden con el manifiesto: {bad[:5]}")
        raise typer.Exit(1)
    logger.success(f"Datos verificados: data_version={manifest['data_version']}")


if __name__ == "__main__":
    app()
