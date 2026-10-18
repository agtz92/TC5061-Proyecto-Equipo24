"""Pruebas de la normalización de la ingesta con los formatos reales del portal."""

import pandas as pd

from mibici.dataset import clean_trips, decode_bytes, parse_dates


def test_decode_latin1_and_utf8():
    assert decode_bytes("Año".encode("utf-8")) == ("Año", "utf-8")
    text, enc = decode_bytes("Año".encode("cp1252"))
    assert enc == "latin-1" and text == "Año"


def test_parse_both_date_formats():
    s = pd.Series(["2024-02-01 00:01:46", "01/11/2023 0:01", "30/11/2023 9:59"])
    out, fmts = parse_dates(s)
    assert out.notna().all()
    assert out.iloc[1] == pd.Timestamp("2023-11-01 00:01")  # día primero, no mes
    assert out.iloc[2] == pd.Timestamp("2023-11-30 09:59")


def test_clean_drops_personal_columns_and_failed_docks():
    df = pd.DataFrame({
        "viaje_id": ["1", "2", "3", "3"],
        "usuario_id": ["9", "9", "9", "9"], "genero": ["M"] * 4, "anio_nacimiento": ["1990"] * 4,
        "inicio": pd.to_datetime(["2025-01-01 08:00"] * 4),
        "fin": pd.to_datetime(["2025-01-01 08:10:00", "2025-01-01 08:00:30", "2025-01-01 08:20:00", "2025-01-01 08:20:00"]),
        "origen_id": ["2", "2", "2", "2"], "destino_id": ["3", "2", "3", "3"],
    })
    out, stats = clean_trips(df, valid_stations={2, 3})
    assert not {"usuario_id", "genero", "anio_nacimiento"} & set(out.columns)
    assert len(out) == 2                      # duplicado y anclaje fallido fuera
    assert stats["dropped_failed_dock"] == 1
