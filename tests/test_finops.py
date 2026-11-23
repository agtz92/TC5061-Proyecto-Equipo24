"""Pruebas del análisis de recursos y FinOps (mibici.finops).

No requieren Docker ni Locust: usan series sintéticas con el mismo formato que los CSV reales.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mibici import finops


def historial(usuarios=(10, 25), seg=20, t0=1000):
    """Serie de Locust con escalones de `seg` segundos y una rampa en 0 usuarios al inicio."""
    filas = [{"Timestamp": t0, "User Count": 0}]
    t = t0 + 1
    for u in usuarios:
        for _ in range(seg):
            filas.append({"Timestamp": t, "User Count": u})
            t += 1
    return pd.DataFrame(filas)


def stats_docker(t0=1000, n=45, paso=1):
    filas = []
    for i in range(n):
        t = t0 + i * paso
        cpu = 20.0 if t <= t0 + 20 else 80.0
        filas.append({"timestamp": float(t), "contenedor": "mibici-api", "cpu_pct": cpu,
                      "mem_mib": 200.0 + i, "mem_limite_mib": 1024.0})
        filas.append({"timestamp": float(t), "contenedor": "mibici-mlflow", "cpu_pct": 1.0,
                      "mem_mib": 670.0, "mem_limite_mib": 17868.8})
    return pd.DataFrame(filas)


@pytest.mark.parametrize("texto, mib", [
    ("202.2MiB", 202.2), ("1GiB", 1024.0), ("512KiB", 0.5), ("17.45GiB", 17.45 * 1024),
    ("1048576B", 1.0), ("0B", 0.0),
])
def test_a_mib_convierte_unidades_de_docker(texto, mib):
    assert finops.a_mib(texto) == pytest.approx(mib)


def test_a_mib_rechaza_unidad_desconocida():
    with pytest.raises(ValueError):
        finops.a_mib("3XB")


@pytest.mark.parametrize("encabezado", [True, False])
def test_leer_docker_stats_con_y_sin_encabezado(tmp_path, encabezado):
    lineas = ["1795395265,mibici-api,0.19%,202.2MiB / 1GiB",
              "1795395265,mibici-mlflow,0.02%,668.7MiB / 17.45GiB"]
    if encabezado:
        lineas.insert(0, "timestamp,contenedor,cpu,memoria")
    ruta = tmp_path / "docker_stats.csv"
    ruta.write_text("\n".join(lineas) + "\n")
    df = finops.leer_docker_stats(ruta)
    assert list(df["contenedor"]) == ["mibici-api", "mibici-mlflow"]
    assert df["cpu_pct"].tolist() == pytest.approx([0.19, 0.02])
    assert df["mem_mib"].iloc[0] == pytest.approx(202.2)
    assert df["mem_limite_mib"].iloc[0] == pytest.approx(1024.0)


def test_ventanas_ignoran_rampa_y_primeros_segundos():
    v = finops.ventanas_etapas(historial())
    assert v["usuarios"].tolist() == [10, 25]
    assert v["t_ini"].iloc[0] == 1003  # primer segundo con 10 usuarios es 1001; se descartan 2
    assert v["t_fin"].iloc[0] == 1020


def test_recursos_por_etapa_calcula_cpu_por_peticion():
    etapas = pd.DataFrame({"usuarios": [10, 25], "rps": [10.0, 20.0], "p95_ms": [90.0, 600.0],
                           "error_pct": [0.0, 0.0]})
    r = finops.recursos_por_etapa(stats_docker(), historial(), etapas)
    assert r["cpu_prom_pct"].tolist() == [20.0, 80.0]
    # 20% de un núcleo / 10 pet/s = 20 ms por petición; 80% / 20 pet/s = 40 ms
    assert r["cpu_ms_por_peticion"].tolist() == [20.0, 40.0]


def test_predicciones_por_peticion_cuenta_lotes():
    stats = pd.DataFrame({"Name": ["GET /health", "POST /predict", "POST /predict/lote",
                                   "Aggregated"], "Request Count": [10, 60, 30, 100]})
    assert finops.predicciones_por_peticion(stats, lote=24) == pytest.approx((60 + 30 * 24) / 100)


def test_capacidad_con_slo_interpola():
    e = pd.DataFrame({"usuarios": [10, 25, 50], "rps": [10.0, 20.0, 25.0],
                      "p95_ms": [100.0, 900.0, 2000.0]})
    # 500 ms queda a la mitad entre 100 y 900 ms -> a la mitad entre 10 y 20 pet/s
    assert finops.capacidad_con_slo(e, 500) == pytest.approx(15.0)
    assert finops.capacidad_con_slo(e, 5000) == pytest.approx(25.0)  # todos cumplen
    assert pd.isna(finops.capacidad_con_slo(e, 50))  # ninguno cumple


def test_costo_tarea_y_por_millon():
    precio = finops.Precio("prueba", vcpu_hora=0.04, gb_hora=0.005)
    tarea = finops.Tarea("t", vcpu=1, gb=2)
    assert tarea.costo_hora(precio) == pytest.approx(0.05)
    # 0.05 USD/h atendiendo 10 pet/s = 36,000 pet/h -> 1.3889 USD por millón
    assert finops.costo_por_millon(0.05, 10) == pytest.approx(0.05 / 36_000 * 1e6)
    assert pd.isna(finops.costo_por_millon(0.05, 0))


def test_costo_por_trafico_agrega_tareas_y_baja_utilizacion():
    t = finops.costo_por_trafico(20.0, [1.0, 20.0, 50.0])
    assert t["tareas"].tolist() == [1, 1, 3]
    assert t["utilizacion_pct"].tolist() == [5.0, 100.0, pytest.approx(83.3)]
    # con poco tráfico la inferencia sale mucho más cara
    assert t["usd_por_millon_peticiones"].iloc[0] > 10 * t["usd_por_millon_peticiones"].iloc[1]


def test_generar_escribe_graficas_csv_y_secciones(tmp_path):
    hist = historial()
    hist["Name"] = "Aggregated"
    etapas = pd.DataFrame({"usuarios": [10, 25], "rps": [10.0, 20.0], "p95_ms": [90.0, 600.0],
                           "error_pct": [0.0, 0.0]})
    stats = pd.DataFrame({"Name": ["POST /predict", "POST /predict/lote", "Aggregated"],
                          "Request Count": [80, 20, 100]})
    ruta = tmp_path / "docker_stats.csv"
    d = stats_docker()
    pd.DataFrame({"timestamp": d["timestamp"].astype(int), "contenedor": d["contenedor"],
                  "cpu": d["cpu_pct"].map(lambda c: f"{c}%"),
                  "memoria": [f"{m}MiB / {lim}MiB" for m, lim in
                              zip(d["mem_mib"], d["mem_limite_mib"])]}).to_csv(ruta, index=False)
    md = "\n".join(finops.generar(ruta, stats, hist, etapas, tmp_path))
    for archivo in ["recursos_tiempo.png", "recursos_por_etapa.png", "recursos_por_etapa.csv",
                    "costos_inferencia.png", "costos_inferencia.csv", "costo_por_trafico.csv"]:
        assert (tmp_path / archivo).exists(), archivo
    assert "## Consumo de recursos durante la prueba" in md
    assert "## Análisis FinOps: costo por inferencia" in md
    assert "### Cómo optimizarlo" in md
