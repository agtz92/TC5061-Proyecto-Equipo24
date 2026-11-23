"""Genera el reporte de la prueba de carga a partir de los CSV de Locust.

Lee los archivos que Locust escribe con `--csv <prefijo> --csv-full-history`:
    <prefijo>_stats.csv          totales por endpoint
    <prefijo>_stats_history.csv  serie de tiempo (una fila por segundo)
    <prefijo>_failures.csv       fallas agrupadas por causa

y escribe en la carpeta de salida:
    latencia_tiempo.png         latencia p50 / p95 / p99 en el tiempo
    rps_tiempo.png              peticiones por segundo en el tiempo
    errores_tiempo.png          tasa de error (%) en el tiempo
    latencia_por_endpoint.png   p50 / p95 / p99 por endpoint
    capacidad_por_etapa.png     throughput y p95 por número de usuarios
    resumen_etapas.csv          métricas por escalón (insumo del análisis FinOps)
    reporte_carga.md            tablas, gráficas y observaciones

Uso:
    python loadtest/generar_reporte.py                       # reports/carga/locust_*.csv
    python loadtest/generar_reporte.py --prefijo ruta/locust --salida ruta/
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Paleta categórica validada (slots 1-3) y tintas neutras
SERIE = ["#2a78d6", "#eb6834", "#1baf7a"]
TINTA, TINTA_2, REJILLA, FONDO = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
P95_OBJETIVO_MS = 500  # latencia p95 aceptable para una consulta interactiva

plt.rcParams.update({
    "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
    "axes.edgecolor": REJILLA, "axes.labelcolor": TINTA_2, "axes.titlecolor": TINTA,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": TINTA_2, "ytick.color": TINTA_2, "text.color": TINTA,
    "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "font.size": 10, "lines.linewidth": 2,
})


def leer(prefijo: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stats = pd.read_csv(f"{prefijo}_stats.csv")
    hist = pd.read_csv(f"{prefijo}_stats_history.csv")
    fallas_csv = Path(f"{prefijo}_failures.csv")
    fallas = pd.read_csv(fallas_csv) if fallas_csv.exists() else pd.DataFrame()

    hist = hist[hist["Name"] == "Aggregated"].copy()
    for c in hist.columns.drop(["Type", "Name"]):
        hist[c] = pd.to_numeric(hist[c], errors="coerce")
    hist["t"] = hist["Timestamp"] - hist["Timestamp"].min()
    hist["error_pct"] = (100 * hist["Failures/s"] / hist["Requests/s"]).where(
        hist["Requests/s"] > 0)
    return stats, hist, fallas


def etapas(hist: pd.DataFrame) -> pd.DataFrame:
    """Métricas por escalón: tramos consecutivos con el mismo número de usuarios."""
    h = hist.dropna(subset=["User Count"]).copy()
    h["tramo"] = (h["User Count"] != h["User Count"].shift()).cumsum()
    filas = []
    for _, g in h.groupby("tramo"):
        if len(g) < 5 or g["User Count"].iloc[0] == 0:
            continue  # rampas de subida o bajada
        g = g.iloc[2:]  # descarta los primeros segundos del escalón (todavía en transición)
        dt = g["Timestamp"].iloc[-1] - g["Timestamp"].iloc[0]
        dreq = g["Total Request Count"].iloc[-1] - g["Total Request Count"].iloc[0]
        dfail = g["Total Failure Count"].iloc[-1] - g["Total Failure Count"].iloc[0]
        filas.append({
            "usuarios": int(g["User Count"].iloc[0]),
            "duracion_s": int(dt),
            "peticiones": int(dreq),
            "rps": round(dreq / dt, 2) if dt > 0 else float("nan"),
            "error_pct": round(100 * dfail / dreq, 3) if dreq > 0 else 0.0,
            "p50_ms": round(g["50%"].median(), 1),
            "p95_ms": round(g["95%"].median(), 1),
            "p99_ms": round(g["99%"].median(), 1),
        })
    return pd.DataFrame(filas)


def marcar_etapas(ax, hist: pd.DataFrame) -> None:
    """Líneas punteadas donde cambia el número de usuarios, con la etiqueta del escalón."""
    h = hist.dropna(subset=["User Count"]).copy()
    h["tramo"] = (h["User Count"] != h["User Count"].shift()).cumsum()
    for _, g in h.groupby("tramo"):
        if len(g) < 5 or g["User Count"].iloc[0] == 0:
            continue  # solo escalones estables, no cada paso de la rampa
        ax.axvline(g["t"].iloc[0], color=TINTA_2, lw=0.8, ls=":", zorder=0)
        ax.annotate(f"{int(g['User Count'].iloc[0])} usuarios", (g["t"].iloc[0], 1),
                    xycoords=("data", "axes fraction"), xytext=(3, -12),
                    textcoords="offset points", fontsize=8, color=TINTA_2)


def grafica_tiempo(hist, series, titulo, ylabel, archivo, linea_ref=None, techo_min=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    for (col, etiqueta), color in zip(series, SERIE):
        ax.plot(hist["t"], hist[col], color=color, label=etiqueta)
    # La línea de referencia solo se dibuja si los datos se acercan a ella (no aplasta la escala)
    if linea_ref is not None and hist[[c for c, _ in series]].max().max() >= 0.4 * linea_ref[0]:
        valor, texto = linea_ref
        ax.axhline(valor, color=TINTA_2, lw=1, ls="--")
        ax.annotate(texto, (1, valor), xycoords=("axes fraction", "data"), xytext=(-4, 3),
                    textcoords="offset points", fontsize=8, color=TINTA_2, ha="right")
    marcar_etapas(ax, hist)
    ax.set_title(titulo)
    ax.set_xlabel("Segundos desde el inicio (líneas punteadas: cambio de usuarios)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    if techo_min is not None:  # p. ej. 0% de error no debe verse como una escala de 0 a 0.05
        ax.set_ylim(top=max(ax.get_ylim()[1], techo_min))
    if len(series) > 1:
        ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncols=len(series))
    fig.tight_layout()
    fig.savefig(archivo, dpi=150)
    plt.close(fig)


def grafica_endpoints(stats: pd.DataFrame, archivo: Path) -> None:
    s = stats[stats["Name"] != "Aggregated"].set_index("Name")[["50%", "95%", "99%"]]
    s = s.apply(pd.to_numeric, errors="coerce").sort_values("95%")
    fig, ax = plt.subplots(figsize=(10, 1.2 + 0.9 * len(s)))
    alto = 0.26
    for i, (col, color) in enumerate(zip(["50%", "95%", "99%"], SERIE)):
        y = [k + (1 - i) * (alto + 0.02) for k in range(len(s))]
        barras = ax.barh(y, s[col], height=alto, color=color, label=f"p{col[:-1]}")
        if col == "95%":
            ax.bar_label(barras, fmt="%.0f ms", padding=3, fontsize=8, color=TINTA_2)
    ax.set_yticks(range(len(s)), s.index)
    ax.set_xlabel("Tiempo de respuesta (ms)")
    ax.set_title("Latencia por endpoint (toda la prueba)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", ncols=3)
    fig.tight_layout()
    fig.savefig(archivo, dpi=150)
    plt.close(fig)


def grafica_capacidad(et: pd.DataFrame, archivo: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(et["usuarios"], et["rps"], color=SERIE[0], marker="o", ms=6)
    a1.set_title("Throughput por escalón")
    a1.set_xlabel("Usuarios concurrentes")
    a1.set_ylabel("Peticiones por segundo")
    a2.plot(et["usuarios"], et["p95_ms"], color=SERIE[0], marker="o", ms=6)
    if et["p95_ms"].max() >= 0.4 * P95_OBJETIVO_MS:
        a2.axhline(P95_OBJETIVO_MS, color=TINTA_2, lw=1, ls="--")
        a2.annotate(f"objetivo p95 = {P95_OBJETIVO_MS} ms", (et["usuarios"].min(), P95_OBJETIVO_MS),
                    xytext=(0, 4), textcoords="offset points", fontsize=8, color=TINTA_2)
    a2.set_title("Latencia p95 por escalón")
    a2.set_xlabel("Usuarios concurrentes")
    a2.set_ylabel("ms")
    for a in (a1, a2):
        a.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(archivo, dpi=150)
    plt.close(fig)


def tabla_md(df: pd.DataFrame) -> str:
    cab = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    filas = ["| " + " | ".join(map(str, f)) + " |" for f in df.itertuples(index=False)]
    return "\n".join([cab, sep, *filas])


def observaciones(et: pd.DataFrame) -> list[str]:
    if et.empty:
        return ["- No hubo escalones estables suficientes para analizar la capacidad."]
    ok = et[(et["p95_ms"] <= P95_OBJETIVO_MS) & (et["error_pct"] < 1)]
    obs = []
    if ok.empty:
        obs.append(f"- Ningún escalón cumplió p95 ≤ {P95_OBJETIVO_MS} ms con menos de 1% de error.")
    else:
        mejor = ok.loc[ok["rps"].idxmax()]
        obs.append(
            f"- **Capacidad sostenible:** {mejor['rps']:.1f} peticiones/s con "
            f"{int(mejor['usuarios'])} usuarios, p95 = {mejor['p95_ms']:.0f} ms y "
            f"{mejor['error_pct']:.2f}% de error. Es el dato que usa el análisis FinOps.")
    base = et["p95_ms"].iloc[0]
    # Saturación sostenida: desde ese escalón, todos los siguientes tienen p95 > 2x la inicial
    # (y por encima de 100 ms, para no confundir el ruido de latencias muy bajas con saturación)
    arriba = (et["p95_ms"] > max(2 * base, 100)).tolist()
    sostenido = [all(arriba[i:]) for i in range(len(arriba))]
    if any(sostenido):
        r = et.iloc[sostenido.index(True)]
        obs.append(
            f"- **Saturación:** a partir de {int(r['usuarios'])} usuarios la p95 pasa de "
            f"{base:.0f} ms a {r['p95_ms']:.0f} ms (más del doble) mientras el throughput "
            f"llega a {r['rps']:.1f} peticiones/s: la API está cerca de su límite de CPU.")
    else:
        obs.append("- La p95 no llegó al doble de la del primer escalón: la API no se saturó "
                   "con esta carga (se puede subir CARGA_USUARIOS).")
    peor = et.loc[et["error_pct"].idxmax()]
    if peor["error_pct"] > 0:
        obs.append(f"- La mayor tasa de error fue {peor['error_pct']:.2f}% con "
                   f"{int(peor['usuarios'])} usuarios (ver la tabla de fallas).")
    else:
        obs.append("- No hubo errores en ningún escalón.")
    return obs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--prefijo", default="reports/carga/locust", help="Prefijo de los CSV de Locust")
    p.add_argument("--salida", default="reports/carga", help="Carpeta del reporte")
    args = p.parse_args()

    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)
    stats, hist, fallas = leer(Path(args.prefijo))
    et = etapas(hist)
    et.to_csv(salida / "resumen_etapas.csv", index=False)

    grafica_tiempo(hist, [("50%", "p50"), ("95%", "p95"), ("99%", "p99")],
                   "Latencia en el tiempo (todas las peticiones)", "Tiempo de respuesta (ms)",
                   salida / "latencia_tiempo.png",
                   linea_ref=(P95_OBJETIVO_MS, f"objetivo p95 = {P95_OBJETIVO_MS} ms"))
    grafica_tiempo(hist, [("Requests/s", "Peticiones/s")], "Peticiones por segundo",
                   "Peticiones/s", salida / "rps_tiempo.png")
    grafica_tiempo(hist, [("error_pct", "Tasa de error")], "Tasa de error",
                   "% de peticiones fallidas", salida / "errores_tiempo.png", techo_min=5)
    grafica_endpoints(stats, salida / "latencia_por_endpoint.png")
    if not et.empty:
        grafica_capacidad(et, salida / "capacidad_por_etapa.png")

    s = stats.copy()
    s["Error %"] = (100 * s["Failure Count"] / s["Request Count"].where(s["Request Count"] > 0)
                    ).fillna(0).round(3)
    por_endpoint = s[["Name", "Request Count", "Failure Count", "Error %", "Requests/s",
                      "Average Response Time", "50%", "95%", "99%", "Max Response Time"]].rename(
        columns={"Name": "Endpoint", "Request Count": "Peticiones", "Failure Count": "Fallas",
                 "Requests/s": "Pet/s", "Average Response Time": "Promedio ms",
                 "50%": "p50 ms", "95%": "p95 ms", "99%": "p99 ms", "Max Response Time": "Máx ms"})
    por_endpoint["Pet/s"] = por_endpoint["Pet/s"].round(2)
    por_endpoint["Promedio ms"] = por_endpoint["Promedio ms"].round(1)
    por_endpoint["Máx ms"] = por_endpoint["Máx ms"].round(0)

    total = s[s["Name"] == "Aggregated"].iloc[0]
    por_endpoint["Endpoint"] = por_endpoint["Endpoint"].replace({"Aggregated": "**Total**"})
    duracion = hist["t"].max()
    md = [
        "# Reporte de la prueba de carga · API de demanda (P1)",
        "",
        f"Generado el {datetime.now():%Y-%m-%d %H:%M} con `loadtest/generar_reporte.py` "
        f"a partir de `{args.prefijo}_*.csv`.",
        "",
        "## Escenario",
        "",
        "Definido en [`loadtest/locustfile.py`](../../loadtest/locustfile.py): usuarios virtuales que "
        "piden pronósticos con una pausa de 0.5 a 1.5 s entre acciones. Mezcla de tareas: "
        "`POST /predict` (peso 10), `POST /predict/lote` con 24 estación-horas (peso 2), "
        "`GET /health` (peso 1) y una entrada inválida que debe responder 422 (peso 1, cuenta como "
        "éxito). La carga sube en escalones de usuarios; la API corre con los límites del "
        "`docker-compose.yml` (por defecto 1 CPU y 1 GB).",
        "",
        "## Resultados globales",
        "",
        f"- Duración: {duracion:.0f} s · peticiones: {int(total['Request Count']):,} · "
        f"fallas: {int(total['Failure Count']):,} "
        f"({100 * total['Failure Count'] / max(total['Request Count'], 1):.2f}%)",
        f"- Throughput promedio: {total['Requests/s']:.1f} peticiones/s · "
        f"latencia p50 / p95 / p99: {total['50%']:.0f} / {total['95%']:.0f} / "
        f"{total['99%']:.0f} ms",
        "",
        tabla_md(por_endpoint),
        "",
        "## Por escalón de usuarios",
        "",
        "Métricas de los segundos estables de cada escalón (sin las rampas). La latencia es la "
        "mediana de los percentiles móviles que Locust reporta cada segundo.",
        "",
        tabla_md(et) if not et.empty else "_Sin escalones estables._",
        "",
        "### Observaciones",
        "",
        *observaciones(et),
        "",
        "## Gráficas",
        "",
        "![Latencia en el tiempo](latencia_tiempo.png)",
        "",
        "![Peticiones por segundo](rps_tiempo.png)",
        "",
        "![Tasa de error](errores_tiempo.png)",
        "",
        "![Latencia por endpoint](latencia_por_endpoint.png)",
        "",
    ]
    if not et.empty:
        md += ["![Capacidad por escalón](capacidad_por_etapa.png)", ""]
    md += ["## Fallas", ""]
    if fallas.empty:
        md.append("No se registraron fallas.")
    else:
        md.append(tabla_md(fallas[["Method", "Name", "Error", "Occurrences"]].rename(columns={
            "Method": "Método", "Name": "Endpoint", "Occurrences": "Ocurrencias"})))
    md += ["", "El reporte HTML interactivo de Locust está en [`locust.html`](locust.html).", ""]
    (salida / "reporte_carga.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Reporte escrito en {salida / 'reporte_carga.md'}")


if __name__ == "__main__":
    main()
