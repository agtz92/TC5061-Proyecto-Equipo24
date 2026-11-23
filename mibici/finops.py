"""Consumo de recursos y análisis FinOps de la API de P1 a partir de la prueba de carga.

Cruza tres fuentes que deja la prueba de carga en reports/carga/:
    docker_stats.csv             muestras de `docker stats` / `podman stats` (CPU % y memoria)
    locust_stats_history.csv     serie de tiempo de Locust (usuarios, peticiones/s, latencia)
    locust_stats.csv             totales por endpoint (mezcla de peticiones individuales y lotes)

y calcula, por escalón de usuarios, el CPU y la memoria de cada contenedor, los milisegundos de CPU
que cuesta cada petición y el costo por inferencia con precios de referencia de AWS Fargate.

Lo usa loadtest/generar_reporte.py para agregar las secciones de recursos y FinOps al reporte.
Todo es determinista: no consulta precios en línea (están fijos abajo, con su fuente y fecha).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CONTENEDOR_API = "mibici-api"
CONTENEDOR_MLFLOW = "mibici-mlflow"
HORAS_MES = 730  # promedio de horas en un mes (365 * 24 / 12)
P95_OBJETIVO_MS = 500  # mismo objetivo que el reporte de carga
REGISTROS_POR_LOTE = 24  # CARGA_LOTE por defecto del locustfile


@dataclass(frozen=True)
class Precio:
    """Tarifa por hora de un contenedor en la nube (USD)."""

    nombre: str
    vcpu_hora: float
    gb_hora: float


# AWS Fargate, Linux, región US East (N. Virginia), consultado el 2026-11-22 en
# https://aws.amazon.com/fargate/pricing/ (precio por segundo x 3600). Spot cambia con la oferta
# y la demanda: es una referencia, no un precio garantizado.
FARGATE_X86 = Precio("Fargate x86 bajo demanda", 0.04048, 0.004445)
FARGATE_ARM = Precio("Fargate ARM (Graviton) bajo demanda", 0.03238, 0.00356)
FARGATE_SPOT = Precio("Fargate Spot x86", 0.0124419, 0.00136621)
PRECIOS = (FARGATE_X86, FARGATE_ARM, FARGATE_SPOT)


@dataclass(frozen=True)
class Tarea:
    """Tamaño de un contenedor. Fargate solo acepta ciertas combinaciones de CPU y memoria."""

    nombre: str
    vcpu: float
    gb: float

    def costo_hora(self, precio: Precio) -> float:
        return self.vcpu * precio.vcpu_hora + self.gb * precio.gb_hora


# El compose limita la API a 1 CPU y 1 GB, pero en Fargate 1 vCPU exige de 2 a 8 GB: el tamaño
# facturable más cercano es 1 vCPU / 2 GB. MLflow cabe en la tarea más chica con 1 GB.
TAREA_API = Tarea("API (1 vCPU / 2 GB)", 1.0, 2.0)
TAREA_API_CHICA = Tarea("API ajustada (0.5 vCPU / 1 GB)", 0.5, 1.0)
TAREA_MLFLOW = Tarea("MLflow (0.25 vCPU / 1 GB)", 0.25, 1.0)

_UNIDADES = {"b": 1 / 2**20, "kib": 1 / 1024, "kb": 1e3 / 2**20, "mib": 1.0, "mb": 1e6 / 2**20,
             "gib": 1024.0, "gb": 1e9 / 2**20}


def a_mib(texto: str) -> float:
    """Convierte '202.2MiB', '1GiB' o '512kB' (formato de docker/podman stats) a MiB."""
    t = str(texto).strip().lower().replace(" ", "")
    num = t.rstrip("abcdefghijklmnopqrstuvwxyz")
    unidad = t[len(num):] or "b"
    if unidad not in _UNIDADES:
        raise ValueError(f"Unidad de memoria desconocida: {texto!r}")
    return float(num) * _UNIDADES[unidad]


def leer_docker_stats(ruta: Path) -> pd.DataFrame:
    """Lee el CSV de recursos: timestamp, contenedor, CPU % y 'uso / límite' de memoria.

    Acepta el archivo con encabezado (el que escribe loadtest/monitor_recursos.py) o sin él.
    """
    crudo = pd.read_csv(ruta, header=None, dtype=str)
    if not crudo.iloc[0, 0].strip().isdigit():
        crudo = crudo.iloc[1:]
    crudo = crudo.iloc[:, :4]
    crudo.columns = ["timestamp", "contenedor", "cpu", "memoria"]
    uso_lim = crudo["memoria"].str.split("/", n=1, expand=True)
    return pd.DataFrame({
        "timestamp": crudo["timestamp"].astype(float).to_numpy(),
        "contenedor": crudo["contenedor"].str.strip().to_numpy(),
        "cpu_pct": crudo["cpu"].str.strip().str.rstrip("%").astype(float).to_numpy(),
        "mem_mib": uso_lim[0].map(a_mib).to_numpy(),
        "mem_limite_mib": uso_lim[1].map(a_mib).to_numpy(),
    })


def ventanas_etapas(hist: pd.DataFrame) -> pd.DataFrame:
    """Inicio y fin (timestamp) de la parte estable de cada escalón de usuarios.

    Usa la misma regla que el reporte de carga: tramos de 5 s o más con el mismo número de
    usuarios, sin los primeros 2 s del tramo.
    """
    h = hist.dropna(subset=["User Count"]).copy()
    h["tramo"] = (h["User Count"] != h["User Count"].shift()).cumsum()
    filas = []
    for _, g in h.groupby("tramo"):
        if len(g) < 5 or g["User Count"].iloc[0] == 0:
            continue
        g = g.iloc[2:]
        filas.append({"usuarios": int(g["User Count"].iloc[0]),
                      "t_ini": g["Timestamp"].iloc[0], "t_fin": g["Timestamp"].iloc[-1]})
    return pd.DataFrame(filas)


def recursos_por_etapa(recursos: pd.DataFrame, hist: pd.DataFrame,
                       etapas: pd.DataFrame) -> pd.DataFrame:
    """CPU y memoria de la API en cada escalón y milisegundos de CPU por petición.

    `etapas` es el resumen por escalón del reporte de carga (usuarios, rps, p95_ms, ...).
    CPU 100% equivale a un núcleo completo, que es el límite de la API en el compose.
    """
    api = recursos[recursos["contenedor"] == CONTENEDOR_API]
    filas = []
    for v in ventanas_etapas(hist).itertuples(index=False):
        m = api[(api["timestamp"] >= v.t_ini) & (api["timestamp"] <= v.t_fin)]
        filas.append({
            "usuarios": v.usuarios,
            "muestras": len(m),
            "cpu_prom_pct": round(m["cpu_pct"].mean(), 1) if len(m) else np.nan,
            "cpu_max_pct": round(m["cpu_pct"].max(), 1) if len(m) else np.nan,
            "mem_max_mib": round(m["mem_mib"].max(), 1) if len(m) else np.nan,
        })
    r = etapas.merge(pd.DataFrame(filas), on="usuarios", how="left")
    r["cpu_ms_por_peticion"] = (10 * r["cpu_prom_pct"] / r["rps"]).round(1)  # %/100 * 1000 ms
    return r[["usuarios", "rps", "p95_ms", "error_pct", "muestras", "cpu_prom_pct",
              "cpu_max_pct", "mem_max_mib", "cpu_ms_por_peticion"]]


def predicciones_por_peticion(stats: pd.DataFrame, lote: int = REGISTROS_POR_LOTE) -> float:
    """Predicciones (estación-hora) que entrega en promedio cada petición de la prueba.

    /predict da 1, /predict/lote da `lote`; /health y la entrada inválida (422) no predicen.
    """
    s = stats[stats["Name"] != "Aggregated"].set_index("Name")["Request Count"]
    total = s.sum()
    individuales = s.get("POST /predict", 0)
    lotes = s.get("POST /predict/lote", 0)
    return float((individuales + lote * lotes) / total) if total else 0.0


def capacidad_con_slo(etapas: pd.DataFrame, p95_objetivo: float = P95_OBJETIVO_MS) -> float:
    """Peticiones/s que la API sostiene con p95 en el objetivo (interpolación lineal).

    Entre el último escalón que cumple y el primero que no; si todos cumplen, el mayor rps medido.
    """
    e = etapas.sort_values("usuarios").reset_index(drop=True)
    ok = e["p95_ms"] <= p95_objetivo
    if ok.all():
        return float(e["rps"].max())
    if not ok.any():
        return float("nan")
    j = int(np.argmax(~ok.to_numpy()))  # primer escalón que no cumple
    if j == 0:
        return float("nan")
    a, b = e.iloc[j - 1], e.iloc[j]
    frac = (p95_objetivo - a["p95_ms"]) / (b["p95_ms"] - a["p95_ms"])
    return float(a["rps"] + frac * (b["rps"] - a["rps"]))


def escenarios(etapas: pd.DataFrame) -> pd.DataFrame:
    """Throughput de referencia de una tarea de la API para calcular el costo por inferencia."""
    e = etapas.sort_values("usuarios")
    filas = [
        ("Escalón más bajo (10 usuarios)", float(e["rps"].iloc[0]),
         f"p95 {e['p95_ms'].iloc[0]:.0f} ms; la API va sobrada (la carga, no el CPU, limita)"),
        (f"Capacidad con p95 ≤ {P95_OBJETIVO_MS} ms (estimada)", capacidad_con_slo(etapas),
         "interpolada entre el último escalón que cumple y el primero que no"),
        ("Máximo medido (saturada)", float(e["rps"].max()),
         f"p95 {e.loc[e['rps'].idxmax(), 'p95_ms']:.0f} ms: no es aceptable para usuarios"),
    ]
    return pd.DataFrame(filas, columns=["escenario", "rps", "nota"])


def costo_por_millon(costo_hora: float, rps: float) -> float:
    """USD por millón de peticiones de una tarea que atiende `rps` peticiones/s todo el tiempo."""
    return costo_hora / (rps * 3600) * 1e6 if rps > 0 else float("nan")


def tabla_costos(esc: pd.DataFrame, pred_por_pet: float, tarea: Tarea = TAREA_API,
                 precios: tuple[Precio, ...] = PRECIOS) -> pd.DataFrame:
    """Costo por millón de peticiones y de predicciones para cada escenario y tarifa."""
    filas = []
    for p in precios:
        ch = tarea.costo_hora(p)
        for x in esc.itertuples(index=False):
            cpm = costo_por_millon(ch, x.rps)
            filas.append({
                "tarifa": p.nombre, "escenario": x.escenario, "rps": round(x.rps, 1),
                "usd_hora": round(ch, 5), "usd_mes_24x7": round(ch * HORAS_MES, 2),
                "usd_por_millon_peticiones": round(cpm, 3),
                "usd_por_millon_predicciones": round(cpm / pred_por_pet, 3) if pred_por_pet else np.nan,
            })
    return pd.DataFrame(filas)


def costo_por_trafico(capacidad_rps: float, trafico_rps: list[float], tarea: Tarea = TAREA_API,
                      precio: Precio = FARGATE_X86, minimo_tareas: int = 1) -> pd.DataFrame:
    """Costo real por millón de peticiones según el tráfico promedio que llegue.

    Se paga por tarea encendida: con poco tráfico la tarea mínima está casi ociosa y el costo
    por inferencia sube; con mucho, se agregan tareas (una por cada `capacidad_rps`).
    """
    filas = []
    for t in trafico_rps:
        n = max(minimo_tareas, int(np.ceil(t / capacidad_rps))) if capacidad_rps > 0 else np.nan
        ch = n * tarea.costo_hora(precio)
        filas.append({"trafico_rps": t, "tareas": n,
                      "utilizacion_pct": round(100 * t / (n * capacidad_rps), 1),
                      "usd_mes": round(ch * HORAS_MES, 2),
                      "usd_por_millon_peticiones": round(costo_por_millon(ch, t), 2)})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------- gráficas

def _estilo(ax, barras: bool = False) -> None:
    ax.set_ylim(bottom=0)
    ax.set_axisbelow(True)
    if barras:
        ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)


def grafica_recursos_tiempo(recursos: pd.DataFrame, hist: pd.DataFrame, archivo: Path,
                            colores: tuple[str, str] = ("#2a78d6", "#eb6834")) -> None:
    """CPU (%) y memoria (MiB) de la API y de MLflow durante la prueba, con los escalones."""
    t0 = hist["Timestamp"].min()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for cont, color in zip([CONTENEDOR_API, CONTENEDOR_MLFLOW], colores):
        r = recursos[recursos["contenedor"] == cont].sort_values("timestamp")
        if r.empty:
            continue
        t = r["timestamp"] - t0
        a1.plot(t, r["cpu_pct"], color=color, marker="o", ms=3, label=cont)
        a2.plot(t, r["mem_mib"], color=color, marker="o", ms=3, label=cont)
    a1.axhline(100, color="#52514e", lw=1, ls="--", label="límite de la API (1 CPU)")
    for v in ventanas_etapas(hist).itertuples(index=False):
        for ax in (a1, a2):
            ax.axvline(v.t_ini - t0, color="#52514e", lw=0.8, ls=":", zorder=0)
        a1.annotate(f"{v.usuarios} usuarios", (v.t_ini - t0, 1), xycoords=("data", "axes fraction"),
                    xytext=(3, -12), textcoords="offset points", fontsize=8, color="#52514e")
    a1.set_title("Uso de CPU por contenedor")
    a1.set_ylabel("CPU (% de un núcleo)")
    a2.set_title("Memoria por contenedor")
    a2.set_ylabel("MiB")
    a2.set_xlabel("Segundos desde el inicio de la prueba (líneas punteadas: cambio de usuarios)")
    for ax in (a1, a2):
        _estilo(ax)
    a1.set_ylim(top=115)
    a1.legend(loc="lower right", bbox_to_anchor=(1, 1.08), ncols=3, frameon=False)
    fig.tight_layout()
    fig.savefig(archivo, dpi=150)
    plt.close(fig)


def grafica_recursos_etapa(r: pd.DataFrame, archivo: Path, color: str = "#2a78d6") -> None:
    """CPU promedio de la API y ms de CPU por petición en cada escalón."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(r["usuarios"], r["cpu_prom_pct"], color=color, marker="o", ms=6)
    a1.axhline(100, color="#52514e", lw=1, ls="--")
    a1.set_title("CPU promedio de la API por escalón")
    a1.set_ylabel("CPU (% de un núcleo)")
    a1.set_ylim(top=110)
    a2.bar(r["usuarios"].astype(str), r["cpu_ms_por_peticion"], color=color)
    a2.bar_label(a2.containers[0], fmt="%.1f ms", fontsize=8, color="#52514e")
    a2.set_title("CPU que cuesta cada petición")
    a2.set_ylabel("ms de CPU por petición")
    for ax, barras in ((a1, False), (a2, True)):
        ax.set_xlabel("Usuarios concurrentes")
        _estilo(ax, barras)
    a1.set_ylim(top=110)
    fig.tight_layout()
    fig.savefig(archivo, dpi=150)
    plt.close(fig)


def grafica_costos(costos: pd.DataFrame, archivo: Path,
                   colores: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a")) -> None:
    """USD por millón de predicciones por escenario y tarifa (barras agrupadas)."""
    esc = list(dict.fromkeys(costos["escenario"]))
    tar = list(dict.fromkeys(costos["tarifa"]))
    fig, ax = plt.subplots(figsize=(10, 4))
    ancho = 0.8 / len(tar)
    x = np.arange(len(esc))
    for i, (t, color) in enumerate(zip(tar, colores)):
        v = costos[costos["tarifa"] == t].set_index("escenario").loc[esc,
                                                                     "usd_por_millon_predicciones"]
        b = ax.bar(x + (i - (len(tar) - 1) / 2) * ancho, v, width=ancho, color=color, label=t)
        ax.bar_label(b, fmt="$%.2f", fontsize=8, color="#52514e", padding=2)
    ax.set_xticks(x, [e.replace(" (", "\n(") for e in esc], fontsize=8)
    ax.set_ylabel("USD por millón de predicciones")
    ax.set_title("Costo por inferencia de una tarea de la API (1 vCPU / 2 GB, 24x7)")
    ax.legend(frameon=False, fontsize=8)
    _estilo(ax, barras=True)
    fig.tight_layout()
    fig.savefig(archivo, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- reporte

def _tabla_md(df: pd.DataFrame) -> str:
    cab = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    filas = ["| " + " | ".join(map(str, f)) + " |" for f in df.itertuples(index=False)]
    return "\n".join([cab, sep, *filas])


def _usd(x: float, dec: int = 2) -> str:
    return f"${x:,.{dec}f}"


def generar(ruta_recursos: Path, stats: pd.DataFrame, hist: pd.DataFrame,
            etapas: pd.DataFrame, salida: Path) -> list[str]:
    """Escribe gráficas y CSV de recursos y FinOps en `salida` y devuelve las secciones en Markdown."""
    recursos = leer_docker_stats(ruta_recursos)
    r = recursos_por_etapa(recursos, hist, etapas)
    r.to_csv(salida / "recursos_por_etapa.csv", index=False)
    grafica_recursos_tiempo(recursos, hist, salida / "recursos_tiempo.png")
    grafica_recursos_etapa(r, salida / "recursos_por_etapa.png")

    ppp = predicciones_por_peticion(stats)
    esc = escenarios(etapas)
    costos = tabla_costos(esc, ppp)
    costos.to_csv(salida / "costos_inferencia.csv", index=False)
    grafica_costos(costos, salida / "costos_inferencia.png")
    cap = capacidad_con_slo(etapas)
    trafico = costo_por_trafico(cap, [0.5, 2, round(cap / 2, 1), round(cap, 1), round(3 * cap, 1)])
    trafico.to_csv(salida / "costo_por_trafico.csv", index=False)

    def res(cont: str) -> dict[str, float]:
        m = recursos[recursos["contenedor"] == cont]
        return {"cpu_prom": m["cpu_pct"].mean(), "cpu_max": m["cpu_pct"].max(),
                "mem_min": m["mem_mib"].min(), "mem_max": m["mem_mib"].max(),
                "mem_lim": m["mem_limite_mib"].iloc[0], "n": len(m)}

    api, mlf = res(CONTENEDOR_API), res(CONTENEDOR_MLFLOW)
    dur = recursos["timestamp"].max() - recursos["timestamp"].min()
    base = r.iloc[0]
    sat = r[r["cpu_prom_pct"] >= 90]
    x86 = costos[costos["tarifa"] == FARGATE_X86.nombre].set_index("escenario")
    esc_cap = esc["escenario"].iloc[1]
    c_cap = x86.loc[esc_cap]
    c_api_mes = TAREA_API.costo_hora(FARGATE_X86) * HORAS_MES
    c_mlf_mes = TAREA_MLFLOW.costo_hora(FARGATE_X86) * HORAS_MES
    c_chica = TAREA_API_CHICA.costo_hora(FARGATE_X86)
    arm = costos[(costos["tarifa"] == FARGATE_ARM.nombre) & (costos["escenario"] == esc_cap)].iloc[0]
    spot = costos[(costos["tarifa"] == FARGATE_SPOT.nombre)
                  & (costos["escenario"] == esc_cap)].iloc[0]
    lote_pred = REGISTROS_POR_LOTE

    tabla_r = r.rename(columns={
        "usuarios": "Usuarios", "rps": "Pet/s", "p95_ms": "p95 ms", "error_pct": "Error %",
        "muestras": "Muestras", "cpu_prom_pct": "CPU prom %", "cpu_max_pct": "CPU máx %",
        "mem_max_mib": "Memoria máx MiB", "cpu_ms_por_peticion": "ms CPU / petición"})
    tabla_c = costos.drop(columns=["usd_hora"]).rename(columns={
        "tarifa": "Tarifa", "escenario": "Escenario", "rps": "Pet/s",
        "usd_mes_24x7": "USD/mes 24x7", "usd_por_millon_peticiones": "USD / millón de peticiones",
        "usd_por_millon_predicciones": "USD / millón de predicciones"})
    tabla_t = trafico.rename(columns={
        "trafico_rps": "Tráfico promedio (pet/s)", "tareas": "Tareas de la API",
        "utilizacion_pct": "Utilización %", "usd_mes": "USD/mes",
        "usd_por_millon_peticiones": "USD / millón de peticiones"})

    md = [
        "## Consumo de recursos durante la prueba",
        "",
        f"Muestras de `docker stats` cada ~6 s durante {dur:.0f} s ({api['n']} por contenedor), "
        "tomadas con [`loadtest/monitor_recursos.py`](../../loadtest/monitor_recursos.py) mientras "
        "corría Locust. El CPU se expresa en % de un núcleo: 100% es el límite de la API en el "
        "compose. Datos crudos en [`docker_stats.csv`](docker_stats.csv); por escalón en "
        "[`recursos_por_etapa.csv`](recursos_por_etapa.csv).",
        "",
        "| Contenedor | CPU prom % | CPU máx % | Memoria MiB (mín → máx) | Límite de memoria |",
        "|---|---|---|---|---|",
        f"| `{CONTENEDOR_API}` | {api['cpu_prom']:.1f} | {api['cpu_max']:.1f} | "
        f"{api['mem_min']:.0f} → {api['mem_max']:.0f} | {api['mem_lim']:.0f} MiB |",
        f"| `{CONTENEDOR_MLFLOW}` | {mlf['cpu_prom']:.1f} | {mlf['cpu_max']:.1f} | "
        f"{mlf['mem_min']:.0f} → {mlf['mem_max']:.0f} | sin límite (memoria del anfitrión) |",
        "",
        "### Por escalón de usuarios (API)",
        "",
        _tabla_md(tabla_r),
        "",
        "![Recursos en el tiempo](recursos_tiempo.png)",
        "",
        "![Recursos por escalón](recursos_por_etapa.png)",
        "",
        "### Interpretación",
        "",
        f"- **El cuello de botella es el CPU, no la memoria.** Con {int(base['usuarios'])} usuarios "
        f"la API usa {base['cpu_prom_pct']:.0f}% de su núcleo"
        + (f"; desde {int(sat['usuarios'].iloc[0])} usuarios se queda pegada entre "
           f"{sat['cpu_prom_pct'].min():.0f}% y {sat['cpu_prom_pct'].max():.0f}%, y a partir de ahí "
           "más usuarios solo alargan la cola: el throughput apenas sube y la p95 se dispara."
           if not sat.empty else "."),
        f"- **La memoria sobra.** La API ocupa {api['mem_max']:.0f} MiB de "
        f"{api['mem_lim']:.0f} MiB ({100 * api['mem_max'] / api['mem_lim']:.0f}%) y casi no crece "
        "con la carga: el modelo se carga una vez y cada predicción no retiene memoria.",
        f"- **Cada petición se vuelve más cara al saturar.** Con poca carga cuesta "
        f"{base['cpu_ms_por_peticion']:.1f} ms de CPU; con la API saturada sube a "
        f"{r['cpu_ms_por_peticion'].max():.1f} ms. Lo más probable es que el CPU extra se vaya en "
        "administrar la cola de peticiones (conexiones abiertas y cambios de contexto en un solo "
        "proceso), no en predecir.",
        f"- **MLflow casi no trabaja** ({mlf['cpu_prom']:.1f}% de CPU en promedio): la API "
        "descarga el modelo al arrancar y después no lo consulta. Ocupa "
        f"~{mlf['mem_max']:.0f} MiB por tener el servidor encendido.",
        "",
        "## Análisis FinOps: costo por inferencia",
        "",
        "### Supuestos",
        "",
        "- **Precio de referencia:** AWS Fargate en Linux, región US East (N. Virginia), consultado "
        "el 2026-11-22 en <https://aws.amazon.com/fargate/pricing/>:",
        "",
        "| Tarifa | USD por vCPU-hora | USD por GB-hora |",
        "|---|---|---|",
        *[f"| {p.nombre} | {p.vcpu_hora} | {p.gb_hora} |" for p in PRECIOS],
        "",
        "- **Tamaño de la tarea:** el compose limita la API a 1 CPU y 1 GB, pero en Fargate 1 vCPU "
        f"exige de 2 a 8 GB; se factura la combinación válida más cercana, {TAREA_API.nombre}: "
        f"{_usd(TAREA_API.costo_hora(FARGATE_X86), 4)} por hora o {_usd(c_api_mes)} al mes "
        "encendida 24x7 en x86 bajo demanda.",
        f"- **Inferencia:** una predicción de demanda (una estación en una hora). En la mezcla de la "
        f"prueba cada petición entrega en promedio **{ppp:.2f} predicciones**, porque `/predict/lote` "
        f"trae {lote_pred} por petición. Se reporta el costo por millón de peticiones y por millón "
        "de predicciones.",
        "- **Costo por inferencia** = costo por hora de la tarea ÷ peticiones atendidas por hora. "
        "Supone la tarea siempre encendida y atendiendo el throughput del escenario.",
        "- No incluye red (transferencia de datos), balanceador, registro de imágenes ni logs, que "
        "en un servicio chico pueden pesar más que el cómputo.",
        "",
        "### Costo por escenario de carga",
        "",
        _tabla_md(esc.assign(rps=esc["rps"].round(1)).rename(
            columns={"escenario": "Escenario", "rps": "Pet/s", "nota": "Nota"})),
        "",
        _tabla_md(tabla_c),
        "",
        "![Costo por inferencia](costos_inferencia.png)",
        "",
        f"**Resultado.** Con la API atendiendo su capacidad útil (~{c_cap['rps']:.0f} pet/s con "
        f"p95 ≤ {P95_OBJETIVO_MS} ms), una tarea de {_usd(c_api_mes)}/mes cuesta "
        f"**{_usd(c_cap['usd_por_millon_peticiones'])} por millón de peticiones**, o "
        f"**{_usd(c_cap['usd_por_millon_predicciones'])} por millón de predicciones** "
        f"({_usd(c_cap['usd_por_millon_peticiones'] / 1e6, 8)} por petición). Subir hasta la "
        "saturación abarata la inferencia, pero con latencias de segundos que no sirven a un "
        "usuario: el costo correcto es el de la capacidad que cumple el objetivo de latencia.",
        "",
        "### El costo real depende del tráfico",
        "",
        "El cómputo se paga por tarea encendida, no por petición. Con poco tráfico la tarea mínima "
        "pasa casi todo el tiempo ociosa y el costo por inferencia se multiplica; con mucho tráfico se "
        f"agregan tareas de ~{cap:.0f} pet/s cada una (x86 bajo demanda, sin contar MLflow):",
        "",
        _tabla_md(tabla_t),
        "",
        f"Con el sistema completo encendido 24x7 (la API más MLflow en una tarea de "
        f"{TAREA_MLFLOW.vcpu} vCPU / {TAREA_MLFLOW.gb:.0f} GB) el piso es "
        f"{_usd(c_api_mes + c_mlf_mes)}/mes en x86 bajo demanda, aunque no llegue ninguna "
        "petición.",
        "",
        "### Cómo optimizarlo",
        "",
        "| Acción | Efecto estimado | Costo o riesgo |",
        "|---|---|---|",
        f"| Usar `/predict/lote` para tableros y procesos batch | {lote_pred} predicciones por "
        "petición: el costo por predicción cae varias veces frente a pedirlas una a una | Solo "
        "aplica a clientes que piden muchas estaciones u horas a la vez |",
        "| Precalcular cada hora el pronóstico de todas las estaciones y servirlo desde caché "
        "(Redis o un archivo) | El modelo corre una vez por estación-hora y no una vez por consulta; "
        "la API solo lee | Requiere un job programado; la predicción ya no usa datos en tiempo real |",
        f"| Ajustar la tarea: la API usa ~{api['mem_max']:.0f} MiB | {TAREA_API_CHICA.nombre} cuesta "
        f"{_usd(c_chica * HORAS_MES)}/mes, la mitad que 1 vCPU / 2 GB | Con la mitad del CPU "
        "atiende cerca de la mitad del tráfico: conviene si el tráfico real es bajo |",
        f"| Fargate ARM (Graviton) | {_usd(arm['usd_por_millon_predicciones'])} por millón de "
        f"predicciones (~{100 * (1 - arm['usd_hora'] / c_cap['usd_hora']):.0f}% menos) | "
        "Construir la imagen para `linux/arm64` y repetir la prueba de carga |",
        f"| Fargate Spot | {_usd(spot['usd_por_millon_predicciones'])} por millón de "
        f"predicciones (~{100 * (1 - spot['usd_hora'] / c_cap['usd_hora']):.0f}% menos) | AWS "
        "puede interrumpir la tarea con 2 minutos de aviso: sirve para el challenger del A/B, "
        "batch o tareas extra, no como única réplica |",
        "| Autoescalar con CPU objetivo de ~60-70% | Se paga una tarea en horas valle y se agregan "
        "en horas pico (la demanda de MiBici es muy horaria) | El arranque tarda lo que tarda en "
        "descargar el modelo (~60 s de `start_period`) |",
        f"| Empaquetar el modelo en la imagen (`MODEL_URI` local) y apagar MLflow fuera de "
        f"entrenamiento | Quita {_usd(c_mlf_mes)}/mes del servicio en línea y una dependencia del "
        "arranque | Cada versión nueva del modelo requiere reconstruir la imagen |",
        "| Más workers de Uvicorn dentro de la misma vCPU | No baja el costo: el límite es el CPU | "
        "Solo ayuda si se da más de 1 vCPU por tarea |",
        "",
        "**Recomendación.** Para el tráfico de un tablero interno de MiBici basta una tarea; lo que "
        "más abarata es dejar de predecir por consulta (lote o precálculo por hora con caché), pasar "
        "a ARM y quitar MLflow del camino de la inferencia. Para tráfico público variable, autoescalar "
        "por CPU y usar Spot solo para las tareas adicionales.",
        "",
    ]
    return md
