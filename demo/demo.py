"""Demo de la API de demanda (P1): peticiones reales y comparación contra lo que pasó.

Manda a la API los ejemplos de demo/ejemplos (un día de septiembre de 2026 que el modelo no vio al
entrenar) y compara cada predicción con los viajes reales. Las peticiones usan solo la biblioteca
estándar; si hay matplotlib, `--grafica` guarda la curva de 24 horas predicha contra la real. La URL por
defecto es API_URL de mibici/config.py (variable MIBICI_API_URL).

Uso, con la API arriba (docker compose up -d):
    python demo/demo.py
    python demo/demo.py --url <URL del challenger, puerto 8001> --pausa
    python demo/demo.py --grafica demo/demo_24h.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from mibici.config import API_URL

EJEMPLOS = Path(__file__).parent / "ejemplos"


def llamar(url: str, ruta: str, cuerpo: dict | None = None) -> tuple[int, dict, float]:
    """GET (sin cuerpo) o POST JSON. Devuelve (código HTTP, respuesta, milisegundos)."""
    datos = json.dumps(cuerpo).encode() if cuerpo is not None else None
    req = urllib.request.Request(url + ruta, data=datos, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:  # nosec B310 (URL local que da el usuario)
            return r.status, json.loads(r.read()), (time.perf_counter() - t0) * 1000
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), (time.perf_counter() - t0) * 1000


def cargar(nombre: str) -> dict:
    return json.loads((EJEMPLOS / nombre).read_text(encoding="utf-8"))


def reales() -> dict[str, dict]:
    archivo = next(EJEMPLOS.glob("real_*.csv"))
    with archivo.open(encoding="utf-8") as f:
        return {fila["ejemplo"]: fila for fila in csv.DictReader(f)}


def paso(titulo: str, pausa: bool) -> None:
    if pausa:
        input("\n[Enter para continuar] ")
    print(f"\n=== {titulo} " + "=" * max(0, 70 - len(titulo)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--url", default=API_URL, help="URL base de la API (MIBICI_API_URL)")
    p.add_argument("--pausa", action="store_true", help="Esperar Enter entre pasos (para presentar)")
    p.add_argument("--grafica", default=None, help="Ruta PNG para la curva de 24 h (requiere matplotlib)")
    args = p.parse_args()
    url = args.url.rstrip("/")
    real = reales()

    paso("1. Salud del servicio: GET /health", args.pausa)
    codigo, r, ms = llamar(url, "/health")
    print(f"HTTP {codigo} en {ms:.0f} ms -> {r}")
    if codigo != 200:
        raise SystemExit("La API no está lista. Revisa `docker compose ps` y los logs del contenedor.")

    casos = [
        ("2. Hora pico: estación concurrida a las 7:00", "01_hora_pico.json", "01_hora_pico"),
        ("3. Madrugada: la misma estación a las 3:00", "02_madrugada.json", "02_madrugada"),
        ("4. Estación nueva (abierta en 2026), hora pico", "03_estacion_nueva.json", "03_estacion_nueva"),
    ]
    for titulo, archivo, clave in casos:
        paso(f"{titulo}: POST /predict", args.pausa)
        fila = real[clave]
        print(f"{fila['estacion']}, {fila['fecha']} a las {int(fila['hour']):02d}:00")
        codigo, r, ms = llamar(url, "/predict", cargar(archivo))
        pred = r.get("viajes_estimados")
        print(f"HTTP {codigo} en {ms:.0f} ms -> predicción {pred:.1f} viajes | reales: {fila['viajes_reales']}")

    paso("5. Entrada inválida (hour = 25): la API debe responder 422", args.pausa)
    codigo, r, ms = llamar(url, "/predict", cargar("04_invalido.json"))
    detalle = r.get("detail", [{}])
    print(f"HTTP {codigo} en {ms:.0f} ms -> {detalle[0].get('msg', detalle) if isinstance(detalle, list) else detalle}")

    paso("6. Pronóstico de las 24 horas en una sola petición: POST /predict/lote", args.pausa)
    lote = cargar("05_lote_24h.json")
    codigo, r, ms = llamar(url, "/predict/lote", lote)
    preds = [x["viajes_estimados"] for x in r["predicciones"]]
    horas = [reg["hour"] for reg in lote["registros"]]
    obs = [int(real[f"05_lote_24h_h{h:02d}"]["viajes_reales"]) for h in horas]
    fila = real["05_lote_24h_h00"]
    print(f"HTTP {codigo} en {ms:.0f} ms | modelo {r['modelo']} v{r['version']} | {fila['estacion']}, {fila['fecha']}")
    print(f"{'Hora':>5} {'Predicho':>9} {'Real':>6}  Barra (# = real, * = predicho)")
    for h, pr, ob in zip(horas, preds, obs):
        print(f"{h:>4}h {pr:>9.1f} {ob:>6}  {'#' * ob}")
        if pr >= 0.5:
            print(f"{'':>24}{'*' * round(pr)}")
    total_p, total_o = sum(preds), sum(obs)
    wape = sum(abs(a - b) for a, b in zip(preds, obs)) / max(total_o, 1) * 100
    print(f"\nTotal del día: predicho {total_p:.0f} viajes | real {total_o} | WAPE {wape:.1f}%")

    if args.grafica:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(horas, obs, color="#c9d7ea", label="Viajes reales")
        ax.plot(horas, preds, color="#2a78d6", marker="o", lw=2, label=f"Predicción de la API (v{r['version']})")
        ax.set_xticks(range(0, 24, 2))
        ax.set_xlabel("Hora del día")
        ax.set_ylabel("Viajes que inician")
        ax.set_title(f"{fila['estacion']}, {fila['fecha']}: real {total_o}, predicho {total_p:.0f} (WAPE {wape:.0f}%)",
                     fontsize=10)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(args.grafica, dpi=150)
        print(f"Gráfica guardada en {args.grafica}")


if __name__ == "__main__":
    main()
