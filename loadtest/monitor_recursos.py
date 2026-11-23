"""Registra el CPU y la memoria de los contenedores mientras corre la prueba de carga.

Toma una muestra de `docker stats` (o `podman stats`) cada pocos segundos y la agrega a un CSV con
encabezado: timestamp, contenedor, cpu, memoria. Es la entrada de las secciones de recursos y FinOps
del reporte (mibici/finops.py).

Uso, en otra terminal, justo antes de lanzar Locust:
    python loadtest/monitor_recursos.py                          # hasta Ctrl+C
    python loadtest/monitor_recursos.py --duracion 320           # se detiene solo
    python loadtest/monitor_recursos.py --motor podman --intervalo 2
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import time

CONTENEDORES = ["mibici-api", "mibici-mlflow"]
FORMATO = "{{.Name}},{{.CPUPerc}},{{.MemUsage}}"


def muestra(motor: str, contenedores: list[str]) -> list[list[str]]:
    """Una lectura de stats: [timestamp, contenedor, cpu, memoria] por contenedor."""
    salida = subprocess.run(
        [motor, "stats", "--no-stream", "--format", FORMATO, *contenedores],
        capture_output=True, text=True, check=True).stdout
    ts = str(int(time.time()))
    return [[ts, *linea.split(",", 2)] for linea in salida.strip().splitlines() if linea]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--salida", default="reports/carga/docker_stats.csv")
    p.add_argument("--motor", default="docker", choices=["docker", "podman"])
    p.add_argument("--intervalo", type=float, default=5, help="Segundos entre muestras")
    p.add_argument("--duracion", type=float, default=None, help="Segundos; sin valor, hasta Ctrl+C")
    p.add_argument("--contenedores", nargs="+", default=CONTENEDORES)
    args = p.parse_args()

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    fin = time.time() + args.duracion if args.duracion else float("inf")
    with salida.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "contenedor", "cpu", "memoria"])
        print(f"Registrando {', '.join(args.contenedores)} en {salida} (Ctrl+C para terminar)")
        try:
            while time.time() < fin:
                inicio = time.time()
                w.writerows(muestra(args.motor, args.contenedores))
                f.flush()
                time.sleep(max(0.0, args.intervalo - (time.time() - inicio)))
        except KeyboardInterrupt:
            pass
    print(f"Listo: {salida}")


if __name__ == "__main__":
    main()
