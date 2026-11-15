"""Diagnóstico del modelo P1: qué versiones hay en el registry y cuál está descargada en local.

Responde a la inconsistencia entre el reporte de Evidently (modelo en models/p1) y la
interpretación del drift (models:/mibici-equipo24-p1-demanda@champion):

1. Lista los modelos registrados y sus alias.
2. Muestra las versiones de P1 bajo el nombre oficial y el nombre heredado con mayúsculas.
3. Identifica a qué versión del registry corresponde cada modelo descargado en models/p1,
   comparando el run_id de su archivo MLmodel con el de cada versión.

Uso:
    python diagnosticop1.py
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models import Model

from mibici.config import MLFLOW_TRACKING_URI, MODELS_DIR, REGISTRY_NAME

NOMBRE_OFICIAL = REGISTRY_NAME["p1"]
NOMBRE_HEREDADO = "Mibici-equipo24-P1-demanda"  # usado por guardarmodelos.py / verificarmodelos.py
CARPETA_LOCAL = MODELS_DIR / "p1"


def ocultar_credenciales(uri: str) -> str:
    """Quita usuario y contraseña de la URI antes de imprimirla."""
    partes = urlsplit(uri)
    if partes.username or partes.password:
        host = partes.hostname or ""
        if partes.port:
            host += f":{partes.port}"
        return urlunsplit(partes._replace(netloc=f"***@{host}"))
    return uri


def listar_registrados(client: mlflow.MlflowClient) -> None:
    print("\n=== Modelos registrados ===")
    modelos = client.search_registered_models()
    if not modelos:
        print("  (ninguno)")
    for rm in modelos:
        print(f"- {rm.name}  aliases={dict(rm.aliases)}")


def versiones_de(client: mlflow.MlflowClient, nombre: str) -> list[dict]:
    """Versiones del modelo con los datos de su corrida. Un error en una versión no
    detiene las demás."""
    print(f"\n[{nombre}]")
    try:
        versiones = client.search_model_versions(f"name='{nombre}'")
    except MlflowException as e:
        print(f"  ERROR al consultar versiones: {e}")
        return []
    # El filtro puede no distinguir mayúsculas según el backend; se filtra exacto aquí.
    versiones = sorted((mv for mv in versiones if mv.name == nombre), key=lambda mv: int(mv.version))
    if not versiones:
        print("  (sin versiones registradas con este nombre)")
        return []

    filas = []
    for mv in versiones:
        fila = {"nombre": nombre, "version": mv.version, "run_id": mv.run_id,
                "aliases": list(mv.aliases)}
        try:
            r = client.get_run(mv.run_id)
            fila |= {"corrida": r.info.run_name, "modelo": r.data.tags.get("model"),
                     "train_end": r.data.params.get("train_end")}
        except MlflowException as e:
            fila |= {"corrida": f"ERROR: {e}", "modelo": None, "train_end": None}
        alias = ",".join(f"@{a}" for a in fila["aliases"]) or "-"
        print(f"  v{fila['version']}  {alias:<12} run={(fila['run_id'] or '')[:8]}  "
              f"corrida={fila['corrida']}  modelo={fila['modelo']}  train_end={fila['train_end']}")
        filas.append(fila)
    return filas


def revisar_carpeta_local(versiones: list[dict]) -> None:
    print(f"\n=== Carpeta local {CARPETA_LOCAL} ===")
    archivos = sorted(CARPETA_LOCAL.rglob("MLmodel")) if CARPETA_LOCAL.exists() else []
    if not archivos:
        print("  (no hay modelos descargados; ejecuta guardarmodelos.py)")
        return

    por_run = {v["run_id"]: v for v in versiones if v["run_id"]}
    for archivo in archivos:
        carpeta = archivo.parent
        print(f"\n- {carpeta.relative_to(MODELS_DIR.parent)}")
        try:
            m = Model.load(str(archivo))
        except Exception as e:  # MLmodel corrupto o de un formato no soportado
            print(f"  ERROR al leer MLmodel: {e}")
            continue
        print(f"  run_id={m.run_id}  sabores={sorted(m.flavors)}  creado={m.utc_time_created}")
        v = por_run.get(m.run_id)
        if v:
            alias = ",".join(f"@{a}" for a in v["aliases"]) or "sin alias"
            print(f"  => corresponde a {v['nombre']} v{v['version']} ({alias}, modelo={v['modelo']})")
            if v["nombre"] != NOMBRE_OFICIAL:
                print(f"  AVISO: no es el modelo oficial {NOMBRE_OFICIAL}.")
        else:
            print("  AVISO: su run_id no coincide con ninguna versión de P1 en el registry.")
        if carpeta != CARPETA_LOCAL:
            print(f"  AVISO: mibici/monitoring.py carga '{CARPETA_LOCAL.name}/' directamente, "
                  f"no esta subcarpeta.")


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    print("Tracking URI:", ocultar_credenciales(MLFLOW_TRACKING_URI))

    listar_registrados(client)

    print("\n=== Versiones de P1 ===")
    versiones = []
    for nombre in (NOMBRE_OFICIAL, NOMBRE_HEREDADO):
        versiones += versiones_de(client, nombre)

    revisar_carpeta_local(versiones)


if __name__ == "__main__":
    main()
