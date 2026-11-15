"""P1, A/B testing entre la versión vigente (@champion) y la alternativa (@challenger).

Diseño
------
Ambas versiones del modelo `mibici-equipo24-p1-demanda` predicen exactamente las mismas
filas: todas las estación-horas de AB_EVAL_START a AB_EVAL_END (julio a septiembre de 2026),
sin muestreo. Ninguna de las dos usó ese periodo para entrenar ni para elegir
hiperparámetros (se verifica con validate_windows), así que la comparación es justa.

Es una comparación pareada fuera de línea (backtest): cada estación-hora tiene la
predicción de las dos versiones, lo que elimina la varianza de asignar tráfico distinto
a cada brazo y permite pruebas pareadas.

Incertidumbre
-------------
Los errores de una misma estación, y los de un mismo día, están correlacionados, así que
un intervalo que trate cada estación-hora como independiente sería demasiado optimista.
La diferencia de WAPE se acompaña de dos intervalos bootstrap al 95%:

- por estación (se remuestrean estaciones completas) y
- por día (se remuestrean días completos: clima, eventos, días festivos),

y la decisión usa el más conservador de los dos. Como apoyo se reporta la prueba de
Diebold-Mariano sobre el error absoluto diario (la misma de la entrega 2).

Decisión
--------
Criterios fijados en config.py antes de ver resultados (AB_*). El challenger queda como
versión vigente solo si cumple todos; si no, se conserva el champion.

Uso:
    python -m mibici.modeling.train ab-test              # compara y documenta
    python -m mibici.modeling.train ab-test --promote true   # además mueve @champion si gana
    mlflow run . -e ab_test --env-manager local
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
import mlflow
import numpy as np
import pandas as pd

from mibici.config import (
    AB_BOOTSTRAP_REPS,
    AB_EVAL_END,
    AB_EVAL_START,
    AB_MIN_MEJORA_WAPE,
    AB_TOLERANCIA_SEGMENTO,
    AB_TOLERANCIA_SESGO,
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    MLFLOW_TRACKING_URI,
    P1_FEATURES,
    PREVIOUS_CHAMPION_ALIAS,
    PROJ_ROOT,
    REGISTRY_NAME,
    REPORTS_DIR,
    SEED,
    TRAIN_END,
)
from mibici.modeling.p1_demand import TARGET, diebold_mariano_pvalue
from mibici.modeling.p1_retrain import evaluate, load_ab_eval_frame, validate_windows
from mibici.modeling.tracking import log_table, start_run

AB_REPORT_DIR = REPORTS_DIR / "ab_test"
JUSTIFICACION_MD = PROJ_ROOT / "docs" / "justificacion_version_vigente.md"
MARCA_INICIO = "<!-- RESULTADOS_AB:INICIO (generado por mibici.modeling.p1_ab_test; no editar) -->"
MARCA_FIN = "<!-- RESULTADOS_AB:FIN -->"
SEGMENTOS = ("existentes", "nuevas")


# ---------------------------------------------------------------------------
# Versiones del registry
# ---------------------------------------------------------------------------
def version_info(client: mlflow.MlflowClient, alias: str, name: str = REGISTRY_NAME["p1"]) -> dict:
    """Versión a la que apunta el alias y datos de la corrida que la produjo."""
    mv = client.get_model_version_by_alias(name, alias)
    run = client.get_run(mv.run_id)
    return {
        "alias": alias,
        "version": str(mv.version),
        "run_id": mv.run_id,
        "run_name": run.info.run_name,
        "model": run.data.tags.get("model") or run.data.params.get("estimator"),
        "train_end": run.data.params.get("train_end", TRAIN_END),
        "uri": f"models:/{name}/{mv.version}",
    }


def predecir(uri: str, X: pd.DataFrame) -> np.ndarray:
    """Predicción no negativa con el modelo del registry (misma interfaz que usa la API)."""
    model = mlflow.pyfunc.load_model(uri)
    return np.clip(np.asarray(model.predict(X), dtype=float), 0, None)


# ---------------------------------------------------------------------------
# Estadística pareada
# ---------------------------------------------------------------------------
def bootstrap_delta_wape(y: np.ndarray, err_a: np.ndarray, err_b: np.ndarray, groups: np.ndarray,
                         n_boot: int = AB_BOOTSTRAP_REPS, seed: int = SEED) -> tuple[float, float]:
    """Intervalo bootstrap al 95% de WAPE(a) - WAPE(b), remuestreando grupos completos.

    err_a y err_b son errores absolutos. Un valor positivo indica que b tiene menor error.
    Se agregan las sumas por grupo y cada réplica pondera los grupos con conteos
    multinomiales, lo que equivale a remuestrear grupos con reemplazo sin copiar filas.
    """
    df = pd.DataFrame({"g": groups, "y": y, "a": err_a, "b": err_b}).groupby("g").sum()
    n = len(df)
    if n < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    w = rng.multinomial(n, np.full(n, 1 / n), size=n_boot).astype(float)
    den = np.maximum(w @ df["y"].to_numpy(), 1)
    delta = (w @ df["a"].to_numpy() - w @ df["b"].to_numpy()) / den
    lo, hi = np.percentile(delta, [2.5, 97.5])
    return float(lo), float(hi)


def comparar(y: np.ndarray, pred_cp: np.ndarray, pred_ch: np.ndarray, base: np.ndarray,
             segment: np.ndarray, station: np.ndarray, day: np.ndarray,
             n_boot: int = AB_BOOTSTRAP_REPS, seed: int = SEED) -> dict:
    """Métricas de ambas versiones sobre las mismas filas y diferencias pareadas.

    Las diferencias se expresan como champion - challenger en WAPE (positivo: el
    challenger mejora) y como |sesgo challenger| - |sesgo champion| (positivo: empeora).
    """
    m_cp = evaluate(y, pred_cp, base, segment, prefix="champion")
    m_ch = evaluate(y, pred_ch, base, segment, prefix="challenger")
    e_cp, e_ch = np.abs(y - pred_cp), np.abs(y - pred_ch)

    ic_est = bootstrap_delta_wape(y, e_cp, e_ch, station, n_boot, seed)
    ic_dia = bootstrap_delta_wape(y, e_cp, e_ch, day, n_boot, seed)
    # El más conservador: el de límite inferior más bajo (más cercano a "no mejora").
    ic = min(ic_est, ic_dia, key=lambda t: t[0])
    res = {
        **m_cp, **m_ch,
        "delta_wape": m_cp["champion_wape"] - m_ch["challenger_wape"],
        "delta_wape_ic_estacion": list(ic_est),
        "delta_wape_ic_dia": list(ic_dia),
        "delta_wape_ic": list(ic),
        "delta_abs_sesgo": abs(m_ch["challenger_bias"]) - abs(m_cp["champion_bias"]),
        "dm_pvalue": diebold_mariano_pvalue(e_cp, e_ch, pd.Series(day)),
    }
    for seg in SEGMENTOS:
        mask = segment == seg
        if not mask.any():
            continue
        res[f"delta_wape_{seg}"] = m_cp[f"champion_wape_{seg}"] - m_ch[f"challenger_wape_{seg}"]
        res[f"delta_wape_{seg}_ic"] = list(bootstrap_delta_wape(
            y[mask], e_cp[mask], e_ch[mask], station[mask], n_boot, seed))
        # Proporción de estaciones del segmento en las que el challenger tiene menor WAPE
        g = pd.DataFrame({"s": station[mask], "y": y[mask], "a": e_cp[mask], "b": e_ch[mask]})
        g = g.groupby("s").sum()
        res[f"estaciones_mejoran_{seg}"] = float((g["b"] < g["a"]).mean())
    return res


def pp(x: float) -> str:
    """Diferencia de WAPE en puntos porcentuales, con signo."""
    return f"{x * 100:+.2f} pp"


def decidir(r: dict, min_mejora: float = AB_MIN_MEJORA_WAPE,
            tol_segmento: float = AB_TOLERANCIA_SEGMENTO,
            tol_sesgo: float = AB_TOLERANCIA_SESGO) -> dict:
    """Aplica los criterios fijados de antemano y elige la versión vigente."""
    criterios = [
        {
            "criterio": "Mejora práctica del WAPE total",
            "regla": f"WAPE champion - WAPE challenger >= {min_mejora * 100:.1f} pp",
            "valor": pp(r["delta_wape"]),
            "cumple": r["delta_wape"] >= min_mejora,
        },
        {
            "criterio": "Mejora estadísticamente significativa",
            "regla": "límite inferior del IC 95% (el más conservador) > 0",
            "valor": f"[{pp(r['delta_wape_ic'][0])}, {pp(r['delta_wape_ic'][1])}]",
            "cumple": bool(r["delta_wape_ic"][0] > 0),
        },
    ]
    for seg in SEGMENTOS:
        if f"delta_wape_{seg}" in r:
            criterios.append({
                "criterio": f"Sin degradación en estaciones {seg}",
                "regla": f"WAPE challenger - WAPE champion <= {tol_segmento * 100:.1f} pp",
                "valor": pp(-r[f"delta_wape_{seg}"]),
                "cumple": -r[f"delta_wape_{seg}"] <= tol_segmento,
            })
    criterios += [
        {
            "criterio": "Sesgo controlado",
            "regla": f"sesgo absoluto: challenger - champion <= {tol_sesgo} viajes/h",
            "valor": f"{r['delta_abs_sesgo']:+.3f}",
            "cumple": r["delta_abs_sesgo"] <= tol_sesgo,
        },
        {
            "criterio": "Supera al baseline ingenuo",
            "regla": "WAPE challenger < WAPE de la misma hora de la semana anterior",
            "valor": f"{r['challenger_wape']:.2%} vs {r['challenger_wape_baseline']:.2%}",
            "cumple": r["challenger_wape"] < r["challenger_wape_baseline"],
        },
    ]
    for c in criterios:
        c["cumple"] = bool(c["cumple"])
    gana = all(c["cumple"] for c in criterios)
    return {"vigente": CHALLENGER_ALIAS if gana else CHAMPION_ALIAS, "criterios": criterios}


# ---------------------------------------------------------------------------
# Documentación de resultados
# ---------------------------------------------------------------------------
def render_resultados(res: dict) -> str:
    """Bloque Markdown con las métricas, los criterios y la decisión."""
    cp, ch, r, d = res["champion"], res["challenger"], res["metricas"], res["decision"]

    def fila(nombre: str, a: float, b: float, fmt: str) -> str:
        if fmt.endswith("%"):
            diff = pp(b - a)
        else:
            diff = f"{b - a:{fmt if fmt.startswith('+') else '+' + fmt}}"
        return f"| {nombre} | {a:{fmt}} | {b:{fmt}} | {diff} |"

    lineas = [
        MARCA_INICIO,
        "",
        f"_Generado el {res['fecha']} con `python -m mibici.modeling.train ab-test` "
        f"(corrida de MLflow `{res['run_name']}`, run `{res['run_id']}`)._",
        "",
        "### Versiones comparadas",
        "",
        "| | Champion | Challenger |",
        "|---|---|---|",
        f"| Versión del registry | v{cp['version']} (`@{cp['alias']}`) | v{ch['version']} (`@{ch['alias']}`) |",
        f"| Corrida de origen | `{cp['run_name']}` | `{ch['run_name']}` |",
        f"| Estimador | {cp['model']} | {ch['model']} |",
        f"| Fin del entrenamiento | {cp['train_end']} | {ch['train_end']} |",
        "",
        f"Conjunto de evaluación: {AB_EVAL_START} a {AB_EVAL_END}, "
        f"{int(r['champion_rows']):,} estación-horas idénticas para ambas versiones "
        f"({int(r.get('champion_rows_existentes', 0)):,} de estaciones existentes y "
        f"{int(r.get('champion_rows_nuevas', 0)):,} de estaciones nuevas).",
        "",
        "### Métricas en el mismo conjunto",
        "",
        "| Métrica | Champion | Challenger | Diferencia (challenger - champion) |",
        "|---|---|---|---|",
        fila("WAPE total", r["champion_wape"], r["challenger_wape"], ".2%"),
    ]
    for seg in SEGMENTOS:
        if f"champion_wape_{seg}" in r:
            lineas.append(fila(f"WAPE estaciones {seg}", r[f"champion_wape_{seg}"],
                               r[f"challenger_wape_{seg}"], ".2%"))
    lineas += [
        fila("MAE (viajes/h)", r["champion_mae"], r["challenger_mae"], ".4f"),
        fila("RMSE (viajes/h)", r["champion_rmse"], r["challenger_rmse"], ".4f"),
        fila("Devianza Poisson", r["champion_poisson_deviance"],
             r["challenger_poisson_deviance"], ".4f"),
        fila("Sesgo (viajes/h)", r["champion_bias"], r["challenger_bias"], "+.4f"),
    ]
    for seg in SEGMENTOS:
        if f"champion_bias_{seg}" in r:
            lineas.append(fila(f"Sesgo estaciones {seg}", r[f"champion_bias_{seg}"],
                               r[f"challenger_bias_{seg}"], "+.4f"))
    lineas += [
        f"| WAPE baseline (semana anterior) | {r['champion_wape_baseline']:.2%} | "
        f"{r['challenger_wape_baseline']:.2%} | (mismo baseline) |",
        "",
        "### Diferencia pareada de WAPE (champion - challenger; positivo = el challenger mejora)",
        "",
        "| Alcance | Diferencia | IC 95% | Estaciones donde mejora el challenger |",
        "|---|---|---|---|",
        f"| Total, bootstrap por estación | {pp(r['delta_wape'])} | "
        f"[{pp(r['delta_wape_ic_estacion'][0])}, {pp(r['delta_wape_ic_estacion'][1])}] | |",
        f"| Total, bootstrap por día | {pp(r['delta_wape'])} | "
        f"[{pp(r['delta_wape_ic_dia'][0])}, {pp(r['delta_wape_ic_dia'][1])}] | |",
    ]
    for seg in SEGMENTOS:
        if f"delta_wape_{seg}" in r:
            ic = r[f"delta_wape_{seg}_ic"]
            lineas.append(f"| Estaciones {seg} | {pp(r[f'delta_wape_{seg}'])} | "
                          f"[{pp(ic[0])}, {pp(ic[1])}] | {r[f'estaciones_mejoran_{seg}']:.0%} |")
    lineas += [
        "",
        f"Prueba de Diebold-Mariano (H0: el challenger no reduce el error absoluto diario): "
        f"p = {r['dm_pvalue']:.4f}.",
        "",
        "### Criterios de decisión",
        "",
        "| Criterio | Regla | Resultado | ¿Cumple? |",
        "|---|---|---|---|",
    ]
    lineas += [f"| {c['criterio']} | {c['regla']} | {c['valor']} | {'Sí' if c['cumple'] else '**No**'} |"
               for c in d["criterios"]]
    fallan = [c["criterio"] for c in d["criterios"] if not c["cumple"]]
    vigente = ch if d["vigente"] == CHALLENGER_ALIAS else cp
    lineas += ["", "### Decisión", ""]
    if d["vigente"] == CHALLENGER_ALIAS:
        lineas.append(f"**Versión vigente: v{vigente['version']} (challenger).** Cumple todos los "
                      "criterios: reduce el error de forma práctica y estadísticamente significativa "
                      "sin degradar ningún segmento ni aumentar el sesgo.")
    else:
        lineas.append(f"**Versión vigente: v{vigente['version']} (se conserva el champion).** "
                      f"El challenger no cumple: {', '.join(fallan)}. Sin evidencia suficiente de "
                      "mejora, cambiar de modelo agrega riesgo y costo de despliegue sin beneficio.")
    estado = res.get("promocion")
    if estado:
        lineas += ["", f"Estado del registry: {estado}"]
    lineas += ["", MARCA_FIN]
    return "\n".join(lineas)


def actualizar_md(path: Path, bloque: str) -> None:
    """Sustituye el bloque generado entre las marcas; si no existen, lo agrega al final."""
    texto = path.read_text(encoding="utf-8") if path.exists() else ""
    if MARCA_INICIO in texto and MARCA_FIN in texto:
        antes = texto.split(MARCA_INICIO)[0]
        despues = texto.split(MARCA_FIN, 1)[1]
        texto = antes + bloque + despues
    else:
        texto = texto.rstrip() + "\n\n" + bloque + "\n"
    path.write_text(texto, encoding="utf-8")


# ---------------------------------------------------------------------------
# Registro y promoción
# ---------------------------------------------------------------------------
def promover(client: mlflow.MlflowClient, champ: dict, chall: dict,
             name: str = REGISTRY_NAME["p1"]) -> str:
    """Mueve @champion al challenger y deja la versión desplazada en @champion_anterior.

    No crea versiones nuevas: solo mueve alias, así que revertir es mover el alias de vuelta.
    """
    client.set_registered_model_alias(name, CHAMPION_ALIAS, chall["version"])
    client.set_registered_model_alias(name, PREVIOUS_CHAMPION_ALIAS, champ["version"])
    client.delete_registered_model_alias(name, CHALLENGER_ALIAS)
    return (f"`@{CHAMPION_ALIAS}` movido a v{chall['version']}; v{champ['version']} queda en "
            f"`@{PREVIOUS_CHAMPION_ALIAS}` para revertir. La API toma la nueva versión al reiniciarse.")


def run(promote: bool = False, n_boot: int = AB_BOOTSTRAP_REPS,
        doc_path: Path = JUSTIFICACION_MD) -> dict:
    """Compara champion y challenger, registra la corrida en MLflow y documenta la decisión.

    promote : si es True y el challenger gana, mueve el alias @champion. Si es False no se
        mueve ningún alias; solo se etiquetan las versiones con el resultado.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    name = REGISTRY_NAME["p1"]
    champ, chall = version_info(client, CHAMPION_ALIAS), version_info(client, CHALLENGER_ALIAS)
    if champ["version"] == chall["version"]:
        raise ValueError(f"@champion y @challenger apuntan a la misma versión (v{champ['version']})")
    # Ninguna versión puede haber visto el periodo de evaluación
    validate_windows(champ["train_end"], chall["train_end"])
    logger.info(f"A/B: champion v{champ['version']} vs challenger v{chall['version']} "
                f"en {AB_EVAL_START} a {AB_EVAL_END}")

    ev = load_ab_eval_frame(champ["train_end"])
    X = ev[P1_FEATURES]
    pred_cp, pred_ch = predecir(champ["uri"], X), predecir(chall["uri"], X)
    y = ev[TARGET].to_numpy(float)
    r = comparar(y, pred_cp, pred_ch, ev["dep_lag_168h"].to_numpy(float), ev["segmento"].to_numpy(),
                 ev["station_id"].to_numpy(), ev["ts"].dt.date.to_numpy(), n_boot=n_boot)
    decision = decidir(r)
    logger.info(f"WAPE champion {r['champion_wape']:.4f} | challenger {r['challenger_wape']:.4f} | "
                f"diferencia {r['delta_wape']:+.4f} IC95 {r['delta_wape_ic']}")

    params = {
        "champion_version": champ["version"], "challenger_version": chall["version"],
        "champion_run_id": champ["run_id"], "challenger_run_id": chall["run_id"],
        "test_start": AB_EVAL_START, "test_end": AB_EVAL_END, "train_end": champ["train_end"],
        "challenger_train_end": chall["train_end"], "ab_bootstrap_reps": n_boot,
        "ab_min_mejora_wape": AB_MIN_MEJORA_WAPE, "ab_tolerancia_segmento": AB_TOLERANCIA_SEGMENTO,
        "ab_tolerancia_sesgo": AB_TOLERANCIA_SESGO, "promote": promote,
    }
    run_name = f"p1-ab-test-v{champ['version']}-vs-v{chall['version']}"
    with start_run("P1", run_name, champ["model"], params, stage="ab_testing",
                   stage_of_work="ab_test") as active:
        escalares = {k: float(v) for k, v in r.items() if np.isscalar(v) and np.isfinite(v)}
        escalares |= {"delta_wape_ic_lo": r["delta_wape_ic"][0], "delta_wape_ic_hi": r["delta_wape_ic"][1]}
        mlflow.log_metrics(escalares)
        vigente = chall if decision["vigente"] == CHALLENGER_ALIAS else champ
        mlflow.set_tags({"ab_vigente": decision["vigente"], "ab_version_vigente": vigente["version"]})
        log_table(pd.DataFrame(decision["criterios"]), "ab_criterios.csv")

        promocion = None
        if decision["vigente"] == CHALLENGER_ALIAS and promote:
            promocion = promover(client, champ, chall)
        elif decision["vigente"] == CHALLENGER_ALIAS:
            promocion = ("el challenger ganó, pero `@champion` no se movió (falta `--promote true`). "
                         "Promover con: `python -m mibici.modeling.train ab-test --promote true`.")
        else:
            promocion = f"sin cambios: `@champion` sigue en v{champ['version']}."
        for v, rol in ((champ, "champion"), (chall, "challenger")):
            client.set_model_version_tag(name, v["version"], "ab_test_run_id", active.info.run_id)
            client.set_model_version_tag(name, v["version"], "ab_resultado",
                                         "vigente" if decision["vigente"] == rol else "no_vigente")

        res = {
            "fecha": pd.Timestamp.now(tz="America/Mexico_City").strftime("%Y-%m-%d %H:%M"),
            "run_name": run_name, "run_id": active.info.run_id,
            "champion": champ, "challenger": chall, "metricas": r, "decision": decision,
            "promocion": promocion,
        }
        mlflow.log_dict(res, "ab_test/p1_ab_test.json")
        bloque = render_resultados(res)
        mlflow.log_text(bloque, "ab_test/resultados.md")

    AB_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (AB_REPORT_DIR / "p1_ab_test.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    actualizar_md(doc_path, bloque)
    vigente = chall if decision["vigente"] == CHALLENGER_ALIAS else champ
    logger.success(f"Versión vigente: v{vigente['version']} ({decision['vigente']}). {promocion}")
    return res
