"""Pruebas del A/B testing de P1 (mibici.modeling.p1_ab_test).

No requieren servidor de MLflow ni datos: validan la estadística pareada, la regla de
decisión y la actualización del documento de justificación.
"""

from __future__ import annotations

import numpy as np
import pytest

from mibici.config import CHALLENGER_ALIAS, CHAMPION_ALIAS
from mibici.modeling.p1_ab_test import (
    MARCA_FIN,
    MARCA_INICIO,
    actualizar_md,
    bootstrap_delta_wape,
    comparar,
    decidir,
    render_resultados,
)


def datos_sinteticos(mejora_challenger: float, seed: int = 0, n_est: int = 40, n_dias: int = 30):
    """Panel estación x hora con dos predicciones. mejora_challenger reduce su ruido."""
    rng = np.random.default_rng(seed)
    station = np.repeat(np.arange(n_est), n_dias * 24)
    day = np.tile(np.repeat(np.arange(n_dias), 24), n_est)
    mu = rng.gamma(2.0, 0.8, n_est)[station]
    y = rng.poisson(mu).astype(float)
    pred_cp = np.clip(mu + rng.normal(0, 0.8, len(y)), 0, None)
    pred_ch = np.clip(mu + rng.normal(0, 0.8 * (1 - mejora_challenger), len(y)), 0, None)
    base = rng.poisson(mu).astype(float)
    segment = np.where(station < n_est * 0.8, "existentes", "nuevas")
    return y, pred_cp, pred_ch, base, segment, station, day


def test_bootstrap_detecta_mejora_y_es_determinista():
    y, cp, ch, *_, station, day = datos_sinteticos(0.6)
    e_cp, e_ch = np.abs(y - cp), np.abs(y - ch)
    lo, hi = bootstrap_delta_wape(y, e_cp, e_ch, station, n_boot=500)
    assert 0 < lo < hi
    assert (lo, hi) == bootstrap_delta_wape(y, e_cp, e_ch, station, n_boot=500)


def test_bootstrap_modelos_identicos_incluye_cero():
    y, cp, *_, station, day = datos_sinteticos(0.0)
    e = np.abs(y - cp)
    lo, hi = bootstrap_delta_wape(y, e, e, day, n_boot=200)
    assert lo == pytest.approx(0) and hi == pytest.approx(0)


def test_comparar_mismas_filas_y_segmentos():
    y, cp, ch, base, seg, station, day = datos_sinteticos(0.6)
    r = comparar(y, cp, ch, base, seg, station, day, n_boot=200)
    assert r["champion_rows"] == r["challenger_rows"] == len(y)
    assert r["delta_wape"] == pytest.approx(r["champion_wape"] - r["challenger_wape"])
    # El IC usado para decidir es el más conservador de los dos
    assert r["delta_wape_ic"][0] == min(r["delta_wape_ic_estacion"][0], r["delta_wape_ic_dia"][0])
    for s in ("existentes", "nuevas"):
        assert 0 <= r[f"estaciones_mejoran_{s}"] <= 1
    assert r["dm_pvalue"] < 0.05


def test_decidir_promueve_challenger_claramente_mejor():
    r = comparar(*datos_sinteticos(0.6), n_boot=200)
    d = decidir(r)
    assert d["vigente"] == CHALLENGER_ALIAS
    assert all(c["cumple"] for c in d["criterios"])


def test_decidir_conserva_champion_sin_mejora():
    r = comparar(*datos_sinteticos(0.0), n_boot=200)
    d = decidir(r)
    assert d["vigente"] == CHAMPION_ALIAS
    assert any(not c["cumple"] for c in d["criterios"])


def test_decidir_guardarrail_de_segmento():
    """Una mejora total no basta si un segmento empeora más que la tolerancia."""
    r = comparar(*datos_sinteticos(0.6), n_boot=200)
    r["delta_wape_nuevas"] = -0.02  # el challenger empeora 2 pp en estaciones nuevas
    d = decidir(r)
    assert d["vigente"] == CHAMPION_ALIAS
    fallan = [c["criterio"] for c in d["criterios"] if not c["cumple"]]
    assert fallan == ["Sin degradación en estaciones nuevas"]


def test_render_y_actualizar_md(tmp_path):
    r = comparar(*datos_sinteticos(0.6), n_boot=100)
    info = {"version": "1", "alias": "champion", "run_name": "a", "model": "hgb",
            "train_end": "2025-06-30 23:00"}
    res = {"fecha": "2026-11-15 12:00", "run_name": "p1-ab-test-v1-vs-v2", "run_id": "x",
           "champion": info, "challenger": {**info, "version": "2", "alias": "challenger"},
           "metricas": r, "decision": decidir(r), "promocion": "sin cambios"}
    bloque = render_resultados(res)
    assert bloque.startswith(MARCA_INICIO) and bloque.endswith(MARCA_FIN)
    assert "Versión vigente: v2" in bloque

    doc = tmp_path / "doc.md"
    doc.write_text(f"# Título\n\nTexto fijo.\n\n{MARCA_INICIO}\nviejo\n{MARCA_FIN}\n\n## Final\n",
                   encoding="utf-8")
    actualizar_md(doc, bloque)
    actualizar_md(doc, bloque)  # idempotente: no duplica el bloque
    texto = doc.read_text(encoding="utf-8")
    assert texto.count(MARCA_INICIO) == 1 and "viejo" not in texto
    assert texto.startswith("# Título") and texto.rstrip().endswith("## Final")
