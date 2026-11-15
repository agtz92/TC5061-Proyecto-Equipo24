"""Pruebas de la versión alternativa de P1 (mibici.modeling.p1_retrain).

No requieren servidor de MLflow ni datos: validan las tres piezas que garantizan que
el challenger sea comparable con el champion.

1. cast_hyperparam: los hiperparámetros leídos de MLflow (texto) recuperan su tipo.
2. validate_windows: se rechazan ventanas que no agregan datos o que se traslapan con
   el conjunto de evaluación del A/B testing (fuga de información).
3. evaluate: las métricas por segmento (estaciones existentes y nuevas) son correctas.
"""

from __future__ import annotations

import numpy as np
import pytest

from mibici.config import AB_EVAL_START, HYPERPARAMS, RETRAIN_TRAIN_END, TRAIN_END
from mibici.modeling.p1_retrain import cast_hyperparam, evaluate, validate_windows


def test_cast_hyperparam():
    assert cast_hyperparam("300", HYPERPARAMS["max_iter"]) == 300
    assert cast_hyperparam("12.0", HYPERPARAMS["max_depth"]) == 12
    assert cast_hyperparam("0.05", HYPERPARAMS["learning_rate"]) == pytest.approx(0.05)
    assert cast_hyperparam("None", HYPERPARAMS["max_depth"]) is None
    assert isinstance(cast_hyperparam("50", HYPERPARAMS["min_samples_leaf"]), int)


def test_validate_windows():
    # Configuración del proyecto: válida
    validate_windows(TRAIN_END, RETRAIN_TRAIN_END)
    # El challenger no agrega datos posteriores al champion
    with pytest.raises(ValueError, match="posterior"):
        validate_windows(TRAIN_END, TRAIN_END)
    # El entrenamiento invade el conjunto de evaluación A/B
    with pytest.raises(ValueError, match="traslapa"):
        validate_windows(TRAIN_END, AB_EVAL_START)
    # Ventana de evaluación invertida
    with pytest.raises(ValueError, match="inválida"):
        validate_windows(TRAIN_END, RETRAIN_TRAIN_END, "2026-09-30", "2026-07-01")


def test_evaluate_por_segmento():
    y = np.array([2.0, 2.0, 1.0, 1.0])
    pred = np.array([2.0, 2.0, 0.0, 2.0])
    base = np.array([1.0, 3.0, 1.0, 1.0])
    seg = np.array(["existentes", "existentes", "nuevas", "nuevas"])
    m = evaluate(y, pred, base, seg)

    # Total: error absoluto 2 sobre 6 viajes; sesgo nulo (una sub y una sobrepredicción)
    assert m["ab_eval_wape"] == pytest.approx(2 / 6)
    assert m["ab_eval_bias"] == pytest.approx(0.0)
    assert m["ab_eval_wape_baseline"] == pytest.approx(2 / 6)
    # Por segmento: las existentes sin error; las nuevas con error igual a su demanda
    assert m["ab_eval_wape_existentes"] == pytest.approx(0.0)
    assert m["ab_eval_wape_nuevas"] == pytest.approx(1.0)
    assert m["ab_eval_rows_nuevas"] == 2

    # Sin estaciones nuevas no se generan métricas de ese segmento
    m2 = evaluate(y[:2], pred[:2], base[:2], seg[:2])
    assert "ab_eval_wape_nuevas" not in m2
