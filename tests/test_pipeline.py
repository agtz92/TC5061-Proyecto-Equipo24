"""Pruebas unitarias con PyTest para los componentes clave de MiBici.

Este archivo contiene exactamente 3 funciones de prueba que validan por separado:
1. Procesamiento de datos (mibici.dataset.clean_trips):
   - Casos normales: eliminación de columnas personales, cálculo de duracion_s,
     filtrado de estaciones e integridad de tipos de datos.
   - Casos límite: anclaje fallido en el umbral exacto (<60s vs >=60s), límites
     de duración (0s, negativa, 3h exactas y >3h), estaciones desconocidas,
     duplicados/nulos, exceso de fechas inválidas (ValueError) y DataFrame vacío.

2. Feature Engineering (mibici.features.add_calendar_features):
   - Casos normales: cálculo de hora, día de la semana, mes, fin de semana,
     festivo y quincena en días laborales estándar.
   - Casos límite: festivos oficiales fijos (Art. 74 LFT), festivos móviles
     (puentes de febrero, marzo y noviembre), festivo sexenal de cambio de Poder
     Ejecutivo (2024-10-01 vs 2025-10-01), quincenas en fin de mes ordinario y bisiesto
     (28/29 feb, 30 abr, 31 may), límites de medianoche (00:00:00) y fin de día (23:59:59),
     y registro único en límite de fin de año.

3. Entrenamiento y reproducibilidad (mibici.modeling.p1_demand):
   - Casos normales: construcción de estimadores con build_model y cálculo de
     métricas de evaluación con regression_metrics (MAE, RMSE, WAPE, Poisson deviance).
   - Casos límite: modelo no soportado (ValueError), métricas con predicciones
     perfectas (error 0) y salvaguarda contra división por cero con demanda nula.
   - Reproducibilidad del entrenamiento: dos ejecuciones independientes entrenadas
     con la misma semilla producen exactamente las mismas predicciones y métricas,
     mientras que una semilla distinta genera predicciones diferentes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from mibici.config import (
    MAX_TRIP_HOURS,
    MIN_TRIP_SECONDS_SAME_STATION,
    RAW_COLUMNS,
    SEED,
)
from mibici.dataset import clean_trips
from mibici.features import add_calendar_features
from mibici.modeling.p1_demand import build_model as build_p1_model, regression_metrics


# ===========================================================================
# 1. PRUEBA DE PROCESAMIENTO DE DATOS
# ===========================================================================
def test_data_processing():
    """Valida la función de procesamiento y limpieza clean_trips en casos normales y límite."""
    valid_stations = {2, 3, 4}
    base_ts = pd.Timestamp("2025-01-15 08:00:00")
    max_duration_s = MAX_TRIP_HOURS * 3600  # 3 horas = 10,800 segundos
    min_dock_s = MIN_TRIP_SECONDS_SAME_STATION  # 60 segundos

    # -----------------------------------------------------------------------
    # A) Caso Normal:
    #    Viajes válidos, eliminación de datos personales, cálculo de duracion_s
    #    y asignación de tipos correctos (int64, int32).
    # -----------------------------------------------------------------------
    normal_df = pd.DataFrame({
        "viaje_id": ["101", "102"],
        "usuario_id": ["USR_1", "USR_2"],
        "genero": ["M", "F"],
        "anio_nacimiento": ["1985", "1992"],
        "inicio": [base_ts, base_ts + pd.Timedelta(minutes=5)],
        "fin": [base_ts + pd.Timedelta(minutes=15), base_ts + pd.Timedelta(minutes=25)],
        "origen_id": ["2", "3"],
        "destino_id": ["3", "4"],
    })

    clean_df, stats = clean_trips(normal_df, valid_stations=valid_stations)

    # 1. Se eliminan las columnas con datos personales
    personal_cols = {"usuario_id", "genero", "anio_nacimiento"}
    assert not (personal_cols & set(clean_df.columns))

    # 2. Se conservan todas las filas válidas y se calcula duracion_s
    assert len(clean_df) == 2
    assert stats["rows_clean"] == 2
    assert stats["dropped_total"] == 0
    assert list(clean_df["duracion_s"]) == [900, 1200]

    # 3. Tipos de datos correctos
    assert clean_df["viaje_id"].dtype == np.int64
    assert clean_df["origen_id"].dtype == np.int32
    assert clean_df["destino_id"].dtype == np.int32
    assert clean_df["duracion_s"].dtype == np.int32

    # -----------------------------------------------------------------------
    # B) Casos Límite (Edge Cases):
    # -----------------------------------------------------------------------

    # Edge Case 1: Anclaje fallido en el umbral exacto (59s se descarta, 60s se conserva)
    dock_df = pd.DataFrame({
        "viaje_id": ["201", "202", "203"],
        "usuario_id": ["U1", "U2", "U3"],
        "genero": ["M", "M", "M"],
        "anio_nacimiento": ["1990", "1990", "1990"],
        "inicio": [base_ts] * 3,
        "fin": [
            base_ts + pd.Timedelta(seconds=min_dock_s - 1),  # 59s mismo origen/destino -> DESCARTADO
            base_ts + pd.Timedelta(seconds=min_dock_s),      # 60s mismo origen/destino -> CONSERVADO
            base_ts + pd.Timedelta(seconds=min_dock_s - 1),  # 59s origen != destino   -> CONSERVADO
        ],
        "origen_id": ["2", "2", "2"],
        "destino_id": ["2", "2", "3"],
    })
    clean_dock, stats_dock = clean_trips(dock_df, valid_stations=valid_stations)
    assert stats_dock["dropped_failed_dock"] == 1
    assert set(clean_dock["viaje_id"]) == {202, 203}

    # Edge Case 2: Límites de duración (duración <= 0, límite máximo de 3h y exceso)
    duration_df = pd.DataFrame({
        "viaje_id": ["301", "302", "303", "304", "305"],
        "usuario_id": ["U"] * 5,
        "genero": ["F"] * 5,
        "anio_nacimiento": ["1995"] * 5,
        "inicio": [base_ts] * 5,
        "fin": [
            base_ts - pd.Timedelta(seconds=10),               # Negativa -> descartada
            base_ts,                                          # Duración 0 -> descartada
            base_ts + pd.Timedelta(seconds=max_duration_s),   # Exactamente 3h -> CONSERVADA
            base_ts + pd.Timedelta(seconds=max_duration_s + 1),  # > 3h -> descartada
            base_ts + pd.Timedelta(minutes=10),               # Normal -> CONSERVADA
        ],
        "origen_id": ["2"] * 5,
        "destino_id": ["3"] * 5,
    })
    clean_dur, stats_dur = clean_trips(duration_df, valid_stations=valid_stations)
    assert stats_dur["dropped_duration"] == 3
    assert set(clean_dur["viaje_id"]) == {303, 305}
    assert clean_dur.loc[clean_dur["viaje_id"] == 303, "duracion_s"].iloc[0] == max_duration_s

    # Edge Case 3: Estaciones fuera de la nomenclatura válida
    unknown_station_df = pd.DataFrame({
        "viaje_id": ["401", "402"],
        "usuario_id": ["U", "U"],
        "genero": ["M", "M"],
        "anio_nacimiento": ["1990", "1990"],
        "inicio": [base_ts, base_ts],
        "fin": [base_ts + pd.Timedelta(minutes=10), base_ts + pd.Timedelta(minutes=10)],
        "origen_id": ["2", "9999"],  # 9999 no existe en valid_stations
        "destino_id": ["8888", "3"],  # 8888 no existe en valid_stations
    })
    clean_unk, stats_unk = clean_trips(unknown_station_df, valid_stations=valid_stations)
    assert len(clean_unk) == 0
    assert stats_unk["dropped_unknown_station"] == 2

    # Edge Case 4: Duplicados de viaje_id y valores nulos/corruptos en identificadores
    dups_nulls_df = pd.DataFrame({
        "viaje_id": ["501", "501", None, "texto_invalido"],
        "usuario_id": ["U"] * 4,
        "genero": ["F"] * 4,
        "anio_nacimiento": ["1991"] * 4,
        "inicio": [base_ts] * 4,
        "fin": [base_ts + pd.Timedelta(minutes=10)] * 4,
        "origen_id": ["2", "2", "2", "2"],
        "destino_id": ["3", "3", "3", "3"],
    })
    clean_dups, _ = clean_trips(dups_nulls_df, valid_stations=valid_stations)
    assert len(clean_dups) == 1
    assert clean_dups.iloc[0]["viaje_id"] == 501

    # Edge Case 5: Exceso de fechas inválidas que supera MAX_UNPARSED_DATES_PCT
    bad_dates_df = pd.DataFrame({
        "viaje_id": ["601", "602"],
        "usuario_id": ["U", "U"],
        "genero": ["M", "M"],
        "anio_nacimiento": ["1990", "1990"],
        "inicio": pd.Series([base_ts, pd.NaT]),  # 50% de fechas nulas
        "fin": pd.Series([base_ts + pd.Timedelta(minutes=10), pd.NaT]),
        "origen_id": ["2", "2"],
        "destino_id": ["3", "3"],
    })
    with pytest.raises(ValueError, match="fechas no reconocidas"):
        clean_trips(bad_dates_df, valid_stations=valid_stations)

    # Edge Case 6: DataFrame vacío (0 filas)
    empty_df = pd.DataFrame(columns=RAW_COLUMNS)
    empty_df["inicio"] = pd.to_datetime(empty_df["inicio"])
    empty_df["fin"] = pd.to_datetime(empty_df["fin"])
    clean_empty, stats_empty = clean_trips(empty_df, valid_stations=valid_stations)
    assert len(clean_empty) == 0
    assert stats_empty["rows_clean"] == 0


# ===========================================================================
# 2. PRUEBA DE FEATURE ENGINEERING
# ===========================================================================
def test_feature_engineering():
    """Valida la ingeniería de variables add_calendar_features en casos normales y límite."""
    # -----------------------------------------------------------------------
    # A) Caso Normal:
    #    Día hábil laboral típico: Miércoles 11 de junio de 2025 a las 14:30.
    # -----------------------------------------------------------------------
    normal_ts = pd.Timestamp("2025-06-11 14:30:00")  # Miércoles (dow=2), mes=6, hora=14
    df_normal = pd.DataFrame({"ts": [normal_ts]})
    out_normal = add_calendar_features(df_normal, ts_col="ts")

    row = out_normal.iloc[0]
    assert row["hour"] == 14
    assert row["dow"] == 2
    assert row["month"] == 6
    assert row["is_weekend"] == 0
    assert row["is_holiday"] == 0
    assert row["is_quincena"] == 0

    # -----------------------------------------------------------------------
    # B) Casos Límite (Edge Cases):
    # -----------------------------------------------------------------------

    # Edge Case 1: Festivos oficiales obligatorios fijos (Art. 74 LFT)
    # 1-ene, 1-may, 16-sep, 25-dic
    fixed_holidays = pd.to_datetime([
        "2025-01-01 10:00:00",  # Año Nuevo
        "2025-05-01 10:00:00",  # Día del Trabajo
        "2025-09-16 10:00:00",  # Día de la Independencia
        "2025-12-25 10:00:00",  # Navidad
    ])
    df_fixed = pd.DataFrame({"ts": fixed_holidays})
    out_fixed = add_calendar_features(df_fixed, ts_col="ts")
    assert (out_fixed["is_holiday"] == 1).all()

    # Edge Case 2: Festivos móviles (puentes laborales por LFT)
    # - 1er lunes de febrero (Constitución): 2025-02-03
    # - 3er lunes de marzo (Natalicio de Juárez): 2025-03-17
    # - 3er lunes de noviembre (Revolución): 2025-11-17
    floating_holidays = pd.to_datetime([
        "2025-02-03 12:00:00",
        "2025-03-17 12:00:00",
        "2025-11-17 12:00:00",
    ])
    df_floating = pd.DataFrame({"ts": floating_holidays})
    out_floating = add_calendar_features(df_floating, ts_col="ts")
    assert (out_floating["is_holiday"] == 1).all()

    # Edge Case 3: Festivo sexenal de cambio de Poder Ejecutivo (1 de octubre)
    # En 2024 aplica, en 2025 NO aplica
    presidential_ts = pd.to_datetime(["2024-10-01 12:00:00", "2025-10-01 12:00:00"])
    df_pres = pd.DataFrame({"ts": presidential_ts})
    out_pres = add_calendar_features(df_pres, ts_col="ts")
    assert out_pres.iloc[0]["is_holiday"] == 1  # 2024: sí es festivo
    assert out_pres.iloc[1]["is_holiday"] == 0  # 2025: día hábil normal

    # Edge Case 4: Quincenas y fin de mes en año bisiesto y meses de 28, 30 y 31 días
    quincena_ts = pd.to_datetime([
        "2025-03-15 09:00:00",  # Día 15 -> quincena
        "2025-03-14 09:00:00",  # Día 14 -> no quincena
        "2024-02-29 09:00:00",  # Fin de mes bisiesto (29 de feb) -> quincena
        "2025-02-28 09:00:00",  # Fin de mes ordinario (28 de feb) -> quincena
        "2025-04-30 09:00:00",  # Fin de mes con 30 días -> quincena
        "2025-05-31 09:00:00",  # Fin de mes con 31 días -> quincena
    ])
    df_quincena = pd.DataFrame({"ts": quincena_ts})
    out_quincena = add_calendar_features(df_quincena, ts_col="ts")
    assert list(out_quincena["is_quincena"]) == [1, 0, 1, 1, 1, 1]

    # Edge Case 5: Límites de horas (medianoche exacta 00:00:00 y fin de día 23:59:59)
    boundary_hours = pd.to_datetime(["2025-07-01 00:00:00", "2025-07-01 23:59:59"])
    df_hours = pd.DataFrame({"ts": boundary_hours})
    out_hours = add_calendar_features(df_hours, ts_col="ts")
    assert out_hours.iloc[0]["hour"] == 0
    assert out_hours.iloc[1]["hour"] == 23

    # Edge Case 6: Fin de semana
    weekends = pd.to_datetime(["2025-06-14 10:00:00", "2025-06-15 10:00:00"])  # Sábado y Domingo
    df_weekends = pd.DataFrame({"ts": weekends})
    out_weekends = add_calendar_features(df_weekends, ts_col="ts")
    assert (out_weekends["is_weekend"] == 1).all()

    # Edge Case 7: DataFrame con un solo registro en límite de fin de año
    single_ts = pd.to_datetime(["2024-12-31 23:59:59"])
    df_single = pd.DataFrame({"ts": single_ts})
    out_single = add_calendar_features(df_single, ts_col="ts")
    assert len(out_single) == 1
    assert out_single.iloc[0]["hour"] == 23
    assert out_single.iloc[0]["month"] == 12
    assert out_single.iloc[0]["is_quincena"] == 1


# ===========================================================================
# 3. PRUEBA DE ENTRENAMIENTO Y REPRODUCIBILIDAD
# ===========================================================================
def test_model_training_and_reproducibility():
    """Valida la construcción, métricas y la reproducibilidad determinista del entrenamiento."""
    # -----------------------------------------------------------------------
    # A) Caso Normal y Casos Límite en Funciones de Modelado
    # -----------------------------------------------------------------------
    hp = {
        "n_estimators": 10,
        "max_depth": 4,
        "min_samples_leaf": 2,
        "alpha": 0.01,
        "max_iter": 50,
        "learning_rate": 0.1,
    }

    # Caso normal: build_model crea los estimadores soportados
    model_rf = build_p1_model("rf", hp)
    assert isinstance(model_rf, RandomForestRegressor)
    assert model_rf.random_state == SEED

    # Caso límite: modelo no soportado debe lanzar ValueError
    with pytest.raises(ValueError, match="Modelo no soportado"):
        build_p1_model("modelo_inexistente", hp)

    # Caso normal: regression_metrics calcula MAE, RMSE, WAPE y Poisson Deviance
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 38.0])
    metrics = regression_metrics(y_true, y_pred, prefix="val")
    assert metrics["val_mae"] == pytest.approx(2.25)
    assert metrics["val_rmse"] > 0
    assert metrics["val_wape"] == pytest.approx(9.0 / 100.0)
    assert "val_poisson_deviance" in metrics

    # Caso límite métricas: predicciones perfectas (error 0)
    metrics_perfect = regression_metrics(y_true, y_true, prefix="test")
    assert metrics_perfect["test_mae"] == 0.0
    assert metrics_perfect["test_rmse"] == 0.0
    assert metrics_perfect["test_wape"] == 0.0

    # Caso límite métricas: suma real igual a 0
    zeros_true = np.array([0.0, 0.0, 0.0])
    zeros_pred = np.array([0.1, 0.0, 0.2])
    metrics_zero = regression_metrics(zeros_true, zeros_pred, prefix="zero")
    assert not np.isnan(metrics_zero["zero_wape"])
    assert not np.isinf(metrics_zero["zero_wape"])

    # -----------------------------------------------------------------------
    # B) Reproducibilidad del Entrenamiento (Requisito Clave):
    #    Dos ejecuciones independientes con la misma semilla fija deben
    #    producir exactamente las mismas predicciones y métricas.
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(12345)
    n_samples, n_features = 200, 8
    X_train = rng.normal(size=(n_samples, n_features))
    # Genera variable objetivo no lineal
    y_train = np.maximum(
        0.0,
        3.0 * X_train[:, 0]
        + 1.5 * (X_train[:, 1] ** 2)
        - 2.0 * X_train[:, 2]
        + rng.normal(0, 0.5, size=n_samples),
    )

    X_test = rng.normal(size=(50, n_features))
    y_test = np.maximum(
        0.0,
        3.0 * X_test[:, 0]
        + 1.5 * (X_test[:, 1] ** 2)
        - 2.0 * X_test[:, 2]
        + rng.normal(0, 0.5, size=50),
    )

    target_seed = 42

    # Ejecución 1: entrenar modelo con target_seed
    model_run_1 = RandomForestRegressor(
        n_estimators=15,
        max_depth=5,
        min_samples_leaf=2,
        max_features=0.7,
        random_state=target_seed,
    )
    model_run_1.fit(X_train, y_train)
    preds_run_1 = model_run_1.predict(X_test)
    metrics_run_1 = regression_metrics(y_test, preds_run_1, prefix="test")

    # Ejecución 2: entrenar modelo idéntico desde cero con la MISMA target_seed
    model_run_2 = RandomForestRegressor(
        n_estimators=15,
        max_depth=5,
        min_samples_leaf=2,
        max_features=0.7,
        random_state=target_seed,
    )
    model_run_2.fit(X_train, y_train)
    preds_run_2 = model_run_2.predict(X_test)
    metrics_run_2 = regression_metrics(y_test, preds_run_2, prefix="test")

    # 1. Las predicciones deben ser exactamente idénticas valor por valor
    np.testing.assert_array_equal(
        preds_run_1,
        preds_run_2,
        err_msg="Las predicciones difieren entre dos ejecuciones con la misma semilla",
    )

    # 2. Las métricas de desempeño deben ser idénticas
    assert metrics_run_1 == metrics_run_2

    # 3. Control negativo / sensibilidad:
    #    Una semilla diferente con submuestreo de variables estocástico
    #    debe producir predicciones distintas, garantizando que el generador pseudoaleatorio
    #    está activo y controlando la reproducibilidad.
    model_diff_seed = RandomForestRegressor(
        n_estimators=15,
        max_depth=5,
        min_samples_leaf=2,
        max_features=0.7,
        random_state=999,
    )
    model_diff_seed.fit(X_train, y_train)
    preds_diff = model_diff_seed.predict(X_test)

    assert not np.array_equal(preds_run_1, preds_diff), (
        "El modelo no responde a cambios en la semilla pseudoaleatoria"
    )
