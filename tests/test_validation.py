"""Pruebas de integracion de la preparacion y los reportes Deepchecks."""

import numpy as np
import pandas as pd
import pytest

import mibici.validation as validation
from mibici.validation import (
    _save_integrity_report,
    _save_split_report,
    _supervised_dataset,
    validate,
)


def _sample_frame(start: str, rows: int, offset: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    values = rng.normal(loc=offset, size=rows)
    return pd.DataFrame(
        {
            "ts": pd.date_range(start, periods=rows, freq="h"),
            "feature": values,
            "target": values * 2 + rng.normal(size=rows),
        }
    )


def test_supervised_dataset_uses_explicit_feature_and_label_metadata():
    dataset = _supervised_dataset(_sample_frame("2025-01-01", 20), "target", ["feature"])

    assert dataset.features == ["feature"]
    assert dataset.label_name == "target"
    assert dataset.datetime_name == "ts"


def test_deepchecks_suites_save_html_reports(tmp_path):
    train = _supervised_dataset(_sample_frame("2025-01-01", 100), "target", ["feature"])
    test = _supervised_dataset(_sample_frame("2026-01-01", 100, offset=0.5), "target", ["feature"])

    integrity_path = _save_integrity_report(train, tmp_path / "integrity.html")
    split_path = _save_split_report(train, test, tmp_path / "train_test.html")

    assert integrity_path.is_file()
    assert split_path.is_file()
    assert "<html" in integrity_path.read_text(encoding="utf-8").lower()
    assert "<html" in split_path.read_text(encoding="utf-8").lower()


def test_validate_rejects_unknown_question():
    with pytest.raises(ValueError, match="question debe ser"):
        validate("p4")


def test_p3_profiles_are_validated_and_reported(tmp_path, monkeypatch):
    rows = 40
    profiles = pd.DataFrame(
        {
            "station_id": np.arange(rows),
            "municipio": ["centro", "zapopan", "tlaquepaque", "centro"] * 10,
            "lat": 20.60 + np.arange(rows) / 10_000,
            "lon": -103.40 + np.arange(rows) / 10_000,
            "capacity": 10 + np.arange(rows) % 20,
            "ratio_arr_dep": 0.5 + np.arange(rows) / 100,
        }
    )
    profile_path = tmp_path / "profiles.parquet"
    profiles.to_parquet(profile_path, index=False)
    reports_path = tmp_path / "reports"
    monkeypatch.setattr(validation, "PROFILES_PARQUET", profile_path)
    monkeypatch.setattr(validation, "REPORTS_DIR", reports_path)

    report = validation._validate_profiles()

    assert report == reports_path / "p3_integridad.html"
    assert report.is_file()