"""El entorno de MLflow Projects debe usar las mismas versiones que requirements.txt."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pins(lines):
    out = {}
    for line in lines:
        line = line.split("#")[0].strip().lstrip("- ").strip()
        if "==" in line:
            name, ver = line.split("==")
            out[name.lower()] = ver
    return out


def test_python_env_matches_requirements():
    req = _pins((ROOT / "requirements.txt").read_text().splitlines())
    env = _pins((ROOT / "python_env.yaml").read_text().splitlines())
    assert env, "python_env.yaml no declara dependencias"
    for name, ver in env.items():
        assert req.get(name) == ver, f"{name}: python_env.yaml={ver} requirements.txt={req.get(name)}"


def test_monitoring_requirements_share_pins():
    """El entorno de Evidently (.venv-monitoring) usa las mismas versiones de datos que el proyecto."""
    req = _pins((ROOT / "requirements.txt").read_text().splitlines())
    mon = _pins((ROOT / "requirements-monitoring.txt").read_text().splitlines())
    assert "evidently" in mon, "requirements-monitoring.txt debe fijar la versión de evidently"
    for name in ("pandas", "numpy", "pyarrow"):
        assert mon.get(name) == req.get(name), (
            f"{name}: requirements-monitoring.txt={mon.get(name)} requirements.txt={req.get(name)}")
