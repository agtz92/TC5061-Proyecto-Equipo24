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
