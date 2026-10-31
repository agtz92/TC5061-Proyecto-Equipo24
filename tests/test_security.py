"""Chequeos básicos de seguridad del repositorio.

Verifican que el repositorio no exponga credenciales ni datos sensibles:
- .gitignore excluye .env, datos, bases de MLflow, modelos y llaves.
- Ningún archivo con secretos (.env, *.pem, *.key, mlflow.db...) está versionado.
- Los archivos versionados no contienen llaves, tokens ni contraseñas escritas.
- La configuración sensible (URI de MLflow) se lee de variables de entorno y no
  aparece escrita en el código.
- Las funciones que arman SQL rechazan valores que no son fechas ni columnas.

Se corren con el resto de las pruebas (pytest tests) o junto con bandit y pip-audit
con `python security/run_checks.py`.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess  # nosec B404
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
HAS_GIT = shutil.which("git") is not None and (ROOT / ".git").exists()
needs_git = pytest.mark.skipif(not HAS_GIT, reason="requiere git y el repositorio clonado")


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)  # nosec


def tracked_files() -> list[str]:
    return [f for f in git("ls-files", "-z").stdout.split("\0") if f]


# Archivos binarios o generados que no se revisan línea por línea.
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".docx", ".xlsx", ".pptx", ".parquet"}
SKIP_PATHS = {"tests/test_security.py"}  # contiene los patrones de ejemplo


# ---------------------------------------------------------------------------
# .gitignore y archivos versionados
# ---------------------------------------------------------------------------
@needs_git
@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        "data/raw/datos_abiertos_2025_01.csv",
        "mlflow.db",
        "mlruns/0/meta.yaml",
        "mlartifacts/1/model.pkl",
        "models/p1/model.pkl",
        "llave_privada.pem",
        "servidor.key",
        ".venv/bin/python",
    ],
)
def test_gitignore_excluye_archivos_sensibles(path):
    res = git("check-ignore", "-q", "--no-index", path)
    assert res.returncode == 0, f"{path} no está cubierto por .gitignore"


@needs_git
def test_env_example_si_se_versiona():
    """La plantilla sí se versiona (sin secretos) para que cada quien cree su .env."""
    assert git("check-ignore", "-q", "--no-index", ".env.example").returncode == 1
    assert ".env.example" in tracked_files()


SENSITIVE_NAME = re.compile(
    r"(^|/)(\.env(\..+)?|.*\.(pem|key|p12|pfx|db|sqlite3?|pkl|joblib)|id_rsa.*|credentials.*\.json|\.netrc|\.pypirc)$",
    re.IGNORECASE,
)


@needs_git
def test_no_hay_archivos_sensibles_versionados():
    offenders = [f for f in tracked_files() if SENSITIVE_NAME.search(f) and f != ".env.example"]
    assert not offenders, f"Archivos sensibles en Git: {offenders}"


# ---------------------------------------------------------------------------
# Secretos escritos en el contenido
# ---------------------------------------------------------------------------
SECRET_PATTERNS = {
    "llave privada": re.compile(r"-----BEGIN (RSA |EC |OPENSSH |DSA |)PRIVATE KEY-----"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "token de GitHub": re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b|\bgithub_pat_[A-Za-z0-9_]{60,}"),
    "API key de Google": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "token de Slack": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    "API key OpenAI/Anthropic": re.compile(r"\bsk-(ant-)?[A-Za-z0-9_\-]{32,}"),
    "contraseña escrita": re.compile(
        r"""(?i)\b(password|passwd|contrase[ñn]a|secret|api_?key|token)\b\s*[:=]\s*["'][^"'\s]{6,}["']"""
    ),
    "URI con usuario y contraseña": re.compile(r"\b[a-z][a-z0-9+.-]*://[^/\s:@'\"]+:[^/\s@'\"]+@"),
}


def text_files():
    for rel in tracked_files():
        path = ROOT / rel
        if rel in SKIP_PATHS or path.suffix.lower() in BINARY_SUFFIXES or not path.is_file():
            continue
        try:
            yield rel, path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue


@needs_git
def test_no_hay_secretos_en_archivos_versionados():
    hallazgos = []
    for rel, text in text_files():
        for nombre, patron in SECRET_PATTERNS.items():
            for match in patron.finditer(text):
                linea = text.count("\n", 0, match.start()) + 1
                hallazgos.append(f"{rel}:{linea} ({nombre})")
    assert not hallazgos, "Posibles secretos versionados:\n" + "\n".join(hallazgos)


def test_patrones_detectan_secretos_de_ejemplo():
    """Control positivo: los patrones sí reconocen secretos (valores inventados)."""
    ejemplos = [
        "-----BEGIN RSA PRIVATE KEY-----",
        "AKIA" + "ABCDEFGHIJKLMNOP",
        "ghp_" + "a" * 36,
        'password = "s3cr3t-valor"',
        "postgresql://usuario:clave@servidor/db",
    ]
    for ejemplo in ejemplos:
        assert any(p.search(ejemplo) for p in SECRET_PATTERNS.values()), ejemplo
    assert not any(p.search('MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")') for p in SECRET_PATTERNS.values())


# ---------------------------------------------------------------------------
# Configuración por variables de entorno
# ---------------------------------------------------------------------------
HARDCODED_URI = re.compile(r"""set_tracking_uri\(\s*["']|["']https?://(localhost|127\.0\.0\.1)[:/]""")


@needs_git
def test_uri_de_mlflow_no_esta_escrita_en_el_codigo():
    """Solo config.py define el valor por defecto; el resto lo importa."""
    offenders = []
    for rel, text in text_files():
        if rel.endswith(".py") and rel != "mibici/config.py" and HARDCODED_URI.search(text):
            offenders.append(rel)
    assert not offenders, f"URI de MLflow escrita en: {offenders} (usar mibici.config.MLFLOW_TRACKING_URI)"


def test_config_lee_mlflow_de_variables_de_entorno():
    env = {**os.environ, "MLFLOW_TRACKING_URI": "http://servidor-prueba:9999", "MIBICI_TEAM": "equipo-prueba"}
    code = "from mibici import config as c; print(c.MLFLOW_TRACKING_URI); print(c.MLFLOW_EXPERIMENT)"
    res = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)  # nosec
    assert res.returncode == 0, res.stderr
    uri, experiment = res.stdout.strip().splitlines()[-2:]
    assert uri == "http://servidor-prueba:9999"
    assert experiment == "mibici-equipo-prueba"


def test_env_example_no_contiene_secretos():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        if re.search(r"(?i)password|secret|token|key", key):
            assert value.strip() == "", f"{key} debe ir vacío en .env.example"


@needs_git
def test_servidor_mlflow_documentado_solo_en_localhost():
    """El servidor de MLflow no debe exponerse a la red (--host 0.0.0.0)."""
    offenders = [rel for rel, text in text_files() if re.search(r"--host\s+0\.0\.0\.0", text)]
    assert not offenders, f"Servidor expuesto a la red en: {offenders}"


# ---------------------------------------------------------------------------
# Validación de valores que se insertan en SQL (bandit B608)
# ---------------------------------------------------------------------------
def test_sql_timestamp_normaliza_fechas_validas():
    from mibici.features import sql_timestamp

    assert sql_timestamp("2025-06-30 23:00") == "2025-06-30 23:00:00"
    assert sql_timestamp("2022-01-01") == "2022-01-01 00:00:00"


@pytest.mark.parametrize(
    "valor", ["2025-01-01' OR 1=1 --", "'; DROP TABLE features; --", "", None, "no es fecha"]
)
def test_sql_timestamp_rechaza_inyeccion(valor):
    from mibici.features import sql_timestamp

    with pytest.raises(ValueError):
        sql_timestamp(valor)


def test_sql_columns_acepta_identificadores_y_rechaza_sql():
    from mibici.features import sql_columns

    assert sql_columns(["hour", "dep_lag_24h", "ts"]) == ["hour", "dep_lag_24h", "ts"]
    for malo in (["hour; DROP TABLE x"], ["* FROM read_csv('x')"], ["a b"], ["1col"]):
        with pytest.raises(ValueError):
            sql_columns(malo)
