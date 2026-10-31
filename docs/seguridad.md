# Revisión de seguridad del repositorio

Revisión hecha el 2026-10-30 sobre `main` (commit `b83683c`). Cubre tres frentes: que el repositorio no
exponga credenciales ni datos sensibles, análisis estático del código con **bandit** y vulnerabilidades
conocidas en dependencias con **pip-audit**. Todo se vuelve a correr con un solo comando:

```bash
python security/run_checks.py      # o: make security
```

## 1. Credenciales, datos sensibles y variables de entorno

| Revisión | Resultado |
|---|---|
| Archivos sensibles versionados (`.env`, llaves, `mlflow.db`, modelos) en todo el historial de Git | Ninguno |
| Llaves, tokens o contraseñas escritas (detect-secrets sobre los archivos actuales y búsqueda de patrones en todo el historial) | Ninguno |
| `.gitignore` | Ya excluía `data/`, `mlruns/`, `mlartifacts/`, `*.db`, modelos, `.env` y `*.pem`/`*.key`. Se agregaron `*.p12`, `*.pfx`, `id_rsa*`, `id_ed25519*`, `credentials*.json`, `service-account*.json`, `*secret*.json`, `.netrc` y `*.sqlite*` |
| Configuración sensible | `mibici/config.py` lee `MLFLOW_TRACKING_URI`, `MIBICI_TEAM` y `MLFLOW_EXPERIMENT` del `.env` (no versionado). **Hallazgo:** `guardarmodelos.py` y `verificarmodelos.py` tenían la URI del servidor escrita en el código; ahora la toman de `config.py` |
| Plantilla `.env.example` | Versionada, sin valores secretos; indica dónde irían usuario y contraseña de MLflow si el servidor usa autenticación |
| Datos personales | `Usuario_Id`, `Genero` y `Año_de_nacimiento` se eliminan en la ingesta (`PERSONAL_COLUMNS`); los datos crudos no se suben a Git |
| Servidor de MLflow | Documentado con `--host 127.0.0.1` (no se expone a la red) |

`tests/test_security.py` convierte estas revisiones en pruebas de PyTest para que no se rompan en el futuro.

## 2. Bandit (análisis estático)

Primera corrida: **16 avisos** (12 de severidad media, 4 baja), todos revisados.

| Regla | Dónde | Decisión |
|---|---|---|
| B608 SQL armado con f-strings (12) | `mibici/features.py`, `mibici/modeling/p2_risk.py` | DuckDB no acepta parámetros dentro de `read_parquet()` ni en nombres de columna. Se agregaron `sql_timestamp()` y `sql_columns()` para validar fechas y columnas que llegan por CLI u otros módulos; las consultas que solo usan rutas y constantes de `config.py` llevan `# nosec B608` |
| B404/B603/B607 `subprocess` con `git` | `mibici/modeling/tracking.py` | Argumentos fijos (`git rev-parse HEAD`), sin entrada externa: `# nosec` |
| B110 `try/except/pass` | `mibici/config.py` | Reconfigurar la consola en Windows es opcional: `# nosec B110` |

Resultado final: **0 avisos** (18 suprimidos con justificación en el código). La configuración vive en
`[tool.bandit]` de `pyproject.toml`.

## 3. pip-audit (dependencias)

Primera corrida sobre `requirements.txt`: **48 avisos en 7 paquetes**. Antes de auditar hubo que corregir
un error: `requirements.txt` pedía `setuptool<82` (sin "s"), un paquete que **no existe en PyPI**. La
instalación fallaba y el nombre quedaba expuesto a *typosquatting*. Se corrigió a `setuptools<82`.

| Paquete | Antes | Después | Avisos |
|---|---|---|---|
| requests | 2.32.3 | 2.34.2 | CVE-2024-47081, CVE-2026-25645 |
| python-dotenv | 1.1.0 | 1.2.4 | CVE-2026-28684 |
| jupyterlab | 4.4.1 | 4.6.5 | 11 avisos |
| pytest | 8.3.5 | 9.0.3 | CVE-2025-71176 |

Riesgos aceptados temporalmente (detalle en [`security/riesgos_aceptados.toml`](../security/riesgos_aceptados.toml)):

| Paquete | Avisos | Por qué no se actualiza ahora | Mitigación |
|---|---|---|---|
| mlflow 2.22.0 | 30 | Pasar a 3.x cambia MLproject, Model Registry y API de modelos | La mayoría afecta al servidor, gateway o integraciones que no usamos; el servidor corre solo en `127.0.0.1` y solo se cargan modelos propios |
| pyarrow 19.0.1 | 1 | mlflow 2.22 exige `pyarrow<20` | Solo se leen Parquet generados por el pipeline |
| setuptools 81.0.0 | 2 | deepchecks 0.19.1 usa `pkg_resources` (eliminado en 82) | Solo se usa al instalar dependencias fijadas |

`security/run_checks.py` falla si aparece una vulnerabilidad en un paquete que no esté en esa lista, o si
la versión de un paquete aceptado cambia, o si vence la fecha de revisión (2026-12-01).

## Cómo agregar una excepción

Solo si no se puede actualizar el paquete. Agrega un bloque a `security/riesgos_aceptados.toml` con
`paquete`, `version` exacta, `motivo` (qué afecta y qué lo mitiga) y fecha `revisar`, y explícalo en el PR.
