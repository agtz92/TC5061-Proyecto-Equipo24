# Guía de reproducibilidad del proyecto MiBici

Esta guía explica cómo reproducir el flujo completo del proyecto desde un clon limpio: estructura,
feature engineering, MLflow Project, MLflow Tracking y Model Registry. También reúne las fallas que
encontramos al hacerlo en Windows y cómo las resolvimos. Si te aparece un error, búscalo en la
[sección 6](#6-fallas-conocidas-y-cómo-resolverlas) por su mensaje.

Todos los comandos son para **PowerShell en Windows**, desde la raíz del repositorio. En macOS o Linux
funcionan igual, salvo la activación del entorno (`source .venv/bin/activate`) y que ahí sí existe `make`.

La evidencia de la última verificación está en:

- [Evidencia de Reproducibilidad del flujo del proyecto/](Evidencia%20de%20Reproducibilidad%20del%20flujo%20del%20proyecto/)
- [Evidencia del Model Registry/](Evidencia%20del%20Model%20Registry/)

---

## 1. Requisitos

| Requisito | Versión | Comprobar con |
|---|---|---|
| Python | **3.12** (el proyecto acepta de 3.11 a 3.13; **no funciona con 3.14**) | `py -0p` |
| Git | cualquiera reciente | `git --version` |
| Docker Desktop | opcional; solo para el servidor de MLflow y la API en contenedores | `docker version` |
| `make` | opcional; en Windows no viene instalado (ver [falla 6.2](#62-make-no-se-reconoce)) | `make --version` |

Si `py -0p` no muestra la versión 3.12, instálala desde python.org o con `winget install Python.Python.3.12`.

---

## 2. Preparar el entorno

```powershell
git clone https://github.com/agtz92/TC5061-Proyecto-Equipo24.git
cd TC5061-Proyecto-Equipo24

py -3.12 -m venv .venv            # usar explícitamente 3.12
.venv\Scripts\Activate.ps1
python --version                  # debe decir Python 3.12.x

python -m pip install -U pip
pip install -r requirements.txt
python -c "import numpy, pandas, mlflow, sklearn; print('OK')"

copy .env.example .env
```

Si `Activate.ps1` está bloqueado, ver la [falla 6.3](#63-no-se-puede-ejecutar-activateps1).

---

## 3. Levantar el servidor de MLflow

El proyecto usa un servidor de MLflow en `http://localhost:5001`, con base de datos SQLite
(`mlflow.db`) y artefactos en `mlartifacts/`. Ambos quedan en la raíz del repo y Git los ignora.
**Usa solo una de estas dos opciones**: las dos ocupan el puerto 5001.

**Opción A: Docker** (recomendada si vas a probar también la API)

```powershell
# Docker Desktop debe estar abierto y decir "Engine running"
docker compose up -d mlflow          # solo el servidor
docker compose up -d                 # servidor + API del modelo (puerto 8000)
```

**Opción B: sin Docker** (en una terminal aparte, con el venv activado)

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --serve-artifacts --host 127.0.0.1 --port 5001
```

En la terminal de trabajo, indica a qué servidor y experimento enviar las corridas:

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5001"
$env:MLFLOW_EXPERIMENT_NAME = "mibici-equipo24-prueba"   # usa otro nombre para no mezclar con los runs reales
```

Abre http://localhost:5001 para comprobar que el servidor responde.

---

## 4. Reproducir el flujo

### 4.1 Con datos sintéticos (recomendado para verificar; no descarga nada)

El repositorio genera datos falsos con el mismo formato que los reales. Cubren de 2025-01 a 2026-02,
así que incluyen los periodos de entrenamiento, validación y prueba de `mibici/config.py`.

```powershell
# Prueba de humo (equivale a "make smoke")
python tests/make_sample_data.py data/sample
python -m mibici.dataset --source-dir data/sample --start 2025-01 --end 2026-02
python -m mibici.features
python -m pytest -q tests

# MLflow Project + Tracking
mlflow run . -e features  --env-manager local
mlflow run . -e baselines --env-manager local
mlflow run . -e train     --env-manager local -P question=p1 -P model=rf -P n_estimators=50
```

**Importante:**

- La prueba de humo **sobrescribe** `data/interim`, `data/processed` y `data_manifest.json`. Si en esa
  copia tienes los datos reales, hazla en un clon aparte.
- `data_manifest.json` sí está en Git. Al terminar, restáuralo con `git checkout data_manifest.json`
  (ver [falla 6.11](#611-las-corridas-quedan-con-git_dirty--true)).
- No uses `mlflow run . -e ingest` en este modo: ese entry point siempre descarga del portal (ver
  [falla 6.5](#65-la-ingesta-descarga-demasiados-datos)).

### 4.2 Con datos reales (descarga grande)

Descarga 57 meses del portal de MiBici (cerca de 21 millones de viajes y varios GB):

```powershell
$env:MIBICI_NOMENCLATURA = "nomenclatura_2026_10.csv"   # ver falla 6.4
mlflow run . -e ingest   --env-manager local
mlflow run . -e features --env-manager local
mlflow run . -e train    --env-manager local -P question=p1 -P model=hgb
```

Para descargar menos, limita el rango sin dejar fuera los periodos de entrenamiento, validación y
prueba, por ejemplo `-P start=2025-01 -P end=2026-02`. Los archivos ya descargados en `data/raw` no
se vuelven a bajar.

### 4.3 Resultado esperado

| Paso | Resultado esperado (datos sintéticos) |
|---|---|
| `mibici.dataset` | 153,241 viajes limpios, `data_version=c5ad32197dd6` |
| `mibici.features` | Panel de 254,348 filas estación-hora, 25 estaciones |
| `pytest` | 76 pruebas aprobadas, 0 fallidas (unos 2 minutos) |
| `mlflow run` | Corridas en estado FINISHED; `baselines` crea P1 como corrida padre y P2 y P3 como hijas |

---

## 5. Verificar el Model Registry

El modelo vigente **no está en Git**: vive en la base de MLflow del equipo (`mlflow.db` +
`mlartifacts/`). Un servidor recién creado tiene el registry vacío. Para revisarlo:

```powershell
docker compose stop mlflow
# Respaldar la base actual si tiene corridas que quieras conservar
mkdir respaldo_mlflow; move mlflow.db respaldo_mlflow\; move mlartifacts respaldo_mlflow\
# Copiar la base del equipo (ajusta la ruta de origen)
copy C:\ruta\a\la\base\mlflow.db .
Copy-Item C:\ruta\a\la\base\mlartifacts .\mlartifacts -Recurse
docker compose up -d mlflow
docker compose restart api
```

Sin copiar archivos, también puedes apuntar el contenedor a otra carpeta con
`$env:MLFLOW_STORE = "C:\ruta\a\la\base"` antes de `docker compose up -d mlflow`.

Qué debes encontrar en http://localhost:5001 → **Models** → `mibici-equipo24-p1-demanda`:

| Alias | Versión | Corrida de origen |
|---|---|---|
| `@champion` | v1 | `p1-hgb-it400-lr0.1-d16-leaf50` (versión vigente, la que sirve la API) |
| `@challenger` | v2 | `p1-hgb-retrain-2026-06` (no se promovió tras el A/B test) |

Para comprobar que la API sirve la versión vigente:

```powershell
Invoke-RestMethod http://localhost:8000/health
# estado=ok, modelo=mibici-equipo24-p1-demanda, version=1, uri=models:/...@champion
```

---

## 6. Fallas conocidas y cómo resolverlas

### 6.1 `ModuleNotFoundError: No module named 'numpy'`

```
File "tests\make_sample_data.py", line 19, in <module>
    import numpy as np
ModuleNotFoundError: No module named 'numpy'
```

**Causa.** El `.venv` se creó con **Python 3.14** (el `python` por defecto de la máquina). El proyecto
requiere de 3.11 a 3.13 y las versiones fijas de `requirements.txt` (`numpy==2.2.5`,
`pyarrow==19.0.1`, etc.) no tienen paquetes para 3.14, así que `pip install` falló y el entorno quedó
vacío.

**Cómo confirmarlo.** `type .venv\pyvenv.cfg` muestra `version = 3.14.x`.

**Solución.** Recrear el entorno con 3.12:

```powershell
deactivate
Remove-Item -Recurse -Force .venv
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python --version                      # 3.12.x
python -m pip install -U pip
pip install -r requirements.txt
```

### 6.2 `make` no se reconoce

```
make : El término 'make' no se reconoce como nombre de un cmdlet, función, archivo de script o programa ejecutable.
```

**Causa.** Windows no incluye `make`.

**Solución.** Ejecutar directamente los comandos equivalentes:

| `make ...` | Equivalente en PowerShell |
|---|---|
| `make smoke` | los 4 comandos de la [sección 4.1](#41-con-datos-sintéticos-recomendado-para-verificar-no-descarga-nada) |
| `make data` | `python -m mibici.dataset` |
| `make verify` | `python -m mibici.dataset verify` |
| `make features` | `python -m mibici.features` |
| `make experiment` | `mlflow run . -e baselines --env-manager local` |
| `make test` | `python -m pytest` |
| `make validate` | `python -m mibici.validation --question all` |
| `make security` | `python security/run_checks.py` |
| `make lint` | `ruff format --check; ruff check` |

Otra opción es instalarlo con `winget install ezwinports.make` y abrir una terminal nueva. Aun así,
`make clean` no funciona en Windows porque usa el comando `find` de Unix.

### 6.3 No se puede ejecutar `Activate.ps1`

```
... no se puede cargar porque la ejecución de scripts está deshabilitada en este sistema.
```

**Causa.** La política de ejecución de PowerShell bloquea los scripts.

**Solución.**

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 6.4 `No se encontró nomenclatura_2026_09.csv`

```
RuntimeError: No se encontró nomenclatura_2026_09.csv en https://www.mibici.net/es/datos-abiertos/
```

**Causa.** El portal de MiBici solo publica la nomenclatura de estaciones **vigente** y la reemplaza
cada mes. `config.py` usa por defecto la de septiembre de 2026, que ya no está publicada.

**Solución.** Indicar la nomenclatura que publica hoy el portal. En noviembre de 2026 era
`nomenclatura_2026_10.csv`; para saber cuál es la vigente, entra a
https://www.mibici.net/es/datos-abiertos/.

```powershell
$env:MIBICI_NOMENCLATURA = "nomenclatura_2026_10.csv"     # solo esta terminal
# o de forma permanente, en el archivo .env:
# MIBICI_NOMENCLATURA=nomenclatura_2026_10.csv
```

**Consecuencia.** Al cambiar la nomenclatura cambia el hash del manifiesto y, con él, la
`data_version`. Los resultados no serán idénticos bit a bit a los del modelo registrado.

### 6.5 La ingesta descarga demasiados datos

**Causa.** `mlflow run . -e ingest` descarga de 2022-01 a 2026-09 (57 meses). El entry point solo
acepta `start` y `end`; no tiene una opción para usar archivos locales.

**Solución.** Para verificar el flujo, usa los datos sintéticos
([sección 4.1](#41-con-datos-sintéticos-recomendado-para-verificar-no-descarga-nada)) y ejecuta la
ingesta como módulo con `--source-dir`. Para datos reales, limita el rango con `-P start=... -P end=...`.

### 6.6 Docker: `failed to connect to the docker API`

```
failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine; check if the path is correct and if the daemon is running
```

**Causa.** Docker Desktop no está abierto.

**Solución.**

1. Abrir **Docker Desktop** y esperar a que diga **Engine running**.
2. Comprobar con `docker version`: debe aparecer la sección `Server:`.
3. Si no arranca: clic derecho en el ícono → *Switch to Linux containers* y revisar WSL 2 con
   `wsl --status` (instalarlo con `wsl --install` y reiniciar).

Docker no es obligatorio: puedes usar la opción B de la [sección 3](#3-levantar-el-servidor-de-mlflow).

### 6.7 `PermissionError: [WinError 10013]` al iniciar `mlflow server`

```
PermissionError: [WinError 10013] Intento de acceso a un socket no permitido por sus permisos de acceso
Running the mlflow server failed.
```

**Causa.** El puerto 5001 ya está ocupado. En nuestro caso lo usaba el contenedor `mibici-mlflow` de
Docker Compose.

**Cómo confirmarlo.**

```powershell
docker ps                                                  # ¿aparece mibici-mlflow en 5001?
Get-NetTCPConnection -LocalPort 5001 -State Listen         # ¿qué proceso escucha?
netsh interface ipv4 show excludedportrange protocol=tcp   # ¿Windows reservó el puerto?
```

**Solución.** Si ya corre `mibici-mlflow`, **no hace falta otro servidor**: usa ese, que lee el mismo
`mlflow.db` de la raíz del repo. Si prefieres el servidor local, detén antes el contenedor con
`docker compose stop mlflow`. Si el puerto está reservado por Windows, usa otro (`--port 5002`) y
actualiza `MLFLOW_TRACKING_URI`.

### 6.8 El Model Registry está vacío o `@champion` no existe

**Síntoma.** En **Models** no aparece `mibici-equipo24-p1-demanda`, o la API falla al arrancar porque
no encuentra `models:/mibici-equipo24-p1-demanda@champion`.

**Causa.** El servidor creó un `mlflow.db` nuevo en esta carpeta. El registry del equipo no está en
Git: vive en el `mlflow.db` y `mlartifacts/` de la máquina donde se entrenó.

**Solución.** Copiar esa base o apuntar el servidor a ella, como se explica en la
[sección 5](#5-verificar-el-model-registry). Copia siempre **los dos**, `mlflow.db` y `mlartifacts/`:
sin los artefactos, las versiones aparecen en el registry pero el modelo no se puede cargar.

### 6.9 La API dice servir una versión que no está en el servidor actual

**Causa.** La API carga el modelo **una sola vez, al arrancar**. Si después se cambia la base del
servidor de MLflow, la API sigue sirviendo el modelo anterior desde memoria y falla en el siguiente
reinicio.

**Solución.** Cada vez que cambies la base de MLflow, reinicia la API:

```powershell
docker compose restart api
Invoke-RestMethod http://localhost:8000/health
```

### 6.10 `Can not safely convert float64 to float32` al predecir

```
MlflowException: Failed to enforce schema ... Incompatible input types for column hour.
Can not safely convert float64 to float32.
```

**Causa.** La firma del modelo declara las 13 variables de entrada como `float32`, y pandas crea
`float64` por defecto. No es un error del modelo: MLflow está validando el contrato de entrada.

**Solución.**

```python
X = X.astype("float32")
model.predict(X)
```

### 6.11 Las corridas quedan con `git_dirty = true`

**Causa.** La prueba de humo sobrescribe `data_manifest.json`, que está en Git. Cualquier corrida
posterior registra que el árbol de trabajo tenía cambios.

**Solución.** Restaurar el manifiesto antes de las corridas que quieras conservar o registrar, y no
subir a Git el manifiesto de los datos sintéticos:

```powershell
git checkout data_manifest.json
git status        # no debe haber cambios antes de entrenar un modelo para registrar
```

### 6.12 Aviso de `cloudpickle` al cargar el modelo

```
WARNING mlflow.utils.requirements_utils: Detected one or more mismatches ...
 - cloudpickle (current: 3.1.3, required: cloudpickle==3.1.2)
```

**Causa.** El modelo se guardó con otra versión de `cloudpickle`, que no está fijada en
`requirements.txt`.

**Solución.** Es solo un aviso: el modelo carga y predice bien. Para eliminarlo,
`pip install cloudpickle==3.1.2`.

### 6.13 `python -c "..."` falla en PowerShell con `SyntaxError`

**Causa.** Windows PowerShell 5.1 quita las comillas internas al pasar argumentos a programas
externos. Por ejemplo, `mlflow.set_tracking_uri("http://...")` le llega a Python sin comillas.

**Solución.** Guardar el código en un archivo `.py` y ejecutarlo con `python archivo.py`, o usar
PowerShell 7 (`pwsh`).

---

## 7. Lista de verificación final

- [ ] `python --version` dice 3.12.x dentro del `.venv`.
- [ ] `python -m pytest -q tests` termina con 76 pruebas aprobadas.
- [ ] Las corridas de `features`, `baselines` y `train` aparecen como FINISHED en http://localhost:5001.
- [ ] Cada corrida de entrenamiento tiene los parámetros `data_version` y `seed`, el tag `git_commit`
      y los artefactos `config/`, `data/data_manifest.json`, `plots/` y `model/`.
- [ ] Dos corridas de `train` con los mismos parámetros dan las mismas métricas (función *Compare*).
- [ ] En **Models**, `mibici-equipo24-p1-demanda` tiene `@champion` → v1 y `@challenger` → v2.
- [ ] `Invoke-RestMethod http://localhost:8000/health` devuelve `version = 1`.
- [ ] `git status` no muestra `data_manifest.json` modificado.
