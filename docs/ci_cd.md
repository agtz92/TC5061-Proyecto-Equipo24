# Pipeline de Integración Continua y Despliegue Continuo (CI/CD)

Este documento describe la arquitectura, configuración y funcionamiento del pipeline de **CI/CD** implementado para el proyecto MiBici utilizando **GitHub Actions**.

---

## 1. Objetivos del Pipeline

1. **Garantizar la calidad e integridad del código:** Ejecutar automáticamente las pruebas unitarias, de integración, de calidad de datos y de seguridad ante cada cambio introducido.
2. **Empaquetado reproducible:** Construir la imagen de contenedor OCI/Docker del servicio de la API de demanda (P1) para asegurar que el despliegue sea consistente y esté libre de dependencias rotas.
3. **Validación temprana (*Fail-fast*):** Bloquear integraciones que no cumplan con los estándares de seguridad, linters o pruebas antes de que lleguen a la rama principal (`main`).

---

## 2. Flujo de Trabajo (*Workflow*)

El workflow está definido en [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).

### Eventos disparadores (*Triggers*)

* **`push`:** Se activa automáticamente al empujar commits a la rama `main` o a ramas de trabajo (`feat/**`, `docs/**`, `fix/**`).
* **`pull_request`:** Se activa automáticamente al abrir o actualizar solicitudes de extracción hacia la rama `main`.
* **Control de concurrencia:** Si un desarrollador envía nuevos commits a una misma rama o PR, las ejecuciones anteriores en curso se cancelan automáticamente (`cancel-in-progress: true`), optimizando los recursos de cómputo.

---

## 3. Estructura de Trabajos (*Jobs*)

El pipeline consta de dos etapas secuenciales:

```
[ Push / Pull Request ]
          │
          ▼
   ┌──────────────┐
   │  Job: test   │ ◄─── Checkout, Python 3.12, pip cache, Ruff lint,
   └──────┬───────┘      Security checks (secretos/higiene), Pytest suite
          │
          │ (solo si 'test' pasa exitosamente)
          ▼
   ┌──────────────┐
   │  Job: build  │ ◄─── Checkout, Docker Buildx, construcción de imagen
   └──────────────┘      'mibici-api', smoke test dentro del contenedor
```

### Job 1: `test` (Pruebas unitarias y de calidad)

* **Entorno de ejecución:** `ubuntu-latest`.
* **Versión de Python:** `3.12` con caché de `pip` activada para acelerar los tiempos de instalación.
* **Pasos:**
  1. **Checkout del código:** Clona el repositorio con `actions/checkout@v4`.
  2. **Instalación de dependencias:** Instala `requirements.txt` y registra el paquete local en modo editable (`pip install -e .`).
  3. **Linting con Ruff:** Verifica formateo e importaciones limpias (`ruff check .`).
  4. **Chequeos de seguridad:** Ejecuta [`security/run_checks.py`](../security/run_checks.py) con `--skip-audit` para validar la ausencia de secretos versionados, configuración segura y exclusiones de `.gitignore`.
  5. **Suite de pruebas Pytest:** Ejecuta todas las pruebas unitarias y de integración de la carpeta `tests/` (`test_pipeline.py`, `test_api.py`, `test_dataset.py`, `test_validation.py`, `test_ab_test.py`, `test_retrain.py`, `test_security.py`).

### Job 2: `build` (Construcción y validación de la imagen Docker)

* **Dependencia:** Requiere que el trabajo `test` termine con éxito (`needs: test`).
* **Pasos:**
  1. **Checkout del código:** Clona el repositorio.
  2. **Configuración de Docker Buildx:** Habilita el motor avanzado de construcción de Docker.
  3. **Construcción de la imagen:** Construye la imagen a partir del [`Containerfile`](../Containerfile) (estándar OCI del proyecto) etiquetando con el commit SHA (`mibici-api:${{ github.sha }}`) y con `latest`.
  4. **Smoke test del contenedor:**
     - Arranca una instancia efímera de la imagen con el usuario no privilegiado `appuser`.
     - Verifica que todas las dependencias del servicio (`fastapi`, `uvicorn`, `mlflow`, `sklearn`, `pandas`, `numpy`, `pyarrow`) importen correctamente en el runtime del contenedor.
     - Valida la integridad del módulo de la API (`mibici.api.main`).

---

## 4. Replicación Local

Los mismos pasos que ejecuta el pipeline en la nube se pueden replicar localmente antes de hacer push:

```bash
# 1. Ejecutar linters
ruff check .

# 2. Ejecutar chequeos de seguridad
python security/run_checks.py --skip-audit

# 3. Ejecutar pruebas unitarias
pytest -v

# 4. Construir la imagen del servicio localmente con Docker o Podman
docker build -f Containerfile -t mibici-api:latest .
# o con Podman:
podman build -t mibici-api:latest .
```

