# Chequeos de seguridad (José Alfredo Gutiérrez Guerra)

Rama `feat/seguridad-alfredo`. Parte 3 de la actividad: chequeos de seguridad básicos.

## Qué se hizo

1. **Credenciales y datos sensibles.** Se revisaron los archivos y el historial de Git: no hay
   credenciales ni archivos sensibles versionados. Se amplió `.gitignore` (llaves y credenciales) y
   `.env.example`. `guardarmodelos.py` y `verificarmodelos.py` ya no tienen la URI de MLflow escrita en el
   código: la leen de `.env`.
2. **Bandit (código).** 16 avisos, la mayoría SQL armado con f-strings. Se agregó validación de fechas
   y columnas antes de armar las consultas de DuckDB. Resultado: 0 avisos.
3. **pip-audit (dependencias).** Se corrigió `setuptool` → `setuptools` en `requirements.txt` (el paquete
   no existía y la instalación fallaba). Se actualizaron requests, python-dotenv, jupyterlab y pytest.
   mlflow, pyarrow y setuptools quedan como riesgos aceptados y justificados en
   `security/riesgos_aceptados.toml`.
4. **Pruebas.** `tests/test_security.py` (25 pruebas de PyTest) verifica `.gitignore`, ausencia de
   secretos, uso de variables de entorno y rechazo de inyección SQL.

## Cómo ejecutarlo

```bash
python security/run_checks.py   # o: make security
```

Corre las pruebas de seguridad, bandit y pip-audit; los resultados quedan en `reports/security/`.
Detalle completo en [`docs/seguridad.md`](docs/seguridad.md) y evidencia en el README.
