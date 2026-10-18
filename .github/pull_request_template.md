## Qué cambia

<!-- Resumen en una o dos líneas. Título del PR en formato Conventional Commits, p. ej. feat(features): agrega clima horario -->

## Pregunta que atiende

- [ ] P1 demanda
- [ ] P2 riesgo de vaciado o saturación
- [ ] P3 tipologías de estaciones
- [ ] Infraestructura o documentación

## Cómo se probó

- [ ] `make test`
- [ ] `make smoke` (datos sintéticos)
- [ ] Run en MLflow (enlace):

## Revisión

- [ ] No se suben datos de MiBici, `mlruns/`, `mlartifacts/`, modelos ni `.env`
- [ ] Rutas, fechas y semillas nuevas están en `mibici/config.py`
- [ ] Las features nuevas no usan información posterior al momento de la predicción
