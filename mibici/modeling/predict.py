"""Predicción batch (pendiente para la siguiente entrega).

Aquí vivirá la generación nocturna de pronósticos para el día siguiente:
cargar el modelo registrado en MLflow Model Registry para P1 y P2, construir
las features con mibici.features hasta las 23:59 del día anterior y escribir
reports/predicciones_AAAA_MM_DD.csv. En esta entrega solo se implementa el
experimento base (mibici.modeling.train).
"""
