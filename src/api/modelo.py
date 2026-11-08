"""Carga el modelo champion desde el Model Registry de MLflow y lo guarda en memoria."""

import mlflow
from mlflow import MlflowClient

# Se usa la misma base de datos que train.py y los scripts de registro.
# La ruta es relativa: la API debe arrancarse desde la raíz del proyecto
URI_SEGUIMIENTO = "sqlite:///mlflow.db"
NOMBRE_MODELO_REGISTRADO = "modelo_diabetes"
ALIAS_CAMPEON = "champion"

# Se guarda el modelo cargado en un diccionario a nivel de módulo para que
# se cargue una sola vez al arrancar y todas las peticiones lo reutilicen.
# Mientras "modelo" sea None, la API sabe que no puede predecir
estado = {"modelo": None, "version": None, "run_id": None, "descripcion": None}


def cargar_modelo_champion() -> None:
    """Busca la versión con alias champion, carga su modelo y rellena el estado."""
    mlflow.set_tracking_uri(URI_SEGUIMIENTO)
    cliente = MlflowClient()

    # Se pregunta al registro qué versión tiene el alias; si no existe, MLflow lanza un error
    version = cliente.get_model_version_by_alias(NOMBRE_MODELO_REGISTRADO, ALIAS_CAMPEON)

    # Se carga el modelo con pyfunc, que comprueba que las columnas de entrada
    # coinciden con las del ejemplo guardado en el entrenamiento
    estado["modelo"] = mlflow.pyfunc.load_model(
        f"models:/{NOMBRE_MODELO_REGISTRADO}@{ALIAS_CAMPEON}"
    )
    # Se guarda la versión como texto porque MLflow la devuelve como número entero
    estado["version"] = str(version.version)
    estado["run_id"] = version.run_id
    estado["descripcion"] = version.description
