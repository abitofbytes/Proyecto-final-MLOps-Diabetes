"""Registra en el Model Registry de MLflow el mejor modelo del experimento.

Flujo del script:
1. Se busca, entre todos los runs del experimento, el que tiene menor RMSE.
2. Se obtiene el identificador del modelo guardado en ese run.
3. Se registra ese modelo con un nombre fijo; cada registro crea una versión nueva.
4. Se añade a la versión una descripción con el tipo de modelo y sus métricas.
"""

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run

# Se usan la misma base de datos y el mismo experimento que train.py
# para leer exactamente los runs que creó el entrenamiento
URI_SEGUIMIENTO = "sqlite:///mlflow.db"
NOMBRE_EXPERIMENTO = "diabetes"

# Se usa un nombre fijo para que cada nuevo registro se añada como una
# versión más del mismo modelo, en lugar de crear un modelo distinto cada vez
NOMBRE_MODELO_REGISTRADO = "modelo_diabetes"


def buscar_mejor_run(cliente: MlflowClient) -> Run:
    """Devuelve el run del experimento con menor RMSE."""
    experimento = cliente.get_experiment_by_name(NOMBRE_EXPERIMENTO)
    # Se detiene el registro con un mensaje claro si todavía no se ha entrenado
    if experimento is None:
        raise ValueError(f"No existe el experimento '{NOMBRE_EXPERIMENTO}'. Ejecuta antes train.py")

    # Se ordenan los runs de menor a mayor RMSE y se pide solo el primero,
    # porque un RMSE más bajo significa un error medio más pequeño.
    # Se descartan los runs que no terminaron bien (por ejemplo, interrumpidos
    # con Ctrl+C), porque podrían no tener el modelo guardado
    runs = cliente.search_runs(
        experiment_ids=[experimento.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["metrics.rmse ASC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"El experimento '{NOMBRE_EXPERIMENTO}' no tiene runs terminados. Ejecuta antes train.py")
    return runs[0]


def registrar_modelo(mejor_run: Run) -> str:
    """Registra el modelo del run indicado y devuelve el número de versión creado."""
    # Se toma el identificador del modelo que guardó log_model en ese run.
    # Se registra con la dirección models:/<id> porque en MLflow 3 el modelo
    # es un objeto propio, con identificador, separado del run que lo creó
    id_modelo = mejor_run.outputs.model_outputs[0].model_id
    version = mlflow.register_model(f"models:/{id_modelo}", NOMBRE_MODELO_REGISTRADO)
    return version.version


def describir_version(cliente: MlflowClient, mejor_run: Run, version: str) -> None:
    """Añade a la versión registrada una descripción con su origen y sus métricas."""
    # Se guarda la descripción para que se vea en la UI y la API pueda mostrarla
    # sin tener que buscar el run de origen
    metricas = mejor_run.data.metrics
    descripcion = (
        f"Modelo {mejor_run.info.run_name} del experimento '{NOMBRE_EXPERIMENTO}'. "
        f"RMSE={metricas['rmse']:.2f} MAE={metricas['mae']:.2f} R2={metricas['r2']:.2f}"
    )
    cliente.update_model_version(name=NOMBRE_MODELO_REGISTRADO, version=version,
                                 description=descripcion)


if __name__ == "__main__":
    mlflow.set_tracking_uri(URI_SEGUIMIENTO)
    cliente = MlflowClient()

    mejor_run = buscar_mejor_run(cliente)
    rmse = mejor_run.data.metrics["rmse"]
    print(f"Mejor run: {mejor_run.info.run_name} (RMSE={rmse:.2f})")

    version = registrar_modelo(mejor_run)
    describir_version(cliente, mejor_run, version)
    print(f"Registrado '{NOMBRE_MODELO_REGISTRADO}' versión {version}")