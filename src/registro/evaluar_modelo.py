"""Aplica un control de calidad (quality gate) a la última versión registrada del modelo.
 
Flujo del script:
1. Se busca la última versión registrada del modelo.
2. Se lee el RMSE del run del que procede esa versión.
3. Si el RMSE es menor que el umbral, se asigna el alias champion a esa versión.
   Si no lo es, el script termina con código de salida 1 y no se asigna el alias.
"""
 
import sys
 
import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
 
# Se usan la misma base de datos y el mismo nombre de modelo que registrar_modelo.py
URI_SEGUIMIENTO = "sqlite:///mlflow.db"
NOMBRE_MODELO_REGISTRADO = "modelo_diabetes"
 
# Se usa un alias en lugar de un número de versión fijo para que la API cargue
# siempre "el modelo campeón" sin tener que cambiar su código en cada versión nueva
ALIAS_CAMPEON = "champion"
 
# Se fija el umbral en 60 porque un modelo que siempre predijera la media del
# target tendría un RMSE de unos 73 con estos mismos datos de prueba.
# Exigir menos de 60 obliga al modelo a mejorar al menos un 18 % esa referencia
# y deja margen sobre el RMSE actual (≈ 53.7)
UMBRAL_RMSE = 60.0
 
 
def buscar_ultima_version(cliente: MlflowClient) -> ModelVersion:
    """Devuelve la versión más reciente del modelo registrado."""
    # Se ordenan las versiones de mayor a menor número y se pide solo la primera,
    # que es la que acaba de registrar registrar_modelo.py
    versiones = cliente.search_model_versions(
        filter_string=f"name='{NOMBRE_MODELO_REGISTRADO}'",
        order_by=["version_number DESC"],
        max_results=1,
    )
    # Se detiene la evaluación con un mensaje claro si todavía no se ha registrado nada
    if not versiones:
        raise ValueError(f"No hay versiones de '{NOMBRE_MODELO_REGISTRADO}'. Ejecuta antes registrar_modelo.py")
    return versiones[0]
 
 
def leer_rmse(cliente: MlflowClient, version: ModelVersion) -> float:
    """Devuelve el RMSE guardado en el run del que procede la versión."""
    # Se lee la métrica del run original en lugar de volver a calcularla,
    # para evaluar exactamente el mismo resultado que se guardó al entrenar
    run = cliente.get_run(version.run_id)
    return run.data.metrics["rmse"]
 
 
def supera_control_calidad(rmse: float) -> bool:
    """Indica si el RMSE es menor que el umbral de calidad."""
    return rmse < UMBRAL_RMSE
 
 
if __name__ == "__main__":
    mlflow.set_tracking_uri(URI_SEGUIMIENTO)
    cliente = MlflowClient()
 
    version = buscar_ultima_version(cliente)
    rmse = leer_rmse(cliente, version)
    print(f"Versión {version.version}: RMSE={rmse:.2f} (umbral {UMBRAL_RMSE:.2f})")
 
    # Se termina con código de salida 1 si no se supera el control, para que el
    # orquestador y GitHub Actions detecten el fallo y detengan el proceso
    if not supera_control_calidad(rmse):
        print(f"No supera el quality gate: no se asigna el alias '{ALIAS_CAMPEON}'")
        sys.exit(1)
 
    # Se asigna el alias a esta versión; si otra versión lo tenía, MLflow se lo
    # quita, porque un alias solo puede apuntar a una versión a la vez
    cliente.set_registered_model_alias(NOMBRE_MODELO_REGISTRADO, ALIAS_CAMPEON, version.version)
    print(f"Supera el quality gate: la versión {version.version} es ahora '{ALIAS_CAMPEON}'")