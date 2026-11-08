"""Entrena modelos de regresión y registra cada entrenamiento en MLflow.

Flujo del script:
1. Se leen los datos limpios de data/processed/.
2. Se separan las variables de entrada (X) de la columna objetivo (y).
3. Se dividen los datos en entrenamiento y prueba.
4. Se entrena cada modelo dentro de su propio run de MLflow, donde se
   registran sus parámetros, sus métricas y el modelo entrenado.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

# Se guarda el entrenamiento en src/entrenamiento/, separado de src/datos/,
# para mantener cada responsabilidad en su propia carpeta y que el proyecto sea más claro
RUTA_DATOS_LIMPIOS = Path("data/processed/diabetes.csv")
COLUMNA_OBJETIVO = "target"

# Se usan el mismo 20 % de prueba y la misma semilla que en el notebook
# para que los resultados sean comparables y reproducibles
PROPORCION_PRUEBA = 0.2
SEMILLA = 42

# Se guarda el seguimiento en una base de datos SQLite local porque es el
# almacenamiento que exige MLflow 3 (la carpeta mlruns/ como almacén está en
# mantenimiento) y deja todos los runs en un único archivo fácil de consultar.
# En mlflow.db se guardan los experimentos, los runs, los parámetros y las métricas;
# los modelos entrenados (artefactos) se guardan como archivos en la carpeta mlruns/
URI_SEGUIMIENTO = "sqlite:///mlflow.db"

# Se agrupan todos los runs en un mismo experimento para poder compararlos entre sí
NOMBRE_EXPERIMENTO = "diabetes"

# Se marca como confiable solo el tipo Tree de los modelos de árboles porque
# los modelos los crea este mismo script y no vienen de una fuente externa.
# MLflow 3 guarda los modelos de scikit-learn con el formato seguro skops, que
# rechaza por defecto este tipo porque un archivo manipulado podría ser peligroso
TIPOS_CONFIABLES = ["sklearn.tree._tree.Tree"]


def leer_datos(ruta: Path) -> pd.DataFrame:
    """Lee el CSV de datos limpios generado por data_cleaning.py."""
    # Se detiene el entrenamiento con un mensaje claro si falta el archivo
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Ejecuta antes data_cleaning.py")
    return pd.read_csv(ruta)


def separar_variables(datos: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa las variables de entrada (X) de la columna objetivo (y)."""
    # Se quita la columna objetivo para quedarse solo con las 10 variables de entrada.
    # drop devuelve una tabla nueva, por eso se guarda en X y datos no cambia
    X = datos.drop(columns=[COLUMNA_OBJETIVO])
    y = datos[COLUMNA_OBJETIVO]
    return X, y


def dividir_datos(X: pd.DataFrame, y: pd.Series) -> list:
    """Divide los datos en entrenamiento y prueba.

    Devuelve cuatro partes siempre en este orden:
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba.
    """
    # Se reserva una parte de los datos que el modelo no ve al entrenar para
    # medir su error con datos nuevos; medirlo con los datos de entrenamiento
    # daría un resultado demasiado optimista
    return train_test_split(X, y, test_size=PROPORCION_PRUEBA, random_state=SEMILLA)


def crear_modelos() -> dict[str, object]:
    """Crea los tres modelos de regresión que se van a comparar.

    La clave del diccionario es el nombre del modelo, que se usa como nombre del run.
    """
    # Se comparan un modelo lineal sencillo y dos modelos de árboles más complejos
    # para comprobar si la complejidad extra mejora el error.
    # Los hiperparámetros no indicados toman el valor por defecto de scikit-learn
    return {
        "regresion_lineal": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=SEMILLA),
        "gradient_boosting": GradientBoostingRegressor(random_state=SEMILLA),
    }


def calcular_metricas(y_real: pd.Series, y_predicho) -> dict[str, float]:
    """Calcula RMSE, MAE y R² comparando los valores reales con las predicciones.

    - RMSE: error medio en las unidades del target, penaliza más los errores grandes.
    - MAE: error medio absoluto en las unidades del target.
    - R²: proporción de la variación del target que explica el modelo (1 es perfecto).
    """
    # Se usa RMSE como métrica principal porque penaliza más los errores grandes
    return {
        "rmse": root_mean_squared_error(y_real, y_predicho),
        "mae": mean_absolute_error(y_real, y_predicho),
        "r2": r2_score(y_real, y_predicho),
    }


def entrenar_y_registrar(nombre_modelo: str, modelo, X_entrenamiento: pd.DataFrame,
                         X_prueba: pd.DataFrame, y_entrenamiento: pd.Series,
                         y_prueba: pd.Series) -> dict[str, float]:
    """Entrena un modelo y registra sus parámetros, métricas y artefacto en un run de MLflow."""
    # Se abre un run nuevo con el nombre del modelo; el bloque with lo cierra
    # automáticamente al terminar, incluso si se produce un error dentro
    with mlflow.start_run(run_name=nombre_modelo):
        # Se entrena el modelo con los datos de entrenamiento y se evalúa con los de prueba
        modelo.fit(X_entrenamiento, y_entrenamiento)
        metricas = calcular_metricas(y_prueba, modelo.predict(X_prueba))

        # Se registran todos los hiperparámetros del modelo, también los que toman
        # el valor por defecto, para dejar constancia completa de su configuración
        mlflow.log_params(modelo.get_params())

        # Se registran el RMSE, el MAE y el R² para comparar los runs entre sí
        mlflow.log_metrics(metricas)

        # Se guarda el modelo entrenado como artefacto para poder registrarlo
        # y cargarlo más adelante sin volver a entrenar.
        # El ejemplo de entrada permite a MLflow anotar qué columnas espera el modelo
        mlflow.sklearn.log_model(sk_model=modelo, name="modelo",
                                 input_example=X_entrenamiento.head(3),
                                 skops_trusted_types=TIPOS_CONFIABLES)
    return metricas


if __name__ == "__main__":
    # Se indica a MLflow dónde guardar el seguimiento y en qué experimento;
    # si el experimento no existe se crea, y si existe se reutiliza
    mlflow.set_tracking_uri(URI_SEGUIMIENTO)
    mlflow.set_experiment(NOMBRE_EXPERIMENTO)

    datos = leer_datos(RUTA_DATOS_LIMPIOS)
    X, y = separar_variables(datos)
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = dividir_datos(X, y)

    # Se recorre el diccionario de modelos: cada vuelta entrena un modelo
    # y crea un run en MLflow, por lo que se obtiene un run por modelo
    for nombre_modelo, modelo in crear_modelos().items():
        metricas = entrenar_y_registrar(nombre_modelo, modelo, X_entrenamiento,
                                        X_prueba, y_entrenamiento, y_prueba)
        print(f"{nombre_modelo}: RMSE={metricas['rmse']:.2f} (registrado en MLflow)")
