"""Extracción de datos: carga el dataset diabetes de scikit-learn y lo guarda en data/raw/."""

# Path, de la librería estándar de Python, para manejar rutas de archivos
from pathlib import Path

# Librería pandas para trabajar con tablas, importada con el alias "pd"
import pandas as pd

# Función de scikit-learn que proporciona el dataset diabetes
from sklearn.datasets import load_diabetes

# Ruta del CSV de datos crudos, relativa a la carpeta desde la que se ejecuta el script
RUTA_DATOS_CRUDOS = Path("data/raw/diabetes.csv")


# Definición de la función (no se ejecuta hasta que se llama); "-> pd.DataFrame" indica el tipo del resultado: una tabla de pandas
def cargar_datos() -> pd.DataFrame:
    """Carga el dataset diabetes y lo devuelve como tabla con la columna target."""

    # 1) Se carga el paquete del dataset: contiene .data, .feature_names y .target por separado
    datos_diabetes = load_diabetes()

    # 2) Se construye la tabla con los valores de .data y los nombres de columna de .feature_names
    tabla_diabetes = pd.DataFrame(datos_diabetes.data, columns=datos_diabetes.feature_names)

    # 3) Se añade la variable objetivo (progresión de la enfermedad) como columna "target"
    tabla_diabetes["target"] = datos_diabetes.target

    # Se devuelve la tabla completa: 10 features + target
    return tabla_diabetes


# Función que recibe la tabla y una ruta; "-> None" indica que no devuelve nada, solo escribe en disco
def guardar_datos(tabla: pd.DataFrame, ruta: Path = RUTA_DATOS_CRUDOS) -> None:
    """Guarda la tabla en formato CSV, creando la carpeta de destino si no existe."""

    # Se crea la carpeta data/raw (y data si falta); exist_ok=True evita el error si ya existe
    ruta.parent.mkdir(parents=True, exist_ok=True)

    # Se escribe la tabla en CSV; index=False evita guardar la columna de números de fila
    tabla.to_csv(ruta, index=False)


# Bloque principal: solo se ejecuta al lanzar "python data_extraction.py", no al importar el archivo
if __name__ == "__main__":
    # Se cargan los datos en memoria
    tabla_diabetes = cargar_datos()

    # Se guardan los datos en data/raw/diabetes.csv
    guardar_datos(tabla_diabetes)

    # Se muestra un mensaje de confirmación con la ruta y el tamaño de la tabla
    print(f"Datos guardados en {RUTA_DATOS_CRUDOS}: {tabla_diabetes.shape[0]} filas y {tabla_diabetes.shape[1]} columnas")