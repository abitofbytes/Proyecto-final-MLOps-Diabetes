"""Validación de datos: comprueba que data/raw/diabetes.csv tiene el esquema esperado y detiene el pipeline si no."""

# Path, de la librería estándar de Python, para manejar rutas de archivos
from pathlib import Path

# Librería pandas para trabajar con tablas, importada con el alias "pd"
import pandas as pd

# Ruta del CSV de datos crudos, relativa a la carpeta desde la que se ejecuta el script
RUTA_DATOS_CRUDOS = Path("data/raw/diabetes.csv")

# Nombre de la columna que el modelo debe predecir
COLUMNA_OBJETIVO = "target"

# Columnas de entrada que el modelo necesita (nombres originales del dataset)
COLUMNAS_REQUERIDAS = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]


# Función que recibe una ruta y devuelve la tabla leída; "-> pd.DataFrame" indica el tipo del resultado
def leer_datos(ruta: Path = RUTA_DATOS_CRUDOS) -> pd.DataFrame:
    """Lee el CSV de datos crudos y lo devuelve como tabla."""

    # Se carga el CSV en una tabla de pandas
    tabla = pd.read_csv(ruta)

    # Se devuelve la tabla leída
    return tabla


# Función que recibe la tabla; "-> None" indica que no devuelve nada: o pasa en silencio o lanza una excepción
def validar_datos(tabla: pd.DataFrame) -> None:
    """Comprueba el esquema de la tabla y lanza una excepción si alguna comprobación falla."""

    # 1) Se comprueba que la tabla tiene al menos una fila
    if len(tabla) == 0:
        raise ValueError("El conjunto de datos está vacío.")

    # 2) Se comprueba que existe la columna objetivo
    if COLUMNA_OBJETIVO not in tabla.columns:
        raise ValueError(f"Falta la columna objetivo '{COLUMNA_OBJETIVO}'.")

    # 3) Se anotan en una lista las columnas requeridas que no aparecen en la tabla
    columnas_faltantes = []
    for columna in COLUMNAS_REQUERIDAS:
        if columna not in tabla.columns:
            columnas_faltantes.append(columna)

    # Se detiene la validación si falta alguna, indicando todas a la vez
    if columnas_faltantes:
        raise ValueError(f"Faltan columnas requeridas: {columnas_faltantes}")

    # 4) Se obtienen los nombres de las columnas que no son numéricas
    columnas_no_numericas = tabla.select_dtypes(exclude="number").columns.tolist()

    # Se detiene la validación si existe alguna columna no numérica
    if columnas_no_numericas:
        raise TypeError(f"Hay columnas no numéricas: {columnas_no_numericas}")

    # 5) Se cuentan los valores nulos de toda la tabla (primero por columna y luego en total)
    total_nulos = int(tabla.isna().sum().sum())

    # Se detiene la validación si hay algún valor nulo
    if total_nulos > 0:
        raise ValueError(f"Hay {total_nulos} valores nulos en los datos.")


# Bloque principal: solo se ejecuta al lanzar "python data_validation.py", no al importar el archivo
if __name__ == "__main__":
    # Se leen los datos crudos desde data/raw/diabetes.csv
    tabla_diabetes = leer_datos()

    # Se validan los datos; si algo falla, la excepción detiene el script aquí con código de salida 1
    validar_datos(tabla_diabetes)

    # Se muestra un mensaje de confirmación con la ruta y el tamaño de la tabla
    print(f"Validación superada en {RUTA_DATOS_CRUDOS}: {tabla_diabetes.shape[0]} filas y {tabla_diabetes.shape[1]} columnas")