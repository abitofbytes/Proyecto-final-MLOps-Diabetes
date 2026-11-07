"""Limpieza de datos: lee data/raw/diabetes.csv, corrige problemas sencillos y guarda el resultado en data/processed/."""

# Path, de la librería estándar de Python, para manejar rutas de archivos
from pathlib import Path

# Librería pandas para trabajar con tablas, importada con el alias "pd"
import pandas as pd

# Ruta del CSV de datos crudos (entrada), relativa a la carpeta desde la que se ejecuta el script
RUTA_DATOS_CRUDOS = Path("data/raw/diabetes.csv")

# Ruta del CSV de datos limpios (salida); la carpeta processed indica que ya están limpios
RUTA_DATOS_PROCESADOS = Path("data/processed/diabetes.csv")

# Nombre de la columna que el modelo debe predecir
COLUMNA_OBJETIVO = "target"


# Función que recibe una ruta y devuelve la tabla leída; "-> pd.DataFrame" indica el tipo del resultado
def leer_datos(ruta: Path = RUTA_DATOS_CRUDOS) -> pd.DataFrame:
    """Lee el CSV de datos crudos y lo devuelve como tabla."""

    # Se carga el CSV en una tabla de pandas
    tabla = pd.read_csv(ruta)

    # Se devuelve la tabla leída
    return tabla

# Función que recibe la tabla cruda y devuelve una tabla nueva ya limpia
def limpiar_datos(tabla: pd.DataFrame) -> pd.DataFrame:
    """Corrige problemas sencillos: tipos no numéricos, nulos, duplicados y objetivo inválido."""

    # Se trabaja sobre una copia para no modificar la tabla original
    tabla_limpia = tabla.copy()

    # 1) Se convierte cada columna a número; lo que no se pueda convertir pasa a ser nulo (NaN)
    for columna in tabla_limpia.columns:
        tabla_limpia[columna] = pd.to_numeric(tabla_limpia[columna], errors="coerce")

    # 2) Se eliminan las filas con algún valor nulo; se elige borrar y no imputar porque el dataset
    #    no tiene nulos y la validación ya los rechaza (con muchos nulos se imputaría la mediana)
    tabla_limpia = tabla_limpia.dropna()

    # 3) Se eliminan las filas repetidas, conservando la primera aparición
    tabla_limpia = tabla_limpia.drop_duplicates()

    # 4) Se conservan solo las filas con objetivo positivo: el target mide el nivel de la enfermedad
    #    (rango real 25-346), así que un valor <= 0 se considera un dato roto
    tabla_limpia = tabla_limpia[tabla_limpia[COLUMNA_OBJETIVO] > 0]

    # Se renumeran las filas de 0 en adelante tras los borrados; drop=True descarta la numeración antigua
    tabla_limpia = tabla_limpia.reset_index(drop=True)

    # Se devuelve la tabla limpia
    return tabla_limpia

# Función que recibe la tabla y una ruta; "-> None" indica que no devuelve nada, solo escribe en disco
def guardar_datos(tabla: pd.DataFrame, ruta: Path = RUTA_DATOS_PROCESADOS) -> None:
    """Guarda la tabla limpia en CSV, creando la carpeta de destino si no existe."""

    # Se crea la carpeta data/processed si no existe; exist_ok=True evita el error si ya existe
    ruta.parent.mkdir(parents=True, exist_ok=True)

    # Se escribe la tabla en CSV; index=False evita guardar la columna de números de fila
    tabla.to_csv(ruta, index=False)


# Bloque principal: solo se ejecuta al lanzar "python data_cleaning.py", no al importar el archivo
if __name__ == "__main__":
    # Se leen los datos crudos desde data/raw/diabetes.csv
    tabla_cruda = leer_datos()

    # Se limpian los datos
    tabla_limpia = limpiar_datos(tabla_cruda)

    # Se guardan los datos limpios en data/processed/diabetes.csv
    guardar_datos(tabla_limpia)

    # Se calcula cuántas filas ha eliminado la limpieza
    filas_eliminadas = len(tabla_cruda) - len(tabla_limpia)

    # Se muestra un mensaje de confirmación con la ruta, el tamaño y las filas eliminadas
    print(f"Datos limpios guardados en {RUTA_DATOS_PROCESADOS}: {tabla_limpia.shape[0]} filas y {tabla_limpia.shape[1]} columnas ({filas_eliminadas} filas eliminadas)")