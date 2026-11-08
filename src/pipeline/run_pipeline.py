"""Ejecuta en orden todos los pasos del pipeline, de los datos al modelo campeón."""

import subprocess
import sys
from pathlib import Path

# Se listan los scripts en el orden en que deben ejecutarse, porque cada paso
# lee lo que dejó en disco el anterior (CSV en data/, runs y versiones en mlflow.db)
PASOS = [
    Path("src/datos/data_extraction.py"),
    Path("src/datos/data_validation.py"),
    Path("src/datos/data_cleaning.py"),
    Path("src/entrenamiento/train.py"),
    Path("src/registro/registrar_modelo.py"),
    Path("src/registro/evaluar_modelo.py"),
]


def ejecutar_paso(ruta: Path) -> int:
    """Ejecuta un script como proceso aparte y devuelve su código de salida."""
    print(f"\n=== Ejecutando {ruta} ===", flush=True)
    # Se usa sys.executable (el Python del venv activo) en lugar de escribir "python",
    # para que cada paso use exactamente las mismas librerías que el orquestador
    resultado = subprocess.run([sys.executable, str(ruta)])
    return resultado.returncode


if __name__ == "__main__":
    for ruta in PASOS:
        codigo = ejecutar_paso(ruta)
        # Se detiene el pipeline en el primer paso que falla, porque los siguientes
        # trabajarían con datos o modelos incorrectos o inexistentes.
        # Se sale con código 1 para que GitHub Actions detecte también el fallo
        if codigo != 0:
            print(f"\nFalló {ruta} (código {codigo}). Se detiene el pipeline")
            sys.exit(1)
    print("\nPipeline completado: todos los pasos terminaron bien")