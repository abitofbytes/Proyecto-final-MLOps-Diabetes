"""API que sirve el modelo champion: estado del servicio, información del modelo y predicciones."""

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from esquemas import DatosPaciente, RespuestaPrediccion
from modelo import ALIAS_CAMPEON, NOMBRE_MODELO_REGISTRADO, cargar_modelo_champion, estado

logging.basicConfig(level=logging.INFO)
registro = logging.getLogger("api")


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    """Carga el modelo una sola vez al arrancar la API."""
    # Se captura cualquier error de carga para que la API arranque igualmente:
    # /health seguirá respondiendo y /predict devolverá 503 en vez de caerse
    try:
        cargar_modelo_champion()
        registro.info("Modelo cargado: %s versión %s", NOMBRE_MODELO_REGISTRADO, estado["version"])
    except Exception as error:
        registro.error("No se pudo cargar el modelo champion: %s", error)
    yield


app = FastAPI(title="API de predicción de diabetes", lifespan=ciclo_de_vida)


def comprobar_modelo_cargado() -> None:
    """Lanza un error 503 si el modelo no está disponible."""
    if estado["modelo"] is None:
        raise HTTPException(status_code=503,
                            detail="Modelo no disponible. Ejecuta antes run_pipeline.py")


@app.get("/health")
def comprobar_salud() -> dict:
    """Indica que el servicio está vivo, aunque el modelo no esté cargado."""
    return {"status": "ok"}


@app.get("/model-info")
def informar_modelo() -> dict:
    """Devuelve qué modelo se está sirviendo y sus datos de origen."""
    comprobar_modelo_cargado()
    return {
        "nombre": NOMBRE_MODELO_REGISTRADO,
        "version": estado["version"],
        "alias": ALIAS_CAMPEON,
        "run_id": estado["run_id"],
        "descripcion": estado["descripcion"],
    }


@app.post("/predict")
def predecir(datos: DatosPaciente) -> RespuestaPrediccion:
    """Devuelve la predicción del modelo para los datos de un paciente."""
    comprobar_modelo_cargado()

    # Se convierte el JSON validado en una tabla de una fila con las columnas
    # en el mismo orden que el esquema, que es el orden del entrenamiento
    tabla_entrada = pd.DataFrame([datos.model_dump()])

    # Se captura cualquier fallo del modelo para devolver un 500 con un mensaje
    # claro y dejar el detalle técnico en el log del servidor
    try:
        prediccion = estado["modelo"].predict(tabla_entrada)[0]
    except Exception:
        registro.exception("Error al generar la predicción")
        raise HTTPException(status_code=500, detail="Error al generar la predicción")

    return RespuestaPrediccion(prediccion=float(prediccion), version_modelo=estado["version"])
