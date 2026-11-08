"""Define la forma de los datos que entran y salen de la API."""

from pydantic import BaseModel, ConfigDict


class DatosPaciente(BaseModel):
    """Variables de entrada de un paciente, con los mismos nombres que
    data/processed/diabetes.csv (sin la columna objetivo)."""

    # Se rechaza cualquier campo que el modelo no conoce (por ejemplo "target")
    # para que un error de escritura en un nombre no pase desapercibido
    model_config = ConfigDict(extra="forbid")

    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


class RespuestaPrediccion(BaseModel):
    """Lo que devuelve /predict: la predicción y la versión del modelo que la hizo."""

    prediccion: float
    version_modelo: str