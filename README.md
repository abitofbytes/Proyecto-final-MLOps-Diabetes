# Proyecto final MLOps: predicción de la progresión de la diabetes

Proyecto MLOps básico de principio a fin: un problema de regresión con scikit-learn que pasa de un notebook de exploración a un repositorio con pipeline de datos, seguimiento y registro de modelos con MLflow, una API con FastAPI y automatización con GitHub Actions.

---

## Índice

1. [Objetivo del proyecto](#1-objetivo-del-proyecto)
2. [Dataset](#2-dataset)
3. [Estructura del repositorio](#3-estructura-del-repositorio)
4. [Instalación](#4-instalación)
5. [Ejecutar todo el pipeline](#5-ejecutar-todo-el-pipeline)
6. [Pipeline de datos](#6-pipeline-de-datos)
7. [Entrenamiento y seguimiento con MLflow](#7-entrenamiento-y-seguimiento-con-mlflow)
8. [Registro del modelo y selección del champion](#8-registro-del-modelo-y-selección-del-champion)
9. [API con FastAPI](#9-api-con-fastapi)
10. [Probar /health, /model-info y /predict](#10-probar-health-model-info-y-predict)
11. [CI/CD con GitHub Actions](#11-cicd-con-github-actions)
12. [Errores que gestiona la API](#12-errores-que-gestiona-la-api)
13. [Conclusiones](#13-conclusiones)

---

## 1. Objetivo del proyecto

Partir de un notebook de exploración y modelado y convertirlo en un repositorio MLOps que:

1. Extrae, valida y limpia los datos con scripts reutilizables.
2. Entrena tres modelos de regresión.
3. Registra los experimentos en MLflow (parámetros, métricas y modelos).
4. Registra el mejor modelo en el Model Registry de MLflow y le asigna el alias `champion` solo si supera un umbral de calidad (quality gate).
5. Sirve el modelo `champion` con una API de FastAPI, con manejo básico de errores.
6. Ejecuta el pipeline completo con GitHub Actions en cada push a `main` o `master`.

## 2. Dataset

Dataset *Diabetes* de scikit-learn (`sklearn.datasets.load_diabetes`):

- 442 pacientes y 10 variables numéricas de entrada, ya escaladas por scikit-learn: edad (`age`), sexo (`sex`), índice de masa corporal (`bmi`), presión arterial media (`bp`) y seis medidas de suero sanguíneo (`s1` a `s6`).
- Variable objetivo `target`: nivel de progresión de la enfermedad un año después, con valores entre 25 y 346 (media ≈ 152).
- No contiene valores nulos ni filas duplicadas.

Se conservan los nombres originales de las columnas en todo el proyecto (limpieza, entrenamiento y esquema de la API), para que todos los pasos usen exactamente los mismos nombres.

### Notebook de partida

`notebooks/Proyecto_MLOps.ipynb` contiene la exploración (estadísticas, nulos, duplicados, gráficas) y la comparación de modelos en los que se basan los scripts. Resultados del notebook (20 % de prueba, semilla 42):

| Modelo | RMSE |
|---|---|
| Gradient Boosting | 53.84 |
| Regresión lineal | 53.85 |
| Random Forest | 54.33 |

Los tres modelos quedan prácticamente empatados. En el repositorio la elección es automática: `registrar_modelo.py` selecciona el de menor RMSE (sección 8). Las gráficas y el análisis exploratorio se quedan solo en el notebook, porque no hacen falta para entrenar ni para servir el modelo.

## 3. Estructura del repositorio

```
Proyecto-final-MLOps-Diabetes/
├── .github/
│   └── workflows/
│       └── ci.yml                  # CI: ejecuta el pipeline en cada push
├── .gitignore                      # Archivos que Git no sube (entorno, datos, MLflow)
├── notebooks/
│   └── Proyecto_MLOps.ipynb        # Notebook de exploración de partida
├── README.md                       # Este documento
├── requirements.txt                # Dependencias con versiones fijadas
└── src/
    ├── datos/                      # Lógica de datos
    │   ├── data_extraction.py      # Carga el dataset y lo guarda en data/raw/
    │   ├── data_validation.py      # Comprueba la estructura de los datos crudos
    │   └── data_cleaning.py        # Limpia los datos y los guarda en data/processed/
    ├── entrenamiento/              # Lógica de entrenamiento
    │   └── train.py                # Entrena 3 modelos y los registra en MLflow
    ├── registro/                   # Lógica de registro y evaluación
    │   ├── registrar_modelo.py     # Registra el run con menor RMSE en el Model Registry
    │   └── evaluar_modelo.py       # Quality gate: asigna "champion" si RMSE < 60
    ├── pipeline/                   # Orquestación
    │   └── run_pipeline.py         # Ejecuta los 6 pasos en orden y para si uno falla
    └── api/                        # Servicio del modelo
        ├── esquemas.py             # Esquemas Pydantic de entrada y salida
        ├── modelo.py               # Carga el modelo champion desde MLflow
        └── main.py                 # Endpoints /health, /model-info y /predict
```

Cada carpeta de `src/` tiene una sola responsabilidad. Los pasos no se importan entre sí: se comunican solo a través de lo que dejan en disco (los CSV de `data/` y la base de datos de MLflow).

El código está en castellano. Los scripts de cada paso (`data_extraction.py`, `data_validation.py`, `data_cleaning.py`, `train.py`), los endpoints y los archivos de configuración mantienen la nomenclatura estándar en inglés, igual que `run_pipeline.py` y `main.py` (nombre habitual de una aplicación FastAPI). Los identificadores solo usan caracteres ASCII para evitar problemas de codificación entre sistemas.

Al clonar el repositorio no aparecen `.venv/`, `data/`, `mlflow.db`, `mlruns/` ni `mlartifacts/`. Están en `.gitignore` porque se generan en local al instalar y ejecutar el proyecto, y así cada ejecución (también la del CI) parte de cero.

## 4. Instalación

Requisitos: Python 3.12 y Git. Los comandos son para Windows (cmd). En Linux o macOS, cambia `\` por `/` y activa el entorno con `source .venv/bin/activate`.

```cmd
REM Descarga una copia del repositorio en una carpeta nueva
git clone https://github.com/abitofbytes/Proyecto-final-MLOps-Diabetes.git

REM Entra en la carpeta del proyecto
cd Proyecto-final-MLOps-Diabetes

REM Crea un entorno virtual aislado en la carpeta .venv
python -m venv .venv

REM Activa el entorno: a partir de aquí python y pip usan .venv
.venv\Scripts\activate

REM Instala las librerías con las versiones exactas de requirements.txt
pip install -r requirements.txt
```

Comprobación, con el entorno activado:

```cmd
python --version
pip show mlflow
```

Debe mostrar `Python 3.12.x` y `Version: 3.18.0`.

| Librería | Versión | Para qué se usa |
|---|---|---|
| pandas | 3.1.1 | Leer, validar y limpiar los datos |
| scikit-learn | 1.9.1 | Dataset, modelos y métricas |
| mlflow | 3.18.0 | Seguimiento de experimentos y Model Registry |
| fastapi | 0.162.0 | API |
| uvicorn | 0.54.0 | Servidor que ejecuta la API |
| pydantic | 2.14.3 | Validación del JSON de entrada |

Las versiones se fijan con `==` para que cualquier persona (y el CI) instale exactamente las versiones con las que se ha probado el proyecto. Las librerías de gráficas del notebook no se incluyen porque el pipeline y la API no las necesitan.

**Todos los comandos se ejecutan desde la raíz del proyecto**, porque las rutas a `data/` y a `mlflow.db` son relativas a esa carpeta.

## 5. Ejecutar todo el pipeline

Un solo comando ejecuta los seis pasos en orden: extracción → validación → limpieza → entrenamiento → registro → evaluación (quality gate).

```cmd
python src\pipeline\run_pipeline.py
```

Al final debe aparecer `Pipeline completado: todos los pasos terminaron bien`. Para comprobar el código de salida:

```cmd
echo %errorlevel%
```

`0` significa que todo fue bien; `1` significa que algún paso falló y el pipeline se detuvo en él.

### Cómo funciona el orquestador

- `run_pipeline.py` guarda la lista de pasos en un solo sitio (`PASOS`) y ejecuta cada script como un proceso aparte con `subprocess.run`.
- **Se usa `subprocess` en lugar de `import`** porque cada script ya funciona por sí solo con `python <script>.py`; el orquestador lo ejecuta exactamente igual que una persona en la terminal, y un fallo en un paso no deja estado a medias en memoria para el siguiente.
- **Se usa `sys.executable`** en lugar de escribir `python` para que cada paso use el mismo Python del entorno virtual que el orquestador.
- **Se comprueba `codigo != 0`** y no `codigo == 1`, porque un script también puede fallar con otros códigos (por ejemplo, `2` si la ruta del archivo no existe).
- Si un paso falla, el orquestador termina con su propio `sys.exit(1)`: el código del proceso hijo solo lo ve el orquestador, no la terminal ni GitHub Actions.
- Se usa `print(..., flush=True)` para que el título de cada paso aparezca antes que la salida del propio paso.

Las secciones 6, 7 y 8 explican cada paso y cómo ejecutarlo por separado.

## 6. Pipeline de datos

```cmd
REM Carga el dataset de scikit-learn y lo guarda en data/raw/diabetes.csv
python src\datos\data_extraction.py

REM Comprueba que los datos crudos tienen la estructura esperada
python src\datos\data_validation.py

REM Limpia los datos y los guarda en data/processed/diabetes.csv
python src\datos\data_cleaning.py
```

| Script | Entrada | Salida |
|---|---|---|
| `data_extraction.py` | `load_diabetes()` de scikit-learn | `data/raw/diabetes.csv` |
| `data_validation.py` | `data/raw/diabetes.csv` | Mensaje de validación superada o error |
| `data_cleaning.py` | `data/raw/diabetes.csv` | `data/processed/diabetes.csv` |

Salida esperada del último paso: `Datos limpios guardados en data\processed\diabetes.csv: 442 filas y 11 columnas (0 filas eliminadas)`.

### Validación

La validación se ejecuta antes de la limpieza: detiene el pipeline ante problemas que no tienen arreglo automático, de modo que la limpieza solo corrige problemas pequeños sobre datos con la estructura correcta. `data_validation.py` termina con un error (código de salida 1) si:

1. La tabla está vacía.
2. Falta la columna objetivo `target`.
3. Falta alguna de las columnas requeridas.
4. Alguna columna no es numérica.
5. Hay valores nulos.

Los errores se lanzan con `raise` y no con `assert`, porque Python ignora los `assert` al ejecutarse con la opción `-O`.

### Limpieza

| Operación | Motivo |
|---|---|
| Convertir todas las columnas a número (`pd.to_numeric(..., errors="coerce")`) | Los valores no convertibles pasan a nulo y se eliminan en el paso siguiente; es una red de seguridad por si los datos llegan sin validar |
| Eliminar filas con nulos (`dropna`) | El dataset no tiene nulos, por lo que no hace falta imputar. Con muchos nulos se imputaría la mediana de las variables de entrada, calculada solo con los datos de entrenamiento para evitar fuga de datos (data leakage); `target` nunca se imputa |
| Eliminar filas duplicadas | Una fila repetida pesa el doble en el entrenamiento |
| Eliminar filas con `target` menor o igual que 0 | `target` mide el nivel de la enfermedad, no un cambio, así que un valor menor o igual que 0 es un dato erróneo |
| Renumerar las filas (`reset_index`) | Deja un índice continuo tras eliminar filas |

No se aplica ingeniería de variables: las 10 variables ya vienen escaladas por scikit-learn y se usan igual que en el notebook, para que los resultados sean comparables.

## 7. Entrenamiento y seguimiento con MLflow

`train.py` lee `data/processed/diabetes.csv`, separa los datos en entrenamiento y prueba, entrena tres modelos y registra cada uno como un run del experimento `diabetes`.

```cmd
python src\entrenamiento\train.py
```

Salida esperada:

```
regresion_lineal: RMSE=53.85 (registrado en MLflow)
random_forest: RMSE=54.33 (registrado en MLflow)
gradient_boosting: RMSE=53.73 (registrado en MLflow)
```

El split es el mismo que en el notebook (20 % de prueba, semilla 42) para que los resultados sean reproducibles y comparables. Se comparan una regresión lineal (`LinearRegression`), como referencia sencilla, y dos modelos de árboles (`RandomForestRegressor` y `GradientBoostingRegressor`), para comprobar si la complejidad extra reduce el error. Las métricas se calculan con los datos de prueba, porque con los de entrenamiento el modelo ya conoce las respuestas y saldrían demasiado optimistas.

En cada run se registran:

| Qué | Contenido |
|---|---|
| Parámetros | Todos los hiperparámetros del modelo con `get_params()`, también los valores por defecto, para poder reproducir el modelo exacto |
| Métricas | `rmse` (principal: penaliza más los errores grandes y está en las unidades de `target`), `mae` y `r2`, sobre los datos de prueba |
| Artefacto | El modelo entrenado, con el nombre `modelo` y un ejemplo de entrada (`input_example`) del que MLflow deduce la firma (columnas y tipos) que después comprueba la API |

### Almacenamiento

- `mlflow.db` (SQLite): experimentos, runs, parámetros, métricas, modelos registrados, versiones y alias. Se usa SQLite porque MLflow 3.18 no admite la carpeta `mlruns/` como almacén de seguimiento y el Model Registry necesita una base de datos (MLflow 2.x no es compatible con pandas 3.1.1).
- `mlruns/`: los archivos de los modelos entrenados (artefactos).

MLflow 3 guarda los modelos de scikit-learn con el formato seguro skops, que rechaza por defecto los modelos de árboles. Por eso `log_model` recibe `skops_trusted_types=["sklearn.tree._tree.Tree"]`: los modelos los crea el propio script y no proceden de una fuente externa.

Para ver los resultados en la interfaz web de MLflow, en otra terminal con el entorno activado:

```cmd
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Se abre en http://127.0.0.1:5000, dentro del experimento `diabetes`. La pestaña *Models* muestra el modelo registrado `modelo_diabetes` y qué versión tiene el alias `champion`.

> El RMSE de Gradient Boosting es 53.73 frente a 53.84 en el notebook, porque los cálculos internos de este modelo varían ligeramente entre versiones de scikit-learn. Los otros dos modelos coinciden.

## 8. Registro del modelo y selección del champion

Dos scripts, en este orden:

```cmd
REM Registra en el Model Registry el run con menor RMSE
python src\registro\registrar_modelo.py

REM Aplica el quality gate y asigna el alias champion si lo supera
python src\registro\evaluar_modelo.py
```

Salida esperada (el número de versión aumenta en cada ejecución):

```
Mejor run: gradient_boosting (RMSE=53.73)
Registrado 'modelo_diabetes' versión 1
Versión 1: RMSE=53.73 (umbral 60.00)
Supera el quality gate: la versión 1 es ahora 'champion'
```

### Cómo se elige el champion

1. **`registrar_modelo.py`** busca, entre los runs terminados (`FINISHED`) del experimento `diabetes`, el de **menor RMSE**. Registra su modelo con la dirección `models:/<id_modelo>` (en MLflow 3 el modelo es un objeto propio, separado del run) y con el nombre fijo `modelo_diabetes`, de modo que cada registro crea una versión nueva del mismo modelo. Añade a la versión una descripción con el tipo de modelo y sus métricas, que la API muestra en `/model-info`.
2. **`evaluar_modelo.py`** toma la última versión registrada y lee el RMSE guardado en su run de origen, sin recalcularlo, para evaluar exactamente el resultado del entrenamiento. Después aplica el **quality gate**:
   - Si `RMSE < 60`: asigna el alias `champion` a esa versión. Como un alias solo puede apuntar a una versión, MLflow se lo quita a la anterior.
   - Si no: no asigna el alias y termina con **código de salida 1**, de modo que el orquestador y el CI se detienen y quedan en rojo. La versión `champion` anterior se conserva, así que la API sigue sirviendo un modelo que sí pasó el control.

Se usa un alias y no los stages (`Staging`, `Production`), obsoletos en MLflow 3: la API carga siempre `models:/modelo_diabetes@champion` sin cambiar su código en cada versión nueva. Con los datos actuales gana Gradient Boosting (RMSE 53.73), que supera el gate.

### Por qué el umbral es 60

- Un modelo que siempre predijera la media del `target` (sin aprender nada) tendría un RMSE de unos **73.2** con estos mismos datos de prueba. Es la referencia mínima (baseline).
- Exigir **RMSE < 60** obliga al modelo a mejorar esa referencia en al menos un 18 %, es decir, a aprender algo útil de las variables.
- Deja margen sobre el RMSE actual (≈ 53.7), para que pequeñas variaciones entre versiones de librerías no hagan fallar el CI, pero sí lo hagan fallar un cambio que estropee el modelo de verdad.

Para ver el gate en acción, basta con bajar temporalmente `UMBRAL_RMSE` a 50 en `evaluar_modelo.py`: el script termina con código 1, la nueva versión se queda sin alias y el champion anterior no cambia.

## 9. API con FastAPI

La API sirve la versión del modelo que tiene el alias `champion`. Antes de arrancarla hay que ejecutar el pipeline al menos una vez (sección 5).

```cmd
uvicorn main:app --app-dir src\api --port 8000
```

- `main:app` = archivo `main.py`, objeto `app`.
- `--app-dir src\api` = carpeta donde están los archivos de la API. Se arranca desde la raíz para que la ruta relativa a `mlflow.db` funcione.
- La documentación interactiva queda en http://127.0.0.1:8000/docs. Para pararla: `Ctrl+C`.

| Método | Ruta | Qué devuelve |
|---|---|---|
| `GET` | `/health` | `{"status": "ok"}` si el servicio está vivo |
| `GET` | `/model-info` | Nombre, versión, alias, `run_id` y descripción (métricas) del modelo servido |
| `POST` | `/predict` | Predicción y versión del modelo para los datos de un paciente |

### Cómo está organizada

- `esquemas.py`: `DatosPaciente` (las 10 variables como `float`, con los mismos nombres que el dataset) y `RespuestaPrediccion` (`prediccion` y `version_modelo`).
- `modelo.py`: busca qué versión tiene el alias `champion` y la carga con `mlflow.pyfunc.load_model`, la forma genérica de cargar modelos de MLflow, que además comprueba que las columnas de entrada coinciden con la firma guardada en el entrenamiento.
- `main.py`: los tres endpoints. El modelo se carga **una sola vez al arrancar** (función `ciclo_de_vida`, pasada como `lifespan`), no en cada petición, porque cargarlo es lento. Si la carga falla, el error queda en el log y la API arranca igualmente: `/health` responde y `/predict` devuelve 503.

Las carpetas de `src/` no son paquetes (no tienen `__init__.py`), por eso los módulos se importan directamente (`from esquemas import ...`) y la API se arranca con `--app-dir`.

## 10. Probar /health, /model-info y /predict

Con la API arrancada, la forma más sencilla es http://127.0.0.1:8000/docs: se elige un endpoint, *Try it out* → *Execute*, y se ve el código y el cuerpo de la respuesta.

También desde otra terminal con `curl` (incluido en Windows 10 y 11):

```cmd
REM Comprueba que el servicio está vivo
curl http://127.0.0.1:8000/health

REM Muestra qué modelo se está sirviendo
curl http://127.0.0.1:8000/model-info

REM Pide una predicción para un paciente (primera fila del dataset)
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"age\": 0.0381, \"sex\": 0.0507, \"bmi\": 0.0617, \"bp\": 0.0219, \"s1\": -0.0442, \"s2\": -0.0348, \"s3\": -0.0434, \"s4\": -0.0026, \"s5\": 0.0199, \"s6\": -0.0176}"
```

En cmd las comillas del JSON van escapadas con `\"`. En Linux o macOS se puede escribir el JSON entre comillas simples sin escapar.

Respuestas esperadas (el número de versión y la predicción dependen de la ejecución):

```json
{"status": "ok"}
```

```json
{"nombre": "modelo_diabetes", "version": "1", "alias": "champion", "run_id": "…", "descripcion": "Modelo gradient_boosting del experimento 'diabetes'. RMSE=53.73 MAE=… R2=…"}
```

```json
{"prediccion": …, "version_modelo": "1"}
```

Cuerpo de ejemplo para `/predict` en `/docs`:

```json
{
  "age": 0.0381, "sex": 0.0507, "bmi": 0.0617, "bp": 0.0219, "s1": -0.0442,
  "s2": -0.0348, "s3": -0.0434, "s4": -0.0026, "s5": 0.0199, "s6": -0.0176
}
```

## 11. CI/CD con GitHub Actions

El flujo está en `.github/workflows/ci.yml`:

- **Cuándo se ejecuta:** automáticamente en cada `push` a `main` o `master`.
- **Dónde:** en una máquina Linux limpia (`ubuntu-latest`) que GitHub crea para cada ejecución y destruye al terminar. Por eso no necesita subir `data/` ni `mlflow.db`: el pipeline los genera desde cero cada vez.
- **Pasos:**
  1. `actions/checkout@v7`: descarga el código del repositorio.
  2. `actions/setup-python@v7`: instala Python 3.12 (la misma versión que en local) con caché de pip para acelerar las siguientes ejecuciones. Se usan las versiones `@v7` de las acciones, que funcionan con Node.js 24.
  3. `pip install -r requirements.txt`: instala las versiones exactas.
  4. `python src/pipeline/run_pipeline.py`: ejecuta en orden extracción → validación → limpieza → entrenamiento → registro → evaluación. Se llama al orquestador y no a los seis scripts sueltos para que el orden esté definido en un solo sitio y el CI ejecute el mismo comando que se usa en local.
- **Resultado:** si cualquier paso falla (por ejemplo, la validación o el quality gate), ese script termina con código 1, el orquestador también sale con 1 y GitHub marca la ejecución en **rojo**. Si todo va bien, sale en **verde**.

El resultado se ve en la pestaña *Actions* del repositorio.

## 12. Errores que gestiona la API

| Caso | Código | Cómo se gestiona |
|---|---|---|
| Petición válida | **200 OK** | Devuelve la predicción y la versión del modelo |
| Falta una variable, tipo incorrecto (por ejemplo `"bmi": "alto"`) o campo desconocido (por ejemplo `"target"`) | **422 Unprocessable Entity** | Automático con Pydantic: FastAPI valida el JSON contra `DatosPaciente` antes de llamar a la función e indica qué campo falla. Los campos desconocidos se rechazan con `extra="forbid"`, para que un nombre mal escrito no pase desapercibido |
| Modelo no cargado (no existe `mlflow.db` o ninguna versión tiene el alias `champion`) | **503 Service Unavailable** | `comprobar_modelo_cargado()` se llama en `/model-info` y `/predict`. La API sigue viva y `/health` responde 200 |
| Fallo al predecir | **500 Internal Server Error** | `try/except` alrededor de `predict`: devuelve un mensaje claro y deja el detalle técnico en el log del servidor |

Cómo provocar el 503 en local (desde la raíz, con la API parada):

```cmd
REM Se aparta la base de datos para que la API no encuentre el modelo
ren mlflow.db mlflow_copia.db

REM Se arranca la API: /health da 200 y /predict da 503. Parar con Ctrl+C
uvicorn main:app --app-dir src\api --port 8000

REM Se borra la base vacía que crea MLflow al arrancar y se recupera la original
del mlflow.db
ren mlflow_copia.db mlflow.db
```

## 13. Conclusiones

- El proyecto sigue las prácticas habituales de un flujo MLOps: cada etapa (datos, entrenamiento, registro, servicio y automatización) vive en su propio módulo, los experimentos quedan registrados y son comparables, el modelo se versiona en un registro y solo se publica tras pasar un control de calidad, y todo el proceso es reproducible gracias a las versiones fijadas, las semillas y la integración continua.
- El flujo completo, de los datos crudos a una API que sirve el modelo `champion`, se ejecuta con un solo comando y de forma idéntica en local y en GitHub Actions.
- Los tres modelos obtienen resultados muy parecidos (RMSE entre 53.7 y 54.3). Gradient Boosting queda primero por poco, pero la regresión lineal, mucho más simple, rinde prácticamente igual: con 442 pacientes y 10 variables, la complejidad extra apenas aporta.
- El mejor modelo mejora en torno a un 27 % la referencia de predecir siempre la media (RMSE ≈ 73.2). Aun así, un error medio de unos 54 puntos sobre una escala de 25 a 346 indica que estas variables solo explican una parte de la progresión de la enfermedad.
- El quality gate y los códigos de salida hacen que un modelo que no alcanza el umbral nunca llegue a la API: el CI se marca en rojo y se sigue sirviendo el último champion válido.
