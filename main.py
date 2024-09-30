from Src.preprocesamiento import preprocesar_datos
from Src.entrenamiento import entrenar_modelo
import pandas as pd
import os
import uvicorn


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

