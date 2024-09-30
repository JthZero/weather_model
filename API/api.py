from fastapi import FastAPI, UploadFile, File
import os
from Src.preprocesamiento import preprocesar_datos
from Src.entrenamiento import entrenar_modelo
from Src.prediccion import hacer_predicciones_generales
from dotenv import load_dotenv


# Definir rutas relevantes
MODEL_PATH = os.getenv('MODEL_PATH')
RESULTADOS_PATH = os.getenv('RESULTADOS_PATH')
LOG_DIR = os.getenv('LOG_DIR')
DATASET_DIR_TRAIN = os.getenv('DATASET_DIR_TRAIN')
DATASET_DIR_PREDICT = os.getenv('DATASET_DIR_PREDICT')

app = FastAPI()
# Definir tus rutas/endpoints aquí
@app.get("/")
def read_root():
    return {"message": "API is running"}
# Endpoint para entrenar el modelo
@app.post("/entrenar/")
async def entrenar_modelo_endpoint(file: UploadFile = File(...)):
    """
    Endpoint para entrenar el modelo.
    Recibe un archivo CSV con los datos de entrenamiento y ejecuta el pipeline de preprocesamiento y entrenamiento.
    """
    try:
        # Guardar el archivo subido en el directorio de datasets
        dataset_path = os.path.join(DATASET_DIR_TRAIN, file.filename)
        with open(dataset_path, "wb") as f:
            f.write(await file.read())
        
        # Preprocesar los datos de entrenamiento
        df_dict_Loc, features = preprocesar_datos(dataset_path, log_dir=LOG_DIR)
        
        # Entrenar el modelo con los datos preprocesados
        mejor_modelo, val_met = entrenar_modelo(
            df_dict_Loc=df_dict_Loc,
            features=features,
            modelo_path=MODEL_PATH,
            log_dir=LOG_DIR
        )

        return {"mensaje": f"Modelo entrenado con éxito. Guardado en {MODEL_PATH}"}
    
    except Exception as e:
        return {"error": f"Error durante el entrenamiento: {str(e)}"}

# Endpoint para realizar predicciones
@app.post("/predicciones/")
async def realizar_predicciones(file: UploadFile = File(...)):
    """
    Endpoint para realizar predicciones generales.
    Sube un archivo con datos para hacer predicciones y ejecuta el pipeline completo.
    """
    try:
        # Guardar el archivo subido en el directorio de datasets
        dataset_path = os.path.join(DATASET_DIR_PREDICT, file.filename)
        with open(dataset_path, "wb") as f:
            f.write(await file.read())
        
        # Llamar a la función de hacer predicciones generales
        hacer_predicciones_generales(
            modelo_path=MODEL_PATH,
            datos_nuevos_path=dataset_path,
            resultados_path=RESULTADOS_PATH,
            log_dir=LOG_DIR
        )
        os.remove(dataset_path)
        return {"mensaje": f"Predicciones realizadas con éxito. Resultados guardados en {RESULTADOS_PATH}"}
    
    except Exception as e:
        return {"error": f"Error al realizar predicciones: {str(e)}"}

# Endpoint para eliminar el dataset de entrenamiento
@app.delete("/dataset/")
async def eliminar_dataset():
    """
    Endpoint para eliminar el dataset de entrenamiento existente.
    """
    try:
        # Eliminar todos los archivos dentro de la carpeta del dataset
        for file in os.listdir(DATASET_DIR_TRAIN):
            file_path = os.path.join(DATASET_DIR_TRAIN, file)
            os.remove(file_path)
        
        return {"mensaje": "Dataset de entrenamiento eliminado con éxito."}
    
    except Exception as e:
        return {"error": f"Error al eliminar el dataset: {str(e)}"}

