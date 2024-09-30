import pandas as pd
import numpy as np
import joblib
from Src.preprocesamiento import preprocesar_datos
from Src.logger_config import configurar_logger
import os
import logging


def hacer_predicciones_generales(modelo_path, datos_nuevos_path, resultados_path, log_dir):
    try:

        log_dir = 'weather_model/Logs'
        log_file = os.path.join(log_dir, 'predicciones.log')
    
        # Crear la carpeta de logs si no existe
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Configuración del logger específico para entrenamiento
        logger_prediccion = logging.getLogger('predicciones')
        logger_prediccion.setLevel(logging.INFO)

        # Handler para escribir en archivo
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        # Agregar el handler al logger
        logger_prediccion.addHandler(file_handler)
        

        logger_prediccion.info("Iniciando predicciones...")

        # Cargar el modelo entrenado
        modelo_cargado = joblib.load(modelo_path)
        logger_prediccion.info(f"Modelo cargado desde {modelo_path}")

        # Llamar al preprocesamiento para los nuevos datos
        df_dict_Loc, features_generadas = preprocesar_datos(datos_nuevos_path, log_dir)

        # Crear un DataFrame para almacenar las predicciones de todas las ubicaciones
        predicciones_totales = pd.DataFrame()
        logger_prediccion.info(f"Realizando predicciones")
        # Iterar sobre todas las ubicaciones para hacer predicciones
        for ubicacion in df_dict_Loc.keys():
            

            # Obtener las características preprocesadas para esta ubicación
            datos_ubicacion = df_dict_Loc[ubicacion].copy()

            # Verificar si las características de esta ubicación fueron generadas por el preprocesamiento
            if ubicacion in features_generadas:
                features_ubicacion = features_generadas[ubicacion]

                # Seleccionar solo las columnas necesarias para hacer predicciones
                X_nuevos_datos_ubicacion = datos_ubicacion[features_ubicacion].copy()

                # Hacer predicciones con el modelo cargado
                predicciones = modelo_cargado.predict(X_nuevos_datos_ubicacion)

                # Añadir las predicciones al DataFrame original
                datos_ubicacion['Prediccion_RainTomorrow'] = predicciones

                # Concatenar las predicciones de esta ubicación a las predicciones totales
                predicciones_totales = pd.concat([predicciones_totales, datos_ubicacion])

            else:
                logger_prediccion.warning(f"No hay características definidas para {ubicacion}. Saltando...")

        # Guardar todas las predicciones en un archivo CSV
        predicciones_totales[['Date', 'Location', 'RainTomorrow', 'Prediccion_RainTomorrow']].to_csv(resultados_path, index=False)

        logger_prediccion.info(f"Predicciones guardadas en '{resultados_path}'")
        print(f"Predicciones guardadas en '{resultados_path}'")

    except Exception as e:
        logger_prediccion.error(f"Error durante las predicciones: {str(e)}")
        print(f"Error durante las predicciones: {str(e)}")
        raise