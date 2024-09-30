import pandas as pd
import numpy as np
import logging
import os
from Src.logger_config import configurar_logger


def preprocesar_datos(file_path, log_dir): #No elimina todo
    # Ruta del archivo de log para el preprocesamiento
    log_dir = 'weather_model/Logs'
    log_file = os.path.join(log_dir, 'preprocesamiento.log')

    # Crear la carpeta de logs si no existe
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configuración del logger específico para preprocesamiento
    logger_preproc = logging.getLogger('preprocesamiento')
    logger_preproc.setLevel(logging.INFO)

    # Handler para escribir en archivo
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Agregar el handler al logger
    logger_preproc.addHandler(file_handler)

    

    
    try:
        
        logger_preproc.info("Iniciando preprocesamiento de datos.")

       
        # Leer el archivo CSV
        df = pd.read_csv(file_path)
        
        # Copia del DataFrame, cambiamos valores de 'Yes' y 'No' a 1 y 0
        data = df.copy()
        if 'RainToday' in data.columns and 'RainTomorrow' in data.columns:
            data['RainToday'] = (data['RainToday'] == 'Yes') * 1
            data['RainTomorrow'] = (data['RainTomorrow'] == 'Yes') * 1
        else:
            raise ValueError("Las columnas 'RainToday' o 'RainTomorrow' no existen en el DataFrame.")
        
        # Mapeo de direcciones del viento a grados
        Ab_WD = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        WD = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5]
        wind_mapping = dict(zip(Ab_WD, WD))
        
        # Convertir las direcciones del viento en grados
        Col_WindDir = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
        for col in Col_WindDir:
            if col in data.columns:
                data[col] = data[col].map(wind_mapping)
        
        # Filtrar las columnas numéricas
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Obtener las ubicaciones únicas
        Loc = data['Location'].unique()
        df_dict_Loc = {key: group for key, group in data.groupby('Location')}
        
        # Procesar valores nulos: eliminar columnas con 100% nulos y rellenar con moda/media
        for key in df_dict_Loc.keys():
            # Eliminar columnas con 100% nulos
            null_perc = (df_dict_Loc[key].isnull().sum() * 100) / len(df_dict_Loc[key])
            cols_to_drop = null_perc[null_perc == 100.0].index
            df_dict_Loc[key] = df_dict_Loc[key].drop(cols_to_drop, axis=1)
            
            # Rellenar valores nulos: moda para columnas categóricas y media para numéricas
            for col in df_dict_Loc[key].columns:
                if df_dict_Loc[key][col].isnull().any():
                    if df_dict_Loc[key][col].dtype == 'object':
                        df_dict_Loc[key][col] = df_dict_Loc[key][col].fillna(df_dict_Loc[key][col].mode()[0])
                    else:
                        df_dict_Loc[key][col] = df_dict_Loc[key][col].fillna(df_dict_Loc[key][col].mean())
        
        # Calcular correlación
        coef_corr = {key: df_dict_Loc[key].select_dtypes(include=[np.number]).corr() for key in df_dict_Loc.keys()}
        
        # Filtrar columnas con correlación >= 0.1 con 'RainTomorrow'
        coef_corr_abs = {}
        features = {}
        for key in df_dict_Loc.keys():
            coef_corr_abs[key] = coef_corr[key][coef_corr[key]['RainTomorrow'].abs() >= 0.1]
            if 'RainTomorrow' in coef_corr_abs[key].index:
                coef_corr_abs[key] = coef_corr_abs[key].drop(['RainTomorrow'], axis=0)
            features[key] = coef_corr_abs[key].index.tolist()
        #logger_preproc.info(f"Features: {features}")
        logger_preproc.info("Preprocesamiento completado con éxito.")
        return df_dict_Loc, features

    except Exception as e:
        # Registrar el error en el archivo log
        logger_preproc.info(f"Error durante el preprocesamiento: {str(e)}")
        raise


