import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import os
import traceback
import logging
from Src.logger_config import configurar_logger
import pickle




def entrenar_modelo(df_dict_Loc, features,modelo_path, log_dir):  # Agregamos log_dir para definir la carpeta de logs
    # Ruta del archivo de log para el entrenamiento
    log_dir = 'weather_model/Logs'
    log_file = os.path.join(log_dir, 'entrenamiento.log')
    
    # Crear la carpeta de logs si no existe
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configuración del logger específico para entrenamiento
    logger_train = logging.getLogger('entrenamiento')
    logger_train.setLevel(logging.INFO)

    # Handler para escribir en archivo
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Agregar el handler al logger
    logger_train.addHandler(file_handler)

    # Ejemplo de uso
    try:
        
        logger_train.info("Iniciando entrenamiento del modelo.")
        # Combinar los datos de todas las ubicaciones en un solo DataFrame
        X_total = pd.DataFrame()
        y_total = pd.Series(dtype='float64')

        for key in df_dict_Loc.keys():
            # Combinar las características (features) de cada ubicación
            X_total = pd.concat([X_total, df_dict_Loc[key][features[key]].copy()])
            # Combinar las etiquetas ('RainTomorrow') de cada ubicación
            y_total = pd.concat([y_total, df_dict_Loc[key]['RainTomorrow'].copy()])

        logger_train.info("Datos combinados de todas las ubicaciones con éxito.")
       
       
        # Dividir los datos en conjunto de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.33, random_state=324)
        


        # Crear un DataFrame para almacenar los resultados de precisión
        val_met = pd.DataFrame(columns=['node', 'accuracy'], index=range(22))

        # Ajustar el modelo de árbol de decisión con diferentes cantidades de nodos
        for i in range(3, 25):
            Rain_Tomorrow = DecisionTreeClassifier(max_leaf_nodes=i, random_state=0)
            Rain_Tomorrow.fit(X_train, y_train)
            predictions = Rain_Tomorrow.predict(X_test)

            # Guardar los resultados de precisión
            j = i - 3
            val_met.iloc[j, 0] = i  # Número de nodos máximos
            val_met.iloc[j, 1] = accuracy_score(y_true=y_test, y_pred=predictions)  # Precisión

            # Log de la precisión de cada modelo
            logger_train.info(f"Nodos: {i}, Precisión: {accuracy_score(y_true=y_test, y_pred=predictions)}")

        # Asegurarnos de que no haya NaN en la columna de precisión
        val_met['accuracy'] = pd.to_numeric(val_met['accuracy'], errors='coerce')

        # Encontrar el valor de nodos máximos que dio la mejor precisión
        mejor_nodos = val_met.iloc[val_met['accuracy'].idxmax()]['node']
        mejor_precision = val_met['accuracy'].max()

        # Entrenar el modelo nuevamente con el mejor número de nodos
        mejor_modelo = DecisionTreeClassifier(max_leaf_nodes=int(mejor_nodos), random_state=0)
        mejor_modelo.fit(X_train, y_train)

        # Guardar el mejor modelo
        modelo_path = os.path.join(modelo_path, 'mejor_modelo_arbol_decision.pkl')
        joblib.dump(mejor_modelo, modelo_path)

        # Log final con el éxito del entrenamiento
        logger_train.info(f"El modelo con {mejor_nodos} nodos y precisión de {mejor_precision} ha sido guardado en {modelo_path}.")

        return mejor_modelo, val_met

    except Exception as e:
        logger_train.info(f"Error durante el entrenamiento: {str(e)}")
        raise
