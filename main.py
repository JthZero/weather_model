from Src.preprocesamiento import preprocesar_datos
from Src.entrenamiento import entrenar_modelo
#from Src.prediccion import hacer_predicciones
import pandas as pd
import os
import uvicorn

file_path = os.path.join('weather_model/dataset',os.listdir('weather_model/dataset')[0])
# Paso 1: Preprocesar los datos
df_dict_Loc, features = preprocesar_datos(file_path,log_dir = 'weather_model/Logs')

# Paso 2: Entrenar el modelo
mejor_nodos, mejor_precision = entrenar_modelo(df_dict_Loc, features, modelo_path = 'weather_model/Modelo', log_dir = 'weather_model/Logs')

# predicciones = hacer_predicciones(model_file='modelo_entrenado.joblib', X_test=X_test)

# # Paso 4: Mostrar los resultados
# resultados = pd.DataFrame({'Real': y_test, 'Predicción': predicciones})
# print(resultados.head())

# # Guardar los resultados en un archivo CSV
# resultados.to_csv('resultados_predicciones.csv', index=False)
# print("Resultados guardados en 'resultados_predicciones.csv'")



if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

