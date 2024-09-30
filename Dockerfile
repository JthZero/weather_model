# Usamos una imagen base de Python con la versión 3.9 (puedes cambiarla según tu necesidad)
FROM python:3.10-slim

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos el archivo de requisitos al directorio de trabajo
COPY requirements.txt .

# Instalamos las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el contenido del directorio actual al contenedor
COPY . .

# Exponemos el puerto en el que correrá la aplicación (8000 es el puerto por defecto de FastAPI)
EXPOSE 8000

# Comando para correr la aplicación utilizando Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

