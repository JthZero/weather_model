import logging
import os

def configurar_logger(log_dir, log_file):
    # Crear la carpeta de logs si no existe
    os.makedirs(log_dir, exist_ok=True)
    
    log_path = os.path.join(log_dir, log_file)

    # Configurar el logger solo si no se ha configurado previamente
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),    # Log a archivo
                logging.StreamHandler()           # Log a consola
            ]
        )
    return logger
