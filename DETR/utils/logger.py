"""
Creating a logging setup for the detr motivated learning experimentation.
Since this is a learning process the target is get familiar with logger instead of using print(). 
Do not want to print and run the same script everytime.
"""


import logging
import os
from datetime import datetime


def setup_logger( name = 'detr', log_dir = 'outputs/logs', level = logging.INFO):
    logger = logging.getLogger(name)
    
    if not logger.hasHandlers():
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s, %(name)s, %(levelname)s, %(message)s')
        
        timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
        log_filename = f"{name}_{timestamp}.log"
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
