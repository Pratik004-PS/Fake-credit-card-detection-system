import os
import logging
from logging.handlers import RotatingFileHandler

def get_logger(name: str, log_file: str = "logs/app.log", level: int = logging.INFO) -> logging.Logger:
    """
    Setup and return a logger with standard console and rotating file handlers.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if logger is already configured
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler (rotating log files up to 5MB, max 5 backup copies)
    file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
