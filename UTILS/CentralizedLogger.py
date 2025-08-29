import logging
import sys
from datetime import datetime

class CentralizedLogger:
    
    def __init__(self, name='MLProject', log_level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(f'logs/ml_project_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def get_logger(self):
        return self.logger

# Singleton pattern for global access
_logger_instance = None

def get_logger(name='MLProject'):
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = CentralizedLogger(name)
    return _logger_instance.get_logger()