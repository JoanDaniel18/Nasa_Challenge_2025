"""
Configuration settings for the Climate Prediction System
"""
import os
from datetime import datetime, timedelta

# NASA HARMONY API Configuration
NASA_EDL_USERNAME = os.getenv("NASA_EDL_USERNAME", "")
NASA_EDL_PASSWORD = os.getenv("NASA_EDL_PASSWORD", "")
HARMONY_BASE_URL = "https://harmony.earthdata.nasa.gov/"

# Ecuador bounding box coordinates
ECUADOR_BBOX = {
    'west': -81.0,   # -81°W
    'south': -5.0,   # -5°S  
    'east': -75.0,   # -75°W
    'north': 5.0     # 5°N (changed from -5°N to 5°N for proper coverage)
}

# NASA Collections for Climate Data
NASA_COLLECTIONS = {
    'precipitation': 'C1598621096-GES_DISC',  # GPM IMERG precipitation
    'temperature': 'C1276812863-GES_DISC',   # AIRS temperature
    'humidity': 'C1276812863-GES_DISC',      # AIRS humidity
    'pressure': 'C1276812863-GES_DISC'      # AIRS pressure
}

# Model configuration
MODEL_CONFIG = {
    'train_test_split': 0.8,
    'random_state': 42,
    'n_estimators': 100,
    'max_depth': 10,
    'learning_rate': 0.1
}

# Cache configuration
CACHE_CONFIG = {
    'data_retention_days': 7,
    'model_retention_days': 30,
    'max_memory_mb': 500
}

# API configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'title': 'Ecuador Climate Prediction API',
    'version': '1.0.0'
}

# Data directories
DATA_DIR = './data'
MODELS_DIR = './models'
CACHE_DIR = './cache'

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Time ranges for data fetching
DEFAULT_TIME_RANGE = {
    'start': datetime.now() - timedelta(days=365),  # Last year of data
    'end': datetime.now()
}
