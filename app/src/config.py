from pathlib import Path
from typing import Set, List

class Config:
    MODEL_PATH: Path = Path("../ml/model/lstm_model.h5")
    WINDOW_SIZE: int = 672
    FEATURES: List[str] = ['energy', 'power']
    TEMP_DIR: Path = Path("temp")
    VALID_FACTORIES: Set[str] = {'factory1', 'factory2', 'factory3', 'factory4', 'factory5'}
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

# Create temp directory on import
Config.TEMP_DIR.mkdir(exist_ok=True)

# Singleton instance
config = Config()