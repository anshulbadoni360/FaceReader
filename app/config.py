"""Configuration settings."""

from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Set


class Settings(BaseSettings):
    """Application settings."""
    
    APP_NAME: str = "Face Emotion API"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Model paths
    MODEL_PATH: str = "checkpoints/best_model.pt"
    SCALER_PATH: str = "checkpoints/scaler_params.json"
    
    # Upload settings
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: Set[str] = {"jpg", "jpeg", "png", "webp", "bmp"}
    
    class Config:
        env_file = ".env"


settings = Settings()

# Create upload directory
Path(settings.UPLOAD_DIR).mkdir(exist_ok=True)
