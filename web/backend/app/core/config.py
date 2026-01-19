# Copyright (c) US Inc. All rights reserved.
"""Application configuration settings - Non-sensitive settings only."""

import os
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings - non-sensitive values only."""
    
    # API Settings
    APP_NAME: str = "USF BIOS API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS Settings
    CORS_ORIGINS: List[str] = ["*"]
    
    # Data Paths (user data locations - not sensitive)
    DATA_DIR: Path = Path("/app/data") if os.path.exists("/app") else Path("./data")
    
    # Training Settings
    MAX_CONCURRENT_JOBS: int = 3
    JOB_TIMEOUT_HOURS: int = 72
    
    # Security
    API_KEY: Optional[str] = None
    DISABLE_CLI: bool = True
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
    )
    
    @property
    def UPLOAD_DIR(self) -> Path:
        return self.DATA_DIR / "uploads"
    
    @property
    def OUTPUT_DIR(self) -> Path:
        return self.DATA_DIR / "outputs"
    
    @property
    def MODELS_DIR(self) -> Path:
        return self.DATA_DIR / "models"


settings = Settings()

# Create directories if they don't exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize system validator (loads settings from environment in binary)
from .capabilities import init_validator
init_validator()
