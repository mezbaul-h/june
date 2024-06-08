"""
This module provides settings related to the application.
"""

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    HF_DEVICE_MAP: str = "auto"
    HF_TOKEN: str = ""
    TORCH_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


settings = Settings()
