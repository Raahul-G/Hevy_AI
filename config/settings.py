"""Application settings and configuration."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM Configuration
    openai_api_key: str
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000

    # Profile Storage
    profile_storage_path: str = "data/profiles/"

    def __init__(self, **kwargs):
        """Initialize settings and ensure profile directory exists."""
        super().__init__(**kwargs)
        # Ensure profile storage directory exists
        profile_path = Path(self.profile_storage_path)
        profile_path.mkdir(parents=True, exist_ok=True)

    @property
    def profile_dir(self) -> Path:
        """Get the profile storage directory as a Path object."""
        return Path(self.profile_storage_path)


# Global settings instance
settings = Settings()

