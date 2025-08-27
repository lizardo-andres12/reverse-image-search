import os
from urllib.parse import quote_plus

import chromadb
import redis
from chromadb.config import Settings
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """Central configuration wrapper for database connection opts. Missing opts cause errors."""

    # .env parse config
    model_config = SettingsConfigDict(
        env_file="docker/.env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Postgres
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_DB: str
    POSTGRES_MIN_CONNECTIONS: int
    POSTGRES_MAX_CONNECTIONS: int

    # Redis
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_PASSWORD: str
    REDIS_DB: str

    # ChromaDB
    CHROMA_HOST: str
    CHROMA_PORT: int
    CHROMA_SERVER_AUTH_TOKEN: str

    def get_postgres_url(self) -> str:
        """
        Get Postgres connection DSN

        Args:
            async_mode (bool): If the connection should be async or not.
        Returns:
            str: DSN connection URL
        """
        password = quote_plus(self.POSTGRES_PASSWORD)

        return (
            f"postgresql://{self.POSTGRES_USER}:{password}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    def get_redis_url(self) -> str:
        """
        Get Redis connection DSN

        Returns:
            str: DSN connection URL
        """
        return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
