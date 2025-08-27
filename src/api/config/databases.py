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

    # Redis
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_PASSWORD: str
    REDIS_DB: str

    # ChromaDB
    CHROMA_HOST: str
    CHROMA_PORT: int
    CHROMA_SERVER_AUTH_TOKEN: str

    @classmethod
    def get_postgres_url(cls, async_mode: bool = False) -> str:
        """
        Get Postgres connection DSN

        Args:
            async_mode (bool): If the connection should be async or not.
        Returns:
            str: DSN connection URL
        """
        password = quote_plus(cls.POSTGRES_PASSWORD)

        if async_mode:
            return (
                f"postgresql+asyncpg://{cls.POSTGRES_USER}:{password}@"
                f"{cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DB}"
            )
        else:
            return (
                f"postgresql://{cls.POSTGRES_USER}:{password}@"
                f"{cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DB}"
            )

    @classmethod
    def get_redis_url(cls) -> str:
        """
        Get Redis connection DSN

        Returns:
            str: DSN connection URL
        """
        return f"redis://:{cls.REDIS_PASSWORD}@{cls.REDIS_HOST}:{cls.REDIS_PORT}/{cls.REDIS_DB}"
