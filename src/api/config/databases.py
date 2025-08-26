import os
from urllib.parse import quote_plus

import chromadb
import redis
from chromadb.config import Settings
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv("docker/.env")


class DatabaseConfig(BaseSettings):
    """Central configuration wrapper for database connection opts. Missing opts cause errors."""

    # Postgres
    POSTGRES_USER = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT")
    POSTGRES_DB = os.getenv("POSTGRES_DB")

    # Redis
    REDIS_HOST = os.getenv("REDIS_HOST")
    REDIS_PORT = os.getenv("REDIS_PORT")
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
    REDIS_DB = os.getenv("REDIS_HOST")

    # ChromaDB
    CHROMA_HOST = os.getenv("CHROMA_HOST")
    CHROMA_PORT = os.getenv("CHROMA_PORT")
    CHROMA_AUTH_TOKEN = os.getenv("CHROMA_SERVER_AUTH_TOKEN")

    class Config:
        env_file = 'docker/.env'
        case_sensitive = True

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
