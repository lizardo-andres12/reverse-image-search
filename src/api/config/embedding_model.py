from pydantic_settings import BaseSettings, SettingsConfigDict


class CLIPConfig(BaseSettings):
    """CLIP model config"""

    model_config = SettingsConfigDict(
        env_file="docker/.env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    MODEL_NAME: str = "openai/clip-vit-base-patch32"
    DEVICE: str = ""
    CACHE_DIR: str = ""
    SUPPORTED_FORMATS: tuple[str, str, str] = "RGB", "RGBA", "L"
