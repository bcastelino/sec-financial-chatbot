from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Filing Room API"
    environment: str = "development"
    api_shared_secret: str = ""
    sec_identity: str = "Filing Room admin@bcastelino.com"
    sec_requests_per_second: float = Field(default=5.0, ge=0.5, le=5.0)
    openrouter_api_key: str = ""
    openrouter_model: str = "deepseek/deepseek-v4-flash-0731"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    allowed_origins: str = "http://localhost:5173,https://sec.bcastelino.com"
    local_data_dir: str = ".data"
    object_store_backend: str = "local"
    vector_index_backend: str = "local"
    metadata_backend: str = "local"
    r2_endpoint_url: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket_name: str = "filing-room-filings"
    max_context_chunks: int = Field(default=8, ge=1, le=12)
    max_query_length: int = Field(default=2_000, ge=100, le=4_000)

    @property
    def origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
