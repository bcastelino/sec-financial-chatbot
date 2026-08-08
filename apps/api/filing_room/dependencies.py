from functools import lru_cache

from fastapi import Header, HTTPException, status

from filing_room.config import get_settings
from filing_room.ingestion import IngestionService
from filing_room.llm import OpenRouterClient
from filing_room.metadata import CloudflareMetadataClient
from filing_room.repository import Repository
from filing_room.sec.client import SecClient
from filing_room.storage import create_object_store
from filing_room.vector_index import VectorIndexer


@lru_cache
def get_repository() -> Repository:
    settings = get_settings()
    metadata = CloudflareMetadataClient() if settings.metadata_backend == "cloudflare" else None
    return Repository(metadata)


@lru_cache
def get_llm() -> OpenRouterClient:
    return OpenRouterClient(get_settings())


@lru_cache
def get_sec_client() -> SecClient:
    return SecClient(get_settings())


@lru_cache
def get_ingestion() -> IngestionService:
    settings = get_settings()
    return IngestionService(
        get_sec_client(),
        get_repository(),
        create_object_store(settings),
        VectorIndexer(settings.vector_index_backend),
    )


async def require_gateway(x_filing_room_secret: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if settings.environment != "development" and x_filing_room_secret != settings.api_shared_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Gateway authentication required"
        )
