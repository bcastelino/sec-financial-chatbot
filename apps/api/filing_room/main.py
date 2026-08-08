from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from filing_room.config import get_settings
from filing_room.dependencies import get_llm, get_sec_client
from filing_room.routes import router
from filing_room.sec.parser import initialize_edgartools


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    initialize_edgartools(settings.sec_identity, settings.sec_requests_per_second)
    yield
    await get_llm().close()
    await get_sec_client().close()


settings = get_settings()
app = FastAPI(
    title="Filing Room API",
    version="0.1.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Filing-Room-Secret"],
)
app.include_router(router)
