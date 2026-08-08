import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from filing_room.dependencies import get_ingestion, get_llm, get_repository, require_gateway
from filing_room.ingestion import IngestionService
from filing_room.llm import OpenRouterClient
from filing_room.models import (
    ChatRequest,
    CompanyOverview,
    CompanyRef,
    FilingSummary,
    IngestionStatus,
    QuotaStatus,
    SourceRef,
)
from filing_room.repository import Repository
from filing_room.retrieval import retrieve

router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_gateway)])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "filing-room-api"}


@router.get("/ready")
async def ready() -> dict[str, bool]:
    return {"ready": True}


@router.get("/companies/search", response_model=list[CompanyRef])
async def search_companies(
    q: str = Query(min_length=1, max_length=100), repository: Repository = Depends(get_repository)
) -> list[CompanyRef]:
    return await repository.search_companies(q)


@router.post("/catalog/refresh")
async def refresh_catalog(ingestion: IngestionService = Depends(get_ingestion)) -> dict[str, int]:
    return {"companies": await ingestion.refresh_company_catalog()}


@router.get("/companies/{ticker}/overview", response_model=CompanyOverview)
async def company_overview(
    ticker: str, repository: Repository = Depends(get_repository)
) -> CompanyOverview:
    overview = await repository.overview(ticker)
    if not overview:
        raise HTTPException(status_code=404, detail="Company not found")
    return overview


@router.get("/companies/{ticker}/filings", response_model=list[FilingSummary])
async def company_filings(
    ticker: str, repository: Repository = Depends(get_repository)
) -> list[FilingSummary]:
    return await repository.list_filings(ticker)


@router.get("/sources/{source_id}", response_model=SourceRef)
async def source_detail(
    source_id: str, repository: Repository = Depends(get_repository)
) -> SourceRef:
    chunk = await repository.source(source_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="Source not found")
    return SourceRef(
        id=chunk.chunk_id,
        ticker=chunk.ticker,
        accession=chunk.accession,
        form=chunk.form,
        filingDate=chunk.filing_date,
        section=chunk.section,
        excerpt=chunk.text,
        r2Locator=f"parsed/{chunk.accession}/{chunk.chunk_id}.json.gz",
        secUrl=chunk.sec_url,
    )


@router.get("/quota", response_model=QuotaStatus)
async def local_quota() -> QuotaStatus:
    reset = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    return QuotaStatus(remaining=5, resetsAt=reset.isoformat())


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    repository: Repository = Depends(get_repository),
    llm: OpenRouterClient = Depends(get_llm),
    ingestion: IngestionService = Depends(get_ingestion),
) -> EventSourceResponse:
    request_id = str(uuid.uuid4())

    async def events() -> AsyncIterator[dict[str, str]]:
        yield _event("retrieval.status", {"message": "Searching selected filings"})
        tickers = {company.ticker.upper() for company in request.scope.companies}
        if not repository.has_scope_chunks(
            tickers, set(request.scope.forms), set(request.scope.fiscalYears)
        ):
            queue: asyncio.Queue[IngestionStatus] = asyncio.Queue()

            async def update(status: IngestionStatus) -> None:
                await queue.put(status)

            task = asyncio.create_task(ingestion.ensure_scope(request.scope, update))
            while not task.done() or not queue.empty():
                try:
                    progress = await asyncio.wait_for(queue.get(), timeout=0.25)
                    yield _event("retrieval.status", progress.model_dump(mode="json"))
                except TimeoutError:
                    continue
            try:
                await task
            except Exception:
                yield _event(
                    "retrieval.status",
                    {"message": "Live SEC ingestion was unavailable; searching cached sources"},
                )
        sources = retrieve(request.query, request.scope, repository.chunks)
        yield _event(
            "answer.sources", {"sources": [source.model_dump(mode="json") for source in sources]}
        )
        async for delta in llm.stream_answer(request.query, request.messages, sources):
            yield _event("answer.delta", {"delta": delta})
        yield _event(
            "quota.updated",
            {
                "remaining": 4,
                "limit": 5,
                "resetsAt": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
                "budgetAvailable": True,
            },
        )
        yield _event("done", {"requestId": request_id})

    return EventSourceResponse(
        events(), ping=15, headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"}
    )


def _event(event: str, data: object) -> dict[str, str]:
    return {"event": event, "data": json.dumps(data, separators=(",", ":"))}
