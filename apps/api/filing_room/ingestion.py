import asyncio
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any

from filing_room.models import (
    CompanyRef,
    FilingSummary,
    IngestionStage,
    IngestionStatus,
    ResearchScope,
)
from filing_room.repository import Repository
from filing_room.sec.client import SecClient
from filing_room.sec.facts import select_financial_facts
from filing_room.sec.parser import FilingParser
from filing_room.sec.urls import company_facts_url, filing_url, submissions_url
from filing_room.storage import ObjectStore
from filing_room.vector_index import VectorIndexer

ProgressCallback = Callable[[IngestionStatus], Awaitable[None]]


class IngestionService:
    def __init__(
        self, sec: SecClient, repository: Repository, store: ObjectStore, vectors: VectorIndexer
    ) -> None:
        self.sec = sec
        self.repository = repository
        self.store = store
        self.vectors = vectors
        self.parser = FilingParser()
        self._locks: dict[str, asyncio.Lock] = {}

    async def refresh_company_catalog(self) -> int:
        payload = await self.sec.get_json("https://www.sec.gov/files/company_tickers.json")
        companies = [
            CompanyRef(
                ticker=str(row["ticker"]).upper(),
                cik=str(row["cik_str"]).zfill(10),
                name=str(row["title"]),
            )
            for row in payload.values()
            if isinstance(row, dict)
        ]
        await self.repository.upsert_companies(companies)
        return len(companies)

    async def ensure_scope(self, scope: ResearchScope, progress: ProgressCallback) -> None:
        for company in scope.companies:
            await self.ingest_company(company, set(scope.forms), set(scope.fiscalYears), progress)

    async def ingest_company(
        self, company: CompanyRef, forms: set[str], years: set[int], progress: ProgressCallback
    ) -> None:
        lock = self._locks.setdefault(company.cik, asyncio.Lock())
        async with lock:
            await progress(
                IngestionStatus(accession=company.cik, stage=IngestionStage.FETCHING, progress=8)
            )
            submissions, facts_payload = await asyncio.gather(
                self.sec.get_json(submissions_url(company.cik)),
                self.sec.get_json(company_facts_url(company.cik)),
            )
            facts = select_financial_facts(facts_payload, company.cik)
            filings = self._filings_from_submissions(company, submissions, forms, years)
            chunks = []
            total = max(1, len(filings))
            for index, filing in enumerate(filings):
                await progress(
                    IngestionStatus(
                        accession=filing.accession,
                        stage=IngestionStage.FETCHING,
                        progress=15 + round(55 * index / total),
                    )
                )
                html = await self.sec.get_text(filing.secUrl)
                await self.store.put_text(
                    f"raw/{filing.accession}/{filing.primaryDocument}", html, "text/html"
                )
                await progress(
                    IngestionStatus(
                        accession=filing.accession,
                        stage=IngestionStage.PARSING,
                        progress=25 + round(55 * index / total),
                    )
                )
                parsed = self.parser.parse(
                    html,
                    ticker=company.ticker,
                    accession=filing.accession,
                    form=filing.form,
                    filing_date=filing.filingDate,
                    fiscal_year=filing.fiscalYear,
                    sec_url=filing.secUrl,
                )
                chunks.extend(parsed.chunks)
                await self.store.put_json_gzip(
                    f"parsed/{filing.accession}/document.json.gz",
                    {
                        "title": parsed.title,
                        "chunks": [chunk.model_dump(mode="json") for chunk in parsed.chunks],
                    },
                )
                filing.ingestion = IngestionStatus(
                    accession=filing.accession, stage=IngestionStage.EMBEDDING, progress=85
                )
            await progress(
                IngestionStatus(accession=company.cik, stage=IngestionStage.EMBEDDING, progress=90)
            )
            await self.vectors.index(chunks)
            await self.repository.replace_company_data(company.ticker, facts, filings, chunks)
            for filing in filings:
                filing.ingestion = IngestionStatus(
                    accession=filing.accession, stage=IngestionStage.READY, progress=100
                )
            await progress(
                IngestionStatus(accession=company.cik, stage=IngestionStage.READY, progress=100)
            )

    @staticmethod
    def _filings_from_submissions(
        company: CompanyRef, payload: dict[str, Any], forms: set[str], years: set[int]
    ) -> list[FilingSummary]:
        recent = payload.get("filings", {}).get("recent", {})
        keys = ("accessionNumber", "form", "filingDate", "reportDate", "primaryDocument")
        length = min((len(recent.get(key, [])) for key in keys), default=0)
        filings: list[FilingSummary] = []
        for index in range(length):
            form = str(recent["form"][index])
            base_form = form.replace("/A", "")
            if base_form not in forms:
                continue
            report_date = date.fromisoformat(recent["reportDate"][index])
            if years and report_date.year not in years:
                continue
            accession = str(recent["accessionNumber"][index])
            document = str(recent["primaryDocument"][index])
            filings.append(
                FilingSummary(
                    accession=accession,
                    form=form,
                    filingDate=date.fromisoformat(recent["filingDate"][index]),
                    reportDate=report_date,
                    fiscalYear=report_date.year,
                    fiscalQuarter=None
                    if base_form == "10-K"
                    else ((report_date.month - 1) // 3 + 1),
                    primaryDocument=document,
                    isAmendment=form.endswith("/A"),
                    ingestion=IngestionStatus(
                        accession=accession, stage=IngestionStage.QUEUED, progress=0
                    ),
                    secUrl=filing_url(company.cik, accession, document),
                )
            )
        return filings[:20]
