from __future__ import annotations

from datetime import date

from filing_room.metadata import CloudflareMetadataClient
from filing_room.models import (
    Chunk,
    CompanyOverview,
    CompanyRef,
    FilingSummary,
    FinancialFact,
)


class Repository:
    """Storage boundary used by local tests and replaced by D1/R2/Vectorize adapters in production."""

    def __init__(self, metadata: CloudflareMetadataClient | None = None) -> None:
        self.metadata = metadata
        self.companies: dict[str, CompanyRef] = {
            "AAPL": CompanyRef(ticker="AAPL", cik="0000320193", name="Apple Inc."),
            "MSFT": CompanyRef(ticker="MSFT", cik="0000789019", name="Microsoft Corporation"),
            "NVDA": CompanyRef(ticker="NVDA", cik="0001045810", name="NVIDIA Corporation"),
            "AMZN": CompanyRef(ticker="AMZN", cik="0001018724", name="Amazon.com, Inc."),
            "GOOGL": CompanyRef(ticker="GOOGL", cik="0001652044", name="Alphabet Inc."),
        }
        self.facts: dict[str, list[FinancialFact]] = {}
        self.filings: dict[str, list[FilingSummary]] = {}
        self.chunks: list[Chunk] = self._demo_chunks()

    async def search_companies(self, query: str, limit: int = 8) -> list[CompanyRef]:
        if self.metadata:
            matches = await self.metadata.search_companies(query, limit)
            self.companies.update({company.ticker: company for company in matches})
            return matches
        normalized = query.lower().strip()
        matches = [
            company
            for company in self.companies.values()
            if normalized in company.ticker.lower() or normalized in company.name.lower()
        ]
        return sorted(
            matches, key=lambda item: (not item.ticker.lower().startswith(normalized), item.ticker)
        )[:limit]

    async def upsert_companies(self, companies: list[CompanyRef]) -> None:
        self.companies.update({company.ticker.upper(): company for company in companies})
        if self.metadata:
            await self.metadata.upsert_companies(companies)

    async def replace_company_data(
        self,
        ticker: str,
        facts: list[FinancialFact],
        filings: list[FilingSummary],
        chunks: list[Chunk],
    ) -> None:
        normalized = ticker.upper()
        self.facts[normalized] = facts
        self.filings[normalized] = filings
        accessions = {filing.accession for filing in filings}
        self.chunks = [chunk for chunk in self.chunks if chunk.accession not in accessions] + chunks
        if self.metadata:
            company = self.companies[normalized]
            await self.metadata.save_company_data(company, facts, filings)

    def has_scope_chunks(self, tickers: set[str], forms: set[str], years: set[int]) -> bool:
        return any(
            chunk.ticker.upper() in tickers
            and chunk.form.replace("/A", "") in forms
            and (not years or chunk.fiscal_year in years)
            for chunk in self.chunks
        )

    async def overview(self, ticker: str) -> CompanyOverview | None:
        if self.metadata:
            overview = await self.metadata.overview(ticker.upper())
            if overview:
                return overview
        company = self.companies.get(ticker.upper())
        if not company:
            return None
        return CompanyOverview(
            company=company,
            description="Public-company financials and disclosures sourced exclusively from SEC filings.",
            facts=self.facts.get(company.ticker, []),
            filings=self.filings.get(company.ticker, []),
        )

    async def list_filings(self, ticker: str) -> list[FilingSummary]:
        if self.metadata:
            overview = await self.metadata.overview(ticker.upper())
            if overview:
                return overview.filings
        return self.filings.get(ticker.upper(), [])

    async def source(self, source_id: str) -> Chunk | None:
        return next((chunk for chunk in self.chunks if chunk.chunk_id == source_id), None)

    def _demo_chunks(self) -> list[Chunk]:
        return [
            Chunk(
                chunk_id="demo-aapl-risk",
                ticker="AAPL",
                accession="0000320193-24-000123",
                form="10-K",
                filing_date=date(2024, 11, 1),
                fiscal_year=2024,
                section="Item 1A. Risk Factors",
                text="The Company's business and performance depend substantially on global supply chains, developer ecosystems, product introductions, component availability, and the ability to compete in highly competitive markets.",
                token_start=0,
                token_end=28,
                sec_url="https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm",
            ),
            Chunk(
                chunk_id="demo-msft-ai",
                ticker="MSFT",
                accession="0000950170-24-087843",
                form="10-K",
                filing_date=date(2024, 7, 30),
                fiscal_year=2024,
                section="Business",
                text="Microsoft is investing across cloud and artificial intelligence infrastructure and products. Demand for cloud services requires continued investment in datacenters and computing capacity.",
                token_start=0,
                token_end=24,
                sec_url="https://www.sec.gov/Archives/edgar/data/789019/000095017024087843/msft-20240630.htm",
            ),
            Chunk(
                chunk_id="demo-nvda-demand",
                ticker="NVDA",
                accession="0001045810-25-000023",
                form="10-K",
                filing_date=date(2025, 2, 26),
                fiscal_year=2025,
                section="Management's Discussion and Analysis",
                text="Demand for accelerated computing and generative AI drove growth in Data Center revenue, while supply availability and product transitions remained important operating considerations.",
                token_start=0,
                token_end=24,
                sec_url="https://www.sec.gov/Archives/edgar/data/1045810/000104581025000023/nvda-20250126.htm",
            ),
        ]
