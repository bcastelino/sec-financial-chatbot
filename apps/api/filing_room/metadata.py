from typing import Any

import httpx

from filing_room.models import CompanyOverview, CompanyRef, FilingSummary, FinancialFact


class CloudflareMetadataClient:
    def __init__(self) -> None:
        self.client = httpx.AsyncClient(timeout=30)

    async def search_companies(self, query: str, limit: int) -> list[CompanyRef]:
        response = await self.client.get(
            "http://metadata.d1/companies/search",
            params={"q": query, "limit": limit},
        )
        response.raise_for_status()
        return [CompanyRef.model_validate(item) for item in response.json()]

    async def overview(self, ticker: str) -> CompanyOverview | None:
        response = await self.client.get(f"http://metadata.d1/companies/{ticker}/overview")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return CompanyOverview.model_validate(response.json())

    async def upsert_companies(self, companies: list[CompanyRef]) -> None:
        response = await self.client.post(
            "http://metadata.d1/companies",
            json=[company.model_dump(mode="json") for company in companies],
        )
        response.raise_for_status()

    async def save_company_data(
        self,
        company: CompanyRef,
        facts: list[FinancialFact],
        filings: list[FilingSummary],
    ) -> None:
        payload: dict[str, Any] = {
            "company": company.model_dump(mode="json"),
            "facts": [fact.model_dump(mode="json") for fact in facts],
            "filings": [filing.model_dump(mode="json") for filing in filings],
        }
        response = await self.client.post("http://metadata.d1/company-data", json=payload)
        response.raise_for_status()
