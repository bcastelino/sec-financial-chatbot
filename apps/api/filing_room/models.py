from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class FilingForm(StrEnum):
    TEN_K = "10-K"
    TEN_Q = "10-Q"
    TEN_K_A = "10-K/A"
    TEN_Q_A = "10-Q/A"


class CompanyRef(BaseModel):
    ticker: str
    cik: str
    name: str


class ResearchScope(BaseModel):
    companies: list[CompanyRef] = Field(min_length=1, max_length=3)
    forms: list[str] = Field(default_factory=lambda: ["10-K", "10-Q"])
    fiscalYears: list[int] = Field(default_factory=list, max_length=5)
    accessions: list[str] | None = None
    sections: list[str] | None = None

    @field_validator("forms")
    @classmethod
    def validate_forms(cls, value: list[str]) -> list[str]:
        if not value or any(form not in {"10-K", "10-Q"} for form in value):
            raise ValueError("Only 10-K and 10-Q forms are supported")
        return list(dict.fromkeys(value))


class FinancialFact(BaseModel):
    concept: str
    displayLabel: str
    value: float
    unit: str
    periodStart: date | None = None
    periodEnd: date
    fiscalYear: int
    fiscalQuarter: int | None = None
    form: str
    filedDate: date
    accession: str
    sourceUrl: str


class IngestionStage(StrEnum):
    QUEUED = "queued"
    FETCHING = "fetching"
    PARSING = "parsing"
    EMBEDDING = "embedding"
    READY = "ready"
    FAILED = "failed"


class IngestionStatus(BaseModel):
    accession: str
    stage: IngestionStage
    progress: int = Field(ge=0, le=100)
    errorCode: str | None = None


class FilingSummary(BaseModel):
    accession: str
    form: str
    filingDate: date
    reportDate: date
    fiscalYear: int
    fiscalQuarter: int | None = None
    primaryDocument: str
    isAmendment: bool
    ingestion: IngestionStatus
    secUrl: str


class SourceRef(BaseModel):
    id: str
    ticker: str
    accession: str
    form: str
    filingDate: date
    section: str
    excerpt: str
    r2Locator: str
    secUrl: str


class CompanyOverview(BaseModel):
    company: CompanyRef
    description: str | None = None
    sic: str | None = None
    fiscalYearEnd: str | None = None
    facts: list[FinancialFact]
    filings: list[FilingSummary]


class ChatMessage(BaseModel):
    role: str
    content: str = Field(max_length=8_000)

    @field_validator("role")
    @classmethod
    def role_is_safe(cls, value: str) -> str:
        if value not in {"user", "assistant"}:
            raise ValueError("Invalid chat role")
        return value


class ChatRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2_000)
    scope: ResearchScope
    messages: list[ChatMessage] = Field(default_factory=list, max_length=8)
    turnstileToken: str


class Chunk(BaseModel):
    chunk_id: str
    ticker: str
    accession: str
    form: str
    filing_date: date
    fiscal_year: int
    section: str
    text: str
    token_start: int
    token_end: int
    sec_url: str
    is_table: bool = False


class QuotaStatus(BaseModel):
    remaining: int
    limit: int = 5
    resetsAt: str
    budgetAvailable: bool = True
