export type FilingForm = "10-K" | "10-Q" | "10-K/A" | "10-Q/A";

export interface CompanyRef {
  ticker: string;
  cik: string;
  name: string;
}

export interface ResearchScope {
  companies: CompanyRef[];
  forms: Array<"10-K" | "10-Q">;
  fiscalYears: number[];
  accessions?: string[];
  sections?: string[];
}

export interface FinancialFact {
  concept: string;
  displayLabel: string;
  value: number;
  unit: string;
  periodStart?: string;
  periodEnd: string;
  fiscalYear: number;
  fiscalQuarter?: number;
  form: FilingForm;
  filedDate: string;
  accession: string;
  sourceUrl: string;
}

export interface SourceRef {
  id: string;
  ticker: string;
  accession: string;
  form: FilingForm;
  filingDate: string;
  section: string;
  excerpt: string;
  r2Locator: string;
  secUrl: string;
}

export type IngestionStage =
  | "queued"
  | "fetching"
  | "parsing"
  | "embedding"
  | "ready"
  | "failed";

export interface IngestionStatus {
  accession: string;
  stage: IngestionStage;
  progress: number;
  errorCode?: string;
}

export interface FilingSummary {
  accession: string;
  form: FilingForm;
  filingDate: string;
  reportDate: string;
  fiscalYear: number;
  fiscalQuarter?: number;
  primaryDocument: string;
  isAmendment: boolean;
  ingestion: IngestionStatus;
  secUrl: string;
}

export interface CompanyOverview {
  company: CompanyRef;
  description?: string;
  sic?: string;
  fiscalYearEnd?: string;
  facts: FinancialFact[];
  filings: FilingSummary[];
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  query: string;
  scope: ResearchScope;
  messages: ChatMessage[];
  turnstileToken: string;
}

export interface QuotaStatus {
  remaining: number;
  limit: number;
  resetsAt: string;
  budgetAvailable: boolean;
}

export type ChatEvent =
  | { type: "retrieval.status"; data: IngestionStatus | { message: string } }
  | { type: "answer.delta"; data: { delta: string } }
  | { type: "answer.sources"; data: { sources: SourceRef[] } }
  | { type: "quota.updated"; data: QuotaStatus }
  | { type: "done"; data: { requestId: string } }
  | { type: "error"; data: { code: string; message: string } };
