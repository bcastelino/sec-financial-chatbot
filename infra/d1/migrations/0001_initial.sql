PRAGMA foreign_keys = ON;

CREATE TABLE companies (
  cik TEXT PRIMARY KEY,
  ticker TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  sic TEXT,
  fiscal_year_end TEXT,
  updated_at TEXT NOT NULL
);

CREATE INDEX companies_name_idx ON companies(name);

CREATE TABLE filings (
  accession TEXT PRIMARY KEY,
  cik TEXT NOT NULL REFERENCES companies(cik),
  form TEXT NOT NULL,
  filing_date TEXT NOT NULL,
  report_date TEXT NOT NULL,
  primary_document TEXT NOT NULL,
  is_amendment INTEGER NOT NULL DEFAULT 0,
  raw_r2_key TEXT,
  parsed_r2_key TEXT,
  updated_at TEXT NOT NULL
);

CREATE INDEX filings_company_date_idx ON filings(cik, filing_date DESC);

CREATE TABLE financial_facts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  cik TEXT NOT NULL REFERENCES companies(cik),
  concept TEXT NOT NULL,
  display_label TEXT NOT NULL,
  value REAL NOT NULL,
  unit TEXT NOT NULL,
  period_start TEXT,
  period_end TEXT NOT NULL,
  fiscal_year INTEGER NOT NULL,
  fiscal_quarter INTEGER,
  form TEXT NOT NULL,
  filed_date TEXT NOT NULL,
  accession TEXT NOT NULL,
  source_url TEXT NOT NULL,
  UNIQUE(cik, concept, period_end, form, accession)
);

CREATE INDEX financial_facts_company_year_idx ON financial_facts(cik, fiscal_year, concept);

CREATE TABLE ingestion_jobs (
  accession TEXT PRIMARY KEY REFERENCES filings(accession),
  stage TEXT NOT NULL,
  progress INTEGER NOT NULL DEFAULT 0,
  retry_count INTEGER NOT NULL DEFAULT 0,
  error_code TEXT,
  updated_at TEXT NOT NULL
);

CREATE TABLE visitor_quotas (
  visitor_hash TEXT NOT NULL,
  quota_day TEXT NOT NULL,
  answer_count INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY(visitor_hash, quota_day)
);

CREATE TABLE sync_state (
  sync_key TEXT PRIMARY KEY,
  cursor TEXT,
  last_success_at TEXT,
  last_error_code TEXT
);
