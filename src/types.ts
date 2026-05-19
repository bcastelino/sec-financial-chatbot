export type Role = 'system' | 'user' | 'assistant'

export interface ChatMessage {
  id: string
  role: Role
  content: string
  /** Sources referenced when assembling this message (assistant only). */
  sources?: SourceRef[]
  /** True while assistant message is still streaming. */
  pending?: boolean
  /** Error string if generation failed. */
  error?: string
  createdAt: number
}

export interface SourceRef {
  label: string
  url: string
  /** e.g. "10-K", "10-Q". */
  form?: string
  accession?: string
  filingDate?: string
  ticker?: string
  fiscalPeriod?: string
}

export interface ModelOption {
  id: string
  name: string
  description: string
  badge?: 'Default' | 'Pro' | 'Fast' | 'Custom'
  /** Tag to pick an icon in the UI. */
  iconKey: 'zap' | 'sparkles' | 'brain' | 'bolt'
}

export interface TickerEntry {
  cik: string         // 10-digit zero-padded
  ticker: string      // upper-case
  name: string
}

/** Subset of fields we care about from data.sec.gov/submissions */
export interface FilingRecord {
  accessionNumber: string
  form: string
  filingDate: string         // YYYY-MM-DD
  reportDate: string         // YYYY-MM-DD
  primaryDocument: string
  primaryDocDescription: string
}

/** A single XBRL fact (one period of one concept) */
export interface XbrlFact {
  concept: string            // us-gaap tag, e.g. Revenues
  label?: string
  unit: string               // e.g. USD, shares
  value: number
  start?: string             // YYYY-MM-DD
  end: string                // YYYY-MM-DD
  fy?: number
  fp?: string                // e.g. FY, Q1, Q2
  form: string
  accession: string
  filed: string
}

export interface IntentExtraction {
  companies: { ticker: string; cik?: string }[]
  years: number[]            // empty = latest
  forms: string[]            // default: ["10-K"]
  kind: 'numeric' | 'narrative' | 'mixed'
  sections: string[]         // e.g. ["1A", "7"] for 10-K Items
  notes?: string
}

export type LlmProvider = 'openrouter' | 'openai'

export interface LlmSettings {
  provider: LlmProvider
  apiKey: string
  model: string
}
