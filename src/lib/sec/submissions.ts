import { fetchJsonCached } from './rateLimiter'
import type { FilingRecord } from '../../types'

interface RawSubmissions {
  cik: string
  name: string
  tickers?: string[]
  filings?: {
    recent?: {
      accessionNumber: string[]
      filingDate: string[]
      reportDate: string[]
      form: string[]
      primaryDocument: string[]
      primaryDocDescription: string[]
    }
  }
}

export interface CompanySubmissions {
  cik: string
  name: string
  tickers: string[]
  filings: FilingRecord[]
}

function parseRecent(raw: RawSubmissions): FilingRecord[] {
  const r = raw.filings?.recent
  if (!r) return []
  const n = r.accessionNumber.length
  const out: FilingRecord[] = []
  for (let i = 0; i < n; i++) {
    out.push({
      accessionNumber: r.accessionNumber[i],
      form: r.form[i],
      filingDate: r.filingDate[i],
      reportDate: r.reportDate[i],
      primaryDocument: r.primaryDocument[i],
      primaryDocDescription: r.primaryDocDescription[i],
    })
  }
  return out
}

export async function getSubmissions(cik: string): Promise<CompanySubmissions> {
  const url = `https://data.sec.gov/submissions/CIK${cik}.json`
  const raw = await fetchJsonCached<RawSubmissions>(url)
  return {
    cik,
    name: raw.name,
    tickers: raw.tickers ?? [],
    filings: parseRecent(raw),
  }
}

/**
 * Filter filings by form and year(s). If years is empty, returns the most recent
 * matching filing only.
 */
export function selectFilings(
  filings: FilingRecord[],
  forms: string[],
  years: number[],
): FilingRecord[] {
  const formSet = new Set(forms.map((f) => f.toUpperCase()))
  const matched = filings.filter((f) => formSet.has(f.form.toUpperCase()))
  if (years.length === 0) {
    return matched.slice(0, 1)
  }
  const yearSet = new Set(years)
  return matched.filter((f) => {
    const y = parseInt((f.reportDate || f.filingDate).slice(0, 4), 10)
    return yearSet.has(y)
  })
}

/**
 * Build the canonical URL for a primary filing document. Accession numbers come
 * in dashed form (0001234567-89-012345); the archive path uses the dash-stripped
 * form for the directory but keeps the dashed form for the index.
 */
export function filingDocumentUrl(cik: string, filing: FilingRecord): string {
  const noDashes = filing.accessionNumber.replace(/-/g, '')
  // CIK in archive URL is unpadded.
  const cikUnpadded = String(parseInt(cik, 10))
  return `https://www.sec.gov/Archives/edgar/data/${cikUnpadded}/${noDashes}/${filing.primaryDocument}`
}

export function filingIndexUrl(cik: string, filing: FilingRecord): string {
  const noDashes = filing.accessionNumber.replace(/-/g, '')
  const cikUnpadded = String(parseInt(cik, 10))
  return `https://www.sec.gov/Archives/edgar/data/${cikUnpadded}/${noDashes}/`
}
