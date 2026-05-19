import { fetchJsonCached } from './rateLimiter'
import type { XbrlFact } from '../../types'

/**
 * The companyfacts endpoint returns every us-gaap/dei XBRL fact a company has
 * ever filed. Shape (simplified):
 * {
 *   cik, entityName,
 *   facts: {
 *     "us-gaap": {
 *       "Revenues": {
 *         label, description,
 *         units: { "USD": [ { end, val, accn, fy, fp, form, filed, start? }, ... ] }
 *       }, ...
 *     },
 *     "dei": { ... }
 *   }
 * }
 */

interface RawFactValue {
  start?: string
  end: string
  val: number
  accn: string
  fy?: number
  fp?: string
  form: string
  filed: string
}

interface RawFactConcept {
  label?: string
  description?: string
  units: Record<string, RawFactValue[]>
}

interface RawCompanyFacts {
  cik: number
  entityName: string
  facts: Record<string, Record<string, RawFactConcept>>
}

export interface CompanyFacts {
  cik: string
  name: string
  raw: RawCompanyFacts
}

export async function getCompanyFacts(cik: string): Promise<CompanyFacts> {
  const url = `https://data.sec.gov/api/xbrl/companyfacts/CIK${cik}.json`
  const raw = await fetchJsonCached<RawCompanyFacts>(url)
  return { cik, name: raw.entityName, raw }
}

/**
 * Default us-gaap concepts most useful for high-level company Q&A. We try each
 * in order and pick the first one that has data, since older filings may use a
 * different tag (e.g. RevenueFromContractWithCustomerExcludingAssessedTax).
 */
export const DEFAULT_CONCEPT_GROUPS: { name: string; tags: string[]; unit?: string }[] = [
  { name: 'Revenue', tags: ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet'], unit: 'USD' },
  { name: 'Cost of Revenue', tags: ['CostOfRevenue', 'CostOfGoodsAndServicesSold'], unit: 'USD' },
  { name: 'Gross Profit', tags: ['GrossProfit'], unit: 'USD' },
  { name: 'Operating Income', tags: ['OperatingIncomeLoss'], unit: 'USD' },
  { name: 'Net Income', tags: ['NetIncomeLoss'], unit: 'USD' },
  { name: 'EPS (basic)', tags: ['EarningsPerShareBasic'], unit: 'USD/shares' },
  { name: 'EPS (diluted)', tags: ['EarningsPerShareDiluted'], unit: 'USD/shares' },
  { name: 'Total Assets', tags: ['Assets'], unit: 'USD' },
  { name: 'Total Liabilities', tags: ['Liabilities'], unit: 'USD' },
  { name: "Stockholders' Equity", tags: ['StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'], unit: 'USD' },
  { name: 'Cash & Equivalents', tags: ['CashAndCashEquivalentsAtCarryingValue', 'Cash'], unit: 'USD' },
  { name: 'Operating Cash Flow', tags: ['NetCashProvidedByUsedInOperatingActivities'], unit: 'USD' },
  { name: 'Research & Development', tags: ['ResearchAndDevelopmentExpense'], unit: 'USD' },
]

/**
 * Pull annual facts for the given concept tags for the requested fiscal years.
 * Filters to 10-K forms when possible to avoid duplicate quarterly entries.
 */
export function extractAnnualFacts(
  facts: RawCompanyFacts,
  conceptTags: string[],
  years: number[],
): XbrlFact[] {
  const out: XbrlFact[] = []
  const us = facts.facts['us-gaap'] ?? {}
  for (const tag of conceptTags) {
    const concept = us[tag]
    if (!concept) continue
    for (const unit of Object.keys(concept.units)) {
      for (const v of concept.units[unit]) {
        if (v.form !== '10-K') continue
        if (years.length > 0 && (!v.fy || !years.includes(v.fy))) continue
        out.push({
          concept: tag,
          label: concept.label,
          unit,
          value: v.val,
          start: v.start,
          end: v.end,
          fy: v.fy,
          fp: v.fp,
          form: v.form,
          accession: v.accn,
          filed: v.filed,
        })
      }
    }
  }
  return dedupeFacts(out)
}

function dedupeFacts(facts: XbrlFact[]): XbrlFact[] {
  const seen = new Map<string, XbrlFact>()
  for (const f of facts) {
    const key = `${f.concept}|${f.fy ?? f.end}|${f.unit}`
    const prev = seen.get(key)
    if (!prev || prev.filed < f.filed) seen.set(key, f)
  }
  return [...seen.values()].sort((a, b) => (a.fy ?? 0) - (b.fy ?? 0) || a.concept.localeCompare(b.concept))
}

/**
 * Produce a compact markdown table of the default concept groups for the given
 * years, suitable to inject into an LLM prompt as numeric context.
 */
export function summarizeFactsAsMarkdown(
  ticker: string,
  facts: RawCompanyFacts,
  years: number[],
): { markdown: string; cited: XbrlFact[] } {
  const effectiveYears = years.length > 0 ? years : inferLatestYears(facts, 3)
  const rows: string[] = []
  const cited: XbrlFact[] = []

  rows.push(`**${ticker} — ${facts.entityName}** annual facts (source: SEC XBRL companyfacts)`)
  rows.push('')
  rows.push(`| Concept | Unit | ${effectiveYears.map((y) => `FY${y}`).join(' | ')} | Source |`)
  rows.push(`|---|---|${effectiveYears.map(() => '---:').join('|')}|---|`)

  for (const group of DEFAULT_CONCEPT_GROUPS) {
    const pulled = extractAnnualFacts(facts, group.tags, effectiveYears)
    if (pulled.length === 0) continue
    const byYear = new Map<number, XbrlFact>()
    for (const f of pulled) if (f.fy != null) byYear.set(f.fy, f)
    const cells = effectiveYears.map((y) => {
      const f = byYear.get(y)
      if (!f) return '—'
      cited.push(f)
      return formatValue(f.value, f.unit)
    })
    const refs = [...new Set(pulled.map((f) => f.accession))].slice(0, 1).join(', ')
    rows.push(`| ${group.name} | ${pulled[0]?.unit ?? group.unit ?? ''} | ${cells.join(' | ')} | ${refs} |`)
  }

  return { markdown: rows.join('\n'), cited }
}

function formatValue(v: number, unit: string): string {
  if (unit === 'USD' || unit === 'usd') {
    const abs = Math.abs(v)
    if (abs >= 1e9) return `$${(v / 1e9).toFixed(2)}B`
    if (abs >= 1e6) return `$${(v / 1e6).toFixed(2)}M`
    if (abs >= 1e3) return `$${(v / 1e3).toFixed(2)}K`
    return `$${v.toFixed(2)}`
  }
  if (unit.includes('shares')) return v.toLocaleString()
  return v.toString()
}

function inferLatestYears(facts: RawCompanyFacts, n: number): number[] {
  const ys = new Set<number>()
  const us = facts.facts['us-gaap'] ?? {}
  for (const tag of ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'NetIncomeLoss']) {
    const c = us[tag]
    if (!c) continue
    for (const u of Object.keys(c.units)) {
      for (const v of c.units[u]) {
        if (v.form === '10-K' && v.fy) ys.add(v.fy)
      }
    }
  }
  return [...ys].sort((a, b) => b - a).slice(0, n).reverse()
}
