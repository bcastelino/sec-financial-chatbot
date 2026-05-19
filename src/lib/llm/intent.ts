import { ensureTickerIndex, searchByName } from '../sec/tickers'
import type { IntentExtraction } from '../../types'
import { SECTION_PATTERNS } from '../sec/filingDoc'

const NARRATIVE_KEYWORDS = [
  'risk', 'risks', 'risk factors', 'strategy', 'discussion', 'management',
  'mda', "md&a", 'outlook', 'segment', 'segments', 'business', 'competition',
  'litigation', 'legal', 'properties', 'controls', 'overview', 'narrative',
  'describe', 'summary', 'summarize', 'compare', 'comparison',
]

const NUMERIC_KEYWORDS = [
  'revenue', 'revenues', 'sales', 'income', 'profit', 'loss', 'eps',
  'earnings per share', 'assets', 'liabilities', 'equity', 'cash',
  'r&d', 'research', 'operating', 'gross', 'margin', 'how much',
  'how many', 'value', 'amount', 'total',
]

const QUARTER_HINTS = ['quarter', 'quarterly', 'q1', 'q2', 'q3', 'q4', '10-q', '10q']
const ANNUAL_HINTS = ['10-k', '10k', 'annual', 'fy', 'fiscal year', 'yearly']

/**
 * Cheap, deterministic intent extraction:
 *  - Tickers: explicit upper-case symbol (1-5 letters) OR known company name lookup.
 *  - Years: any 4-digit year in 1995..(current+1).
 *  - Forms: 10-K default, switches to 10-Q on quarter hints.
 *  - Sections: 10-K Item ids detected via shared SECTION_PATTERNS.
 *  - Kind: heuristic from keywords; defaults to numeric when ambiguous.
 */
export async function extractIntent(query: string, history: string[] = []): Promise<IntentExtraction> {
  const text = `${history.slice(-4).join(' ')} ${query}`.toLowerCase()
  const idx = await ensureTickerIndex()

  // Tickers: explicit 1-5 letter all-caps tokens in the *original* query
  const explicitTickers = new Set<string>()
  for (const m of query.matchAll(/\b[A-Z]{1,5}\b/g)) {
    const t = m[0]
    if (idx.has(t)) explicitTickers.add(t)
  }

  // Company-name lookups for things like "apple", "alphabet", "meta"
  if (explicitTickers.size === 0) {
    // Try a handful of capitalized words from the original query
    const candidates = query.match(/\b[A-Za-z][A-Za-z.&'\- ]{2,40}\b/g) ?? []
    for (const c of candidates.slice(0, 6)) {
      const hits = await searchByName(c, 1)
      if (hits[0]) explicitTickers.add(hits[0].ticker)
    }
  }

  // Years
  const yearSet = new Set<number>()
  const nowYear = new Date().getFullYear()
  for (const m of text.matchAll(/\b(19\d{2}|20\d{2})\b/g)) {
    const y = parseInt(m[1], 10)
    if (y >= 1995 && y <= nowYear + 1) yearSet.add(y)
  }
  // "from 2022 to 2024" → expand
  const range = text.match(/from\s+(20\d{2})\s+(?:to|through|-)\s+(20\d{2})/)
  if (range) {
    const [a, b] = [parseInt(range[1], 10), parseInt(range[2], 10)].sort((x, y) => x - y)
    for (let y = a; y <= b; y++) yearSet.add(y)
  }

  // Forms
  const isQuarter = QUARTER_HINTS.some((k) => text.includes(k))
  const isAnnual = ANNUAL_HINTS.some((k) => text.includes(k))
  const forms = isQuarter && !isAnnual ? ['10-Q'] : ['10-K']

  // Sections
  const sections: string[] = []
  for (const s of SECTION_PATTERNS) {
    if (s.patterns.some((p) => p.test(text))) sections.push(s.id)
  }

  // Kind
  const hasNarrative = NARRATIVE_KEYWORDS.some((k) => text.includes(k)) || sections.length > 0
  const hasNumeric = NUMERIC_KEYWORDS.some((k) => text.includes(k))
  let kind: IntentExtraction['kind'] = 'numeric'
  if (hasNarrative && hasNumeric) kind = 'mixed'
  else if (hasNarrative) kind = 'narrative'
  else if (hasNumeric) kind = 'numeric'

  const companies = await Promise.all(
    [...explicitTickers].map(async (t) => {
      const e = idx.get(t)
      return { ticker: t, cik: e?.cik }
    }),
  )

  return {
    companies,
    years: [...yearSet].sort((a, b) => a - b),
    forms,
    kind,
    sections,
  }
}
