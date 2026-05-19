import { fetchJsonCached } from './rateLimiter'
import type { TickerEntry } from '../../types'

/**
 * The SEC publishes a JSON map of (ticker → CIK → name). We:
 *   1. Try a fast static snapshot bundled with the app (public/data/company_tickers.json).
 *   2. Fall back to fetching the live file from sec.gov on demand.
 *
 * The raw shape is: { "0": { cik_str: 320193, ticker: "AAPL", title: "Apple Inc." }, ... }
 */

const LIVE_URL = 'https://www.sec.gov/files/company_tickers.json'
const STATIC_URL = `${import.meta.env.BASE_URL}data/company_tickers.json`

interface RawEntry {
  cik_str: number
  ticker: string
  title: string
}

type RawMap = Record<string, RawEntry>

let cache: Map<string, TickerEntry> | null = null

function normalize(raw: RawMap): Map<string, TickerEntry> {
  const map = new Map<string, TickerEntry>()
  for (const key of Object.keys(raw)) {
    const r = raw[key]
    if (!r || typeof r.cik_str !== 'number' || !r.ticker) continue
    const entry: TickerEntry = {
      cik: String(r.cik_str).padStart(10, '0'),
      ticker: r.ticker.toUpperCase(),
      name: r.title,
    }
    map.set(entry.ticker, entry)
  }
  return map
}

async function loadStatic(): Promise<Map<string, TickerEntry> | null> {
  try {
    const res = await fetch(STATIC_URL, { headers: { Accept: 'application/json' } })
    if (!res.ok) return null
    const raw = (await res.json()) as RawMap
    return normalize(raw)
  } catch {
    return null
  }
}

async function loadLive(): Promise<Map<string, TickerEntry> | null> {
  try {
    const raw = await fetchJsonCached<RawMap>(LIVE_URL)
    return normalize(raw)
  } catch {
    return null
  }
}

export async function ensureTickerIndex(): Promise<Map<string, TickerEntry>> {
  if (cache && cache.size > 0) return cache
  cache = (await loadStatic()) ?? (await loadLive()) ?? new Map()
  // Best-effort live refresh after static load (non-blocking).
  if (cache.size > 0) {
    loadLive().then((live) => {
      if (live && live.size > 0) cache = live
    })
  }
  return cache
}

export async function lookupTicker(ticker: string): Promise<TickerEntry | null> {
  const idx = await ensureTickerIndex()
  return idx.get(ticker.toUpperCase()) ?? null
}

/**
 * Best-effort name → entry resolution. Matches common variants like
 * "apple" or "alphabet". Returns the highest-scoring matches.
 */
export async function searchByName(query: string, limit = 5): Promise<TickerEntry[]> {
  const idx = await ensureTickerIndex()
  const q = query.trim().toLowerCase()
  if (!q) return []
  const scored: { e: TickerEntry; score: number }[] = []
  for (const e of idx.values()) {
    const name = e.name.toLowerCase()
    let score = 0
    if (name === q) score += 100
    else if (name.startsWith(q)) score += 50
    else if (name.includes(q)) score += 20
    if (e.ticker.toLowerCase() === q) score += 80
    if (score > 0) scored.push({ e, score })
  }
  scored.sort((a, b) => b.score - a.score)
  return scored.slice(0, limit).map((s) => s.e)
}
