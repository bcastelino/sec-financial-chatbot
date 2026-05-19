import { getProxyCandidates } from './proxy'

/**
 * Try a fetch through each configured proxy in turn. The first proxy that
 * returns a 2xx response wins. We only fall through on network-level errors
 * or 5xx/4xx-from-proxy responses; once we have any 2xx we stop. The last
 * error is rethrown if every candidate fails.
 */
async function fetchWithProxyFallback(
  url: string,
  init: RequestInit,
  label: string,
): Promise<Response> {
  const candidates = getProxyCandidates()
  let lastErr: unknown = null
  for (const base of candidates) {
    const fetchUrl = base ? base + encodeURIComponent(url) : url
    try {
      const res = await fetch(fetchUrl, init)
      if (res.ok) return res
      // Non-2xx: capture and try next proxy. If this was the last one, the
      // caller will receive this response back via the thrown error path.
      lastErr = new Error(`HTTP ${res.status} from ${fetchUrl}`)
      console.warn(`[sec] ${label} non-OK via proxy`, { url, fetchUrl, status: res.status })
      // If the SEC origin itself returned a real 4xx (not a proxy error),
      // trying another proxy won't help. Heuristic: 404 means not found at
      // SEC; surface immediately.
      if (res.status === 404) return res
    } catch (e) {
      lastErr = e
      const msg = (e as Error)?.message ?? String(e)
      console.warn(`[sec] ${label} network error via proxy`, { url, fetchUrl, error: msg })
    }
  }
  const msg = (lastErr as Error)?.message ?? String(lastErr)
  console.error(`[sec] ${label} all proxies failed`, { url, error: msg })
  throw new Error(`Network/CORS error fetching ${url}: ${msg}`)
}

/**
 * Simple promise-chain rate limiter to keep us under SEC's ~10 req/s fair-use
 * limit. We target 5 rps with a small jitter to be polite from a browser.
 */

const MIN_INTERVAL_MS = 200 // ~5 req/s

let chain: Promise<unknown> = Promise.resolve()

export function throttle<T>(fn: () => Promise<T>): Promise<T> {
  const next = chain.then(async () => {
    const result = await fn()
    await sleep(MIN_INTERVAL_MS)
    return result
  })
  // Don't let one rejection break the entire chain.
  chain = next.catch(() => undefined)
  return next as Promise<T>
}

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms))
}

const cache = new Map<string, { at: number; data: unknown }>()
const CACHE_TTL_MS = 5 * 60 * 1000

/**
 * Cached + throttled fetch+json helper. Caches successful JSON responses for
 * 5 minutes to minimize repeat hits on EDGAR.
 */
export async function fetchJsonCached<T>(url: string, init?: RequestInit): Promise<T> {
  const cached = cache.get(url)
  if (cached && Date.now() - cached.at < CACHE_TTL_MS) {
    return cached.data as T
  }
  const data = await throttle(async () => {
    const res = await fetchWithProxyFallback(
      url,
      {
        ...init,
        headers: {
          Accept: 'application/json',
          ...(init?.headers ?? {}),
        },
      },
      'fetchJsonCached',
    )
    if (!res.ok) {
      const body = await safeText(res)
      console.error('[sec] fetchJsonCached HTTP error', { url, status: res.status, body: body.slice(0, 200) })
      throw new Error(`SEC request failed (${res.status}) ${url} :: ${body.slice(0, 160)}`)
    }
    return (await res.json()) as T
  })
  cache.set(url, { at: Date.now(), data })
  return data
}

/**
 * Cached + throttled fetch+text helper for filing documents (HTML/text).
 */
export async function fetchTextCached(url: string, init?: RequestInit): Promise<string> {
  const cached = cache.get(url)
  if (cached && Date.now() - cached.at < CACHE_TTL_MS) {
    return cached.data as string
  }
  const data = await throttle(async () => {
    const res = await fetchWithProxyFallback(
      url,
      {
        ...init,
        headers: {
          Accept: 'text/html,application/xhtml+xml,*/*',
          ...(init?.headers ?? {}),
        },
      },
      'fetchTextCached',
    )
    if (!res.ok) {
      const body = await safeText(res)
      console.error('[sec] fetchTextCached HTTP error', { url, status: res.status, body: body.slice(0, 200) })
      throw new Error(`SEC request failed (${res.status}) ${url} :: ${body.slice(0, 160)}`)
    }
    return await res.text()
  })
  cache.set(url, { at: Date.now(), data })
  return data
}

async function safeText(res: Response): Promise<string> {
  try { return await res.text() } catch { return '' }
}
