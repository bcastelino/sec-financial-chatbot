/**
 * SEC's endpoints (both data.sec.gov and www.sec.gov) do NOT send
 * Access-Control-Allow-Origin headers, so a static SPA running in a browser
 * cannot read their responses directly. We route requests through a public
 * CORS proxy that adds the CORS header.
 *
 * Default: https://corsproxy.io  (free, no key, GET-only)
 * User override: localStorage["sec-chat:proxy-base"] — e.g. a self-hosted
 * Cloudflare Worker URL like "https://my-proxy.example.com/?url="
 *
 * Set to empty string (clearProxyBase) to bypass the proxy and call SEC
 * directly (only works if SEC ever enables CORS, or you are testing locally
 * with a browser CORS extension).
 */

const KEY = 'sec-chat:proxy-base'

/**
 * Ordered list of public CORS proxies tried in sequence. The first that
 * succeeds for a given request wins. Each entry must accept a
 * URL-encoded target URL appended to the base.
 *
 * We keep multiple because any single public proxy can (and regularly does)
 * suffer outages, certificate issues, or rate limits.
 */
const FALLBACK_PROXIES: string[] = [
  'https://api.allorigins.win/raw?url=',
  'https://api.codetabs.com/v1/proxy?quest=',
  'https://corsproxy.io/?url=',
]

const DEFAULT_PROXY = FALLBACK_PROXIES[0]

export function getProxyBase(): string {
  try {
    const v = localStorage.getItem(KEY)
    if (v === null) return DEFAULT_PROXY
    return v
  } catch {
    return DEFAULT_PROXY
  }
}

/**
 * Returns the ordered list of proxy bases to try. If the user explicitly
 * configured an override, that override is the only entry (we trust the
 * user's choice and don't leak requests to other proxies). Otherwise we
 * return the full fallback list.
 */
export function getProxyCandidates(): string[] {
  try {
    const v = localStorage.getItem(KEY)
    if (v === null) return [...FALLBACK_PROXIES]
    // Empty string => no proxy (direct request).
    if (v === '') return ['']
    // User override: try it first, then fall back to the public proxies
    // in case the override is itself down.
    const rest = FALLBACK_PROXIES.filter((p) => p !== v)
    return [v, ...rest]
  } catch {
    return [...FALLBACK_PROXIES]
  }
}

export function setProxyBase(value: string): void {
  try {
    localStorage.setItem(KEY, value)
  } catch {
    // ignore
  }
}

export function clearProxyBase(): void {
  try {
    localStorage.removeItem(KEY)
  } catch {
    // ignore
  }
}

/**
 * Wrap an SEC URL with the active CORS proxy. If the proxy base is empty
 * the URL is returned unchanged.
 */
export function viaProxy(url: string): string {
  const base = getProxyBase()
  if (!base) return url
  return base + encodeURIComponent(url)
}
