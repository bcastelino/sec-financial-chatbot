import { fetchTextCached } from './rateLimiter'

/**
 * Sections we can identify inside a 10-K. Keyed by canonical Item id used in
 * intent extraction. The patterns match common heading variants.
 */
export const SECTION_PATTERNS: { id: string; title: string; patterns: RegExp[] }[] = [
  { id: '1', title: 'Item 1. Business', patterns: [/item\s*1\b\.?\s*(business)?/i] },
  { id: '1A', title: 'Item 1A. Risk Factors', patterns: [/item\s*1a\b\.?\s*(risk\s*factors)?/i, /\brisk\s*factors\b/i] },
  { id: '1B', title: 'Item 1B. Unresolved Staff Comments', patterns: [/item\s*1b\b\.?/i] },
  { id: '2', title: 'Item 2. Properties', patterns: [/item\s*2\b\.?\s*(properties)?/i] },
  { id: '3', title: 'Item 3. Legal Proceedings', patterns: [/item\s*3\b\.?\s*(legal\s*proceedings)?/i] },
  { id: '5', title: 'Item 5. Market for Registrant Common Equity', patterns: [/item\s*5\b\.?/i] },
  { id: '7', title: "Item 7. Management's Discussion and Analysis", patterns: [/item\s*7\b\.?\s*(management('s)?\s*discussion)?/i, /\bmd&a\b/i] },
  { id: '7A', title: 'Item 7A. Quantitative and Qualitative Disclosures About Market Risk', patterns: [/item\s*7a\b\.?/i] },
  { id: '8', title: 'Item 8. Financial Statements', patterns: [/item\s*8\b\.?\s*(financial\s*statements)?/i] },
  { id: '9A', title: 'Item 9A. Controls and Procedures', patterns: [/item\s*9a\b\.?/i] },
]

/**
 * Strip HTML tags + style/script and collapse whitespace to a plain text
 * representation suitable for naive section extraction. We do this client-side
 * because we don't have a server.
 */
export function htmlToText(html: string): string {
  // Remove iXBRL / inline metadata blocks aggressively first
  const cleaned = html
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<!--[\s\S]*?-->/g, ' ')
    .replace(/<\/(p|div|tr|li|h[1-6]|br)>/gi, '\n')
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<[^>]+>/g, ' ')

  return cleaned
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/[ \t]+/g, ' ')
    .replace(/\n[ \t]+/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

export interface ExtractedSection {
  id: string
  title: string
  text: string
}

/**
 * Locate requested Items inside a 10-K plain-text body. The strategy is:
 *  1. Find the first occurrence of the requested Item heading
 *  2. Find the next Item heading that comes after it (any item)
 *  3. Slice between the two
 *
 * This is intentionally simple; SEC HTML is messy and we trade precision for
 * robustness across different filers.
 */
export function extractSections(text: string, ids: string[]): ExtractedSection[] {
  const idSet = new Set(ids.map((i) => i.toUpperCase()))
  const headings = findItemHeadings(text)
  const wanted: ExtractedSection[] = []
  for (let i = 0; i < headings.length; i++) {
    const h = headings[i]
    if (!idSet.has(h.id.toUpperCase())) continue
    const next = headings[i + 1]
    const slice = text.slice(h.index, next ? next.index : Math.min(text.length, h.index + 60_000))
    wanted.push({ id: h.id, title: h.title, text: slice.trim() })
  }
  return wanted
}

interface HeadingHit {
  id: string
  title: string
  index: number
}

function findItemHeadings(text: string): HeadingHit[] {
  const hits: HeadingHit[] = []
  // Match "Item 1A.", "Item 7", "ITEM 1A.", etc. at line-ish boundaries.
  const re = /(^|\n|\.\s)\s*item\s+(\d{1,2}[a-c]?)\b\.?\s*([^\n]{0,120})/gi
  let m: RegExpExecArray | null
  while ((m = re.exec(text)) !== null) {
    const id = m[2].toUpperCase()
    const tail = (m[3] || '').trim()
    hits.push({ id, title: `Item ${id}. ${tail}`.slice(0, 140), index: m.index })
  }
  // Dedupe by id keeping first occurrence (typically the actual section, not TOC).
  // Actually the first hit is usually the TOC, so prefer the *second*.
  const byId = new Map<string, HeadingHit[]>()
  for (const h of hits) {
    const arr = byId.get(h.id) ?? []
    arr.push(h)
    byId.set(h.id, arr)
  }
  const chosen: HeadingHit[] = []
  for (const [, arr] of byId) {
    chosen.push(arr.length > 1 ? arr[1] : arr[0])
  }
  return chosen.sort((a, b) => a.index - b.index)
}

/**
 * Fetch a filing document and extract requested section texts. Truncates each
 * section to a per-section char budget (default 12k) to fit LLM context.
 */
export async function fetchFilingSections(
  url: string,
  sectionIds: string[],
  perSectionBudget = 12_000,
): Promise<ExtractedSection[]> {
  const html = await fetchTextCached(url)
  const text = htmlToText(html)
  const sections = extractSections(text, sectionIds.length > 0 ? sectionIds : ['1A', '7'])
  return sections.map((s) => ({
    ...s,
    text: s.text.length > perSectionBudget ? s.text.slice(0, perSectionBudget) + '\n…[truncated]' : s.text,
  }))
}
