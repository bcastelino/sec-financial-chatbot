import { getSubmissions, selectFilings, filingDocumentUrl, filingIndexUrl } from '../sec/submissions'
import { getCompanyFacts, summarizeFactsAsMarkdown } from '../sec/facts'
import { fetchFilingSections } from '../sec/filingDoc'
import { extractIntent } from './intent'
import type { ChatMessage, IntentExtraction, SourceRef } from '../../types'

export const SYSTEM_PROMPT = `You are SEC Chat, a careful financial analyst grounded in SEC EDGAR filings.

Rules:
- ONLY use the facts and excerpts provided in the CONTEXT block below. Do not invent numbers, dates, or quotes.
- If the context is missing what the user asked for, say so plainly and suggest a more specific question (e.g. specify a year or filing).
- Cite sources inline using the markers provided in the context (e.g. "[AAPL FY2024 10-K, accession 0000320193-24-000123]").
- Use Markdown: bold for emphasis, bullet lists, and **tables** when comparing multiple companies or years.
- Be concise but complete. Round large USD values to billions/millions where it improves readability.
- If the user asks about something not contained in SEC filings (e.g. live stock price, opinions), say it's out of scope.`

export interface BuiltContext {
  intent: IntentExtraction
  contextMarkdown: string
  sources: SourceRef[]
  warnings: string[]
}

/**
 * Gather SEC context for a single user turn:
 *  1. Extract intent
 *  2. For each company, fetch submissions + (if numeric/mixed) company facts
 *  3. For each requested year+form, optionally fetch and extract narrative sections
 *  4. Combine into a single markdown CONTEXT block
 */
export async function buildContext(
  userQuery: string,
  history: ChatMessage[],
): Promise<BuiltContext> {
  const historyTexts = history.filter((m) => m.role !== 'system').map((m) => m.content)
  const intent = await extractIntent(userQuery, historyTexts)
  const warnings: string[] = []
  const sources: SourceRef[] = []
  const blocks: string[] = []

  if (intent.companies.length === 0) {
    warnings.push('No company recognized. Mention a ticker (e.g. AAPL) or company name.')
  }

  for (const { ticker, cik } of intent.companies) {
    if (!cik) {
      warnings.push(`Unknown CIK for ${ticker}`)
      continue
    }

    // Numeric / mixed → XBRL facts
    if (intent.kind === 'numeric' || intent.kind === 'mixed') {
      try {
        const facts = await getCompanyFacts(cik)
        const { markdown, cited } = summarizeFactsAsMarkdown(ticker, facts.raw, intent.years)
        blocks.push(markdown)
        for (const f of cited.slice(0, 8)) {
          sources.push({
            label: `${ticker} ${f.form} FY${f.fy ?? f.end.slice(0, 4)} — ${f.concept}`,
            url: `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=${cik}&type=${f.form}&dateb=&owner=include&count=40`,
            form: f.form,
            accession: f.accession,
            ticker,
            fiscalPeriod: `FY${f.fy ?? ''}`,
          })
        }
      } catch (e) {
        warnings.push(`XBRL facts fetch failed for ${ticker}: ${(e as Error).message}`)
      }
    }

    // Narrative / mixed → fetch primary 10-K (or 10-Q) and pull sections
    if (intent.kind === 'narrative' || intent.kind === 'mixed') {
      try {
        const subs = await getSubmissions(cik)
        const filings = selectFilings(subs.filings, intent.forms, intent.years)
        if (filings.length === 0) {
          warnings.push(`No ${intent.forms.join('/')} found for ${ticker} in years ${intent.years.join(', ') || '(latest)'}`)
        }
        for (const filing of filings.slice(0, 3)) {
          const docUrl = filingDocumentUrl(cik, filing)
          try {
            const sections = await fetchFilingSections(docUrl, intent.sections)
            const heading = `**${ticker} ${filing.form} — filed ${filing.filingDate}** (accession ${filing.accessionNumber})`
            const body = sections
              .map((s) => `### ${s.title}\n\n${s.text}`)
              .join('\n\n')
            blocks.push(`${heading}\n\n${body || '_No matching sections extracted._'}`)
            sources.push({
              label: `${ticker} ${filing.form} ${filing.filingDate}`,
              url: filingIndexUrl(cik, filing),
              form: filing.form,
              accession: filing.accessionNumber,
              filingDate: filing.filingDate,
              ticker,
            })
          } catch (e) {
            warnings.push(`Filing fetch failed (${ticker} ${filing.accessionNumber}): ${(e as Error).message}. This is often a CORS restriction on www.sec.gov from the browser.`)
          }
        }
      } catch (e) {
        warnings.push(`Submissions fetch failed for ${ticker}: ${(e as Error).message}`)
      }
    }
  }

  const warningBlock = warnings.length > 0
    ? `\n\n**DATA FETCH WARNINGS** (these prevented full retrieval; mention them in your answer if relevant):\n- ${warnings.join('\n- ')}`
    : ''

  const contextMarkdown = blocks.length > 0
    ? `CONTEXT (from SEC EDGAR):\n\n${blocks.join('\n\n---\n\n')}${warningBlock}`
    : `CONTEXT: (no SEC data could be retrieved for this query)${warningBlock}`

  return { intent, contextMarkdown, sources, warnings }
}

/**
 * Compose the final chat-completions messages array.
 */
export function composeMessages(
  history: ChatMessage[],
  userQuery: string,
  contextMarkdown: string,
): { role: 'system' | 'user' | 'assistant'; content: string }[] {
  const msgs: { role: 'system' | 'user' | 'assistant'; content: string }[] = []
  msgs.push({ role: 'system', content: SYSTEM_PROMPT })
  // Keep last N turns (excluding pending/failed) for follow-ups.
  const turns = history.filter((m) => m.role !== 'system' && !m.pending && !m.error).slice(-8)
  for (const t of turns) {
    msgs.push({ role: t.role as 'user' | 'assistant', content: t.content })
  }
  msgs.push({
    role: 'user',
    content: `${contextMarkdown}\n\n---\n\nUSER QUESTION: ${userQuery}`,
  })
  return msgs
}
