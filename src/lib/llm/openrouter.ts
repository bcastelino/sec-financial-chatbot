import type { LlmSettings } from '../../types'

interface ChatPart {
  role: 'system' | 'user' | 'assistant'
  content: string
}

export interface StreamOptions {
  settings: LlmSettings
  messages: ChatPart[]
  signal?: AbortSignal
  onDelta: (delta: string) => void
  temperature?: number
}

function endpoint(settings: LlmSettings): string {
  return settings.provider === 'openai'
    ? 'https://api.openai.com/v1/chat/completions'
    : 'https://openrouter.ai/api/v1/chat/completions'
}

/**
 * SSE streaming chat completion. Calls onDelta for each incremental token chunk.
 * Returns the full assembled text. Throws on HTTP errors or aborts.
 */
export async function streamChat(opts: StreamOptions): Promise<string> {
  const { settings, messages, onDelta, signal, temperature = 0.2 } = opts
  if (!settings.apiKey) throw new Error('Missing API key')
  if (!settings.model) throw new Error('Missing model id')

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${settings.apiKey}`,
  }
  if (settings.provider === 'openrouter') {
    headers['HTTP-Referer'] = location.origin
    headers['X-Title'] = 'SEC Financial Chatbot'
  }

  const res = await fetch(endpoint(settings), {
    method: 'POST',
    headers,
    body: JSON.stringify({
      model: settings.model,
      messages,
      stream: true,
      temperature,
    }),
    signal,
  })

  if (!res.ok || !res.body) {
    const detail = await safeText(res)
    throw new Error(`LLM request failed (${res.status}): ${detail.slice(0, 240)}`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  let full = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })

    // SSE frames are separated by blank lines.
    let idx: number
    while ((idx = buf.indexOf('\n\n')) !== -1) {
      const frame = buf.slice(0, idx)
      buf = buf.slice(idx + 2)
      const line = frame.split('\n').find((l) => l.startsWith('data:'))
      if (!line) continue
      const payload = line.slice(5).trim()
      if (payload === '[DONE]') {
        return full
      }
      try {
        const obj = JSON.parse(payload)
        const delta = obj?.choices?.[0]?.delta?.content
        if (typeof delta === 'string' && delta.length > 0) {
          full += delta
          onDelta(delta)
        }
      } catch {
        // ignore keep-alive comments / partial frames
      }
    }
  }

  return full
}

async function safeText(res: Response): Promise<string> {
  try {
    return await res.text()
  } catch {
    return ''
  }
}
