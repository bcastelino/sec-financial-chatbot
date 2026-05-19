import type { LlmSettings } from '../types'

const KEY_LLM = 'sec-chat:llm-settings:v1'
const KEY_HISTORY = 'sec-chat:history:v1'

export function loadLlmSettings(): LlmSettings | null {
  try {
    const raw = localStorage.getItem(KEY_LLM)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    if (!parsed || typeof parsed.apiKey !== 'string') return null
    return parsed as LlmSettings
  } catch {
    return null
  }
}

export function saveLlmSettings(s: LlmSettings): void {
  localStorage.setItem(KEY_LLM, JSON.stringify(s))
}

export function clearLlmSettings(): void {
  localStorage.removeItem(KEY_LLM)
}

export function loadHistory<T>(): T | null {
  try {
    const raw = localStorage.getItem(KEY_HISTORY)
    if (!raw) return null
    return JSON.parse(raw) as T
  } catch {
    return null
  }
}

export function saveHistory<T>(data: T): void {
  try {
    localStorage.setItem(KEY_HISTORY, JSON.stringify(data))
  } catch {
    // ignore quota errors
  }
}

export function clearHistory(): void {
  localStorage.removeItem(KEY_HISTORY)
}
