import { useEffect, useMemo, useRef, useState } from 'react'
import { Landing } from './components/Landing'
import { ChatRoom } from './components/ChatRoom'
import { ApiKeyModal } from './components/ApiKeyModal'
import { DEFAULT_MODELS } from './lib/llm/models'
import { buildContext, composeMessages } from './lib/llm/prompt'
import { streamChat } from './lib/llm/openrouter'
import { loadLlmSettings, saveLlmSettings, loadHistory, saveHistory, clearHistory } from './lib/storage'
import type { ChatMessage, LlmSettings } from './types'

interface PersistedState {
  messages: ChatMessage[]
  selectedModelId: string
}

function uid() {
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36)
}

export default function App() {
  const persisted = loadHistory<PersistedState>()
  // If the persisted model id is no longer one of the curated DEFAULT_MODELS,
  // fall back to the default so removed/renamed ids don't cause 404s.
  const knownIds = new Set(DEFAULT_MODELS.map((m) => m.id))
  const initialModelId = persisted?.selectedModelId && knownIds.has(persisted.selectedModelId)
    ? persisted.selectedModelId
    : DEFAULT_MODELS[0].id
  const [messages, setMessages] = useState<ChatMessage[]>(persisted?.messages ?? [])
  const [selectedModelId, setSelectedModelId] = useState<string>(initialModelId)
  const [settings, setSettings] = useState<LlmSettings | null>(loadLlmSettings())
  const [showKeyModal, setShowKeyModal] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  // Persist chat state
  useEffect(() => {
    saveHistory<PersistedState>({ messages, selectedModelId })
  }, [messages, selectedModelId])

  const showLanding = messages.length === 0
  const effectiveSettings = useMemo<LlmSettings | null>(() => {
    if (!settings) return null
    // Always sync the active model with the selector unless the user typed a custom one.
    return { ...settings, model: settings.model || selectedModelId }
  }, [settings, selectedModelId])

  function openSettings() {
    setShowKeyModal(true)
  }

  function newChat() {
    if (abortRef.current) abortRef.current.abort()
    setMessages([])
    setIsStreaming(false)
  }

  function clearChat() {
    if (abortRef.current) abortRef.current.abort()
    setMessages([])
    setIsStreaming(false)
    clearHistory()
  }

  function stop() {
    abortRef.current?.abort()
  }

  async function handleSend(content: string) {
    const text = content.trim()
    if (!text || isStreaming) return

    // Require API key before any LLM call.
    const current = settings ?? loadLlmSettings()
    if (!current?.apiKey) {
      setShowKeyModal(true)
      return
    }

    const userMsg: ChatMessage = {
      id: uid(),
      role: 'user',
      content: text,
      createdAt: Date.now(),
    }
    const asstId = uid()
    const asstMsg: ChatMessage = {
      id: asstId,
      role: 'assistant',
      content: '',
      pending: true,
      createdAt: Date.now(),
    }
    setMessages((prev) => [...prev, userMsg, asstMsg])
    setIsStreaming(true)

    const controller = new AbortController()
    abortRef.current = controller

    try {
      // 1) Build SEC context
      const built = await buildContext(text, [...messages, userMsg])

      // 2) Compose prompt + stream
      const prompt = composeMessages([...messages, userMsg], text, built.contextMarkdown)

      // selectedModelId is the source of truth; settings.model is just a default.
      const activeSettings: LlmSettings = {
        provider: current.provider,
        apiKey: current.apiKey,
        model: selectedModelId,
      }

      // Reflect intent warnings in the streamed message (will be prepended if no answer).
      if (built.warnings.length > 0) {
        // attach as system hint inside the user message rather than a separate role to keep UI simple
        prompt[prompt.length - 1].content += `\n\nNOTE (data fetch warnings, may affect completeness):\n- ${built.warnings.join('\n- ')}`
      }

      await streamChat({
        settings: activeSettings,
        messages: prompt,
        signal: controller.signal,
        onDelta: (delta) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === asstId ? { ...m, content: m.content + delta, pending: true } : m,
            ),
          )
        },
      })

      setMessages((prev) =>
        prev.map((m) =>
          m.id === asstId
            ? { ...m, pending: false, sources: built.sources }
            : m,
        ),
      )
    } catch (err) {
      const message = (err as Error)?.message ?? 'Unknown error'
      const aborted = controller.signal.aborted
      setMessages((prev) =>
        prev.map((m) =>
          m.id === asstId
            ? {
              ...m,
              pending: false,
              error: aborted ? 'Generation stopped.' : message,
            }
            : m,
        ),
      )
    } finally {
      setIsStreaming(false)
      abortRef.current = null
    }
  }

  return (
    <>
      {showLanding ? (
        <Landing
          selectedModelId={selectedModelId}
          onChangeModel={setSelectedModelId}
          onSend={handleSend}
          onOpenSettings={openSettings}
        />
      ) : (
        <ChatRoom
          messages={messages}
          selectedModelId={selectedModelId}
          onChangeModel={setSelectedModelId}
          onSend={handleSend}
          onOpenSettings={openSettings}
          onNewChat={newChat}
          onClearChat={clearChat}
          isStreaming={isStreaming}
          onStop={stop}
        />
      )}

      {showKeyModal && (
        <ApiKeyModal
          initial={effectiveSettings}
          selectedModelId={selectedModelId}
          onClose={() => setShowKeyModal(false)}
          onSave={(s) => {
            saveLlmSettings(s)
            setSettings(s)
            setSelectedModelId(s.model)
            setShowKeyModal(false)
          }}
        />
      )}
    </>
  )
}
