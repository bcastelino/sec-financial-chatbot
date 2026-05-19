import { useEffect, useRef } from 'react'
import { Plus, Trash2 } from 'lucide-react'
import { Message } from './Message'
import { ChatInput } from './ChatInput'
import type { ChatMessage } from '../types'

interface Props {
  messages: ChatMessage[]
  selectedModelId: string
  onChangeModel: (id: string) => void
  onSend: (message: string) => void
  onOpenSettings: () => void
  onNewChat: () => void
  onClearChat: () => void
  isStreaming: boolean
  onStop: () => void
}

export function ChatRoom({
  messages,
  selectedModelId,
  onChangeModel,
  onSend,
  onOpenSettings,
  onNewChat,
  onClearChat,
  isStreaming,
  onStop,
}: Props) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  }, [messages])

  return (
    <div className="relative flex flex-col h-screen w-full bg-[#0f0f0f]">
      {/* Header */}
      <header className="flex items-center justify-between px-4 sm:px-6 py-3 border-b border-white/[0.06]">
        <div className="flex items-center gap-2">
          <img src={`${import.meta.env.BASE_URL}fevicon.png`} alt="" className="size-6" />
          <span className="text-sm font-semibold text-white">SEC Financial Chatbot</span>
          <span className="hidden sm:inline text-[11px] px-2 py-0.5 rounded-full bg-white/[0.06] text-[#8a8a8f]">live EDGAR</span>
        </div>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={onNewChat}
            title="New chat"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium text-[#a0a0a5] hover:text-white hover:bg-white/5 transition"
          >
            <Plus className="size-4" />
            <span className="hidden sm:inline">New</span>
          </button>
          <button
            type="button"
            onClick={onClearChat}
            title="Clear chat"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium text-[#a0a0a5] hover:text-white hover:bg-white/5 transition"
          >
            <Trash2 className="size-4" />
            <span className="hidden sm:inline">Clear</span>
          </button>
        </div>
      </header>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 sm:px-6 py-6">
        <div className="mx-auto max-w-[860px] flex flex-col gap-5">
          {messages.map((m) => (
            <Message key={m.id} message={m} />
          ))}
          <div className="h-8" />
        </div>
      </div>

      {/* Input */}
      <div className="px-4 sm:px-6 pb-5">
        <ChatInput
          selectedModelId={selectedModelId}
          onChangeModel={onChangeModel}
          onSend={onSend}
          onOpenSettings={onOpenSettings}
          isStreaming={isStreaming}
          onStop={onStop}
        />
        <p className="text-center text-[11px] text-[#5a5a5f] mt-3">
          Numbers come from SEC XBRL (companyfacts). Narrative text is fetched live from the filing. Verify before relying on it.
        </p>
      </div>
    </div>
  )
}
