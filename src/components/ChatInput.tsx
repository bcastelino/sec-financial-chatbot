import { useEffect, useRef, useState } from 'react'
import { SendHorizontal, KeyRound, Square } from 'lucide-react'
import { ModelSelector } from './ModelSelector'

interface Props {
  selectedModelId: string
  onChangeModel: (id: string) => void
  onSend: (message: string) => void
  onOpenSettings: () => void
  placeholder?: string
  isStreaming?: boolean
  onStop?: () => void
  autoFocus?: boolean
}

export function ChatInput({
  selectedModelId,
  onChangeModel,
  onSend,
  onOpenSettings,
  placeholder = 'Ask about an SEC filing — e.g. "AAPL revenue 2022–2024"',
  isStreaming,
  onStop,
  autoFocus,
}: Props) {
  const [message, setMessage] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`
  }, [message])

  useEffect(() => {
    if (autoFocus) textareaRef.current?.focus()
  }, [autoFocus])

  const handleSubmit = () => {
    const trimmed = message.trim()
    if (!trimmed || isStreaming) return
    onSend(trimmed)
    setMessage('')
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="relative w-full max-w-[720px] mx-auto">
      <div className="absolute -inset-[1px] rounded-2xl bg-gradient-to-b from-white/[0.08] to-transparent pointer-events-none" />
      <div className="relative rounded-2xl bg-[#1e1e22] ring-1 ring-white/[0.08] shadow-[0_0_0_1px_rgba(255,255,255,0.05),0_2px_20px_rgba(0,0,0,0.4)]">
        <textarea
          ref={textareaRef}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={placeholder}
          rows={1}
          className="w-full resize-none bg-transparent text-[15px] text-white placeholder-[#5a5a5f] px-5 pt-5 pb-3 focus:outline-none min-h-[80px] max-h-[200px]"
          style={{ height: '80px' }}
        />

        <div className="flex items-center justify-between px-3 pb-3 pt-1">
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={onOpenSettings}
              title="LLM settings"
              className="flex items-center justify-center size-8 rounded-full bg-white/[0.08] hover:bg-white/[0.12] text-[#8a8a8f] hover:text-white transition-all duration-200 active:scale-95"
            >
              <KeyRound className="size-4" />
            </button>
            <ModelSelector selectedModelId={selectedModelId} onChange={onChangeModel} />
          </div>

          <div className="flex items-center gap-2">
            {isStreaming ? (
              <button
                type="button"
                onClick={onStop}
                className="flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium bg-white/10 hover:bg-white/15 text-white transition-all duration-200 active:scale-95"
              >
                <Square className="size-3.5 fill-current" />
                <span className="hidden sm:inline">Stop</span>
              </button>
            ) : (
              <button
                type="button"
                onClick={handleSubmit}
                disabled={!message.trim()}
                className="flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium bg-[#1488fc] hover:bg-[#1a94ff] text-white transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed active:scale-95 shadow-[0_0_20px_rgba(20,136,252,0.3)]"
              >
                <span className="hidden sm:inline">Ask</span>
                <SendHorizontal className="size-4" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
