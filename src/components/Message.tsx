import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { AlertCircle, Bot, User as UserIcon, FileText, ExternalLink } from 'lucide-react'
import type { ChatMessage } from '../types'

interface Props {
  message: ChatMessage
}

export function Message({ message }: Props) {
  const isUser = message.role === 'user'
  return (
    <div className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <div className="flex-shrink-0 size-8 rounded-full bg-blue-500/15 border border-blue-500/30 flex items-center justify-center">
          <Bot className="size-4 text-blue-300" />
        </div>
      )}
      <div className={`max-w-[80ch] ${isUser ? 'order-first' : ''}`}>
        <div
          className={`rounded-2xl px-4 py-3 text-[15px] leading-relaxed ${
            isUser
              ? 'bg-[#1488fc] text-white shadow-[0_0_20px_rgba(20,136,252,0.25)]'
              : 'bg-[#1a1a1e] border border-white/[0.06] text-[#e6e6ea]'
          }`}
        >
          {message.error ? (
            <div className="flex items-start gap-2 text-red-300">
              <AlertCircle className="size-4 mt-0.5 flex-shrink-0" />
              <div>
                <div className="font-medium">Something went wrong</div>
                <div className="text-sm text-red-200/80 mt-1 break-words">{message.error}</div>
              </div>
            </div>
          ) : isUser ? (
            <div className="whitespace-pre-wrap">{message.content}</div>
          ) : (
            <div className="md">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content || (message.pending ? '…' : '')}</ReactMarkdown>
              {message.pending && <span className="inline-block ml-1 w-2 h-4 bg-blue-400/70 animate-pulse align-middle" />}
            </div>
          )}
        </div>

        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {message.sources.slice(0, 8).map((s, i) => (
              <a
                key={`${s.url}-${i}`}
                href={s.url}
                target="_blank"
                rel="noopener noreferrer"
                title={s.accession ? `Accession ${s.accession}` : s.label}
                className="inline-flex items-center gap-1 px-2 py-1 rounded-full border border-white/10 bg-white/[0.04] hover:bg-white/[0.08] text-[11px] text-[#a0a0a5] hover:text-white transition"
              >
                <FileText className="size-3" />
                <span>{s.label}</span>
                <ExternalLink className="size-2.5 opacity-60" />
              </a>
            ))}
          </div>
        )}
      </div>
      {isUser && (
        <div className="flex-shrink-0 size-8 rounded-full bg-white/10 border border-white/10 flex items-center justify-center">
          <UserIcon className="size-4 text-white/80" />
        </div>
      )}
    </div>
  )
}
