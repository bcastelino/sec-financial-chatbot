import { useState } from 'react'
import { ChevronDown, Check, Zap, Sparkles, Brain, Bolt } from 'lucide-react'
import { DEFAULT_MODELS } from '../lib/llm/models'
import type { ModelOption } from '../types'

function iconFor(key: ModelOption['iconKey']) {
  switch (key) {
    case 'zap': return <Zap className="size-4 text-blue-400" />
    case 'sparkles': return <Sparkles className="size-4 text-purple-400" />
    case 'brain': return <Brain className="size-4 text-emerald-400" />
    case 'bolt': return <Bolt className="size-4 text-cyan-400" />
  }
}

interface Props {
  selectedModelId: string
  onChange: (modelId: string) => void
}

export function ModelSelector({ selectedModelId, onChange }: Props) {
  const [isOpen, setIsOpen] = useState(false)
  const selected = DEFAULT_MODELS.find((m) => m.id === selectedModelId) ?? DEFAULT_MODELS[0]

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setIsOpen((v) => !v)}
        className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-full text-xs font-medium transition-all duration-200 text-[#8a8a8f] hover:text-white hover:bg-white/5 active:scale-95"
      >
        {iconFor(selected.iconKey)}
        <span className="truncate max-w-[180px]">{selected.name}</span>
        <ChevronDown className={`size-3.5 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setIsOpen(false)} />
          <div className="absolute bottom-full left-0 mb-2 z-50 min-w-[260px] bg-[#1a1a1e]/95 backdrop-blur-xl border border-white/10 rounded-xl shadow-2xl shadow-black/50 overflow-hidden animate-fade-in animate-slide-in-from-bottom-2">
            <div className="p-1.5">
              <div className="px-2.5 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-[#5a5a5f]">
                Select Model
              </div>
              {DEFAULT_MODELS.map((model) => (
                <button
                  type="button"
                  key={model.id}
                  onClick={() => {
                    onChange(model.id)
                    setIsOpen(false)
                  }}
                  className={`w-full flex items-center gap-3 px-2.5 py-2 rounded-lg text-left transition-all duration-150 ${
                    selected.id === model.id ? 'bg-white/10 text-white' : 'text-[#a0a0a5] hover:bg-white/5 hover:text-white'
                  }`}
                >
                  <div className="flex-shrink-0">{iconFor(model.iconKey)}</div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium">{model.name}</span>
                      {model.badge && (
                        <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium ${
                          model.badge === 'Pro' ? 'bg-purple-500/20 text-purple-300' :
                          model.badge === 'Fast' ? 'bg-emerald-500/20 text-emerald-300' :
                          'bg-blue-500/20 text-blue-300'
                        }`}>
                          {model.badge}
                        </span>
                      )}
                    </div>
                    <span className="block text-[11px] text-[#6a6a6f] truncate">{model.description}</span>
                  </div>
                  {selected.id === model.id && <Check className="size-4 text-blue-400 flex-shrink-0" />}
                </button>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
