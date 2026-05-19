import { useState } from 'react'
import { X, KeyRound, ExternalLink } from 'lucide-react'
import type { LlmProvider, LlmSettings } from '../types'

interface Props {
  initial?: LlmSettings | null
  selectedModelId: string
  onSave: (s: LlmSettings) => void
  onClose: () => void
}

export function ApiKeyModal({ initial, selectedModelId, onSave, onClose }: Props) {
  const [provider, setProvider] = useState<LlmProvider>(initial?.provider ?? 'openrouter')
  const [apiKey, setApiKey] = useState(initial?.apiKey ?? '')
  const [model, setModel] = useState(initial?.model ?? selectedModelId)

  const canSave = apiKey.trim().length > 10 && model.trim().length > 0

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <div className="relative w-full max-w-md rounded-2xl bg-[#1a1a1e] border border-white/10 shadow-2xl">
        <button
          type="button"
          onClick={onClose}
          className="absolute top-3 right-3 size-8 rounded-full hover:bg-white/5 text-[#8a8a8f] hover:text-white flex items-center justify-center transition"
          aria-label="Close"
        >
          <X className="size-4" />
        </button>
        <div className="p-6">
          <div className="flex items-center gap-2 mb-3">
            <div className="size-9 rounded-xl bg-blue-500/15 border border-blue-500/30 flex items-center justify-center">
              <KeyRound className="size-4 text-blue-300" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white">Add your LLM API key</h2>
              <p className="text-xs text-[#8a8a8f]">Stored locally in your browser. Never sent anywhere except the LLM provider.</p>
            </div>
          </div>

          <label className="block text-[11px] uppercase tracking-wider text-[#6a6a6f] mt-4 mb-1">Provider</label>
          <div className="grid grid-cols-2 gap-2">
            {(['openrouter', 'openai'] as LlmProvider[]).map((p) => (
              <button
                key={p}
                type="button"
                onClick={() => setProvider(p)}
                className={`px-3 py-2 rounded-lg text-sm font-medium border transition ${
                  provider === p
                    ? 'border-blue-500/50 bg-blue-500/10 text-white'
                    : 'border-white/10 bg-white/[0.02] text-[#a0a0a5] hover:text-white hover:bg-white/[0.05]'
                }`}
              >
                {p === 'openrouter' ? 'OpenRouter' : 'OpenAI'}
              </button>
            ))}
          </div>

          <label className="block text-[11px] uppercase tracking-wider text-[#6a6a6f] mt-4 mb-1">API key</label>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder={provider === 'openrouter' ? 'sk-or-v1-…' : 'sk-…'}
            autoComplete="off"
            spellCheck={false}
            className="w-full px-3 py-2 rounded-lg bg-[#0f0f0f] border border-white/10 text-sm text-white placeholder-[#5a5a5f] focus:outline-none focus:border-blue-500/50"
          />

          <label className="block text-[11px] uppercase tracking-wider text-[#6a6a6f] mt-4 mb-1">Model</label>
          <input
            type="text"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder="deepseek/deepseek-r1:free"
            className="w-full px-3 py-2 rounded-lg bg-[#0f0f0f] border border-white/10 text-sm text-white placeholder-[#5a5a5f] focus:outline-none focus:border-blue-500/50 font-mono"
          />
          <p className="text-[11px] text-[#6a6a6f] mt-1">
            Any model id supported by your provider works. Default suggestions are listed in the model picker.
          </p>

          <a
            href={provider === 'openrouter' ? 'https://openrouter.ai/keys' : 'https://platform.openai.com/api-keys'}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300 mt-3"
          >
            Get a {provider === 'openrouter' ? 'OpenRouter' : 'OpenAI'} key <ExternalLink className="size-3" />
          </a>

          <div className="mt-6 flex gap-2 justify-end">
            <button
              type="button"
              onClick={onClose}
              className="px-3 py-2 rounded-full text-sm text-[#a0a0a5] hover:text-white hover:bg-white/5 transition"
            >
              Cancel
            </button>
            <button
              type="button"
              disabled={!canSave}
              onClick={() => onSave({ provider, apiKey: apiKey.trim(), model: model.trim() })}
              className="px-4 py-2 rounded-full text-sm font-medium bg-[#1488fc] hover:bg-[#1a94ff] text-white disabled:opacity-40 disabled:cursor-not-allowed transition"
            >
              Save
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
