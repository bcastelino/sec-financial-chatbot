import { Github, BookOpen } from 'lucide-react'

interface Props {
  onSampleQuestion?: (q: string) => void
}

const SAMPLES = [
  'What was AAPL revenue in 2022, 2023, and 2024?',
  'Summarize MSFT risk factors from the latest 10-K',
  'Compare NVDA net income FY2023 vs FY2024',
  "What does GOOGL's MD&A say about AI in the latest 10-K?",
]

export function ImportButtons({ onSampleQuestion }: Props) {
  return (
    <div className="flex flex-col items-center gap-3 justify-center">
      <div className="flex items-center gap-3 justify-center text-sm text-[#6a6a6f]">
        <a
          href="https://github.com/bcastelino/sec-financial-chatbot"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border border-white/10 bg-[#0f0f0f] hover:bg-[#1a1a1e] text-[#8a8a8f] hover:text-white transition-all duration-200 active:scale-95"
        >
          <Github className="size-4" />
          <span>View on GitHub</span>
        </a>
        <a
          href="https://www.sec.gov/search-filings/edgar-application-programming-interfaces"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border border-white/10 bg-[#0f0f0f] hover:bg-[#1a1a1e] text-[#8a8a8f] hover:text-white transition-all duration-200 active:scale-95"
        >
          <BookOpen className="size-4" />
          <span>EDGAR API</span>
        </a>
      </div>
      <div className="flex flex-wrap gap-2 justify-center max-w-[680px]">
        {SAMPLES.map((q) => (
          <button
            key={q}
            type="button"
            onClick={() => onSampleQuestion?.(q)}
            className="px-3 py-1.5 rounded-full text-xs text-[#a0a0a5] hover:text-white border border-white/10 bg-white/[0.02] hover:bg-white/[0.06] transition"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  )
}
