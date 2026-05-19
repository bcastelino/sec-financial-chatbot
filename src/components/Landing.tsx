import { RayBackground } from './RayBackground'
import { AnnouncementBadge } from './AnnouncementBadge'
import { ChatInput } from './ChatInput'
import { ImportButtons } from './ImportButtons'

interface Props {
  selectedModelId: string
  onChangeModel: (id: string) => void
  onSend: (message: string) => void
  onOpenSettings: () => void
}

export function Landing({ selectedModelId, onChangeModel, onSend, onOpenSettings }: Props) {
  return (
    <div className="relative flex flex-col items-center justify-center min-h-screen w-full overflow-hidden bg-[#0f0f0f]">
      <RayBackground />

      <div className="absolute top-[70px]">
        <AnnouncementBadge
          text="Powered by SEC EDGAR — live data, BYOK"
          href="https://www.sec.gov/search-filings/edgar-application-programming-interfaces"
        />
      </div>

      <div className="absolute top-[66%] left-1/2 sm:top-1/2 -translate-x-1/2 -translate-y-1/2 flex flex-col items-center justify-center w-full h-full overflow-hidden px-4">
        <div className="text-center mb-6">
          <h1 className="text-4xl sm:text-5xl font-bold text-white tracking-tight mb-1">
            What will you{' '}
            <span className="bg-gradient-to-b from-[#4da5fc] via-[#4da5fc] to-white bg-clip-text text-transparent italic">
              discover
            </span>{' '}
            today?
          </h1>
          <p className="text-base font-semibold sm:text-lg text-[#8a8a8f]">
            Chat with SEC filings — live EDGAR data, your own LLM key.
          </p>
        </div>

        <div className="w-full max-w-[720px] mb-6 sm:mb-8 mt-2">
          <ChatInput
            selectedModelId={selectedModelId}
            onChangeModel={onChangeModel}
            onSend={onSend}
            onOpenSettings={onOpenSettings}
            autoFocus
          />
        </div>

        <ImportButtons onSampleQuestion={onSend} />
      </div>
    </div>
  )
}
