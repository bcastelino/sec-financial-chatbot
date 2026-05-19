import { Bolt } from 'lucide-react'

export function AnnouncementBadge({ text, href = '#' }: { text: string; href?: string }) {
  const className =
    'relative inline-flex items-center gap-2 px-5 py-2 min-h-[40px] rounded-full text-sm overflow-hidden transition-all duration-300 hover:scale-[1.02] active:scale-[0.98] cursor-pointer'
  const style: React.CSSProperties = {
    background: 'linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05))',
    backdropFilter: 'blur(20px) saturate(140%)',
    boxShadow:
      'inset 0 1px rgba(255,255,255,0.2), inset 0 -1px rgba(0,0,0,0.1), 0 8px 32px -8px rgba(0,0,0,0.1), 0 0 0 1px rgba(255,255,255,0.08)',
  }

  const content = (
    <>
      <span
        className="absolute top-0 left-0 right-0 h-1/2 pointer-events-none opacity-70 mix-blend-overlay"
        style={{ background: 'radial-gradient(ellipse at center top, rgba(255, 255, 255, 0.15) 0%, transparent 70%)' }}
      />
      <span
        className="absolute -top-px left-1/2 -translate-x-1/2 h-[2px] w-[100px] opacity-60"
        style={{
          background:
            'linear-gradient(90deg, transparent 0%, rgba(37, 119, 255, 0.8) 20%, rgba(126, 93, 225, 0.8) 50%, rgba(59, 130, 246, 0.8) 80%, transparent 100%)',
          filter: 'blur(0.5px)',
        }}
      />
      <Bolt className="size-4 relative z-10 text-white" />
      <span className="relative z-10 text-white font-medium">{text}</span>
    </>
  )

  return href !== '#' ? (
    <a href={href} target="_blank" rel="noopener noreferrer" className={className} style={style}>
      {content}
    </a>
  ) : (
    <button className={className} style={style}>
      {content}
    </button>
  )
}
