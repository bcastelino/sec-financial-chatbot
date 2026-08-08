export function Logo({ compact = false }: { compact?: boolean }) {
  return (
    <span className="brand" aria-label="Filing Room">
      <svg className="brand-mark" viewBox="0 0 48 48" aria-hidden="true">
        <path className="brand-paper" d="M8 6h20l12 12v24H8z" />
        <path className="brand-tab" d="M28 6v12h12" />
        <path className="brand-line" d="M14 32l7-7 5 4 9-10" />
        <path className="brand-rule" d="M14 37h20" />
      </svg>
      {!compact && <span className="brand-word">Filing Room</span>}
    </span>
  );
}
