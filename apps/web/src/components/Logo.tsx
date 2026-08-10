export function Logo({ compact = false }: { compact?: boolean }) {
  return (
    <span className="brand" aria-label="Filing Room">
      <img className="brand-mark" src="/fr-logo.png" alt="" aria-hidden="true" />
      {!compact && <span className="brand-word">Filing Room</span>}
    </span>
  );
}
