import type { FinancialFact } from "@filing-room/contracts";

function billions(value: number): string {
  return `$${(value / 1_000_000_000).toFixed(value >= 100_000_000_000 ? 0 : 1)}B`;
}

export function MetricChart({ facts, label = "Revenue" }: { facts: FinancialFact[]; label?: string }) {
  const ordered = [...facts].sort((a, b) => a.fiscalYear - b.fiscalYear);
  const max = Math.max(...ordered.map((fact) => fact.value), 1);
  const points = ordered.map((fact, index) => ({ x: 16 + index * (268 / Math.max(1, ordered.length - 1)), y: 112 - (fact.value / max) * 88, fact }));
  const path = points.map((point, index) => `${index ? "L" : "M"}${point.x},${point.y}`).join(" ");
  return (
    <figure className="metric-chart">
      <figcaption><span>{label}</span><strong>{ordered.at(-1) ? billions(ordered.at(-1)!.value) : "N/A"}</strong><small>Five-year filing history</small></figcaption>
      <svg viewBox="0 0 300 140" role="img" aria-label={`${label} over five fiscal years`}>
        <path className="chart-grid" d="M16 24H284M16 68H284M16 112H284" />
        <path className="chart-area" d={`${path} L284,112 L16,112 Z`} />
        <path className="chart-line" d={path} />
        {points.map(({ x, y, fact }) => <g key={fact.fiscalYear}><circle cx={x} cy={y} r="3" /><text x={x} y="132" textAnchor="middle">{fact.fiscalYear}</text></g>)}
      </svg>
    </figure>
  );
}
