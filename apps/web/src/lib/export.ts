import type { FinancialFact } from "@filing-room/contracts";

export function factsToCsv(facts: FinancialFact[]): string {
  const rows = [["Metric", "Value", "Unit", "Fiscal year", "Quarter", "Form", "Period end", "Filed", "Accession", "SEC source"], ...facts.map((fact) => [fact.displayLabel, String(fact.value), fact.unit, String(fact.fiscalYear), fact.fiscalQuarter ? `Q${fact.fiscalQuarter}` : "FY", fact.form, fact.periodEnd, fact.filedDate, fact.accession, fact.sourceUrl])];
  return rows.map((row) => row.map((cell) => `"${String(cell).replaceAll('"', '""')}"`).join(",")).join("\r\n");
}

export function downloadCsv(filename: string, csv: string): void {
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}
