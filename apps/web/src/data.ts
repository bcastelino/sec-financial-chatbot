import type { CompanyRef, FinancialFact } from "@filing-room/contracts";

export const POPULAR_COMPANIES: CompanyRef[] = [
  { ticker: "AAPL", cik: "0000320193", name: "Apple Inc." },
  { ticker: "MSFT", cik: "0000789019", name: "Microsoft Corporation" },
  { ticker: "NVDA", cik: "0001045810", name: "NVIDIA Corporation" },
  { ticker: "AMZN", cik: "0001018724", name: "Amazon.com, Inc." },
  { ticker: "GOOGL", cik: "0001652044", name: "Alphabet Inc." },
  { ticker: "META", cik: "0001326801", name: "Meta Platforms, Inc." },
  { ticker: "JPM", cik: "0000019617", name: "JPMorgan Chase & Co." },
  { ticker: "COST", cik: "0000909832", name: "Costco Wholesale Corporation" },
  { ticker: "TSLA", cik: "0001318605", name: "Tesla, Inc." },
  { ticker: "BRK.B", cik: "0001067983", name: "Berkshire Hathaway Inc." },
];

const revenueByTicker: Record<string, number[]> = {
  AAPL: [274.5, 365.8, 394.3, 383.3, 391.0],
  MSFT: [143.0, 168.1, 198.3, 211.9, 245.1],
  NVDA: [10.9, 16.7, 26.9, 27.0, 60.9],
};

export function demoFacts(company: CompanyRef): FinancialFact[] {
  const values = revenueByTicker[company.ticker] ?? [42, 47, 51, 58, 66];
  return values.map((billions, index) => {
    const year = 2020 + index;
    return {
      concept: "revenue",
      displayLabel: "Revenue",
      value: billions * 1_000_000_000,
      unit: "USD",
      periodStart: `${year - 1}-10-01`,
      periodEnd: `${year}-09-30`,
      fiscalYear: year,
      form: "10-K",
      filedDate: `${year}-11-01`,
      accession: `${company.cik}-${String(year).slice(2)}-000001`,
      sourceUrl: `https://www.sec.gov/edgar/browse/?CIK=${Number(company.cik)}`,
    };
  });
}

export const SAMPLE_PROMPTS = [
  "What changed in management's risk language year over year?",
  "Compare revenue growth and operating margins for AAPL, MSFT, and NVDA.",
  "What does management say is driving capital expenditure?",
];
