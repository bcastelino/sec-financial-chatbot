import type { ChatEvent, ChatRequest, CompanyOverview, CompanyRef, QuotaStatus } from "@filing-room/contracts";
import { POPULAR_COMPANIES, demoFacts } from "../data";

const BASE = import.meta.env.VITE_API_BASE_URL ?? "/api/v1";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, { ...init, headers: { "Content-Type": "application/json", ...init?.headers } });
  if (!response.ok) throw new Error(`Request failed (${response.status})`);
  return response.json() as Promise<T>;
}

export async function searchCompanies(query: string): Promise<CompanyRef[]> {
  try {
    return await request<CompanyRef[]>(`/companies/search?q=${encodeURIComponent(query)}`);
  } catch {
    const normalized = query.toLowerCase();
    return POPULAR_COMPANIES.filter((company) => company.ticker.toLowerCase().includes(normalized) || company.name.toLowerCase().includes(normalized)).slice(0, 8);
  }
}

export async function getCompanyOverview(ticker: string): Promise<CompanyOverview> {
  try {
    return await request<CompanyOverview>(`/companies/${encodeURIComponent(ticker)}/overview`);
  } catch {
    const company = POPULAR_COMPANIES.find((item) => item.ticker === ticker.toUpperCase()) ?? POPULAR_COMPANIES[0]!;
    return { company, description: "Public-company financials and disclosures sourced exclusively from SEC filings.", facts: demoFacts(company), filings: [] };
  }
}

export async function getQuota(): Promise<QuotaStatus> {
  try {
    return await request<QuotaStatus>("/quota");
  } catch {
    const reset = new Date();
    reset.setUTCDate(reset.getUTCDate() + 1);
    return { remaining: 5, limit: 5, resetsAt: reset.toISOString(), budgetAvailable: true };
  }
}

export async function streamChat(payload: ChatRequest, onEvent: (event: ChatEvent) => void, signal?: AbortSignal): Promise<void> {
  const response = await fetch(`${BASE}/chat/stream`, { method: "POST", headers: { "Content-Type": "application/json", Accept: "text/event-stream" }, body: JSON.stringify(payload), signal });
  if (!response.ok || !response.body) throw new Error(response.status === 429 ? "Your five daily answers have been used." : "Research service is unavailable.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done }).replace(/\r\n/g, "\n");
    let boundary = buffer.indexOf("\n\n");
    while (boundary >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const eventName = block.split("\n").find((line) => line.startsWith("event:"))?.slice(6).trim();
      const data = block.split("\n").filter((line) => line.startsWith("data:")).map((line) => line.slice(5).trim()).join("\n");
      if (eventName && data) onEvent({ type: eventName, data: JSON.parse(data) } as ChatEvent);
      boundary = buffer.indexOf("\n\n");
    }
    if (done) break;
  }
}
