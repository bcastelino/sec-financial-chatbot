import type { ChatMessage, CompanyRef, FinancialFact, QuotaStatus, ResearchScope, SourceRef } from "@filing-room/contracts";
import { Check, ChevronDown, Copy, Download, ExternalLink, FileSearch, History, Menu, PanelRightClose, Plus, Send, Trash2, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { CitationText } from "../components/CitationText";
import { CompanySearch } from "../components/CompanySearch";
import { Turnstile } from "../components/Turnstile";
import { POPULAR_COMPANIES, demoFacts } from "../data";
import { getQuota, streamChat } from "../lib/api";
import { downloadCsv, factsToCsv } from "../lib/export";
import { clearConversations, deleteConversation, listConversations, saveConversation, type LocalConversation } from "../lib/history";

interface ThreadMessage extends ChatMessage { sources?: SourceRef[] }

function companiesFrom(value: string | null): CompanyRef[] {
  const tickers = (value ?? "AAPL").split(",").slice(0, 3);
  return tickers.map((ticker) => POPULAR_COMPANIES.find((company) => company.ticker === ticker.toUpperCase())).filter((company): company is CompanyRef => Boolean(company));
}

export function ResearchPage({ theme }: { theme: "light" | "dark" }) {
  const [params] = useSearchParams();
  const [companies, setCompanies] = useState<CompanyRef[]>(() => companiesFrom(params.get("companies")));
  const [forms, setForms] = useState<Array<"10-K" | "10-Q">>(() => (params.get("forms")?.split(",").filter((form): form is "10-K" | "10-Q" => form === "10-K" || form === "10-Q") ?? ["10-K", "10-Q"]));
  const [years, setYears] = useState<number[]>(() => (params.get("years") ?? "2024,2023,2022").split(",").map(Number).filter(Boolean).slice(0, 5));
  const [query, setQuery] = useState(params.get("q") ?? "");
  const [messages, setMessages] = useState<ThreadMessage[]>([]);
  const [sources, setSources] = useState<SourceRef[]>([]);
  const [activeSource, setActiveSource] = useState<SourceRef | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [status, setStatus] = useState("Ready to search selected filings");
  const [quota, setQuota] = useState<QuotaStatus>({ remaining: 5, limit: 5, resetsAt: "", budgetAvailable: true });
  const [token, setToken] = useState("dev-token");
  const [history, setHistory] = useState<LocalConversation[]>([]);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [scopeOpen, setScopeOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const scope: ResearchScope = useMemo(() => ({ companies, forms, fiscalYears: years }), [companies, forms, years]);
  const facts: FinancialFact[] = useMemo(() => companies.flatMap(demoFacts), [companies]);

  useEffect(() => { void getQuota().then(setQuota); void listConversations().then(setHistory).catch(() => undefined); }, []);

  const runQuery = async () => {
    const question = query.trim();
    if (!question || streaming || companies.length === 0) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setStreaming(true); setStatus("Searching selected filings…"); setSources([]); setActiveSource(null); setQuery("");
    const prior = messages.slice(-8).map(({ role, content }) => ({ role, content }));
    const next: ThreadMessage[] = [...messages, { role: "user", content: question }, { role: "assistant", content: "", sources: [] }];
    setMessages(next);
    try {
      await streamChat({ query: question, scope, messages: prior, turnstileToken: token }, (event) => {
        if (event.type === "retrieval.status") setStatus("message" in event.data ? event.data.message : `Preparing ${event.data.accession}`);
        if (event.type === "answer.sources") { setSources(event.data.sources); setMessages((current) => current.map((item, index) => index === current.length - 1 ? { ...item, sources: event.data.sources } : item)); }
        if (event.type === "answer.delta") setMessages((current) => current.map((item, index) => index === current.length - 1 ? { ...item, content: item.content + event.data.delta } : item));
        if (event.type === "quota.updated") setQuota(event.data);
        if (event.type === "done") setStatus("Answer grounded in selected SEC filings");
        if (event.type === "error") throw new Error(event.data.message);
      }, controller.signal);
      setMessages((current) => {
        const conversation: LocalConversation = { id: crypto.randomUUID(), title: question.slice(0, 70), updatedAt: new Date().toISOString(), scope, messages: current.map(({ role, content }) => ({ role, content })) };
        void saveConversation(conversation).then(() => listConversations().then(setHistory)).catch(() => undefined);
        return current;
      });
    } catch (error) {
      if ((error as Error).name !== "AbortError") setMessages((current) => current.map((item, index) => index === current.length - 1 ? { ...item, content: `I couldn't complete that request. ${(error as Error).message}` } : item));
      setStatus("Research request ended");
    } finally { setStreaming(false); abortRef.current = null; }
  };

  const addCompany = (company: CompanyRef) => { if (companies.length < 3 && !companies.some((item) => item.cik === company.cik)) setCompanies((items) => [...items, company]); };
  const copyShare = async () => {
    const latestQuestion = [...messages].reverse().find((message) => message.role === "user")?.content ?? query;
    const url = new URL("/research", window.location.origin);
    url.searchParams.set("companies", companies.map((company) => company.ticker).join(",")); url.searchParams.set("forms", forms.join(",")); url.searchParams.set("years", years.join(",")); if (latestQuestion) url.searchParams.set("q", latestQuestion);
    await navigator.clipboard.writeText(url.toString()); setCopied(true); window.setTimeout(() => setCopied(false), 1_500);
  };
  const loadHistory = (conversation: LocalConversation) => { setCompanies(conversation.scope.companies); setForms(conversation.scope.forms); setYears(conversation.scope.fiscalYears); setMessages(conversation.messages); setHistoryOpen(false); };
  const onToken = useCallback((value: string) => setToken(value), []);

  return (
    <main className="research-page">
      <aside className={`scope-sidebar ${scopeOpen ? "open" : ""}`}>
        <div className="sidebar-heading"><span>Research scope</span><button className="icon-button mobile-only" onClick={() => setScopeOpen(false)} aria-label="Close scope"><X size={18} /></button></div>
        <div className="scope-block"><label>Companies <small>{companies.length}/3</small></label>{companies.map((company) => <div className="company-chip" key={company.cik}><span>{company.ticker.slice(0, 2)}</span><div><strong>{company.ticker}</strong><small>{company.name}</small></div><button onClick={() => setCompanies((items) => items.filter((item) => item.cik !== company.cik))} aria-label={`Remove ${company.ticker}`}><X size={14} /></button></div>)}{companies.length < 3 && <CompanySearch onSelect={addCompany} />}</div>
        <div className="scope-block"><label>Forms</label><div className="segmented">{(["10-K", "10-Q"] as const).map((form) => <button className={forms.includes(form) ? "active" : ""} key={form} onClick={() => setForms((items) => items.includes(form) ? (items.length > 1 ? items.filter((item) => item !== form) : items) : [...items, form])}>{forms.includes(form) && <Check size={13} />}{form}</button>)}</div></div>
        <div className="scope-block"><label>Fiscal years <small>Max 5</small></label><div className="year-grid">{[2026, 2025, 2024, 2023, 2022].map((year) => <button key={year} className={years.includes(year) ? "active" : ""} onClick={() => setYears((items) => items.includes(year) ? items.filter((item) => item !== year) : [...items, year].slice(0, 5))}>{year}</button>)}</div></div>
        <div className="scope-summary"><FileSearch size={18} /><div><strong>SEC filings only</strong><p>10-K and 10-Q primary documents with accession-level provenance.</p></div></div>
        <button className="history-button" onClick={() => setHistoryOpen(true)}><History size={16} /> Local history <span>{history.length}</span></button>
      </aside>

      <section className="thread-panel">
        <header className="workspace-header"><button className="icon-button mobile-only" onClick={() => setScopeOpen(true)} aria-label="Open scope"><Menu size={19} /></button><div><p>{companies.map((company) => company.ticker).join(" · ") || "No company selected"}</p><span>{forms.join(" + ")} · {years.length} fiscal years</span></div><div className="workspace-actions"><button className="button ghost" onClick={() => downloadCsv("filing-room-comparison.csv", factsToCsv(facts))}><Download size={15} /> Export</button><button className="button ghost" onClick={copyShare}>{copied ? <Check size={15} /> : <Copy size={15} />} {copied ? "Copied" : "Share"}</button></div></header>
        <div className="thread">
          {messages.length === 0 ? <div className="empty-thread"><div className="source-glyph"><FileSearch /></div><p>Scoped to {companies.map((company) => company.name).join(", ")}</p><h1>What do you want to<br /><em>understand from the filings?</em></h1><p>Ask about changes, risks, strategy, financial performance, or compare up to three issuers.</p><div className="suggestion-row">{["Summarize the biggest risks", "What changed year over year?", "Compare revenue growth"].map((suggestion) => <button key={suggestion} onClick={() => setQuery(suggestion)}>{suggestion}</button>)}</div></div> : messages.map((message, index) => <article key={index} className={`message ${message.role}`}><div className="message-label">{message.role === "user" ? "Your question" : "Filing Room"}</div>{message.role === "assistant" ? <CitationText text={message.content || (streaming ? "Researching…" : "")} sources={message.sources ?? []} onSource={(source) => { setActiveSource(source); setSources(message.sources ?? []); }} /> : <p>{message.content}</p>}</article>)}
        </div>
        <footer className="composer"><div className="retrieval-status"><span className={streaming ? "pulse" : ""} />{status}<small>{quota.remaining} of {quota.limit} answers remaining today</small></div><div className="composer-box"><textarea value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void runQuery(); } }} placeholder="Ask a question about the selected filings…" rows={2} maxLength={2000} /><button className="send-button" onClick={() => void runQuery()} disabled={!query.trim() || streaming || companies.length === 0} aria-label="Run research query">{streaming ? <X onClick={() => abortRef.current?.abort()} /> : <Send />}</button></div><Turnstile theme={theme} onToken={onToken} /><p>Answers may be incomplete. Verify material conclusions against the original SEC filing.</p></footer>
      </section>

      <aside className={`source-panel ${activeSource ? "open" : ""}`}>
        <header><div><p>Source inspector</p><span>{sources.length ? `${sources.length} passages retrieved` : "No source selected"}</span></div><button className="icon-button" onClick={() => setActiveSource(null)} aria-label="Close source panel"><PanelRightClose size={18} /></button></header>
        {activeSource ? <div className="source-content"><div className="source-meta"><span>{activeSource.ticker}</span><strong>{activeSource.form}</strong><small>Filed {activeSource.filingDate}</small></div><p className="source-section">{activeSource.section}</p><blockquote>{activeSource.excerpt}</blockquote><dl><div><dt>Accession</dt><dd>{activeSource.accession}</dd></div><div><dt>Source ID</dt><dd>{activeSource.id}</dd></div></dl><a className="button primary full" href={activeSource.secUrl} target="_blank" rel="noreferrer">Open original filing <ExternalLink size={15} /></a></div> : <div className="source-empty"><FileSearch /><h2>Trace every claim.</h2><p>Select a citation in an answer to inspect the exact supporting excerpt and original SEC document.</p></div>}
      </aside>

      {historyOpen && <div className="drawer-backdrop" onClick={() => setHistoryOpen(false)}><aside className="history-drawer" onClick={(event) => event.stopPropagation()}><header><div><p>Local research history</p><span>Stored only in this browser</span></div><button className="icon-button" onClick={() => setHistoryOpen(false)}><X size={18} /></button></header><div className="history-list">{history.length ? history.map((conversation) => <div key={conversation.id}><button onClick={() => loadHistory(conversation)}><strong>{conversation.title}</strong><span>{conversation.scope.companies.map((company) => company.ticker).join(" · ")} · {new Date(conversation.updatedAt).toLocaleDateString()}</span></button><button className="delete" onClick={() => void deleteConversation(conversation.id).then(() => setHistory((items) => items.filter((item) => item.id !== conversation.id)))} aria-label="Delete conversation"><Trash2 size={15} /></button></div>) : <p className="empty-history">Your completed research threads will appear here.</p>}</div>{history.length > 0 && <button className="button danger" onClick={() => void clearConversations().then(() => setHistory([]))}><Trash2 size={15} /> Clear all history</button>}</aside></div>}
    </main>
  );
}
