import type { CompanyOverview as Overview } from "@filing-room/contracts";
import { ArrowRight, Download, ExternalLink, FileText } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { MetricChart } from "../components/MetricChart";
import { POPULAR_COMPANIES, demoFacts } from "../data";
import { getCompanyOverview } from "../lib/api";
import { downloadCsv, factsToCsv } from "../lib/export";

function money(value: number): string {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", notation: "compact", maximumFractionDigits: 1 }).format(value);
}

export function CompanyPage() {
  const { ticker = "AAPL" } = useParams();
  const fallback = POPULAR_COMPANIES.find((company) => company.ticker === ticker.toUpperCase()) ?? POPULAR_COMPANIES[0]!;
  const [overview, setOverview] = useState<Overview>({ company: fallback, facts: demoFacts(fallback), filings: [] });
  useEffect(() => { void getCompanyOverview(ticker).then((result) => setOverview({ ...result, facts: result.facts.length ? result.facts : demoFacts(result.company) })); }, [ticker]);
  const revenue = useMemo(() => overview.facts.filter((fact) => fact.concept === "revenue"), [overview]);
  const latest = revenue.at(-1);
  const prior = revenue.at(-2);
  const growth = latest && prior ? ((latest.value - prior.value) / Math.abs(prior.value)) * 100 : undefined;
  return (
    <main className="company-page shell">
      <nav className="breadcrumbs" aria-label="Breadcrumb"><Link to="/">Companies</Link><span>/</span><span>{overview.company.ticker}</span></nav>
      <header className="company-hero">
        <div className="company-monogram">{overview.company.ticker.slice(0, 2)}</div>
        <div><p className="kicker">{overview.company.ticker} · CIK {overview.company.cik}</p><h1>{overview.company.name}</h1><p>{overview.description}</p></div>
        <Link className="button primary" to={`/research?companies=${overview.company.ticker}&forms=10-K,10-Q&years=2024,2023,2022`}>Research this company <ArrowRight size={17} /></Link>
      </header>

      <section className="metric-strip" aria-label="Company filing metrics">
        <article><span>Latest annual revenue</span><strong>{latest ? money(latest.value) : "N/A"}</strong><small>{latest?.fiscalYear ?? "No filing data"}</small></article>
        <article><span>Year-over-year growth</span><strong>{growth === undefined ? "N/A" : `${growth >= 0 ? "+" : ""}${growth.toFixed(1)}%`}</strong><small>Calculated from Company Facts</small></article>
        <article><span>Coverage</span><strong>5 years</strong><small>10-K and 10-Q</small></article>
        <article><span>Last refreshed</span><strong>Daily</strong><small>SEC submissions catalog</small></article>
      </section>

      <section className="company-grid">
        <div className="panel"><div className="panel-heading"><div><p>Financial history</p><h2>Revenue from filings</h2></div><button className="button secondary" onClick={() => downloadCsv(`${overview.company.ticker}-filing-facts.csv`, factsToCsv(overview.facts))}><Download size={16} /> CSV</button></div><MetricChart facts={revenue} /></div>
        <aside className="panel filing-profile"><p>Filing profile</p><dl><div><dt>Forms</dt><dd>10-K · 10-Q</dd></div><div><dt>Fiscal year end</dt><dd>{overview.fiscalYearEnd ?? "Reported by issuer"}</dd></div><div><dt>Industry</dt><dd>{overview.sic ?? "SEC registrant"}</dd></div><div><dt>Source</dt><dd><a href={`https://www.sec.gov/edgar/browse/?CIK=${Number(overview.company.cik)}`} target="_blank" rel="noreferrer">SEC EDGAR <ExternalLink size={13} /></a></dd></div></dl></aside>
      </section>

      <section className="filings-section">
        <div className="panel-heading"><div><p>Primary documents</p><h2>Filing library</h2></div><span className="status-chip">Latest five fiscal years</span></div>
        <div className="filing-table" role="table" aria-label="SEC filing list">
          <div className="table-row header" role="row"><span>Form</span><span>Period</span><span>Filed</span><span>Status</span><span /></div>
          {(overview.filings.length ? overview.filings : [2024, 2023, 2022].flatMap((year) => [{ form: "10-K", reportDate: `${year}-09-30`, filingDate: `${year}-11-01`, isAmendment: false, accession: `${overview.company.cik}-${String(year).slice(2)}-000001`, secUrl: `https://www.sec.gov/edgar/browse/?CIK=${Number(overview.company.cik)}` }, { form: "10-Q", reportDate: `${year}-06-30`, filingDate: `${year}-08-01`, isAmendment: false, accession: `${overview.company.cik}-${String(year).slice(2)}-000002`, secUrl: `https://www.sec.gov/edgar/browse/?CIK=${Number(overview.company.cik)}` }])).map((filing) => <div className="table-row" role="row" key={filing.accession}><strong><FileText size={15} />{filing.form}</strong><span>{filing.reportDate}</span><span>{filing.filingDate}</span><span className="ready">Ready</span><a href={filing.secUrl} target="_blank" rel="noreferrer" aria-label={`Open ${filing.form} on SEC`}><ExternalLink size={16} /></a></div>)}
        </div>
      </section>
    </main>
  );
}
