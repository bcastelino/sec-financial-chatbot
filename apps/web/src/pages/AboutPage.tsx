import { ExternalLink } from "lucide-react";

export function AboutPage() {
  return (
    <main className="about-page shell">
      <header><p className="eyebrow">Methodology & limitations</p><h1>Research software should<br /><em>show its work.</em></h1><p>Filing Room is a portfolio project built by Brian Castelino to make SEC disclosures easier to inspect, compare, and verify.</p></header>
      <div className="method-grid">
        <nav aria-label="Methodology sections"><a href="#sources">01 · Sources</a><a href="#extraction">02 · Extraction</a><a href="#retrieval">03 · Retrieval</a><a href="#privacy">04 · Privacy</a><a href="#limits">05 · Limitations</a></nav>
        <div className="method-copy">
          <section id="sources"><span>01</span><h2>SEC-only source boundary</h2><p>Filing Room reads public submissions, Company Facts, and primary filing documents from SEC EDGAR. It does not blend filings with prices, news, analyst notes, or general web results.</p><a href="https://www.sec.gov/search-filings/edgar-application-programming-interfaces" target="_blank" rel="noreferrer">SEC EDGAR APIs <ExternalLink size={14} /></a></section>
          <section id="extraction"><span>02</span><h2>Structured extraction</h2><p>Normalized metrics come from inline XBRL and a curated concept registry. Filing HTML is parsed into headings, paragraphs, sections, and standalone tables. Every stored passage retains accession and source provenance.</p></section>
          <section id="retrieval"><span>03</span><h2>Grounded retrieval</h2><p>Financial arithmetic runs deterministically in the backend. Narrative research uses company, form, year, and accession filters before semantic retrieval. The model receives only supported passages labeled with server-issued source IDs.</p></section>
          <section id="privacy"><span>04</span><h2>Anonymous by design</h2><p>Up to 20 conversations stay in your browser’s IndexedDB and can be deleted at any time. Shared URLs include only scope and question, not answers or history. Server logs exclude full prompts, answers, credentials, and raw IP addresses.</p></section>
          <section id="limits"><span>05</span><h2>Important limitations</h2><p>Automated extraction and generative summaries can be incomplete. Filing Room is not affiliated with the SEC and does not provide investment advice. Verify material conclusions against the original filing.</p></section>
        </div>
      </div>
    </main>
  );
}
