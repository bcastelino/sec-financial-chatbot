import { ArrowRight, BookOpen, Database, Quote, ShieldCheck } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { CompanySearch } from "../components/CompanySearch";
import { SAMPLE_PROMPTS } from "../data";

export function LandingPage() {
  const navigate = useNavigate();
  return (
    <main>
      <section className="hero shell">
        <div className="eyebrow"><span>SEC research workspace</span><span className="live-dot" /> Filing-derived data</div>
        <h1>Read the filing.<br /><em>See the evidence.</em></h1>
        <p className="hero-copy">Research public companies across five fiscal years with answers grounded in 10-K and 10-Q filings, not headlines, market chatter, or opaque citations.</p>
        <CompanySearch large />
        <p className="search-caption">Search any SEC registrant by name or ticker · No account required</p>
      </section>

      <section className="prompt-section shell" aria-labelledby="prompt-heading">
        <div className="section-heading"><p>Start with a question</p><h2 id="prompt-heading">A better way into<br />the primary sources.</h2></div>
        <div className="prompt-grid">
          {SAMPLE_PROMPTS.map((prompt, index) => (
            <button key={prompt} onClick={() => navigate(`/research?companies=${index === 1 ? "AAPL,MSFT,NVDA" : "AAPL"}&forms=10-K,10-Q&years=2024,2023,2022&q=${encodeURIComponent(prompt)}`)}>
              <span>0{index + 1}</span><p>{prompt}</p><ArrowRight size={18} />
            </button>
          ))}
        </div>
      </section>

      <section className="proof-band">
        <div className="shell proof-grid">
          <article><Database /><strong>Five fiscal years</strong><p>Normalized SEC Company Facts, connected to the original accession.</p></article>
          <article><Quote /><strong>Source-level citations</strong><p>Every citation opens the supporting excerpt and SEC filing.</p></article>
          <article><ShieldCheck /><strong>Built for verification</strong><p>Deterministic math, bounded scope, and no invented source IDs.</p></article>
        </div>
      </section>

      <section className="recent shell">
        <div className="section-heading inline"><div><p>Recently filed</p><h2>Fresh from EDGAR</h2></div><Link to="/research">Open research workspace <ArrowRight size={16} /></Link></div>
        <div className="filing-list">
          {[
            { ticker: "NVDA", name: "NVIDIA Corporation", form: "10-Q", description: "Quarterly report", date: "Aug 2026" },
            { ticker: "AAPL", name: "Apple Inc.", form: "10-Q", description: "Quarterly report", date: "Aug 2026" },
            { ticker: "MSFT", name: "Microsoft Corporation", form: "10-K", description: "Annual report", date: "Jul 2026" },
          ].map(({ ticker, name, form, description, date }) => <Link key={ticker} to={`/company/${ticker}`}><span className="ticker-tile">{ticker.slice(0, 2)}</span><span><strong>{name}</strong><small>{ticker} · {description}</small></span><b>{form}</b><time>{date}</time><ArrowRight size={18} /></Link>)}
        </div>
        <p className="data-note"><BookOpen size={15} /> Filing dates shown here are illustrative until the production daily catalog sync is connected.</p>
      </section>

      <section className="closing-cta shell">
        <p>Due diligence starts with the source.</p>
        <h2>Ask the filing.<br />Trace the answer.</h2>
        <Link className="button primary" to="/research">Begin research <ArrowRight size={17} /></Link>
      </section>
    </main>
  );
}
