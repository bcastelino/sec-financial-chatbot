import type { CompanyRef } from "@filing-room/contracts";
import { Search } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { searchCompanies } from "../lib/api";

export function CompanySearch({ large = false, onSelect }: { large?: boolean; onSelect?: (company: CompanyRef) => void }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<CompanyRef[]>([]);
  const navigate = useNavigate();
  useEffect(() => {
    if (query.trim().length < 1) return setResults([]);
    const timer = window.setTimeout(() => void searchCompanies(query).then(setResults), 150);
    return () => window.clearTimeout(timer);
  }, [query]);
  const choose = (company: CompanyRef) => {
    setQuery("");
    setResults([]);
    if (onSelect) onSelect(company);
    else navigate(`/company/${company.ticker}`);
  };
  return (
    <div className={`company-search ${large ? "large" : ""}`}>
      <Search aria-hidden="true" size={20} />
      <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search any public company or ticker" aria-label="Search companies" autoComplete="off" />
      {results.length > 0 && (
        <div className="search-results" role="listbox">
          {results.map((company) => <button key={company.cik} onClick={() => choose(company)} role="option"><strong>{company.ticker}</strong><span>{company.name}</span><small>CIK {company.cik}</small></button>)}
        </div>
      )}
    </div>
  );
}
