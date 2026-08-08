# Extraction and retrieval methodology

## Financial facts

The curated registry in `apps/api/filing_room/sec/facts.py` maps product metrics
to explicit US-GAAP concept candidates. Selection filters by form and expected
duration, prefers the most recently filed valid value (including amendments),
and retains unit, period, fiscal label, accession, and SEC provenance.

Growth and margins are calculated in Python. The model is not asked to select
duplicate periods or perform authoritative arithmetic.

## Filing documents

The production image pins EdgarTools 5.x and initializes its SEC identity, rate,
cache, and XBRL integration boundary. A deterministic, network-free HTML walker
currently emits headings, narrative passages, and tables separately. Every chunk
retains company, accession, form, filing date, fiscal year, section, source offsets,
and the SEC URL. Moving document-tree extraction fully onto the EdgarTools 5 model
is a remaining production-hardening item.

## Retrieval and citations

Research scope is deterministic: one to three companies, form filters, no more
than five fiscal years, and optional accession/section filters. The implemented
retriever applies those filters and deterministic lexical ranking; ingestion also
writes 384-dimensional Workers AI embeddings to Vectorize. Metadata-filtered
Vectorize querying, adjacent-context expansion, and reranker calibration remain
production-hardening items. No out-of-scope passage is sent to the model.

Sources are labeled `[S1]`, `[S2]`, and so on for one request. The renderer only
activates citation IDs present in that request's returned source set. Unsupported
claims are omitted or identified as unavailable.

## Evaluation contract

The target suite includes 20+ filing fixtures and 60 labeled questions across
12 issuers. Release gates are exact fixture calculations, retrieval recall@8 of
at least 90%, resolvable citations for every displayed source, and zero invented
citation IDs. Live-network tests are kept outside normal CI.
