# Portfolio case study

## Problem

SEC filings are authoritative but difficult to navigate across issuers and
years. The original project proved the idea in Streamlit, then moved to a static
browser application. That version improved presentation but introduced two
product-level compromises: browser-held model credentials and third-party CORS
proxies for SEC documents.

## Product decision

Filing Room reframes the chatbot as a research workspace. Company scope,
financial history, filing navigation, comparison, and evidence inspection are
first-class surfaces. Chat is one research tool rather than the entire product.

## Engineering decision

The revamp uses a same-origin Cloudflare gateway and scale-to-zero Python
Container. This preserves EdgarTools/Python extraction while placing quotas,
Turnstile, D1, R2, Workers AI, Vectorize, and static delivery in one platform.
Financial calculations are deterministic; the language model synthesizes only
bounded, attributed passages.

## Design decision

The archival-tab mark combines a filing page with a restrained data line. The
ivory/navy system, Newsreader display type, compact mono metadata, and designed
dark theme aim for an editorial finance product rather than a generic AI chat
surface.

## Measures of success

- Financial fixture calculations are exact.
- Narrative retrieval recall@8 is at least 90% on the labeled evaluation set.
- Every rendered citation resolves; invented citation IDs are impossible to
  activate in the client.
- Cached overview p95 is under one second and warm first-token latency is under
  five seconds under normal provider conditions.
- Owner-funded usage remains within the $5 Cloudflare base and $7 model cap.
