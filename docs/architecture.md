# Architecture

## Request path

The React application is deployed with Cloudflare Workers Static Assets. Static
files bypass Worker execution; `/api/*` invokes the gateway first.

The gateway performs origin and size checks, verifies Turnstile on cost-bearing
routes, derives a rotating daily HMAC of the connecting IP, consumes the D1
quota atomically, and proxies the request to one named scale-to-zero API
Container. Provider credentials are passed into the Container as environment
variables and never serialized to the browser.

## Storage ownership

| Store | Data | Retention/behavior |
| --- | --- | --- |
| D1 | issuers, filings, ingestion jobs, sync state, daily quota counts | Mutable catalog metadata |
| R2 | immutable raw HTML and gzip parsed-document JSON | Accession-addressed objects |
| Vectorize | 384-dimension embeddings and small filter metadata | No filing text |
| IndexedDB | up to 20 local research threads | User-clearable; browser only |

The Container accesses R2, Workers AI, and Vectorize through Cloudflare
Container outbound handlers. Local development uses a `.data` directory and a
deterministic lexical retriever, so tests require no Cloudflare account.

## Trust boundaries

- Only `data.sec.gov`, `www.sec.gov/files`, and derived SEC Archives URLs are
  accepted by the SEC client.
- The backend identifies itself and rate limits all SEC calls to at most five
  requests per second, below the SEC fair-access ceiling.
- OpenRouter receives one bounded call per answer and only the retrieved source
  set plus the last eight local messages.
- Markdown is sanitized, citation IDs are matched against the current source
  set, and the Worker adds a restrictive CSP.
- Logs include request ID, timing, counts, stages, scores, and estimated cost;
  they exclude raw IPs, full prompts, answers, and credentials.

## Cold ingestion sequence

1. Resolve the selected issuer from the SEC ticker/CIK catalog.
2. Fetch Submissions and Company Facts concurrently through the shared limiter.
3. Select requested 10-K/10-Q filings, retaining amendments and provenance.
4. Download the known primary document and write immutable raw HTML to R2.
5. Parse sections and standalone tables; chunk narrative at ~400 words with a
   50-word overlap and store compressed parsed JSON.
6. Embed with `bge-small-en-v1.5` and upsert 384-dimensional vectors.
7. Stream each ingestion stage to the browser and continue the pending query.

## Cost envelope

The intended prototype envelope is the $5 Workers Paid base plus an OpenRouter
key capped at $7/month. The daily per-visitor limit and global provider key
guardrails are independent controls. Cached browsing remains available if the
AI budget is exhausted.
