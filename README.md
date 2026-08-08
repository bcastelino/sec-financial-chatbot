# Filing Room

**Ask the filing. Trace the answer.**

Filing Room is an open-source SEC research workspace for exploring public-company
10-K and 10-Q filings. It combines deterministic financial calculations,
filing-scoped retrieval, streamed answers, and source-level citations in an
editorial React interface.

![Filing Room social preview](apps/web/public/social-preview.svg)

[![CI](https://github.com/bcastelino/sec-financial-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/bcastelino/sec-financial-chatbot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-102338.svg)](LICENSE)

> Filing Room is not affiliated with the U.S. Securities and Exchange
> Commission. It is not investment advice. Verify material conclusions against
> the original filing.

## Why Filing Room

SEC filings are authoritative, but researching several issuers and fiscal years
usually means moving between long documents, spreadsheets, and search tools.
Filing Room treats chat as one part of a broader research workspace:

- Begin with an issuer, filing form, and fiscal-year scope.
- Use normalized SEC Company Facts for financial values and calculations.
- Search narrative disclosures without sending whole filings to a model.
- Inspect the exact excerpt and original SEC document behind each citation.
- Keep anonymous research history in the browser instead of on the server.

The prior Streamlit implementation remains available on the
[`streamlit-legacy`](https://github.com/bcastelino/sec-financial-chatbot/tree/streamlit-legacy)
branch. Filing Room replaces the former browser-only React version, including
its client-side API keys and third-party SEC proxy dependency.

## Current implementation

### Working locally

- Responsive landing, company, research, and methodology pages
- Original Filing Room SVG identity, favicon, social preview, and dark theme
- Company, form, and fiscal-year research scope for up to three issuers
- Server-Sent Events for retrieval progress and streamed answers
- Citation validation, exact excerpts, and direct SEC filing links
- SEC Submissions and Company Facts ingestion through an identified backend
- Deterministic XBRL fact selection, amendment handling, growth, and margins
- Structured heading, narrative, and standalone-table chunking
- Server-controlled OpenRouter generation with an offline extractive fallback
- CSV export, scope-only share URLs, and up to 20 IndexedDB conversations
- Cloudflare Worker bindings for D1, R2, Vectorize, Workers AI, and Containers
- Turnstile, origin checks, request limits, CSP, and daily quota implementation

### Production hardening still required

- Connect metadata-filtered Vectorize querying and reranker calibration
- Move document-tree extraction fully onto the EdgarTools 5 document model
- Replace the current comparison export prototype with a complete comparison table
- Expand golden fixtures to 20 or more filings and the evaluation set to 60 questions
- Provision Cloudflare resources, prewarm the issuer cache, and attach `sec.bcastelino.com`

The Cloudflare deployment has not been performed from this repository. Current
status and limitations are documented in
[the methodology](docs/methodology.md) and [deployment guide](docs/deployment.md).

## Architecture

```text
Browser
  |
  +-- React application and static assets
  |     +-- IndexedDB conversation history
  |     +-- sanitized Markdown and citation renderer
  |
  +-- Cloudflare Worker gateway
        +-- Turnstile, origin policy, CSP, and request limits
        +-- D1 company catalog, ingestion state, and quota counters
        +-- R2 raw filings and compressed parsed documents
        +-- Workers AI embeddings and Vectorize index
        |
        +-- scale-to-zero Python Container
              +-- FastAPI and Server-Sent Events
              +-- SEC Submissions, Company Facts, and filing HTML
              +-- structured extraction and deterministic calculations
              +-- one bounded, server-controlled OpenRouter call
```

The SEC client accepts only known SEC hosts and server-derived filing URLs. It
uses an identified `User-Agent`, retries temporary failures, and remains at or
below five SEC requests per second.

## Repository layout

```text
apps/web/             React, TypeScript, Vite, Vitest, and Playwright tooling
apps/api/             FastAPI, SEC ingestion, extraction, retrieval, and tests
packages/contracts/   Shared public TypeScript contracts
workers/gateway/      Cloudflare gateway, security, quota, and Container bindings
infra/d1/             D1 database migrations
infra/prewarm/        Popular-100 issuer seed list
docs/                 Architecture, methodology, deployment, and case study
scripts/              Repository-level validation scripts
```

## Local preview

### Prerequisites

- Node.js 22 or newer
- Corepack
- Python 3.12 or 3.13
- [uv](https://docs.astral.sh/uv/)

### Interface-only preview

The frontend can run without the Python API. Company search and overview pages
use bundled demonstration data when the API is unavailable. Generated chat is
not available in this mode.

```bash
corepack pnpm install --frozen-lockfile
corepack pnpm --filter @filing-room/web dev
```

Open `http://localhost:5173`.

### Full local stack

Install both dependency sets from the repository root:

```bash
corepack pnpm install --frozen-lockfile
cd apps/api
uv sync --frozen
```

Start the API in the first terminal:

```bash
cd apps/api
uv run uvicorn filing_room.main:app --reload --port 8000
```

Start the web application in a second terminal from the repository root:

```bash
corepack pnpm --filter @filing-room/web dev
```

Vite proxies `/api/*` to `http://localhost:8000`. Without an OpenRouter key,
the API returns a clearly labeled extractive answer so retrieval, streaming,
citations, and source inspection can still be reviewed.

### Optional local model configuration

Create `apps/api/.env` and keep it untracked:

```env
OPENROUTER_API_KEY=your-server-side-key
OPENROUTER_MODEL=deepseek/deepseek-v4-flash-0731
SEC_IDENTITY=Filing Room your-monitored-email@example.com
```

Never expose provider credentials through a `VITE_` variable. The supplied
`.env.example` contains the complete configuration reference.

## API surface

| Method | Path                                  | Purpose                |
| ------ | ------------------------------------- | ---------------------- |
| `GET`  | `/api/v1/health`                      | Liveness               |
| `GET`  | `/api/v1/ready`                       | Readiness              |
| `GET`  | `/api/v1/companies/search`            | Issuer search          |
| `GET`  | `/api/v1/companies/{ticker}/overview` | Facts and filings      |
| `GET`  | `/api/v1/companies/{ticker}/filings`  | Filing catalog         |
| `GET`  | `/api/v1/sources/{source_id}`         | Supporting excerpt     |
| `GET`  | `/api/v1/quota`                       | Anonymous answer quota |
| `POST` | `/api/v1/chat/stream`                 | Scoped research stream |

Chat streams `retrieval.status`, `answer.delta`, `answer.sources`,
`quota.updated`, `done`, and `error` events.

## Quality checks

Frontend, contracts, and gateway:

```bash
corepack pnpm lint
corepack pnpm typecheck
corepack pnpm test
corepack pnpm build
```

Python API:

```bash
cd apps/api
uv run ruff check .
uv run mypy filing_room
uv run pytest
```

`pnpm lint` also rejects em dash characters in repository-owned files. CI uses
mocked or fixture-backed tests and does not require live SEC, OpenRouter, or
Cloudflare access.

## Security and privacy

- OpenRouter, Cloudflare, and SEC identity credentials remain server-side.
- Arbitrary outbound SEC URLs are rejected.
- Filing contents are treated as untrusted prompt data.
- Markdown is sanitized and unknown citation IDs are not activated.
- Shared URLs contain scope and a question, not answers or history.
- Browser history is limited to 20 conversations and can be cleared locally.
- Production logs are designed to omit full prompts, answers, raw IPs, and secrets.

Please report vulnerabilities through [SECURITY.md](SECURITY.md), not a public issue.

## Deployment

The target is Cloudflare Workers Static Assets with a Worker gateway and a
scale-to-zero FastAPI Container. D1, R2, Workers AI, and Vectorize provide the
catalog, object storage, embeddings, and index. OpenRouter access is controlled
by a server-side key with a planned $7 monthly limit.

Production setup requires the repository owner's Cloudflare and OpenRouter
accounts, Docker, secret configuration, D1 migrations, staging validation, and
the `sec.bcastelino.com` custom domain. Follow [docs/deployment.md](docs/deployment.md).

## Documentation

- [Architecture](docs/architecture.md)
- [Extraction and retrieval methodology](docs/methodology.md)
- [Cloudflare deployment](docs/deployment.md)
- [Production readiness checklist](docs/production-readiness-checklist.md)
- [Portfolio case study](docs/case-study.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

## License

Released under the [MIT License](LICENSE). Copyright 2026 Brian Castelino.
