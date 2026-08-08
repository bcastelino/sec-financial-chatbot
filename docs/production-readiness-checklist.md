# Filing Room production readiness checklist

This is the release runbook for moving Filing Room from the current portfolio
prototype to a public service at `sec.bcastelino.com`.

Last reviewed: August 8, 2026.

The checklist is intentionally stricter than a basic deployment guide. A checked
item means the task is implemented, tested, documented, and supported by saved
evidence. Do not mark an item complete because a configuration file merely exists.

## How to use this checklist

- Work from top to bottom. Later stages assume all earlier release gates pass.
- Treat every `P0` item as a launch blocker.
- Save evidence in the linked issue, pull request, deployment record, or release
  notes. Evidence can be test output, a dashboard screenshot, a version ID, a D1
  bookmark, or a documented manual result.
- Use separate staging and production resources. Never test migrations, secret
  rotation, quota bypasses, or destructive recovery against production first.
- Keep credentials out of Git, shell history, screenshots, issue comments, and
  Actions logs.
- Recheck linked provider documentation immediately before provisioning because
  platform limits and prices can change.

## Current launch blockers

The following items were confirmed by a source audit on August 8, 2026.

| Priority | Blocker                                                | Current evidence                                                       | Required outcome                                                       |
| -------- | ------------------------------------------------------ | ---------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| P0       | Production mode is not configured                      | `workers/gateway/wrangler.jsonc` sets `ENVIRONMENT` to `development`   | Explicit staging and production environments set non-development modes |
| P0       | Internal refresh can be induced publicly               | The Worker proxies every `/api/*` path and adds the gateway secret     | Public route allowlist, internal route denylist, and tests             |
| P0       | Quota consumption is not atomic                        | Quota is read and then incremented in separate D1 calls                | One atomic reserve operation under concurrency                         |
| P0       | Global AI budget cutoff is not implemented             | `budgetAvailable` always returns `true`                                | Monthly usage ledger and fail-closed cutoff                            |
| P0       | Turnstile validation is incomplete                     | Only `success` is checked                                              | Validate expected hostname and action, handle expiry and replay        |
| P0       | Production retrieval is not durable                    | Answers use in-memory lexical chunks; Vectorize only supports indexing | Vector query, R2 rehydration, reranking, and restart tests             |
| P0       | Internal ingestion coordination is process-local       | Company locks are stored in the Python process                         | Durable D1 job leases with idempotency                                 |
| P0       | Comparison export uses demonstration facts             | `ResearchPage.tsx` calls `demoFacts`                                   | Live scoped facts and a real comparison table                          |
| P0       | Production evaluation gates are incomplete             | Small unit fixture set and no committed Playwright flows               | Required parser, retrieval, citation, accessibility, and E2E suites    |
| P1       | Full EdgarTools document-tree extraction is incomplete | The parser uses a deterministic HTML walker                            | EdgarTools 5 document model with regression fixtures                   |
| P1       | Operational telemetry is incomplete                    | Observability is enabled, but structured product events are sparse     | Privacy-safe logs, dashboards, alerts, and incident runbooks           |
| P1       | Deployment automation is incomplete                    | CI validates code but does not deploy staging or production            | Protected staged deployment workflow with rollback evidence            |

## Gate 1: ownership, accounts, and spending controls

### 1.1 Confirm ownership and contacts

- [ ] Record the production owner: Brian Castelino.
- [ ] Record a monitored technical email for the SEC `User-Agent`.
  - Do not use a placeholder or an inbox that is not checked.
  - Set `SEC_IDENTITY` to a descriptive application name plus the monitored email.
  - Example: `Filing Room sec-operations@bcastelino.com`.
- [ ] Record a backup contact who can rotate credentials and roll back a deployment.
- [ ] Confirm that `bcastelino.com` is an active zone in the Cloudflare account that
      will own the Worker.
- [ ] Create a private inventory containing account owners, billing contacts,
      resource names, secret rotation dates, and incident contacts.
- [ ] Confirm the MIT license and public repository status are intentional.

Verification:

- The monitored address receives a test message.
- Two authorized people, or the owner plus a documented recovery process, can
  access Cloudflare, OpenRouter, and GitHub.

### 1.2 Protect GitHub `main`

- [ ] Create a repository ruleset or branch protection rule for `main`.
- [ ] Require pull requests for future changes after this bootstrap release.
- [ ] Require the `web`, `api`, and `secret-scan` CI jobs.
- [ ] Block force pushes and branch deletion.
- [ ] Require branches to be current before merge.
- [ ] Enable the dependency graph, Dependabot alerts, and Dependabot security
      updates.
- [ ] Add `.github/dependabot.yml` for the pnpm, Python, GitHub Actions, and Docker
      ecosystems.
- [ ] Enable secret scanning and push protection where the repository plan permits.
- [ ] Enable private vulnerability reporting and verify `SECURITY.md` points to it.
- [ ] Review Actions permissions and keep the default `GITHUB_TOKEN` read-only
      unless a job has a specific write requirement.
- [ ] Pin third-party Actions to reviewed commit SHAs for production deployment
      workflows.

Verification:

- A test pull request cannot merge with a failing required check.
- A test branch cannot force push to `main`.
- GitHub Security shows the dependency graph and secret scanning status.

Reference: [GitHub protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches),
[GitHub security features](https://docs.github.com/en/code-security/getting-started/github-security-features).

### 1.3 Establish Cloudflare spending boundaries

- [ ] Confirm the current Workers Paid plan price and included usage before purchase.
- [ ] Upgrade the intended Cloudflare account to Workers Paid.
- [ ] Record the billing renewal date and payment method owner.
- [ ] Configure available billing notifications for Workers, Containers, D1, R2,
      Vectorize, Workers AI, and logs.
- [ ] Set an internal warning threshold below the maximum acceptable monthly spend.
- [ ] Choose an operational response for each threshold:
  - Warning: investigate unusual traffic or ingestion volume.
  - High: disable cold ingestion and reduce logging or prewarming.
  - Critical: disable generated answers while keeping cached browsing available.
- [ ] Document that Cloudflare usage can exceed the $5 base and identify which
      services can create overages.
- [ ] Review usage at least weekly during the first month.

Current pricing reference: [Cloudflare Workers pricing](https://developers.cloudflare.com/workers/platform/pricing/),
[Workers AI pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/).

### 1.4 Create OpenRouter production guardrails

- [ ] Create a dedicated Filing Room workspace or isolated production key.
- [ ] Verify the configured model exists in the current OpenRouter model catalog.
- [ ] Decide whether provider fallback for the same model is acceptable.
- [ ] Create a guardrail assigned directly to the production API key.
- [ ] Restrict the model allowlist to the single approved model.
- [ ] Set the key or guardrail budget to `$7` with a monthly reset.
- [ ] Consider setting the application cutoff below `$7`, such as `$6.50`, to
      preserve headroom for delayed usage reporting.
- [ ] Enable Zero Data Retention only after confirming the selected model has an
      eligible endpoint and the behavior matches the privacy statement.
- [ ] Review provider data policies and disable endpoints that conflict with the
      product privacy statement.
- [ ] Disable the key when the application is not ready for public traffic.
- [ ] Record key creation and planned rotation dates without recording the key.
- [ ] Test that an unapproved model request is rejected.
- [ ] Test that the budget guardrail rejects requests after a small staging limit is
      reached.

References: [OpenRouter guardrails](https://openrouter.ai/docs/guides/features/guardrails/overview),
[OpenRouter Zero Data Retention](https://openrouter.ai/docs/guides/features/zdr),
[OpenRouter model catalog](https://openrouter.ai/api/v1/models).

### 1.5 Confirm SEC access policy

- [ ] Keep the backend rate limit at five requests per second or lower.
- [ ] Use one shared limiter across submissions, facts, filing HTML, retries, and
      prewarming within each running service.
- [ ] Account for multiple Container instances so aggregate SEC traffic cannot exceed
      the intended application limit.
- [ ] Add jittered exponential backoff for `429`, transient `403`, and `5xx` results.
- [ ] Honor `Retry-After` when present.
- [ ] Cache immutable filing documents by accession and avoid repeat downloads.
- [ ] Do not crawl search result pages or arbitrary SEC URLs.
- [ ] Log response status, request class, latency, and request ID without logging
      credentials or full documents.
- [ ] Add an operator switch that stops SEC ingestion without disabling cached data.

The SEC currently asks automated clients to identify themselves and remain below
10 requests per second. Filing Room intentionally targets five or fewer. Reference:
[SEC developer resources](https://www.sec.gov/about/developer-resources).

## Gate 2: secrets and environment isolation

### 2.1 Rotate and inventory credentials

- [ ] Review Git history, GitHub secret scanning, local reflogs, issue attachments,
      and deployment logs for provider credentials.
- [ ] Rotate any key whose history is uncertain.
- [ ] Create separate Turnstile, Worker gateway, visitor hash, and OpenRouter secrets
      for staging and production.
- [ ] Generate `API_SHARED_SECRET` and `VISITOR_HASH_SECRET` from at least 32 random
      bytes.
- [ ] Never reuse the visitor hash secret as an API or provider key.
- [ ] Store production secrets with `wrangler secret put`, not in `wrangler.jsonc`,
      `.dev.vars`, `.env`, or GitHub issue text.
- [ ] Keep local test keys in ignored files only.
- [ ] Document a 90-day rotation review and immediate incident rotation procedure.
- [ ] Confirm the browser bundle contains no secret values or secret variable names
      with populated values.

Example secret commands, run separately for each Wrangler environment:

```bash
cd workers/gateway
corepack pnpm exec wrangler secret put API_SHARED_SECRET --env staging
corepack pnpm exec wrangler secret put VISITOR_HASH_SECRET --env staging
corepack pnpm exec wrangler secret put TURNSTILE_SECRET_KEY --env staging
corepack pnpm exec wrangler secret put OPENROUTER_API_KEY --env staging

corepack pnpm exec wrangler secret put API_SHARED_SECRET --env production
corepack pnpm exec wrangler secret put VISITOR_HASH_SECRET --env production
corepack pnpm exec wrangler secret put TURNSTILE_SECRET_KEY --env production
corepack pnpm exec wrangler secret put OPENROUTER_API_KEY --env production
```

Verification:

- `git grep` finds no real secret.
- A production bundle scan finds no OpenRouter key pattern.
- A staging secret cannot authenticate to production.

### 2.2 Add explicit Wrangler environments

- [ ] Replace the single development configuration with explicit `staging` and
      `production` environments in `workers/gateway/wrangler.jsonc`.
- [ ] Set `ENVIRONMENT=staging` for staging and `ENVIRONMENT=production` for
      production.
- [ ] Never deploy a public environment with `ENVIRONMENT=development`.
- [ ] Use distinct Worker names, D1 databases, R2 buckets, Vectorize indexes,
      Durable Object namespaces, and Turnstile widgets.
- [ ] Use `sec-staging.bcastelino.com` for staging.
- [ ] Use `sec.bcastelino.com` only for the tested production deployment.
- [ ] Set production `SEC_IDENTITY` to the monitored contact.
- [ ] Keep `DAILY_ANSWER_LIMIT=5` in production.
- [ ] Set production API CORS origins to only `https://sec.bcastelino.com`.
- [ ] Remove localhost from production Turnstile hostname configuration.
- [ ] Confirm non-inheritable Wrangler bindings are declared in every environment.

Required negative test:

- A request containing `dev-token` must fail in staging and production.

### 2.3 Separate browser-visible configuration

- [ ] Create environment-specific frontend build variables.
- [ ] Use the staging Turnstile site key only in the staging build.
- [ ] Use the production Turnstile site key only in the production build.
- [ ] Keep `VITE_API_BASE_URL=/api/v1` for same-origin deployment.
- [ ] Add a CI bundle scan that fails if `OPENROUTER_API_KEY`,
      `TURNSTILE_SECRET_KEY`, `API_SHARED_SECRET`, or `VISITOR_HASH_SECRET` values appear
      in `apps/web/dist`.
- [ ] Document which variables are public and which are secrets.

## Gate 3: provision isolated Cloudflare resources

Do this only after environment configuration exists in the repository.

### 3.1 Authenticate and verify prerequisites

- [ ] Install and start Docker.
- [ ] Run `docker info` successfully.
- [ ] Install repository dependencies from the lockfile.
- [ ] Authenticate Wrangler to the correct Cloudflare account.
- [ ] Confirm the active account ID and zone before creating resources.

```bash
corepack pnpm install --frozen-lockfile
cd workers/gateway
corepack pnpm exec wrangler whoami
```

Cloudflare Containers require Docker for the normal Wrangler build and deploy
flow. Reference: [Cloudflare Containers getting started](https://developers.cloudflare.com/containers/get-started/).

### 3.2 Create staging resources

- [ ] Create the staging D1 database.
- [ ] Create the staging R2 bucket.
- [ ] Create the staging 384-dimensional cosine Vectorize index.
- [ ] Copy the returned D1 ID into only the staging binding.
- [ ] Confirm Workers AI is bound in staging.

```bash
cd workers/gateway
corepack pnpm exec wrangler d1 create filing-room-staging
corepack pnpm exec wrangler r2 bucket create filing-room-filings-staging
corepack pnpm exec wrangler vectorize create filing-room-chunks-staging --dimensions=384 --metric=cosine
```

### 3.3 Create production resources

- [ ] Repeat resource creation with production names.
- [ ] Copy the returned production D1 ID into only the production binding.
- [ ] Confirm no staging binding points to a production resource and vice versa.

```bash
cd workers/gateway
corepack pnpm exec wrangler d1 create filing-room-production
corepack pnpm exec wrangler r2 bucket create filing-room-filings-production
corepack pnpm exec wrangler vectorize create filing-room-chunks-production --dimensions=384 --metric=cosine
```

The Vectorize dimension cannot be changed after index creation. Confirm that
`@cf/baai/bge-small-en-v1.5` returns 384 dimensions before creating the production
index. Reference: [creating Vectorize indexes](https://developers.cloudflare.com/vectorize/best-practices/create-indexes/).

### 3.4 Create Vectorize metadata indexes before ingestion

- [ ] Create string metadata indexes for `ticker`, `accession`, `form`, and
      `section`.
- [ ] Create a number metadata index for `fiscal_year`.
- [ ] Decide whether `is_table` needs a boolean metadata index.
- [ ] Create metadata indexes before prewarming. Existing vectors must be upserted
      again if the metadata indexes are created later.
- [ ] Confirm all filters are represented exactly as the ingestion code writes them.

Example for staging:

```bash
cd workers/gateway
corepack pnpm exec wrangler vectorize create-metadata-index filing-room-chunks-staging --property-name=ticker --type=string
corepack pnpm exec wrangler vectorize create-metadata-index filing-room-chunks-staging --property-name=accession --type=string
corepack pnpm exec wrangler vectorize create-metadata-index filing-room-chunks-staging --property-name=form --type=string
corepack pnpm exec wrangler vectorize create-metadata-index filing-room-chunks-staging --property-name=fiscal_year --type=number
corepack pnpm exec wrangler vectorize create-metadata-index filing-room-chunks-staging --property-name=section --type=string
```

Repeat for production after staging queries pass. Reference:
[Vectorize metadata filtering](https://developers.cloudflare.com/vectorize/reference/metadata-filtering/).

### 3.5 Apply and verify D1 migrations

- [ ] Review every migration for forward and backward compatibility.
- [ ] Apply migrations to staging first.
- [ ] Inspect the D1 migrations table and application tables.
- [ ] Run API integration tests against staging.
- [ ] Record the current production D1 Time Travel bookmark before a production
      migration.
- [ ] Apply production migrations only after staging evidence is approved.

```bash
cd workers/gateway
corepack pnpm exec wrangler d1 migrations list filing-room-staging --remote --env staging
corepack pnpm exec wrangler d1 migrations apply filing-room-staging --remote --env staging

corepack pnpm exec wrangler d1 time-travel info filing-room-production
corepack pnpm exec wrangler d1 migrations list filing-room-production --remote --env production
corepack pnpm exec wrangler d1 migrations apply filing-room-production --remote --env production
```

Do not run a Time Travel restore during normal deployment. Restore overwrites the
database and belongs only in the incident runbook. References:
[D1 migrations](https://developers.cloudflare.com/d1/reference/migrations/),
[D1 Time Travel](https://developers.cloudflare.com/d1/reference/time-travel/).

### 3.6 Decide R2 retention behavior

- [ ] Keep the R2 bucket private.
- [ ] Confirm only the Worker binding can access filing objects in normal operation.
- [ ] Decide whether raw SEC filings and parsed documents are retained indefinitely
      or have lifecycle rules.
- [ ] If immutability is a release requirement, evaluate bucket lock rules on a
      staging bucket before production.
- [ ] Document how corrected or reprocessed parsed documents receive new object keys.
- [ ] Test that arbitrary user-provided keys and path traversal are rejected.
- [ ] Record R2 object counts and storage size after prewarming.

Reference: [R2 bucket locks](https://developers.cloudflare.com/r2/buckets/bucket-locks/),
[R2 data security](https://developers.cloudflare.com/r2/reference/data-security/).

## Gate 4: close P0 application security gaps

### 4.1 Replace generic API proxying with a public route allowlist

- [ ] Define an explicit Worker allowlist for public API routes and methods.
- [ ] Permit only intended public reads for company search, overview, filings,
      source retrieval, health, readiness, and quota.
- [ ] Permit `POST /api/v1/chat/stream` only through Turnstile, quota, size, and
      validation checks.
- [ ] Deny `/api/v1/catalog/refresh` from public requests.
- [ ] Deny future ingestion administration, OpenAPI, and debug endpoints unless an
      authenticated internal path explicitly permits them.
- [ ] Do not automatically add `X-Filing-Room-Secret` to arbitrary public paths.
- [ ] Prefer direct Worker-to-Container calls for scheduled internal work.
- [ ] Add route matrix tests covering every method and path.

Required tests:

- Public `POST /api/v1/catalog/refresh` returns `404` or `403` and causes no SEC
  traffic.
- Unknown `/api/*` paths do not reach the Container.
- Browser requests never receive or reflect `X-Filing-Room-Secret`.

### 4.2 Harden Worker request validation

- [ ] Require `Content-Type: application/json` for chat.
- [ ] Return a controlled `400` for malformed JSON.
- [ ] Validate every scope field, ticker format, accession format, form, year, and
      message length before consuming quota.
- [ ] Reject more than eight prior messages.
- [ ] Reject request bodies above the configured byte limit even if
      `Content-Length` is missing or inaccurate.
- [ ] Add a streaming body limit or read a bounded body before parsing.
- [ ] Enforce allowed methods and return `405` consistently.
- [ ] Add `Cache-Control: no-store` to quota, chat, source, and error responses.
- [ ] Add `X-Frame-Options: DENY` for older clients in addition to CSP.
- [ ] Review CSP against the final Turnstile integration and remove unnecessary
      sources.
- [ ] Consider validating `Sec-Fetch-Site` for browser mutation requests.

### 4.3 Complete Turnstile verification

- [ ] Create separate managed widgets for staging and production.
- [ ] Restrict each widget to its exact hostname.
- [ ] Supply an explicit action such as `research_answer` during rendering.
- [ ] Validate `success`, expected `hostname`, and expected `action` in the Worker.
- [ ] Handle `timeout-or-duplicate` by resetting the widget and requesting a fresh
      token.
- [ ] Add an idempotency key if Siteverify retries are implemented.
- [ ] Use Cloudflare test keys only in local and automated test environments.
- [ ] Add tests for success, failure, expiry, replay, wrong hostname, wrong action,
      and Siteverify outage.
- [ ] Fail closed when Siteverify is unavailable on a cost-bearing route.

Turnstile tokens expire after five minutes and are single-use. Server-side
validation is mandatory. References: [Turnstile setup](https://developers.cloudflare.com/turnstile/get-started/),
[Siteverify validation](https://developers.cloudflare.com/turnstile/get-started/server-side-validation/),
[Turnstile testing keys](https://developers.cloudflare.com/turnstile/troubleshooting/testing/).

### 4.4 Make quota reservation atomic

- [ ] Replace the separate `getQuota` and `consumeQuota` sequence with one atomic
      D1 operation that reserves a request only when the count is below the limit.
- [ ] Return the resulting remaining count from the same operation.
- [ ] Test 20 or more simultaneous requests from one visitor and verify no more than
      five reservations succeed.
- [ ] Decide whether provider failures consume quota.
- [ ] If failures should not consume quota, implement an idempotent reservation and
      release state rather than a blind decrement.
- [ ] Attach an idempotency key to each research request so client retries cannot
      consume multiple answers.
- [ ] Add cleanup for expired quota and idempotency rows.
- [ ] Keep the visitor hash daily rotating and do not persist raw IP addresses.
- [ ] Validate `VISITOR_HASH_SECRET` length at Worker startup or deployment time.

### 4.5 Implement the global monthly AI cutoff

- [ ] Add a D1 monthly usage table for estimated and confirmed spend.
- [ ] Reserve estimated cost before sending the OpenRouter request.
- [ ] Reconcile actual usage and cost from the provider response where available.
- [ ] Reject new generated answers when the application threshold is reached.
- [ ] Return `budgetAvailable=false` from quota status when generation is disabled.
- [ ] Keep cached company, filing, financial, and source browsing operational.
- [ ] Handle OpenRouter budget or allowlist `403`, payment `402`, rate-limit `429`,
      and transient `5xx` responses explicitly.
- [ ] Add an operator kill switch that disables model calls immediately.
- [ ] Alert at 50%, 75%, 90%, and 100% of the internal monthly threshold.
- [ ] Reconcile the internal ledger against the OpenRouter Activity dashboard daily
      during the first week.

### 4.6 Harden Container and API boundaries

- [ ] Confirm staging and production Containers receive non-development
      `ENVIRONMENT` values.
- [ ] Confirm the API rejects requests without the exact gateway secret outside
      development.
- [ ] Disable or protect `/api/docs` and `/api/openapi.json` in production if they
      are not intentionally public.
- [ ] Add trusted proxy and host handling appropriate to the Container boundary.
- [ ] Set explicit connection, read, write, and total timeouts for external calls.
- [ ] Close all reusable HTTP clients during application shutdown.
- [ ] Add a maximum filing size before parsing.
- [ ] Bound HTML node count, table size, chunk count, and parsed document size.
- [ ] Keep the Container filesystem non-root and read-only where supported, except
      for the required temporary cache path.
- [ ] Scan the built image for high and critical vulnerabilities.
- [ ] Produce an SBOM for the deployed image.
- [ ] Verify the image starts and passes health checks without network access to
      package registries.

## Gate 5: finish the SEC ingestion pipeline

### 5.1 Make ingestion durable and idempotent

- [ ] Replace process-local locks with D1 job claims or leases.
- [ ] Store job owner, lease expiry, attempt count, next retry time, and stable error
      code.
- [ ] Ensure two Containers cannot process the same accession concurrently.
- [ ] Make each stage safe to retry after a Container sleeps or crashes.
- [ ] Check R2 and D1 state before downloading or parsing again.
- [ ] Store a parser version and embedding version with each parsed document.
- [ ] Reprocess only when the version changes or an operator explicitly requests it.
- [ ] Use content hashes to detect unexpected source changes.
- [ ] Keep original accession provenance after amendments.
- [ ] Add a failed-job retry command with a bounded attempt count.
- [ ] Add a dead-letter state for filings needing manual review.

Required recovery test:

- Kill the Container during fetch, parse, and embedding stages. Restart it and
  verify one correct final record, no duplicate vectors, and no orphaned job lease.

### 5.2 Complete EdgarTools 5 extraction

- [ ] Replace the production HTML walker path with EdgarTools 5 document-tree
      extraction.
- [ ] Preserve headings, sections, paragraphs, tables, inline XBRL context, and
      stable source offsets.
- [ ] Keep the network-free parser adapter for deterministic fixtures where useful.
- [ ] Detect and remove table-of-contents duplicates without removing real sections.
- [ ] Handle 10-K/A and 10-Q/A filings explicitly.
- [ ] Handle unusual fiscal calendars and 52/53-week years.
- [ ] Preserve standalone tables rather than flattening them into narrative text.
- [ ] Record parser warnings and partial extraction state.
- [ ] Reject unsafe active content and never render raw filing HTML in the browser.

### 5.3 Finish deterministic financial facts

- [ ] Expand the curated US-GAAP concept registry for every displayed metric.
- [ ] Define unit, duration, instant, quarter, fiscal-year, and amendment selection
      rules per metric.
- [ ] Add explicit duplicate-period tie-breaking rules.
- [ ] Add scale and sign validation.
- [ ] Keep all growth, margin, and comparison arithmetic in Python.
- [ ] Return a reason code when a metric is unavailable or ambiguous.
- [ ] Show the exact accession and SEC URL for each displayed fact.
- [ ] Add tests for restatements, amendments, missing concepts, custom concepts,
      multiple units, and unusual calendars.

### 5.4 Build an operable prewarm command

- [ ] Add a checked-in CLI or admin job that reads
      `infra/prewarm/popular-100.txt`.
- [ ] Validate every ticker against the refreshed SEC catalog.
- [ ] Seed Company Facts for five fiscal years.
- [ ] Seed the latest 10-K and four latest 10-Q filings per issuer.
- [ ] Limit concurrency so total SEC traffic remains below the configured ceiling.
- [ ] Support resume from a saved cursor.
- [ ] Report successes, skips, retries, and permanent failures.
- [ ] Allow a small batch size for safe staging runs.
- [ ] Dry-run the command without SEC writes.
- [ ] Run production prewarming gradually and monitor storage and embedding cost.

## Gate 6: finish production retrieval and answer generation

### 6.1 Query Vectorize in production

- [ ] Add a query operation to the Worker `vectors.ai` outbound handler.
- [ ] Embed the user query with the same model and version used for document chunks.
- [ ] Apply metadata filters before top-k selection.
- [ ] Filter by selected ticker, form, fiscal year, accession, and section.
- [ ] Query multiple companies without allowing one company to dominate all results.
- [ ] Retrieve more candidates than the final context count for reranking.
- [ ] Return vector ID, score, and filter metadata, not filing text.
- [ ] Define a minimum supported score or fallback behavior based on evaluation data.
- [ ] Track embedding and index versions to support future reindexing.

Reference: [querying Vectorize](https://developers.cloudflare.com/vectorize/best-practices/query-vectors/).

### 6.2 Rehydrate exact chunks from R2

- [ ] Make every vector ID resolve deterministically to an R2 parsed-document object
      and chunk ID.
- [ ] Add object-store read methods to the Python storage abstraction.
- [ ] Load selected chunks after every Container cold start rather than relying on
      process memory.
- [ ] Validate the accession, ticker, form, fiscal year, section, and SEC URL read
      from storage against the vector metadata.
- [ ] Fetch adjacent narrative chunks only within the same filing and section.
- [ ] Bound total decompressed bytes and total context tokens.
- [ ] Return a controlled unavailable result when an index entry points to a missing
      object.
- [ ] Add a reconciliation job for orphaned vectors and R2 objects.

Required restart test:

- Ingest a filing, stop the Container, start a new instance, and answer a narrative
  question using Vectorize plus R2 without reingesting the filing.

### 6.3 Add a bounded reranker

- [ ] Choose and document the reranker implementation.
- [ ] Combine vector score, lexical match, section priority, table intent, and filing
      recency deterministically or through a separately evaluated reranker.
- [ ] Keep reranking local or account for any additional model cost.
- [ ] Prevent amendment and original duplicates from crowding out diverse evidence.
- [ ] Preserve the highest-scoring supported passages and adjacent context.
- [ ] Produce only the top eight final source groups unless evaluation justifies a
      different bound.
- [ ] Log scores and selected source IDs without logging full excerpts.

### 6.4 Enforce citation integrity server-side

- [ ] Generate request-local source IDs only after final retrieval.
- [ ] Accept citations only from the request-local source map.
- [ ] Remove or neutralize invented citation markers in model output.
- [ ] Ensure every returned citation resolves to a stored excerpt and SEC URL.
- [ ] Ensure `GET /sources/{source_id}` uses an opaque signed or server-resolvable ID,
      not a request-local label that cannot survive another request.
- [ ] Define source expiration behavior if signed IDs are used.
- [ ] Omit unsupported claims or clearly say the evidence is unavailable.
- [ ] Add adversarial tests where filing text asks the model to invent or alter
      citations.

### 6.5 Lock down the OpenRouter call

- [ ] Keep the model identifier server-controlled.
- [ ] Disable browser model selection and BYOK paths.
- [ ] Make the maximum input context and maximum output tokens explicit.
- [ ] Set temperature to `0.1` and configure low reasoning if the selected model API
      supports it.
- [ ] Confirm exactly one generation call is made per accepted answer.
- [ ] Decide whether provider fallback is permitted and configure it explicitly.
- [ ] Add request and response timeouts plus a controlled stream error event.
- [ ] Do not retry after answer tokens have started streaming.
- [ ] Strip provider errors of sensitive details before returning them to the browser.
- [ ] Record model, latency, token usage, and estimated cost without storing the full
      prompt or answer.
- [ ] Add a startup or scheduled model-availability check that does not consume user
      quota.

## Gate 7: finish the web product

### 7.1 Replace demonstration data in production

- [ ] Add an explicit build-time flag controlling demonstration fallbacks.
- [ ] Disable silent demo fallback in staging and production.
- [ ] Show a clear unavailable state when the API fails.
- [ ] Replace illustrative recent filings with catalog data.
- [ ] Prevent production pages from showing future or fabricated filing dates.
- [ ] Add a visible staging banner only in staging.

### 7.2 Complete company comparison

- [ ] Fetch live overview facts for all selected companies.
- [ ] Align metrics by fiscal year, period, unit, and form.
- [ ] Render a comparison table for up to three companies.
- [ ] Display missing values as unavailable rather than substituting demo values.
- [ ] Show accession provenance for each comparison value.
- [ ] Export exactly the displayed live table to CSV.
- [ ] Test different currencies, units, fiscal calendars, and missing metrics.

### 7.3 Make shared URLs robust

- [ ] Resolve arbitrary shared tickers through the company search API instead of the
      bundled popular-company list.
- [ ] Preserve only companies, forms, fiscal years, optional accession or section,
      and question.
- [ ] Never include answers, history, Turnstile tokens, visitor IDs, or provider data.
- [ ] Require the recipient to click Run before consuming quota.
- [ ] Validate and normalize malformed URL parameters.
- [ ] Add copy, paste, reload, and cross-device tests.

### 7.4 Complete Turnstile client behavior

- [ ] Render the correct widget for the active environment.
- [ ] Disable Run until a current token is available when required.
- [ ] Reset the widget after success, expiry, duplicate token, or rejected request.
- [ ] Preserve the typed question when verification must be repeated.
- [ ] Provide an accessible error message and retry path.
- [ ] Test light, dark, keyboard, reduced-motion, and mobile behavior.

### 7.5 Accessibility and browser quality

- [ ] Achieve WCAG AA contrast in both themes.
- [ ] Run automated accessibility checks on every primary route.
- [ ] Complete keyboard-only flows for search, filters, chat, drawers, citations,
      history, export, share, and theme selection.
- [ ] Verify focus moves into and returns from mobile drawers and the source panel.
- [ ] Announce streaming status and new answers through appropriate live regions.
- [ ] Respect reduced motion.
- [ ] Verify 200% zoom and narrow mobile layouts without clipped controls.
- [ ] Test current Chromium, Firefox, and WebKit.
- [ ] Set a performance budget for JavaScript, CSS, fonts, and the social preview.

### 7.6 Privacy and disclosures

- [ ] Publish a concise privacy page that describes Turnstile, daily IP-derived HMAC
      quota counters, IndexedDB history, provider processing, and retention.
- [ ] State that prompts and answers are not intentionally persisted server-side.
- [ ] State that filings are public SEC records and may contain sensitive business
      information.
- [ ] Keep the SEC non-affiliation, non-investment-advice, and source-verification
      disclosures visible but proportionate.
- [ ] Provide local history clear and per-thread delete controls.
- [ ] Add a contact for privacy and data questions.

## Gate 8: testing and evaluation release gates

### 8.1 Backend golden fixtures

- [ ] Add at least 20 varied 10-K and 10-Q fixtures.
- [ ] Include amendments, inline XBRL, unusual fiscal years, TOC duplication, missing
      sections, large tables, custom concepts, and malformed but accepted HTML.
- [ ] Store fixture provenance and expected parser version.
- [ ] Test section extraction, table boundaries, source offsets, and chunk overlap.
- [ ] Test idempotent reprocessing and versioned reindexing.
- [ ] Keep CI independent from live SEC traffic.

Release gate:

- All supported deterministic financial calculations match expected fixture values.

### 8.2 Retrieval and answer evaluation

- [ ] Create at least 60 labeled questions across 12 issuers.
- [ ] Cover exact facts, calculations, comparisons, narrative summaries, follow-ups,
      ambiguity, missing facts, amendments, and out-of-scope requests.
- [ ] Label relevant chunks and acceptable supporting citations.
- [ ] Measure recall at top eight.
- [ ] Measure unsupported-claim rate and citation resolvability.
- [ ] Run the suite on every retrieval, parser, chunking, embedding, prompt, or model
      change.
- [ ] Store evaluation version, model ID, parser version, embedding version, and date.

Release gates:

- Narrative retrieval recall at top eight is at least 90%.
- Every displayed citation resolves to stored evidence and the original SEC URL.
- No invented source ID becomes an active citation.
- Unsupported claims are omitted or identified as unavailable.

### 8.3 Security tests

- [ ] Test arbitrary SEC URL rejection and path traversal.
- [ ] Test prompt injection inside headings, paragraphs, tables, and XBRL labels.
- [ ] Test citation spoofing and Markdown HTML injection.
- [ ] Test cross-origin and missing-origin mutation requests.
- [ ] Test oversized, deeply nested, malformed, and duplicate requests.
- [ ] Test Turnstile bypass, replay, wrong hostname, and provider outage.
- [ ] Test quota races, idempotency, daily rotation, and global budget cutoff.
- [ ] Test internal endpoint access from a public browser.
- [ ] Test that logs and error responses do not contain secrets, full prompts, raw
      IPs, or filing bodies.

### 8.4 Frontend component and E2E tests

- [ ] Add Vitest and Testing Library coverage for filters, SSE parsing, quota states,
      citations, exports, share URLs, history limits, and errors.
- [ ] Commit Playwright desktop and mobile tests for search, company overview, warm
      chat, cold progress, comparison, CSV, citations, source panel, history, theme,
      Turnstile failure, and share URLs.
- [ ] Use Turnstile test keys in automated tests.
- [ ] Run automated accessibility assertions in Playwright.
- [ ] Capture traces or screenshots only on failure and keep artifacts free of
      secrets.

### 8.5 Integration and failure recovery

- [ ] Mock D1, R2, Vectorize, Workers AI, SEC, Turnstile, and OpenRouter in CI.
- [ ] Add a staging-only live integration suite with strict spending and request
      limits.
- [ ] Test Container cold starts and scale-to-zero recovery.
- [ ] Test partial R2, D1, Vectorize, SEC, and provider failures.
- [ ] Test SSE disconnect and browser cancellation.
- [ ] Test migration compatibility with the previous Worker version.
- [ ] Test Worker rollback while preserving current storage bindings.

### 8.6 Performance and load tests

- [ ] Measure cached overview p50, p95, and p99.
- [ ] Measure retrieval latency before generation.
- [ ] Measure warm time to first answer token.
- [ ] Measure cold-ingestion progress time and completion time by filing size.
- [ ] Load test quota concurrency and public read endpoints.
- [ ] Confirm SEC aggregate traffic remains under five requests per second.
- [ ] Confirm Container instance count remains within configuration.
- [ ] Measure D1 rows read and written, R2 operations, vector dimensions queried,
      Workers AI usage, log volume, and estimated model cost.

Release targets:

- Cached company overview p95 below one second.
- Warm retrieval p95 below 1.5 seconds before model generation.
- Warm first answer token below five seconds under normal provider conditions.
- Cold ingestion reports progress within one second.
- Typical cold filing completes within 45 seconds or the target is revised with
  measured evidence and honest product copy.

## Gate 9: observability and operations

### 9.1 Add privacy-safe structured logs

- [ ] Generate or propagate a request ID for every request.
- [ ] Log structured JSON, not concatenated free-form strings.
- [ ] Log route class, status, latency, source count, retrieval scores, ingestion
      stage, retry code, model ID, token usage, and estimated cost.
- [ ] Never log full prompts, answers, raw IP addresses, credentials, Turnstile
      tokens, gateway secrets, or complete filing bodies.
- [ ] Hash or classify identifiers only when needed for debugging.
- [ ] Configure an intentional Workers Logs sampling rate.
- [ ] Verify log retention and cost fit the budget.
- [ ] Add a redaction test that inspects captured logs.

Cloudflare recommends structured JSON for Workers Logs. Reference:
[Workers Logs](https://developers.cloudflare.com/workers/observability/logs/workers-logs/).

### 9.2 Make health and readiness meaningful

- [ ] Keep liveness independent of external providers.
- [ ] Make readiness verify required internal configuration and bindings.
- [ ] Report dependency state without exposing secret values or sensitive details.
- [ ] Add a deeper authenticated diagnostic for D1, R2, Vectorize, Workers AI, and
      model availability.
- [ ] Ensure a provider outage does not mark cached browsing unavailable.
- [ ] Add an external synthetic check for landing, company search, quota, and one
      low-frequency staging research query.

### 9.3 Create dashboards and alerts

- [ ] Dashboard Worker request rate, error rate, latency, CPU, and logs.
- [ ] Dashboard Container starts, active time, errors, and cold-start latency.
- [ ] Dashboard D1 operations and failures.
- [ ] Dashboard R2 object count, storage, reads, and writes.
- [ ] Dashboard Vectorize upserts, queries, errors, and dimensions.
- [ ] Dashboard Workers AI usage and errors.
- [ ] Dashboard SEC status codes, backoff, and ingestion failures.
- [ ] Dashboard OpenRouter usage, cost, latency, and rejection codes.
- [ ] Alert on elevated `5xx`, repeated SEC `403` or `429`, ingestion failure spikes,
      citation validation failure, and budget thresholds.
- [ ] Route alerts to a monitored channel with a named responder.

### 9.4 Write incident runbooks

- [ ] AI budget exhausted.
- [ ] OpenRouter outage or model removal.
- [ ] SEC blocking or prolonged outage.
- [ ] Citation integrity regression.
- [ ] D1 migration failure or corruption.
- [ ] R2 object loss or retrieval mismatch.
- [ ] Vector index corruption or reindex requirement.
- [ ] Turnstile outage.
- [ ] Credential exposure.
- [ ] Traffic abuse or quota bypass.
- [ ] Worker or Container rollback.

Each runbook must name detection, immediate containment, user-visible behavior,
recovery, verification, communication, and follow-up ownership.

## Gate 10: CI and deployment automation

### 10.1 Strengthen CI

- [ ] Keep Node and Python lockfile installs frozen.
- [ ] Run punctuation, formatting, lint, type checks, unit tests, builds, and secret
      scanning.
- [ ] Add Docker image build and startup smoke tests.
- [ ] Add dependency and image vulnerability scans.
- [ ] Generate the FastAPI OpenAPI schema in CI.
- [ ] Generate TypeScript clients or contracts and fail on drift.
- [ ] Run database migration checks against an isolated D1 test database.
- [ ] Upload test reports and browser traces only on failure.
- [ ] Cancel superseded workflow runs with a concurrency group.
- [ ] Add Dependabot coverage for every package ecosystem.
- [ ] Require all release gates before deployment.

### 10.2 Create protected deployment environments

- [ ] Create GitHub `staging` and `production` environments.
- [ ] Store only deployment credentials appropriate to each environment.
- [ ] Restrict production deployment to `main` or signed release tags.
- [ ] Require manual approval for production.
- [ ] Prevent staging pull requests from receiving production secrets.
- [ ] Prefer a scoped Cloudflare API token over a global API key.
- [ ] Document token permissions and expiration.

Reference: [GitHub deployment environments](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments),
[GitHub secret types](https://docs.github.com/en/code-security/reference/secret-security/secret-types).

### 10.3 Build a staging deployment workflow

- [ ] Trigger only after required CI succeeds on `main`.
- [ ] Build the exact frontend and Container image once.
- [ ] Apply staging migrations before application promotion.
- [ ] Deploy a version and record Worker and Container version identifiers.
- [ ] Run automated staging smoke and E2E tests.
- [ ] Mark the workflow failed if citations, quota, Turnstile, or cold ingestion fail.
- [ ] Retain deployment evidence and commit SHA.

### 10.4 Build a production promotion workflow

- [ ] Promote the exact tested commit and artifacts.
- [ ] Capture a D1 Time Travel bookmark before migration.
- [ ] Require a manual go or no-go approval.
- [ ] Apply backward-compatible production migrations.
- [ ] Deploy or gradually promote the Worker and Container.
- [ ] Run production smoke tests that do not expose secrets or incur uncontrolled
      model cost.
- [ ] Record domain, commit SHA, Worker version, Container version, D1 bookmark,
      migration list, and approver.

Cloudflare stores Worker versions separately from deployments and supports
gradual promotion and rollback. Storage changes are not part of a Worker version.
Reference: [Workers versions and deployments](https://developers.cloudflare.com/workers/versions-and-deployments/).

## Gate 11: staging validation

### 11.1 Deploy staging

- [ ] Confirm Docker is running.
- [ ] Confirm staging variables and secrets are set.
- [ ] Confirm staging D1 migrations and Vectorize metadata indexes exist.
- [ ] Build from a clean checkout.
- [ ] Deploy with the staging environment.

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm lint
pnpm typecheck
pnpm test
pnpm build
pnpm --filter @filing-room/gateway exec wrangler deploy --env staging
```

- [ ] Wait for Container provisioning before judging readiness.
- [ ] Record the staging Worker and Container versions.

### 11.2 Execute staging acceptance tests

- [ ] Landing page and self-hosted fonts load with no console errors.
- [ ] Company search uses the refreshed SEC catalog.
- [ ] Company overview displays live facts and filings.
- [ ] Warm chat streams sources and an answer.
- [ ] Cold chat shows progress, persists data, and continues automatically.
- [ ] Restarting the Container does not lose retrieval capability.
- [ ] Comparison table and CSV contain live values.
- [ ] Citation buttons open exact excerpts and valid SEC links.
- [ ] Share URL contains only scope and question and does not auto-run.
- [ ] Five-answer quota rejects the sixth request.
- [ ] Concurrent requests cannot exceed quota.
- [ ] Global cutoff disables generation but not cached browsing.
- [ ] Turnstile rejects wrong-hostname, replayed, expired, and test tokens.
- [ ] Internal endpoints are unavailable publicly.
- [ ] CSP, origin, size, and method restrictions behave as expected.
- [ ] Mobile, keyboard, dark theme, reduced motion, and screen-reader flows pass.
- [ ] Logs contain required fields and no prohibited data.

### 11.3 Run a controlled prewarm sample

- [ ] Prewarm three diverse issuers first.
- [ ] Verify D1, R2, Vectorize, and Workers AI counts.
- [ ] Run labeled questions against the prewarmed data.
- [ ] Inspect SEC response codes and request rate.
- [ ] Measure cost and extrapolate to the popular-100 set.
- [ ] Increase batch size only after the measured envelope is acceptable.

## Gate 12: production domain and launch

### 12.1 Prepare the custom domain

- [ ] Confirm no existing CNAME or conflicting Worker route exists for
      `sec.bcastelino.com`.
- [ ] Add a Worker Custom Domain through Wrangler or the Cloudflare dashboard.
- [ ] Let Cloudflare create the DNS record and certificate.
- [ ] Do not create a manual CNAME for an originless Worker Custom Domain.
- [ ] Confirm certificate issuance and HTTPS before public announcement.
- [ ] Confirm HTTP redirects to HTTPS.
- [ ] Confirm the final host is in Turnstile hostname restrictions.
- [ ] Confirm the final host is the only production API origin.

Reference: [Workers Custom Domains](https://developers.cloudflare.com/workers/configuration/routing/custom-domains/).

### 12.2 Final go or no-go review

- [ ] Every P0 item is checked with evidence.
- [ ] Required CI and evaluation gates pass on the release commit.
- [ ] Staging has run the exact release for an agreed soak period.
- [ ] Cloudflare resources and bindings are production-only and correct.
- [ ] Production secrets are present and test secrets are absent.
- [ ] OpenRouter allowlist and $7 monthly guardrail are active.
- [ ] Internal app budget cutoff works.
- [ ] SEC identity and aggregate limiter are verified.
- [ ] Production Turnstile hostname and action validation pass.
- [ ] D1 bookmark is recorded.
- [ ] Rollback versions and operator commands are recorded.
- [ ] Dashboards and alerts are active.
- [ ] Privacy, disclosures, security contact, and limitations are published.
- [ ] The incident responder is available for the launch window.

### 12.3 Deploy and smoke test

- [ ] Deploy the tested release through the protected production environment.
- [ ] Verify the Worker and Container reach healthy states.
- [ ] Verify `https://sec.bcastelino.com` certificate and headers.
- [ ] Run one cached overview request.
- [ ] Run one warm research request.
- [ ] Run one controlled cold request if the launch window allows it.
- [ ] Validate citation resolution and original SEC URLs.
- [ ] Validate quota and Turnstile.
- [ ] Validate cached browsing after temporarily activating the model kill switch.
- [ ] Record all production version identifiers and smoke-test evidence.

### 12.4 Prewarm production gradually

- [ ] Start with the previously validated three issuers.
- [ ] Continue in small batches from `popular-100.txt`.
- [ ] Stop on sustained SEC `403`, `429`, unusual `5xx`, cost anomalies, or ingestion
      failure threshold.
- [ ] Reconcile expected filings, R2 objects, vectors, and facts after each batch.
- [ ] Run sample questions after each batch.
- [ ] Publish no claim of complete popular-100 coverage until reconciliation passes.

## Gate 13: rollback readiness

### 13.1 Practice before launch

- [ ] Deploy two harmless staging versions.
- [ ] Roll back the Worker to the earlier version.
- [ ] Confirm the associated Container behavior.
- [ ] Confirm current D1, R2, and Vectorize data remain compatible.
- [ ] Practice disabling model generation without rolling back cached browsing.
- [ ] Practice restoring service after a bad non-destructive configuration change.
- [ ] Record the exact commands and expected duration.

### 13.2 Production rollback decision tree

- [ ] For Worker or frontend regression, roll back to the previous Worker version.
- [ ] For Container regression, restore the prior compatible Worker and Container
      release together.
- [ ] For retrieval-index regression, stop generation and ingestion, then use the
      known-good index or reindex from immutable R2 data.
- [ ] For bad D1 migration, stop writes and follow the reviewed database incident
      procedure. Do not improvise a Time Travel restore.
- [ ] For credential exposure, disable and rotate the credential before restoring
      traffic.
- [ ] For citation integrity failure, disable generated answers immediately while
      preserving cached source browsing.

```bash
cd workers/gateway
corepack pnpm exec wrangler deployments list --env production
corepack pnpm exec wrangler versions list --env production
corepack pnpm exec wrangler rollback --env production
```

Cloudflare rollbacks do not roll back D1, R2, Vectorize, or other storage state.
Reference: [Workers rollbacks](https://developers.cloudflare.com/workers/versions-and-deployments/rollbacks/).

## Gate 14: post-launch operations

### First hour

- [ ] Watch Worker and Container errors continuously.
- [ ] Watch SEC status codes and request rate.
- [ ] Watch OpenRouter spend and latency.
- [ ] Watch Turnstile failures and quota rejections.
- [ ] Sample citations manually from multiple issuers.
- [ ] Confirm alerts reach the responder.

### First 24 hours

- [ ] Review p50, p95, and p99 latency by route.
- [ ] Reconcile internal model cost against OpenRouter.
- [ ] Review ingestion failures and retry outcomes.
- [ ] Review D1, R2, Vectorize, Workers AI, Container, and logs usage.
- [ ] Confirm no secret or prompt content appears in logs.
- [ ] Confirm no unexpected public route reaches an internal operation.
- [ ] Write a brief launch report and record follow-up issues.

### Weekly

- [ ] Review provider costs and projections.
- [ ] Review dependency and security alerts.
- [ ] Review failed and slow filings.
- [ ] Review retrieval misses and unsupported-answer examples.
- [ ] Run the evaluation suite and compare trends.
- [ ] Verify the model remains available and within the allowlist.
- [ ] Review quota abuse patterns without attempting to identify visitors.

### Monthly

- [ ] Confirm the OpenRouter budget reset and application ledger reset.
- [ ] Review Cloudflare invoices and included usage.
- [ ] Review secret age and access permissions.
- [ ] Test one staging rollback or recovery scenario.
- [ ] Review D1 Time Travel and longer-term backup needs.
- [ ] Review R2 retention and orphan reconciliation.
- [ ] Update this checklist when architecture or provider behavior changes.

## Release evidence template

Copy this into the release issue or deployment record:

```text
Release commit:
Release date and time (UTC):
Approver:
Staging Worker version:
Staging Container version:
Production Worker version:
Production Container version:
Production D1 bookmark:
Applied migrations:
OpenRouter model:
OpenRouter guardrail reviewed:
Turnstile hostname and action verified:
Evaluation report:
Accessibility report:
Performance report:
Security test report:
Smoke test result:
Rollback version:
Known limitations:
Incident responder:
```

## Definition of production ready

Filing Room is production ready only when:

- Every P0 blocker is closed with automated or saved manual evidence.
- The application remains useful when generation or ingestion is disabled.
- Every active citation resolves to stored evidence and an original SEC URL.
- Concurrent traffic cannot bypass visitor or global spending limits.
- A Container restart does not lose access to previously indexed research data.
- Staging and production have isolated resources and secrets.
- Required evaluation, security, accessibility, performance, and recovery gates
  pass on the exact release commit.
- The owner can detect, contain, roll back, and explain a production failure.
