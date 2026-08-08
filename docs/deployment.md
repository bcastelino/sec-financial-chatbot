# Cloudflare deployment

These steps require the owner's Cloudflare and OpenRouter accounts. They are
intentionally not automated from a contributor workstation.

This page is the concise deployment sequence. Complete the full
[production readiness checklist](production-readiness-checklist.md) before
attaching the public domain.

## 1. Prepare provider guardrails

Create a dedicated OpenRouter key restricted to the configured model and set a
$7 monthly limit. Rotate the prior key if a repository-history scan shows it was
ever committed.

## 2. Create Cloudflare resources

From `workers/gateway`:

```bash
pnpm wrangler d1 create filing-room
pnpm wrangler r2 bucket create filing-room-filings
pnpm wrangler vectorize create filing-room-chunks --dimensions=384 --metric=cosine
```

Replace only the placeholder D1 `database_id` in `wrangler.jsonc`; do not commit
account IDs or secrets. Apply migrations:

```bash
pnpm wrangler d1 migrations apply filing-room --remote
```

Create a managed Turnstile widget for `sec.bcastelino.com` and configure its
site key as `VITE_TURNSTILE_SITE_KEY` during the web build.

## 3. Set secrets

```bash
pnpm wrangler secret put OPENROUTER_API_KEY
pnpm wrangler secret put API_SHARED_SECRET
pnpm wrangler secret put TURNSTILE_SECRET_KEY
pnpm wrangler secret put VISITOR_HASH_SECRET
```

Use distinct random values of at least 32 bytes for the last two application
secrets. Never paste secrets into `wrangler.jsonc` or GitHub Actions logs.

## 4. Build and deploy staging

Docker must be running because Wrangler builds the Python Container image.

```bash
pnpm install --frozen-lockfile
pnpm build
pnpm --filter @filing-room/gateway exec wrangler deploy --env staging
```

Wait for Container provisioning, apply D1 migrations, then verify `/api/v1/health`,
catalog refresh, a warm answer, a cold answer, citations, quota rejection, CSV,
share URL, mobile layout, CSP, and dark mode.

## 5. Attach the custom domain

Add a Worker custom domain for `sec.bcastelino.com` in the same Cloudflare zone,
then deploy the exact tested Worker and Container versions to production. Do not
create a manual proxied CNAME for a Worker custom domain.

## 6. Prewarm and monitor

The daily cron refreshes the issuer catalog. Seed `infra/prewarm/popular-100.txt`
in small batches so SEC traffic stays under five requests per second. Monitor
Container starts, D1/R2/Vectorize operations, SEC response codes, retrieval
latency, OpenRouter spend, quota rejections, and citation validation.

## Rollback

Keep the prior Worker deployment and Container image. If error or citation rates
regress, roll back both versions together; storage keys are immutable by
accession, and D1 migrations must remain backward compatible for one release.
