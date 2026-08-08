import { Container } from "@cloudflare/containers";
import { env as workerEnv } from "cloudflare:workers";
import type { Env } from "./env";

const runtimeEnv = workerEnv as unknown as Env;

export class ApiContainer extends Container<Env> {
  defaultPort = 8080;
  sleepAfter = "5m";
  envVars = {
    ENVIRONMENT: runtimeEnv.ENVIRONMENT,
    API_SHARED_SECRET: runtimeEnv.API_SHARED_SECRET,
    SEC_IDENTITY: runtimeEnv.SEC_IDENTITY,
    OPENROUTER_API_KEY: runtimeEnv.OPENROUTER_API_KEY,
    OPENROUTER_MODEL: runtimeEnv.OPENROUTER_MODEL,
    OBJECT_STORE_BACKEND: "cloudflare",
    VECTOR_INDEX_BACKEND: "cloudflare",
    METADATA_BACKEND: runtimeEnv.METADATA_BACKEND,
  };

  override onError(error: unknown) {
    console.error("api_container_error", { message: error instanceof Error ? error.message : "unknown" });
  }
}

ApiContainer.outboundByHost = {
  "filings.r2": async (request, rawBindings) => {
    const bindings = rawBindings as unknown as Env;
    const key = decodeURIComponent(new URL(request.url).pathname.slice(1));
    if (!key || key.includes("..")) return new Response("Invalid key", { status: 400 });
    if (request.method === "PUT") {
      await bindings.FILINGS.put(key, request.body, {
        httpMetadata: {
          contentType: request.headers.get("content-type") ?? "application/octet-stream",
          contentEncoding: request.headers.get("content-encoding") ?? undefined,
        },
      });
      return new Response(null, { status: 204 });
    }
    if (request.method === "GET") {
      const object = await bindings.FILINGS.get(key);
      return object ? new Response(object.body, { headers: { etag: object.httpEtag } }) : new Response(null, { status: 404 });
    }
    return new Response(null, { status: 405 });
  },
  "vectors.ai": async (request, rawBindings) => {
    const bindings = rawBindings as unknown as Env;
    if (request.method !== "POST") return new Response(null, { status: 405 });
    const body = await request.json<{ chunks: Array<{ id: string; text: string; metadata: Record<string, string | number | boolean> }> }>();
    if (!Array.isArray(body.chunks) || body.chunks.length > 100) return new Response("Invalid batch", { status: 422 });
    const result = await bindings.AI.run("@cf/baai/bge-small-en-v1.5", { text: body.chunks.map((chunk) => chunk.text) }) as { data: number[][] };
    await bindings.VECTORS.upsert(body.chunks.map((chunk, index) => ({ id: chunk.id, values: result.data[index]!, metadata: chunk.metadata })));
    return Response.json({ indexed: body.chunks.length });
  },
  "metadata.d1": async (request, rawBindings) => {
    const bindings = rawBindings as unknown as Env;
    const url = new URL(request.url);
    if (request.method === "GET" && url.pathname === "/companies/search") {
      const rawQuery = (url.searchParams.get("q") ?? "").trim();
      const query = `%${rawQuery}%`;
      const limit = Math.min(20, Math.max(1, Number(url.searchParams.get("limit") ?? 8)));
      const result = await bindings.DB.prepare(
        "SELECT ticker, cik, name FROM companies WHERE ticker LIKE ? OR name LIKE ? ORDER BY CASE WHEN ticker LIKE ? THEN 0 ELSE 1 END, ticker LIMIT ?",
      ).bind(query, query, `${rawQuery}%`, limit).all();
      return Response.json(result.results);
    }
    const overviewMatch = url.pathname.match(/^\/companies\/([A-Za-z0-9.-]+)\/overview$/);
    if (request.method === "GET" && overviewMatch) {
      const company = await bindings.DB.prepare(
        "SELECT ticker, cik, name, sic, fiscal_year_end AS fiscalYearEnd FROM companies WHERE ticker = ?",
      ).bind(overviewMatch[1]!.toUpperCase()).first<Record<string, unknown>>();
      if (!company) return new Response(null, { status: 404 });
      const facts = await bindings.DB.prepare(
        "SELECT concept, display_label AS displayLabel, value, unit, period_start AS periodStart, period_end AS periodEnd, fiscal_year AS fiscalYear, fiscal_quarter AS fiscalQuarter, form, filed_date AS filedDate, accession, source_url AS sourceUrl FROM financial_facts WHERE cik = ? ORDER BY fiscal_year, COALESCE(fiscal_quarter, 5), concept",
      ).bind(company.cik).all();
      const filings = await bindings.DB.prepare(
        "SELECT f.accession, f.form, f.filing_date AS filingDate, f.report_date AS reportDate, CAST(substr(f.report_date, 1, 4) AS INTEGER) AS fiscalYear, f.primary_document AS primaryDocument, f.is_amendment AS isAmendment, j.stage, j.progress, j.error_code AS errorCode FROM filings f LEFT JOIN ingestion_jobs j ON j.accession = f.accession WHERE f.cik = ? ORDER BY f.filing_date DESC",
      ).bind(company.cik).all<Record<string, unknown>>();
      const normalizedFilings = filings.results.map((filing) => ({
        ...filing,
        isAmendment: Boolean(filing.isAmendment),
        ingestion: { accession: filing.accession, stage: filing.stage ?? "queued", progress: filing.progress ?? 0, errorCode: filing.errorCode ?? undefined },
        secUrl: `https://www.sec.gov/Archives/edgar/data/${Number(company.cik)}/${String(filing.accession).replaceAll("-", "")}/${filing.primaryDocument}`,
      }));
      return Response.json({ company: { ticker: company.ticker, cik: company.cik, name: company.name }, sic: company.sic, fiscalYearEnd: company.fiscalYearEnd, description: "Public-company financials and disclosures sourced exclusively from SEC filings.", facts: facts.results, filings: normalizedFilings });
    }
    if (request.method === "POST" && url.pathname === "/companies") {
      const companies = await request.json<Array<{ ticker: string; cik: string; name: string }>>();
      for (let start = 0; start < companies.length; start += 100) {
        await bindings.DB.batch(companies.slice(start, start + 100).map((company) => bindings.DB.prepare(
          "INSERT INTO companies (cik, ticker, name, updated_at) VALUES (?, ?, ?, ?) ON CONFLICT(cik) DO UPDATE SET ticker = excluded.ticker, name = excluded.name, updated_at = excluded.updated_at",
        ).bind(company.cik, company.ticker, company.name, new Date().toISOString())));
      }
      return Response.json({ saved: companies.length });
    }
    if (request.method === "POST" && url.pathname === "/company-data") {
      const body = await request.json<{ company: { ticker: string; cik: string; name: string }; facts: Array<Record<string, unknown>>; filings: Array<Record<string, unknown>> }>();
      const now = new Date().toISOString();
      await bindings.DB.prepare(
        "INSERT INTO companies (cik, ticker, name, updated_at) VALUES (?, ?, ?, ?) ON CONFLICT(cik) DO UPDATE SET ticker = excluded.ticker, name = excluded.name, updated_at = excluded.updated_at",
      ).bind(body.company.cik, body.company.ticker, body.company.name, now).run();
      await bindings.DB.prepare("DELETE FROM financial_facts WHERE cik = ?").bind(body.company.cik).run();
      const statements = [
        ...body.facts.map((fact) => bindings.DB.prepare(
          "INSERT OR REPLACE INTO financial_facts (cik, concept, display_label, value, unit, period_start, period_end, fiscal_year, fiscal_quarter, form, filed_date, accession, source_url) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ).bind(body.company.cik, fact.concept, fact.displayLabel, fact.value, fact.unit, fact.periodStart ?? null, fact.periodEnd, fact.fiscalYear, fact.fiscalQuarter ?? null, fact.form, fact.filedDate, fact.accession, fact.sourceUrl)),
        ...body.filings.flatMap((filing) => [
          bindings.DB.prepare(
            "INSERT OR REPLACE INTO filings (accession, cik, form, filing_date, report_date, primary_document, is_amendment, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
          ).bind(filing.accession, body.company.cik, filing.form, filing.filingDate, filing.reportDate, filing.primaryDocument, filing.isAmendment ? 1 : 0, now),
          bindings.DB.prepare(
            "INSERT OR REPLACE INTO ingestion_jobs (accession, stage, progress, retry_count, updated_at) VALUES (?, 'ready', 100, 0, ?)",
          ).bind(filing.accession, now),
        ]),
      ];
      for (let start = 0; start < statements.length; start += 100) await bindings.DB.batch(statements.slice(start, start + 100));
      return Response.json({ saved: true });
    }
    return new Response(null, { status: 404 });
  },
};
