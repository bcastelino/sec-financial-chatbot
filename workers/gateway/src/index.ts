import type { ChatRequest } from "@filing-room/contracts";
import type { Env } from "./env";
import { consumeQuota, getQuota } from "./quota";
import { securityHeaders, verifyTurnstile, visitorHash } from "./security";
export { ApiContainer } from "./container";

const JSON_HEADERS = { "content-type": "application/json; charset=utf-8" };

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: securityHeaders(new Headers(JSON_HEADERS)) });
}

function validOrigin(request: Request): boolean {
  const origin = request.headers.get("Origin");
  return !origin || new URL(request.url).origin === origin;
}

async function proxy(request: Request, env: Env, path: string, extraHeaders: HeadersInit = {}): Promise<Response> {
  const upstream = new URL(path, env.API_BASE_URL);
  const headers = new Headers(request.headers);
  headers.set("X-Filing-Room-Secret", env.API_SHARED_SECRET);
  Object.entries(extraHeaders).forEach(([key, value]) => headers.set(key, value));
  headers.delete("CF-Connecting-IP");
  headers.delete("X-Forwarded-For");
  const upstreamRequest = new Request(upstream, { method: request.method, headers, body: request.body, redirect: "manual" });
  const response = env.ENVIRONMENT === "development"
    ? await fetch(upstreamRequest)
    : await env.API_CONTAINER.getByName("filing-room-api").fetch(upstreamRequest);
  return new Response(response.body, { status: response.status, headers: securityHeaders(new Headers(response.headers)) });
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (!validOrigin(request)) return json({ detail: "Origin rejected" }, 403);

    if (url.pathname === "/api/v1/quota" && request.method === "GET") {
      return json(await getQuota(env, await visitorHash(request, env)));
    }

    if (url.pathname === "/api/v1/chat/stream" && request.method === "POST") {
      const length = Number(request.headers.get("content-length") ?? 0);
      if (length > 32_768) return json({ detail: "Request too large" }, 413);
      const body = (await request.clone().json()) as ChatRequest;
      if (!body.query?.trim() || body.query.length > 2_000 || body.scope.companies.length > 3) {
        return json({ detail: "Invalid research request" }, 422);
      }
      const visitor = await visitorHash(request, env);
      const quota = await getQuota(env, visitor);
      if (!quota.budgetAvailable || quota.remaining < 1) return json({ detail: "Daily answer limit reached", quota }, 429);
      if (!(await verifyTurnstile(request, body.turnstileToken, env))) return json({ detail: "Verification failed" }, 403);
      await consumeQuota(env, visitor);
      const sanitized = new Request(request.url, { method: "POST", headers: request.headers, body: JSON.stringify({ ...body, turnstileToken: "verified" }) });
      return proxy(sanitized, env, "/api/v1/chat/stream", { "X-Visitor-Hash": visitor });
    }

    if (url.pathname.startsWith("/api/")) return proxy(request, env, url.pathname + url.search);
    const asset = await env.ASSETS.fetch(request);
    return new Response(asset.body, { status: asset.status, headers: securityHeaders(new Headers(asset.headers)) });
  },
  async scheduled(_controller: ScheduledController, env: Env): Promise<void> {
    const headers = new Headers({ "X-Filing-Room-Secret": env.API_SHARED_SECRET });
    const request = new Request(new URL("/api/v1/catalog/refresh", env.API_BASE_URL), { method: "POST", headers });
    const response = env.ENVIRONMENT === "development" ? await fetch(request) : await env.API_CONTAINER.getByName("filing-room-api").fetch(request);
    if (!response.ok) console.error("catalog_refresh_failed", { status: response.status });
  },
} satisfies ExportedHandler<Env>;
