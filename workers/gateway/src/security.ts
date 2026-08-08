import type { Env } from "./env";

const encoder = new TextEncoder();

export async function visitorHash(request: Request, env: Env): Promise<string> {
  const ip = request.headers.get("CF-Connecting-IP") ?? "local";
  const day = new Date().toISOString().slice(0, 10);
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(env.VISITOR_HASH_SECRET),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const digest = await crypto.subtle.sign("HMAC", key, encoder.encode(`${day}:${ip}`));
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

export async function verifyTurnstile(request: Request, token: string, env: Env): Promise<boolean> {
  if (env.ENVIRONMENT === "development" && token === "dev-token") return true;
  const form = new FormData();
  form.set("secret", env.TURNSTILE_SECRET_KEY);
  form.set("response", token);
  const ip = request.headers.get("CF-Connecting-IP");
  if (ip) form.set("remoteip", ip);
  const response = await fetch("https://challenges.cloudflare.com/turnstile/v0/siteverify", {
    method: "POST",
    body: form,
  });
  const result = (await response.json()) as { success?: boolean };
  return result.success === true;
}

export function securityHeaders(headers = new Headers()): Headers {
  headers.set("Content-Security-Policy", "default-src 'self'; script-src 'self' https://challenges.cloudflare.com; frame-src https://challenges.cloudflare.com; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'");
  headers.set("Referrer-Policy", "strict-origin-when-cross-origin");
  headers.set("X-Content-Type-Options", "nosniff");
  headers.set("Permissions-Policy", "camera=(), microphone=(), geolocation=()");
  return headers;
}
