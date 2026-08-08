import type { QuotaStatus } from "@filing-room/contracts";
import type { Env } from "./env";

function resetTime(): string {
  const next = new Date();
  next.setUTCDate(next.getUTCDate() + 1);
  next.setUTCHours(0, 0, 0, 0);
  return next.toISOString();
}

export async function getQuota(env: Env, visitor: string): Promise<QuotaStatus> {
  const day = new Date().toISOString().slice(0, 10);
  const row = await env.DB.prepare("SELECT answer_count FROM visitor_quotas WHERE visitor_hash = ? AND quota_day = ?")
    .bind(visitor, day)
    .first<{ answer_count: number }>();
  const limit = Number.parseInt(env.DAILY_ANSWER_LIMIT || "5", 10);
  return { remaining: Math.max(0, limit - (row?.answer_count ?? 0)), limit, resetsAt: resetTime(), budgetAvailable: true };
}

export async function consumeQuota(env: Env, visitor: string): Promise<QuotaStatus> {
  const day = new Date().toISOString().slice(0, 10);
  await env.DB.prepare(
    "INSERT INTO visitor_quotas (visitor_hash, quota_day, answer_count) VALUES (?, ?, 1) ON CONFLICT(visitor_hash, quota_day) DO UPDATE SET answer_count = answer_count + 1",
  ).bind(visitor, day).run();
  return getQuota(env, visitor);
}
