export interface Env {
  ASSETS: Fetcher;
  DB: D1Database;
  FILINGS: R2Bucket;
  VECTORS: VectorizeIndex;
  AI: Ai;
  API_CONTAINER: DurableObjectNamespace;
  API_BASE_URL: string;
  API_SHARED_SECRET: string;
  TURNSTILE_SECRET_KEY: string;
  VISITOR_HASH_SECRET: string;
  DAILY_ANSWER_LIMIT: string;
  ENVIRONMENT: string;
  SEC_IDENTITY: string;
  OPENROUTER_API_KEY: string;
  OPENROUTER_MODEL: string;
  METADATA_BACKEND: string;
}
