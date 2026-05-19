<p align="center">
  <img src="fevicon.png" alt="SEC Financial Chatbot Icon" width="90"/>
</p>

<h1 align="center">SEC Financial Chatbot</h1>

<p align="center">
  A fully static React chatbot that answers questions about SEC EDGAR filings using live data and your own LLM API key. Hosted on GitHub Pages.
</p>

> ⚠️ **Looking for the original Streamlit / Python RAG implementation?** It lives on the
> [`streamlit-legacy`](https://github.com/bcastelino/sec-financial-chatbot/tree/streamlit-legacy) branch.

---

## ✨ What it does

- Chat naturally about US public-company filings (10-K, 10-Q).
- Pulls **live** structured financials from SEC's XBRL `companyfacts` API.
- Fetches narrative sections (Risk Factors, MD&A, Business, …) directly from the filing on demand.
- Sends only the retrieved context (plus your message) to the LLM you choose.
- Renders Markdown answers with inline citations linking back to sec.gov.
- 100% client-side. No backend. No database. Your API key never leaves your browser except when calling the LLM provider.

## 🧱 Architecture

```
Browser (React SPA on GitHub Pages)
   │
   ├── Ticker → CIK     ── public/data/company_tickers.json snapshot + live refresh
   ├── SEC EDGAR API    ── data.sec.gov (submissions, XBRL facts) + www.sec.gov/Archives (filing HTML)
   └── LLM API (BYOK)   ── OpenRouter or OpenAI, called directly with your key from localStorage
```

For each user turn the app:

1. Extracts intent (companies, years, forms, sections, numeric vs narrative).
2. Pulls just-enough context: XBRL facts for numbers, on-demand 10-K section text for narrative.
3. Builds a single Markdown CONTEXT block and streams a Markdown answer back from the chosen LLM.

## 🚀 Run locally

```bash
npm install
npm run dev
```

Open the printed URL (usually <http://localhost:5173/sec-financial-chatbot/>). On first send, paste an API key when prompted.

## 🌐 Deploy

This repo deploys automatically to GitHub Pages via `.github/workflows/deploy.yml` on every push to `main`.

To enable it in a fresh fork:

1. **Settings → Pages → Build and deployment → Source: GitHub Actions**.
2. Push to `main`.

The Vite `base` is set to `/sec-financial-chatbot/`; change it in `vite.config.ts` if you host under a different path.

## 🔑 BYOK (Bring Your Own Key)

The app calls the LLM provider directly from your browser. We support:

- **OpenRouter** (default) — get a key at <https://openrouter.ai/keys>. Many models, including free tiers.
- **OpenAI** — get a key at <https://platform.openai.com/api-keys>.

Your key is stored only in `localStorage` and is sent only to the provider you select.

Default model suggestions (editable from the modal):

- `openrouter/auto` — OpenRouter chooses the best model for the prompt
- `openrouter/free` — OpenRouter routes to a free-tier model
- `openai/gpt-oss-120b:free`
- `nvidia/nemotron-3-super-120b-a12b:free`
- `google/gemma-4-26b-a4b-it:free`
- `nousresearch/hermes-3-llama-3.1-405b:free`

## 📚 SEC EDGAR endpoints used

| Purpose | Endpoint |
|---|---|
| Ticker → CIK map | `https://www.sec.gov/files/company_tickers.json` |
| Filing index for a company | `https://data.sec.gov/submissions/CIK##########.json` |
| XBRL company facts | `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json` |
| Filing primary document | `https://www.sec.gov/Archives/edgar/data/<cik>/<accession-nodashes>/<file>` |

SEC asks API consumers to identify themselves and stay under ~10 req/s. Browsers can't set `User-Agent`, but we throttle to ~5 req/s and cache responses for 5 minutes. Please be respectful — don't hammer the endpoints.

### CORS / public proxy

SEC's APIs (`data.sec.gov`, `www.sec.gov`) do **not** send `Access-Control-Allow-Origin` headers, so a static SPA in a browser cannot read their responses directly. The app routes all SEC requests through a public CORS proxy by default:

```
https://corsproxy.io/?url=<sec-url>
```

You can override it from DevTools:

```js
localStorage.setItem('sec-chat:proxy-base', 'https://your-proxy.example.com/?url=')
// or to disable the proxy entirely (only works locally with a browser CORS extension):
localStorage.setItem('sec-chat:proxy-base', '')
```

For production / heavy use, deploy your own proxy (e.g. a tiny Cloudflare Worker that forwards GETs to `data.sec.gov` and adds CORS headers) and point the app at it.

### Known limits

- **CORS on `www.sec.gov/Archives`**: filing HTML is fetched via the proxy too, but may occasionally return slowly or fail under proxy load. When that happens you'll see a warning in the answer and only XBRL-based context will be used. Numerical questions still work.
- **Section extraction is heuristic** — it locates Items by scanning for headings. Some filers use unusual TOC structures and a section may come back partial.
- **LLM context window**: section text is truncated to ~12k chars per section. Ask narrower questions for higher-fidelity answers.

## 💡 Example questions

- *What was AAPL revenue in 2022, 2023, and 2024?*
- *Compare MSFT and GOOGL net income for FY2023.*
- *Summarize NVDA risk factors from the latest 10-K.*
- *What does AMZN's MD&A say about AWS margins?*

## 🛠 Project structure

```
sec-financial-chatbot/
├── .github/workflows/deploy.yml      GitHub Pages deploy
├── public/
│   ├── fevicon.png
│   └── data/company_tickers.json     ticker → CIK snapshot
├── src/
│   ├── App.tsx                       top-level state machine
│   ├── main.tsx                      Vite entry
│   ├── components/                   UI (Landing, ChatRoom, ChatInput, ApiKeyModal, …)
│   ├── lib/
│   │   ├── sec/                      tickers, submissions, facts, filingDoc, rateLimiter
│   │   ├── llm/                      OpenRouter streaming, prompt builder, intent extractor
│   │   └── storage.ts                localStorage helpers
│   └── styles/globals.css            Tailwind + chat markdown styling
├── index.html
├── tailwind.config.ts
├── vite.config.ts                    base: '/sec-financial-chatbot/'
└── tsconfig.json
```

## 📄 License

MIT — see [`LICENSE`](LICENSE).

## 🐱‍👤 Author

Built by **Brian Denis Castelino**. Original Streamlit + RAG implementation is preserved on the [`streamlit-legacy`](https://github.com/bcastelino/sec-financial-chatbot/tree/streamlit-legacy) branch.
