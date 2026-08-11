# Options Trading Copilot

FastAPI backend combining option Gamma Exposure (GEX) analytics, an AI trading
copilot (OpenAI Responses API tool calling), and a React frontend. Ships two
market data modes so it can run fully local (real broker data) or in the
cloud (no credentials required):

- **Local**: Moomoo OpenD, real-time quotes and option chains, requires a
  logged-in OpenD instance and brokerage credentials.
- **Cloud**: yfinance, no login required, ~15-20 min delayed data, no
  broker-calculated Greeks (the backend computes gamma via Black-Scholes
  itself in that case).
- **Synthetic/demo**: deterministic mock data is available only when
  `SYNTHETIC_MARKET_DATA_ENABLED=true`. Production should keep it disabled;
  if trusted market data is unavailable, the API fails closed with 503 rather
  than showing fake levels.

An optional **CloudSync** mechanism lets a local instance (with real Moomoo
data) push its computed GEX summaries to a cloud deployment's cache, without
ever sending Moomoo credentials to the cloud — see `CLOUD_SYNC_URL` below.

## Run the backend locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python stockschedule.py
```

Open `http://127.0.0.1:8002/docs` for the API documentation. Override the
development port with `PORT=8000 python stockschedule.py` when needed.

Redis is optional; an in-memory TTL cache is always maintained. Set
`OPENAI_API_KEY` before calling `/api/v1/chat`. `DATABASE_URL` accepts either
SQLite (`sqlite+aiosqlite:///...`, default, good for local dev) or Postgres
(`postgresql+asyncpg://...`, used for the cloud deployment's persistent
storage — chat history, trade journal, profile memory, GEX snapshots).

`CORS_ORIGINS` is a comma-separated allowlist. It defaults to local frontend
development servers on ports 3000 and 5173 for both `localhost` and
`127.0.0.1`.

## Run the frontend locally

```bash
cd web
npm install
npm run dev
```

Opens on `http://127.0.0.1:5173` and talks to the backend via
`VITE_API_BASE_URL`, which defaults to `http://127.0.0.1:8002`, the same port
used by `stockschedule.py`.

## Verify local Moomoo mode

Before trusting local GEX levels, start Moomoo OpenD, keep `MOOMOO_ENABLED=true`,
run the backend, then execute:

```bash
scripts/smoke-local-moomoo.sh
```

This is intentionally stricter than the cloud smoke: it fails unless
`GET /health` reports `market_data_mode: "moomoo"`, then verifies real
expirations and a Moomoo-backed GEX payload for `SMOKE_TICKER` (default
`AAPL`). Override `BACKEND_URL` or `SMOKE_TICKER` when needed.

## Key environment variables

See `.env.example` for the vars it lists, and `app/config.py` for the full
set (everything has a sane default, so `.env.example` only lists the ones
you're likely to actually change). Worth understanding:

- `MOOMOO_ENABLED` — set `false` on any deployment that shouldn't (or can't)
  hold brokerage credentials, e.g. the cloud instance. Falls back to
  yfinance when `YFINANCE_FALLBACK_ENABLED=true`.
- `MOOMOO_OPTION_CHAIN_MAX_CALLS` / `MOOMOO_OPTION_CHAIN_WINDOW_SECONDS` —
  backend-side guard for Moomoo/Futu's option-chain quota. Defaults to
  `8 / 30s` to stay below the documented `10 / 30s` cap.
- `SYNTHETIC_MARKET_DATA_ENABLED` — explicit demo/sandbox opt-in for
  deterministic mock data. Keep `false` for any user-facing paid product.
- `CLOUD_SYNC_URL` / `SYNC_TOKEN` — set on the **local** instance only, to
  push real GEX summaries to a cloud deployment's cache. Never set
  `MOOMOO_*` credentials on the cloud instance itself.
- `CHAT_RATE_LIMIT` — per-client-IP rate limit on `/api/v1/chat` (the only
  endpoint that spends OpenAI budget per call), e.g. `10/minute`.
- `SNAPSHOT_INTERVAL_SECONDS` — throttle for how often a real GEX calculation
  gets persisted to the `gex_snapshots` history table per ticker.

## Main endpoints

- `GET /health` — reports `market_data_mode` (`moomoo` / `yfinance` / `mock` / `unavailable`)
- `GET /api/v1/gex/{ticker}?days_to_expiration=30` — single-expiration GEX summary
- `GET /api/v1/gex/{ticker}/aggregate?expirations=...` — aggregate GEX across up to 6 expirations
- `GET /api/v1/gex/{ticker}/history?limit=100` — persisted GEX snapshot history
- `GET /api/v1/expirations/{ticker}` — available option expirations
- `POST /api/v1/chat` — AI copilot chat turn (GEX-grounded advice, trade plan cards)
- `GET /api/v1/conversations` / `GET /api/v1/conversations/{id}/messages` — chat history
- `GET /api/v1/plans` / `POST /api/v1/plans/save` — saved trade plans (trade journal)
- `GET|PUT /api/v1/profile/{user_id}` — saved risk tolerance / preferences
- `POST /api/v1/sync/gex`, `/sync/expirations`, `/sync/gex/aggregate` — CloudSync push targets (token-protected, local → cloud only)

Moomoo tickers without a market prefix are treated as US symbols, so `AAPL`
becomes `US.AAPL` for OpenD calls.

## Tests

```bash
PYTHONPATH=. pytest -q
cd web
npm run build
npm run test:e2e
npm audit --omit=optional
```

GitHub Actions runs the same product gate on every push to `main` and every
pull request: backend compile + pytest, frontend production build, Playwright
browser regressions, and dependency audit. CI deliberately disables Moomoo,
yfinance fallback, Redis, and synthetic market data so tests cannot pass by
depending on a local broker session, a network data source, or fake levels.

## Production smoke

Before or after a Railway release, run the read-only production gate:

```bash
scripts/smoke-production.sh
```

It verifies the backend is healthy and not serving `mock` / `unavailable`
market data, fetches a real GEX payload, confirms the CloudSync write endpoint
rejects unauthenticated requests, and checks the frontend shell serves built
assets. Override `BACKEND_URL`, `FRONTEND_URL`, `SMOKE_TICKER`, or `SMOKE_DTE`
when testing another environment.
