# Options Trading Copilot Backend

FastAPI backend combining option Gamma Exposure analytics, Moomoo OpenD market
data, OpenAI Responses API tool calling, Redis caching, and SQLite journaling.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python stockschedule.py
```

Open `http://127.0.0.1:8000/docs` for the API documentation.

The service falls back to deterministic mock market data when OpenD is disabled
or unavailable. Redis is optional; an in-memory TTL cache is always maintained.
Set `OPENAI_API_KEY` before calling `/api/v1/chat`.

`CORS_ORIGINS` is a comma-separated allowlist. It defaults to local frontend
development servers on ports 3000 and 5173 for both `localhost` and `127.0.0.1`.

## Main endpoints

- `GET /health`
- `GET /api/v1/gex/{ticker}?days_to_expiration=30`
- `POST /api/v1/chat`
- `POST /api/v1/plans/save`

Moomoo tickers without a market prefix are treated as US symbols, so `AAPL`
becomes `US.AAPL` for OpenD calls.
