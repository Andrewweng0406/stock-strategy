#!/usr/bin/env bash
set -euo pipefail

BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8002}"
SMOKE_TICKER="${SMOKE_TICKER:-AAPL}"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

fail() {
  echo "local moomoo smoke failed: $*" >&2
  exit 1
}

echo "Checking local backend health at ${BACKEND_URL}/health"
health_file="$tmpdir/health.json"
curl -fsS "${BACKEND_URL}/health" -o "$health_file" || fail "backend is not reachable"

python3 - "$health_file" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    payload = json.load(fh)

status = payload.get("status")
mode = payload.get("market_data_mode")
if status != "ok":
    raise SystemExit(f"backend health status is {status!r}")
if mode != "moomoo":
    raise SystemExit(
        "local backend is not using Moomoo real-time data "
        f"(market_data_mode={mode!r}). Start Moomoo OpenD and run the backend "
        "with MOOMOO_ENABLED=true before trusting local GEX values."
    )
if "token" in json.dumps(payload).lower():
    raise SystemExit("health response appears to expose token material")
print("local backend mode: moomoo")
PY

echo "Checking real expirations for ${SMOKE_TICKER}"
exp_file="$tmpdir/expirations.json"
curl -fsS "${BACKEND_URL}/api/v1/expirations/${SMOKE_TICKER}" -o "$exp_file" \
  || fail "expiration request failed"

smoke_dte="$(
  python3 - "$exp_file" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    payload = json.load(fh)

expirations = payload.get("expirations")
if not isinstance(expirations, list) or not expirations:
    raise SystemExit(f"no expirations returned: {payload!r}")

first = expirations[0]
dte = first.get("days_to_expiration")
date = first.get("date")
if not isinstance(dte, int) or dte < 0:
    raise SystemExit(f"invalid first expiration: {first!r}")
if not isinstance(date, str) or not date:
    raise SystemExit(f"invalid first expiration date: {first!r}")
print(dte)
PY
)"

echo "Checking Moomoo-backed GEX for ${SMOKE_TICKER} ${smoke_dte}DTE"
gex_file="$tmpdir/gex.json"
curl -fsS "${BACKEND_URL}/api/v1/gex/${SMOKE_TICKER}?days_to_expiration=${smoke_dte}" \
  -o "$gex_file" || fail "GEX request failed"

python3 - "$gex_file" <<'PY'
import json
import math
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    payload = json.load(fh)

required = ["ticker", "stock_price", "net_gex", "gex_status", "pinning"]
missing = [key for key in required if key not in payload]
if missing:
    raise SystemExit(f"GEX payload missing keys: {missing}")

price = payload.get("stock_price")
net_gex = payload.get("net_gex")
if not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
    raise SystemExit(f"invalid stock_price: {price!r}")
if not isinstance(net_gex, (int, float)) or not math.isfinite(net_gex):
    raise SystemExit(f"invalid net_gex: {net_gex!r}")
if payload.get("gex_status") not in {"POS_GAMMA", "NEG_GAMMA"}:
    raise SystemExit(f"invalid gex_status: {payload.get('gex_status')!r}")
if not isinstance(payload.get("pinning"), dict):
    raise SystemExit(f"pinning analysis missing: {payload.get('pinning')!r}")
if payload.get("data_source") != "MOOMOO":
    raise SystemExit(f"expected MOOMOO payload data_source, got {payload.get('data_source')!r}")
if payload.get("is_delayed") is not False:
    raise SystemExit(f"Moomoo payload must not be delayed: {payload.get('is_delayed')!r}")
if payload.get("is_synthetic") is not False:
    raise SystemExit(f"Moomoo payload must not be synthetic: {payload.get('is_synthetic')!r}")
if payload.get("is_stale") is not False:
    raise SystemExit(f"Moomoo payload must be live, not stale: {payload.get('is_stale')!r}")

print(
    "Moomoo GEX payload ok: "
    f"{payload.get('ticker')} price={price} net_gex={net_gex} "
    f"status={payload.get('gex_status')} source={payload.get('data_source')} stale={payload.get('is_stale')}"
)
PY

echo "Local Moomoo smoke passed"
