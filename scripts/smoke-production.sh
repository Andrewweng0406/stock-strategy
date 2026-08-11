#!/usr/bin/env bash
set -euo pipefail

BACKEND_URL="${BACKEND_URL:-https://backend-production-0e24.up.railway.app}"
FRONTEND_URL="${FRONTEND_URL:-https://frontend-production-e380d.up.railway.app}"
SMOKE_TICKER="${SMOKE_TICKER:-AAPL}"
SMOKE_DTE="${SMOKE_DTE:-11}"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

fail() {
  echo "smoke failed: $*" >&2
  exit 1
}

echo "Checking backend health at ${BACKEND_URL}/health"
health_file="$tmpdir/health.json"
health_headers="$tmpdir/health.headers"
curl -fsS -D "$health_headers" "${BACKEND_URL}/health" -o "$health_file"

python3 - "$health_file" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    payload = json.load(fh)

status = payload.get("status")
mode = payload.get("market_data_mode")
cloud_sync = payload.get("cloud_sync")
if status != "ok":
    raise SystemExit(f"backend health status is {status!r}")
if mode in {"mock", "unavailable", None, ""}:
    raise SystemExit(f"backend market_data_mode is not production-safe: {mode!r}")
if not isinstance(cloud_sync, dict):
    raise SystemExit(f"cloud_sync health is not an object: {cloud_sync!r}")
expected_sync_keys = {"enabled", "last_success_at", "last_error_at", "last_error"}
if set(cloud_sync) != expected_sync_keys:
    raise SystemExit(f"cloud_sync health keys changed: {sorted(cloud_sync)}")
if "token" in json.dumps(payload).lower():
    raise SystemExit("backend health response appears to expose token material")
print(f"backend market data mode: {mode}")
PY

python3 - "$health_headers" <<'PY'
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    raw = fh.read()

headers = {}
for line in raw.replace("\r\n", "\n").split("\n"):
    if ":" not in line:
        continue
    key, value = line.split(":", 1)
    headers[key.strip().lower()] = value.strip()

expected = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}
for header, value in expected.items():
    actual = headers.get(header.lower())
    if actual != value:
        raise SystemExit(f"{header} is {actual!r}; expected {value!r}")
print("backend security headers ok")
PY

echo "Checking trusted GEX response for ${SMOKE_TICKER} ${SMOKE_DTE}DTE"
gex_file="$tmpdir/gex.json"
curl -fsS "${BACKEND_URL}/api/v1/gex/${SMOKE_TICKER}?days_to_expiration=${SMOKE_DTE}" -o "$gex_file"

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

source = payload.get("data_source")
if source in {"MOCK", None, ""}:
    raise SystemExit(f"GEX payload is not production-safe data_source={source!r}")
if payload.get("is_synthetic") is not False:
    raise SystemExit(f"GEX payload must explicitly be non-synthetic: {payload.get('is_synthetic')!r}")
if source == "YFINANCE" and payload.get("is_delayed") is not True:
    raise SystemExit("YFINANCE payload must explicitly be marked delayed")

price = payload.get("stock_price")
if not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
    raise SystemExit(f"invalid stock_price: {price!r}")

net_gex = payload.get("net_gex")
if not isinstance(net_gex, (int, float)) or not math.isfinite(net_gex):
    raise SystemExit(f"invalid net_gex: {net_gex!r}")

pinning = payload.get("pinning")
if not isinstance(pinning, dict) or "score" not in pinning or "regime" not in pinning:
    raise SystemExit(f"invalid pinning object: {pinning!r}")

print(
    "GEX payload ok: "
    f"{payload.get('ticker')} price={price} net_gex={net_gex} "
    f"status={payload.get('gex_status')} source={source}"
)
PY

echo "Checking sync endpoint rejects unauthenticated writes"
sync_status="$(
  curl -sS -o "$tmpdir/sync.json" -w "%{http_code}" \
    -H "content-type: application/json" \
    -X POST "${BACKEND_URL}/api/v1/sync/gex" \
    --data '{"ticker":"SMOKE","days_to_expiration":1,"payload":{"source":"smoke"}}'
)"
if [[ "$sync_status" != "403" ]]; then
  fail "sync endpoint returned HTTP ${sync_status}; expected 403 without token"
fi

echo "Checking frontend shell at ${FRONTEND_URL}"
frontend_file="$tmpdir/frontend.html"
curl -fsS "${FRONTEND_URL}/" -o "$frontend_file"
grep -q "/assets/" "$frontend_file" || fail "frontend HTML does not reference built assets"

echo "Production smoke passed"
