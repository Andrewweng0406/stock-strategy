# Trade Journal & AI Post-Trade Review — Design Spec

Status: Approved. Scope: Sub-project 1 of 2 (git2threads dashboard integration
is a separate, later sub-project — see Out of Scope).

## Purpose

Close the loop between GEX analysis, live trading, and post-trade reflection.
Users log actual executed trades (distinct from the AI-proposed trade plan
cards already produced by chat), optionally close them out, and trigger an
AI-written post-trade review that is grounded in the GEX context captured at
entry and a deterministically computed execution-discipline score.

## Relationship to existing systems

- **`trade_plans` (existing)**: an AI-proposed trade idea (entry/stop/target,
  never executed or not yet executed). Unrelated to whether a real trade
  happened.
- **`trades` (new)**: a real executed position, optionally originating from a
  `trade_plans` row via `source_plan_id` (nullable — freehand manual entries
  are equally valid).
- **`gex_snapshots` (existing)**: historical GEX readings, currently written
  on a best-effort basis (throttled to at most once/hour per ticker by the
  poller). Trade creation writes its own snapshot on demand instead of
  relying on that throttled history, so entry context is always present and
  always accurate to the moment of entry.

## Data Model

Both new tables use plain SQLAlchemy Core types (`String`, `Integer`,
`Float`, `DateTime(timezone=True)`, `Text`) exactly like every existing table
in `app/database.py`, so they work unchanged under both the local SQLite
dev database and the cloud Postgres deployment. No Postgres-only types.

### `trades`

| column | type | notes |
|---|---|---|
| `id` | `String(36)` PK | uuid4, matches `trade_plans.plan_id` convention |
| `user_id` | `String(128)`, indexed | |
| `ticker` | `String(32)`, indexed | |
| `strategy_type` | `String(128)` | e.g. "Bull Put Spread", "Long Call" |
| `source_plan_id` | `String(36)`, nullable, FK → `trade_plans.plan_id` | |
| `entry_date` | `DateTime(timezone=True)` | |
| `exit_date` | `DateTime(timezone=True)`, nullable | |
| `entry_price` | `Float` | |
| `exit_price` | `Float`, nullable | |
| `position_size` | `Integer` | contracts |
| `pnl` | `Float`, nullable | user-entered dollar P/L at close (strategy-specific math is not derived — see Open Question resolution below) |
| `pnl_pct` | `Float`, nullable | backend-computed: `pnl / (entry_price * position_size) * 100` |
| `status` | `String(16)`, indexed | `OPEN` \| `CLOSED` |
| `notes` | `Text`, nullable | |
| `entry_gex_snapshot_id` | `Integer`, nullable, FK → `gex_snapshots.id` | |
| `created_at` | `DateTime(timezone=True)` | |

### `trade_reviews`

| column | type | notes |
|---|---|---|
| `id` | `String(36)` PK | uuid4 |
| `trade_id` | `String(36)`, FK → `trades.id`, unique | one row per trade; re-triggering overwrites |
| `execution_score` | `Integer` | 1-5, backend-computed, never AI-generated |
| `ai_feedback` | `Text` | AI-generated prose |
| `key_takeaways` | `Text` | AI-generated, stored as a JSON-encoded string array |
| `created_at` | `DateTime(timezone=True)` | |

Pydantic models (`app/models.py`): `TradeStatus` (str enum: `OPEN`, `CLOSED`),
`Trade`, `TradeCreate`, `TradeClose`, `TradeList`, `TradeReview`,
`TradeReviewRequest` (empty body — trade_id comes from the path).

## Backend API

All endpoints follow the existing `Services` dependency-injection pattern in
`app/main.py` and the existing repository pattern in `app/database.py`
(`TradeRepository`, `TradeReviewRepository`).

### `POST /api/v1/trades`

Body: `TradeCreate` (`user_id`, `ticker`, `strategy_type`, `source_plan_id?`,
`entry_price`, `position_size`, `notes?`, `days_to_expiration` — used only to
fetch the entry GEX snapshot, defaults to 30). Note: `trade_plans` does not
itself store a DTE, so this is always an explicit request field, never
derived from `source_plan_id`.

On create: call `services.gex_service.get_summary(ticker, days_to_expiration)`
synchronously, persist it via `GEXSnapshotRepository.save_snapshot(...)`
(bypassing the poller's hourly throttle — this call always writes), and
record the resulting snapshot's `id` as `entry_gex_snapshot_id`. Status is
always created as `OPEN`. Returns `Trade`.

### `GET /api/v1/trades?user_id=&ticker=&status=`

`ticker` and `status` are optional filters. Returns `TradeList`.

### `PUT /api/v1/trades/{id}`

Body: `TradeClose` (`exit_price`, `exit_date`, `pnl`, `notes?`). Loads the
trade, verifies `user_id` ownership (same mismatch-raises-`PermissionError`
pattern as `PlanRepository.save_signed_plan`, caught in `main.py` and
mapped to `403` exactly like the existing `/api/v1/plans/save` handler).
Computes `pnl_pct`, sets `status = CLOSED`. Returns updated `Trade`. `404`
if the trade doesn't exist; `409` if it's already `CLOSED` — closing is
one-way in this scope, no reopening.

### `POST /api/v1/trades/{id}/review`

`404` if the trade doesn't exist; `403` if it exists but belongs to a
different `user_id` (same mapping as `PUT /api/v1/trades/{id}` above); `400`
if `status != CLOSED` (reviews only make sense post-close). Steps:

1. **Compute `execution_score` in the backend** (see Scoring below) —
   never delegated to the model.
2. **Call `LLMOrchestrator.review_trade(...)`**: a new orchestrator method
   mirroring the existing forced-tool-call pattern used for
   `generate_trade_plan`. Uses `tool_choice={"type": "function", "name":
   "submit_trade_review"}` with a tool schema of exactly
   `{ai_feedback: str, key_takeaways: list[str]}` — the score is *not* a tool
   argument; it's computed already and passed into the prompt as context for
   the model to explain, not invent. Input context: entry/exit price,
   position_size, strategy_type, entry snapshot's Zero Gamma/Call Wall/Put
   Wall/GEX status, notes, the computed `execution_score`, and whether a
   `source_plan_id` was present (so the model can say plainly when there was
   no predefined plan to grade against, rather than implying there was).
3. Upsert into `trade_reviews` (overwrite on re-trigger, per the `unique`
   constraint on `trade_id`).
4. Return `TradeReview`.

## Execution Score (backend-computed, 1-5)

**With `source_plan_id`** (the linked `trade_plans` row has `stop_loss` and
`target_price`):

```
direction     = BULLISH if target_price > entry_price else BEARISH
planned_risk  = abs(entry_price - stop_loss)
planned_reward = abs(target_price - entry_price)
planned_rr    = planned_reward / planned_risk   (guard: planned_risk == 0 → treat as no-plan fallback)
realized_move = (exit_price - entry_price) if BULLISH else (entry_price - exit_price)
r_multiple    = realized_move / planned_risk

r_multiple >= planned_rr        → 5  (met or beat the planned target)
0 < r_multiple < planned_rr     → 4  (profitable, plan respected, target not fully reached)
-1 <= r_multiple <= 0           → 3  (breakeven/small loss, stayed within planned risk)
-1.5 <= r_multiple < -1         → 2  (stop discipline slipped moderately past plan)
r_multiple < -1.5               → 1  (loss ran well past the planned stop)
```

**Without a usable `source_plan_id`** (fallback, capped at 4 — "5" requires a
verifiable predefined plan that was met):

```
pnl_pct >= 15        → 4
0 <= pnl_pct < 15     → 3
-10 <= pnl_pct < 0    → 2
pnl_pct < -10         → 1
```

The prompt passed to `review_trade()` always states explicitly which path
was used, so the AI's prose never implies a plan comparison that didn't
happen.

## Frontend (existing Vite/React app, `web/`)

New `web/src/TradeJournal.jsx`, added as a tab/route alongside the existing
`TradingTerminalNotebook.jsx`, reusing its existing `BASE_URL` /
fetch-wrapper conventions.

- Quick-add card: ticker, strategy_type, entry_price, position_size, notes,
  optional link to an existing saved plan (dropdown sourced from
  `GET /api/v1/plans`).
- Open trades list: each card has a "平倉" action revealing exit_price/pnl
  inputs; submitting computes and displays `pnl_pct` immediately from the
  `PUT` response.
- Closed trades list: each card shows PnL, and a "🤖 觸發 AI 覆盤分析" button
  that calls `POST /api/v1/trades/{id}/review` and expands the returned
  star rating + `ai_feedback` + `key_takeaways` inline. Re-clicking
  re-triggers and replaces the prior review (matches the upsert behavior).

## Testing Plan

Backend (`tests/test_database.py` or a new `tests/test_trade_journal.py`,
plus `tests/test_openai_orchestrator.py` additions):

- `pnl_pct` computed correctly on close; `pnl`/`pnl_pct` remain `None` while
  `OPEN`.
- Creating a trade writes a fresh `gex_snapshots` row and links its id,
  independent of the poller's last-write throttle (i.e. two trades created
  back-to-back within the same hour each still get their own snapshot).
- Ownership check: closing or reviewing another user's trade raises/returns
  a permission error, mirroring the existing `PlanRepository` test.
- Reviewing a non-`CLOSED` trade returns 400; reviewing a nonexistent trade
  returns 404.
- Execution score, scored path: table-driven tests across the five score
  bands using `source_plan_id`-linked trades with known
  entry/stop/target/exit combinations.
- Execution score, fallback path: table-driven tests across the four
  `pnl_pct` bands with no `source_plan_id`.
- `review_trade()` forces the `submit_trade_review` tool call and parses its
  arguments into `TradeReview`, mirroring
  `test_selected_put_forces_tool_and_builds_card`.
- Re-triggering a review on the same trade overwrites the existing
  `trade_reviews` row rather than inserting a second one.

Frontend: manual smoke test via the running dev server (create → close →
trigger review → confirm the star rating and prose render), per this
project's existing "start the dev server and use the feature" convention
for UI changes — no new automated frontend test tooling introduced.

## Out of Scope (Sub-project 2, separate spec later)

git2threads dashboard integration (Task 4 of the original request): a
"包含今日交易覆盤紀錄" checkbox that pulls today's trade reviews into a
generated Threads post. Deferred until this core journal exists and until
git2threads's actual location/API/auth model is understood — it is not
part of this repository today.
