import { expect, test } from "@playwright/test";

const API_BASE = "http://127.0.0.1:8002";

const gexSummary = {
  ticker: "AAPL",
  stock_price: 230.5,
  zero_gamma: 225,
  call_wall: 240,
  put_wall: 215,
  iv_rank: 22,
  net_gex: 12345678,
  gex_status: "POS_GAMMA",
  calculated_at: "2026-08-10T12:00:00Z",
  pinning: {
    pin_strike: 230,
    pin_strike_matches_max_pain: false,
    distance_pct: 0.2,
    oi_concentration_pct: 8,
    in_positive_gamma: true,
    has_broken_wall: false,
    score: 70,
    label: "高",
    regime: "NEUTRAL",
  },
};

function corsHeaders() {
  return {
    "access-control-allow-origin": "*",
    "access-control-allow-methods": "GET,POST,PUT,OPTIONS",
    "access-control-allow-headers": "content-type",
  };
}

async function installApiStub(page, { initialTrades = [] } = {}) {
  const trades = [...initialTrades];
  await page.route(`${API_BASE}/**`, async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (request.method() === "OPTIONS") {
      await route.fulfill({ status: 204, headers: corsHeaders() });
      return;
    }

    const json = async (body, status = 200) =>
      route.fulfill({
        status,
        headers: { "content-type": "application/json", ...corsHeaders() },
        body: JSON.stringify(body),
      });

    if (url.pathname === "/health") {
      await json({ status: "ok", market_data_mode: "moomoo" });
      return;
    }
    if (url.pathname === "/api/v1/expirations/AAPL") {
      await json({
        ticker: "AAPL",
        expirations: [
          { date: "2026-08-10", days_to_expiration: 0, expiration_type: "0DTE" },
          { date: "2026-08-21", days_to_expiration: 11, expiration_type: "MONTHLY" },
        ],
      });
      return;
    }
    if (url.pathname === "/api/v1/gex/AAPL") {
      await json(gexSummary);
      return;
    }
    if (url.pathname === "/api/v1/gex/AAPL/history") {
      await json({ ticker: "AAPL", snapshots: [] });
      return;
    }
    if (url.pathname === "/api/v1/conversations") {
      await json({ conversations: [] });
      return;
    }
    if (url.pathname === "/api/v1/plans") {
      await json({ plans: [] });
      return;
    }
    if (url.pathname.startsWith("/api/v1/profile/")) {
      await json({
        user_id: url.pathname.split("/").pop(),
        risk_tolerance: null,
        preferred_strategy_types: [],
        notes: "",
      });
      return;
    }
    if (url.pathname === "/api/v1/trades" && request.method() === "GET") {
      await json({ trades });
      return;
    }
    if (url.pathname === "/api/v1/trades" && request.method() === "POST") {
      const payload = request.postDataJSON();
      if (!payload.expiration_date) {
        await json({ detail: "expiration_date is required" }, 422);
        return;
      }
      const trade = {
        id: `trade-${trades.length + 1}`,
        user_id: payload.user_id,
        ticker: payload.ticker,
        strategy_type: payload.strategy_type,
        direction: payload.direction,
        credit_debit: payload.credit_debit,
        expiration_date: payload.expiration_date,
        option_type: payload.option_type,
        strike_price: payload.strike_price,
        contract_symbol: payload.contract_symbol,
        legs: payload.legs || [],
        source_plan_id: payload.source_plan_id,
        entry_date: payload.entry_date || "2026-08-10T12:00:00Z",
        exit_date: null,
        entry_price: payload.entry_price,
        exit_price: null,
        position_size: payload.position_size,
        pnl: null,
        pnl_pct: null,
        status: "OPEN",
        notes: payload.notes,
        entry_gex_snapshot_id: 10,
        created_at: "2026-08-10T12:00:00Z",
      };
      trades.unshift(trade);
      await json(trade);
      return;
    }

    await json({ detail: `Unhandled ${request.method()} ${url.pathname}` }, 500);
  });
}

test("trade journal records and displays option expiration for a new single-leg trade", async ({ page }) => {
  await installApiStub(page);

  await page.goto("/");
  await page.getByTitle("交易日誌").click();

  await expect(page.getByRole("button", { name: /新增交易/ })).toBeVisible();
  await expect(page.locator('input[type="date"]').first()).toHaveValue("2026-08-21");

  await page.getByPlaceholder("策略類型（可挑選或自行輸入）").fill("Long Call");
  await page.getByPlaceholder("履約價").fill("450");
  await page.getByPlaceholder("進場價").fill("5.25");
  await page.getByRole("button", { name: /新增交易/ }).click();

  const card = page.locator("text=AAPL · Long Call").locator("..").locator("..");
  await expect(card).toContainText("到期 8/21");
  await expect(card).toContainText("$450 Call");
  await expect(card).not.toContainText("到期 —");
});

test("legacy trades without expiration are explicitly marked instead of looking valid", async ({ page }) => {
  await installApiStub(page, {
    initialTrades: [
      {
        id: "legacy-trade",
        user_id: "web-client",
        ticker: "AAPL",
        strategy_type: "Long Put",
        direction: "SHORT",
        credit_debit: "DEBIT",
        expiration_date: null,
        option_type: "PUT",
        strike_price: 210,
        contract_symbol: null,
        legs: [],
        source_plan_id: null,
        entry_date: "2026-08-07T17:26:00Z",
        exit_date: null,
        entry_price: 3.1,
        exit_price: null,
        position_size: 1,
        pnl: null,
        pnl_pct: null,
        status: "OPEN",
        notes: null,
        entry_gex_snapshot_id: null,
        created_at: "2026-08-07T17:26:00Z",
      },
    ],
  });

  await page.goto("/");
  await page.getByTitle("交易日誌").click();

  const card = page.locator("text=AAPL · Long Put").locator("..").locator("..");
  await expect(card).toContainText("到期未記錄");
  await expect(card).not.toContainText("到期 —");
});
