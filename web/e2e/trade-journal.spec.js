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

async function installApiStub(
  page,
  { initialTrades = [], onCreateTrade = () => {}, onCloseTrade = () => {} } = {}
) {
  const trades = [...initialTrades];
  const reviews = new Map();
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
      await json({
        status: "ok",
        market_data_mode: "moomoo",
        cloud_sync: {
          enabled: false,
          last_success_at: null,
          last_error_at: null,
          last_error: null,
        },
      });
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
      onCreateTrade(payload);
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
    const tradeMatch = url.pathname.match(/^\/api\/v1\/trades\/([^/]+)$/);
    if (tradeMatch && request.method() === "PUT") {
      const trade = trades.find((t) => t.id === tradeMatch[1]);
      if (!trade) {
        await json({ detail: "Trade not found" }, 404);
        return;
      }
      const payload = request.postDataJSON();
      onCloseTrade(payload, trade);
      const pnlPct = (payload.pnl / (trade.entry_price * 100 * trade.position_size)) * 100;
      Object.assign(trade, {
        exit_price: payload.exit_price,
        exit_date: payload.exit_date,
        pnl: payload.pnl,
        pnl_pct: pnlPct,
        status: "CLOSED",
      });
      await json(trade);
      return;
    }
    const reviewMatch = url.pathname.match(/^\/api\/v1\/trades\/([^/]+)\/review$/);
    if (reviewMatch && request.method() === "GET") {
      await json(reviews.get(reviewMatch[1]) || null);
      return;
    }
    if (reviewMatch && request.method() === "POST") {
      const trade = trades.find((t) => t.id === reviewMatch[1]);
      if (!trade) {
        await json({ detail: "Trade not found" }, 404);
        return;
      }
      if (trade.status !== "CLOSED") {
        await json({ detail: "Trade must be closed before review" }, 400);
        return;
      }
      const review = {
        trade_id: trade.id,
        execution_score: 4,
        ai_feedback: "Good exit discipline. Position sizing and contract identity are clear.",
        key_takeaways: ["Closed with a positive R/R profile", "Contract metadata was preserved"],
        created_at: "2026-08-10T13:00:00Z",
      };
      reviews.set(trade.id, review);
      await json(review);
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

test("single-leg trade can be closed and reviewed with correct debit PnL display", async ({ page }) => {
  const trade = {
    id: "open-debit-trade",
    user_id: "web-client",
    ticker: "AAPL",
    strategy_type: "Long Call",
    direction: "LONG",
    credit_debit: "DEBIT",
    expiration_date: "2026-08-21",
    option_type: "CALL",
    strike_price: 450,
    contract_symbol: "AAPL260821C00450000",
    legs: [],
    source_plan_id: null,
    entry_date: "2026-08-10T12:00:00Z",
    exit_date: null,
    entry_price: 5.25,
    exit_price: null,
    position_size: 2,
    pnl: null,
    pnl_pct: null,
    status: "OPEN",
    notes: null,
    entry_gex_snapshot_id: 10,
    created_at: "2026-08-10T12:00:00Z",
  };
  let closePayload = null;
  await installApiStub(page, {
    initialTrades: [trade],
    onCloseTrade: (payload) => {
      closePayload = payload;
    },
  });

  await page.goto("/");
  await page.getByTitle("交易日誌").click();

  const openCard = page.locator("text=AAPL · Long Call").locator("..").locator("..");
  await openCard.getByRole("button", { name: "平倉" }).click();
  await page.getByPlaceholder("出場價").fill("7.25");
  await expect(openCard).toContainText("$400（+38.10%）");
  await openCard.getByRole("button", { name: "套用" }).click();
  await expect(page.getByPlaceholder("損益（$）")).toHaveValue("400.00");
  await openCard.getByRole("button", { name: "確認平倉" }).click();

  expect(closePayload.exit_price).toBe(7.25);
  expect(closePayload.pnl).toBe(400);
  const closedCard = page.locator("text=AAPL · Long Call").locator("..").locator("..");
  await expect(closedCard).toContainText("+38.10%");
  await expect(closedCard).toContainText("損益 $400");

  await closedCard.getByRole("button", { name: /觸發 AI 覆盤分析/ }).click();
  await expect(closedCard).toContainText("Good exit discipline");
  await expect(closedCard).toContainText("Contract metadata was preserved");
});

test("credit close math uses entry minus exit before applying contract multiplier", async ({ page }) => {
  const trade = {
    id: "open-credit-trade",
    user_id: "web-client",
    ticker: "AAPL",
    strategy_type: "Bull Put Credit Spread",
    direction: "LONG",
    credit_debit: "CREDIT",
    expiration_date: "2026-08-21",
    option_type: "PUT",
    strike_price: 210,
    contract_symbol: null,
    legs: [],
    source_plan_id: null,
    entry_date: "2026-08-10T12:00:00Z",
    exit_date: null,
    entry_price: 2.5,
    exit_price: null,
    position_size: 2,
    pnl: null,
    pnl_pct: null,
    status: "OPEN",
    notes: null,
    entry_gex_snapshot_id: 10,
    created_at: "2026-08-10T12:00:00Z",
  };
  await installApiStub(page, { initialTrades: [trade] });

  await page.goto("/");
  await page.getByTitle("交易日誌").click();

  const card = page.locator("text=AAPL · Bull Put Credit Spread").locator("..").locator("..");
  await card.getByRole("button", { name: "平倉" }).click();
  await page.getByPlaceholder("出場價").fill("0.50");

  await expect(card).toContainText("$400（+80.00%）");
  await expect(card).toContainText("($2.5 收取 − $0.5 買回) × 100 股/口 × 2 口");
});

test("multi-leg trade preserves manual direction, credit/debit, and every leg in payload", async ({ page }) => {
  let createPayload = null;
  await installApiStub(page, {
    onCreateTrade: (payload) => {
      createPayload = payload;
    },
  });

  await page.goto("/");
  await page.getByTitle("交易日誌").click();

  await page.getByPlaceholder("策略類型（可挑選或自行輸入）").fill("Ratio Spread");
  await page.locator("select").nth(1).selectOption("NEUTRAL");
  await page.locator("select").nth(2).selectOption("CREDIT");
  await page.locator("select").nth(3).selectOption("MULTI_LEG");

  const legDates = page.locator('input[type="date"]');
  await legDates.nth(1).fill("2026-08-21");
  await legDates.nth(2).fill("2026-08-28");
  await page.getByPlaceholder("Strike").nth(0).fill("450");
  await page.getByPlaceholder("Strike").nth(1).fill("460");
  await page.getByPlaceholder("Qty").nth(0).fill("1");
  await page.getByPlaceholder("Qty").nth(1).fill("2");
  await page.getByPlaceholder("Price").nth(0).fill("4.20");
  await page.getByPlaceholder("Price").nth(1).fill("2.40");
  await page.getByPlaceholder("進場價").fill("0.60");
  await page.getByRole("button", { name: /新增交易/ }).click();

  expect(createPayload).toMatchObject({
    ticker: "AAPL",
    strategy_type: "Ratio Spread",
    direction: "NEUTRAL",
    credit_debit: "CREDIT",
    expiration_date: "2026-08-21",
    option_type: "MULTI_LEG",
    strike_price: null,
    entry_price: 0.6,
    position_size: 1,
  });
  expect(createPayload.legs).toEqual([
    {
      side: "BUY",
      option_type: "CALL",
      strike_price: 450,
      expiration_date: "2026-08-21",
      quantity: 1,
      price: 4.2,
      contract_symbol: null,
    },
    {
      side: "SELL",
      option_type: "CALL",
      strike_price: 460,
      expiration_date: "2026-08-28",
      quantity: 2,
      price: 2.4,
      contract_symbol: null,
    },
  ]);

  const card = page.locator("text=AAPL · Ratio Spread").locator("..").locator("..");
  await expect(card).toContainText("到期 8/21");
  await expect(card).toContainText("2 legs");
  await expect(card).toContainText("方向 中性");
  await expect(card).toContainText("資流 Credit");
});
