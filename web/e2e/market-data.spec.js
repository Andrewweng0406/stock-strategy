import { expect, test } from "@playwright/test";

const API_BASE = "http://127.0.0.1:8002";

const baseSummary = {
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

const aaplExpirations = [
  { date: "2026-08-10", days_to_expiration: 0, expiration_type: "0DTE" },
  { date: "2026-08-21", days_to_expiration: 11, expiration_type: "MONTHLY" },
  { date: "2026-08-28", days_to_expiration: 18, expiration_type: "WEEKLY" },
  { date: "2026-09-04", days_to_expiration: 25, expiration_type: "WEEKLY" },
  { date: "2026-09-11", days_to_expiration: 32, expiration_type: "WEEKLY" },
  { date: "2026-09-18", days_to_expiration: 39, expiration_type: "MONTHLY" },
  { date: "2026-09-25", days_to_expiration: 46, expiration_type: "WEEKLY" },
];

const tslaExpirations = [
  { date: "2026-08-14", days_to_expiration: 4, expiration_type: "WEEKLY" },
  { date: "2026-08-21", days_to_expiration: 11, expiration_type: "MONTHLY" },
];

function corsHeaders() {
  return {
    "access-control-allow-origin": "*",
    "access-control-allow-methods": "GET,POST,PUT,OPTIONS",
    "access-control-allow-headers": "content-type",
  };
}

function summaryFor(ticker, override = {}) {
  const values = ticker === "TSLA"
    ? { stock_price: 330.25, zero_gamma: 305, call_wall: 345, put_wall: 310, net_gex: -9999999, gex_status: "NEG_GAMMA" }
    : {};
  return { ...baseSummary, ...values, ...override, ticker };
}

async function fulfillJson(route, body, status = 200) {
  await route.fulfill({
    status,
    headers: { "content-type": "application/json", ...corsHeaders() },
    body: JSON.stringify(body),
  });
}

async function installTerminalStub(
  page,
  {
    healthMode = "moomoo",
    expirationsByTicker = { AAPL: aaplExpirations, TSLA: tslaExpirations },
    failExpirations = new Set(),
    failGex = new Set(),
    gexOverridesByTicker = {},
    onGexRequest = () => {},
    onChatRequest = () => {},
    onAggregateRequest = () => {},
  } = {}
) {
  await page.route(`${API_BASE}/**`, async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (request.method() === "OPTIONS") {
      await route.fulfill({ status: 204, headers: corsHeaders() });
      return;
    }

    if (url.pathname === "/health") {
      await fulfillJson(route, { status: "ok", market_data_mode: healthMode });
      return;
    }
    if (url.pathname === "/api/v1/conversations") {
      await fulfillJson(route, { conversations: [] });
      return;
    }
    if (url.pathname === "/api/v1/plans") {
      await fulfillJson(route, { plans: [] });
      return;
    }
    if (url.pathname === "/api/v1/trades") {
      await fulfillJson(route, { trades: [] });
      return;
    }
    if (url.pathname.startsWith("/api/v1/profile/")) {
      await fulfillJson(route, {
        user_id: url.pathname.split("/").pop(),
        risk_tolerance: null,
        preferred_strategy_types: [],
        notes: "",
      });
      return;
    }
    if (url.pathname.match(/^\/api\/v1\/gex\/[^/]+\/history$/)) {
      await fulfillJson(route, { ticker: url.pathname.split("/")[4], snapshots: [] });
      return;
    }
    if (url.pathname.match(/^\/api\/v1\/expirations\/[^/]+$/)) {
      const ticker = url.pathname.split("/").pop();
      if (failExpirations.has(ticker)) {
        await fulfillJson(route, { detail: `${ticker} expirations unavailable` }, 503);
        return;
      }
      await fulfillJson(route, {
        ticker,
        expirations: expirationsByTicker[ticker] || [],
      });
      return;
    }
    if (url.pathname.match(/^\/api\/v1\/gex\/[^/]+\/aggregate$/)) {
      onAggregateRequest(url);
      const ticker = url.pathname.split("/")[4];
      await fulfillJson(route, summaryFor(ticker, { net_gex: 4444444 }));
      return;
    }
    if (url.pathname.match(/^\/api\/v1\/gex\/[^/]+$/)) {
      const ticker = url.pathname.split("/").pop();
      onGexRequest(url);
      if (failGex.has(ticker)) {
        await fulfillJson(route, { detail: `${ticker} GEX unavailable` }, 503);
        return;
      }
      await fulfillJson(route, summaryFor(ticker, gexOverridesByTicker[ticker] || {}));
      return;
    }
    if (url.pathname === "/api/v1/chat" && request.method() === "POST") {
      onChatRequest(request.postDataJSON());
      await fulfillJson(route, {
        assistant_message: "收到",
        gex_summary: summaryFor("AAPL"),
        risk_profile: {
          gex_status: "POS_GAMMA",
          volatility_regime: "LOW_VOL_MEAN_REVERSION",
          risk_level: "NORMAL",
          warnings: [],
          locked_warning: false,
        },
        trade_plan_card: null,
      });
      return;
    }

    await fulfillJson(route, { detail: `Unhandled ${request.method()} ${url.pathname}` }, 500);
  });
}

async function switchTicker(page, ticker) {
  const input = page.getByTitle("輸入代號後按 Enter 或點擊別處套用");
  await input.fill(ticker);
  await input.press("Enter");
}

test("ticker switch waits for that ticker's expirations before requesting GEX", async ({ page }) => {
  const requests = [];
  await installTerminalStub(page, {
    onGexRequest: (url) => requests.push(`${url.pathname}?${url.searchParams.toString()}`),
  });

  await page.goto("/");
  await expect(page.getByTitle("輸入代號後按 Enter 或點擊別處套用")).toHaveValue("AAPL");
  await expect(page.locator("header")).toContainText("$225");

  await switchTicker(page, "TSLA");
  await expect(page.locator("header")).toContainText("$305");

  expect(requests).toContain("/api/v1/gex/TSLA?days_to_expiration=4");
  expect(requests).not.toContain("/api/v1/gex/TSLA?days_to_expiration=11");
});

test("GEX failure clears previous numbers instead of showing stale levels", async ({ page }) => {
  const failGex = new Set(["TSLA"]);
  await installTerminalStub(page, { failGex });

  await page.goto("/");
  await expect(page.locator("header")).toContainText("$225");
  await switchTicker(page, "TSLA");

  await expect(page.getByText("TSLA GEX 資料載入失敗")).toBeVisible();
  await expect(page.getByText("為避免誤導，這裡不會保留上一次查詢的數字。")).toBeVisible();
  await expect(page.locator("header")).not.toContainText("$225");
  await expect(page.locator("header")).not.toContainText("$240");
  await expect(page.locator("header")).not.toContainText("$215");

  failGex.delete("TSLA");
  await page.getByRole("button", { name: "重試 GEX" }).click();
  await expect(page.locator("header")).toContainText("$330.25");
  await expect(page.getByText("TSLA GEX 資料載入失敗")).not.toBeVisible();
});

test("partial GEX payload renders placeholders instead of crashing", async ({ page }) => {
  await installTerminalStub(page, {
    gexOverridesByTicker: {
      AAPL: {
        iv_rank: null,
        zero_gamma: null,
        call_wall: null,
        put_wall: null,
        pinning: {
          ...baseSummary.pinning,
          pin_strike: null,
          distance_pct: null,
          oi_concentration_pct: null,
        },
      },
    },
  });

  await page.goto("/");

  await expect(page.getByText("頁面發生錯誤")).not.toBeVisible();
  await expect(page.getByText("— 無資料 —")).toBeVisible();
  await expect(page.getByText("IV Rank").locator("..")).toContainText("—");
  await expect(page.getByText("Pin Strike").locator("..")).toContainText("—");
  await expect(page.getByText("距離").locator("..")).toContainText("—");
  await expect(page.getByText("OI 集中度").locator("..")).toContainText("—");
});

test("chat sends null DTE when the current ticker has no resolved expiration", async ({ page }) => {
  let chatPayload = null;
  await installTerminalStub(page, {
    failExpirations: new Set(["AAPL"]),
    onChatRequest: (payload) => {
      chatPayload = payload;
    },
  });

  await page.goto("/");
  await expect(page.getByText("AAPL expirations unavailable")).toBeVisible();
  await expect(page.getByText("AAPL 尚無 GEX 資料")).toBeVisible();
  await page.getByPlaceholder("問問左側數據代表什麼…").fill("現在左側資料代表什麼？");
  await page.getByTitle("Send").click();

  await expect.poll(() => chatPayload?.context?.days_to_expiration).toBe(null);
});

test("aggregate mode caps expiration dates sent to the backend", async ({ page }) => {
  const aggregateUrls = [];
  await installTerminalStub(page, {
    onAggregateRequest: (url) => aggregateUrls.push(url),
  });

  await page.goto("/");
  await page.locator("select").first().selectOption("__aggregate__");

  await expect.poll(() => aggregateUrls.length).toBe(1);
  const sentExpirations = aggregateUrls[0].searchParams.getAll("expirations");
  expect(sentExpirations).toEqual(aaplExpirations.slice(0, 5).map((e) => e.date));
});

for (const [mode, label] of [
  ["moomoo", "已連線至 Moomoo 後端（即時）"],
  ["yfinance", "已連線（Yahoo Finance · 約 15-20 分鐘延遲）"],
  ["unavailable", "市場資料源未設定"],
  ["mock", "已連線（Demo/Mock 模式 · 非真實資料）"],
]) {
  test(`health badge labels ${mode} market data mode`, async ({ page }) => {
    await installTerminalStub(page, { healthMode: mode });
    await page.goto("/");
    await expect(page.getByText(label)).toBeVisible();
  });
}
