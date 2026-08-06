import { useEffect, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Calendar,
  CheckCircle2,
  Mic,
  PenTool,
  Send,
  ShieldAlert,
  Smile,
  Target,
  Wifi,
} from "lucide-react";

/**
 * Trading Terminal Notebook
 * Single-file React component. Tailwind (arbitrary-value classes carry the
 * palette, so no tailwind.config changes are required) + lucide-react.
 *
 * Live-wired to the local FastAPI backend (app/main.py):
 *   GET  /health                                     -> HealthResponse
 *   GET  /api/v1/gex/{ticker}?days_to_expiration=  -> OptionGEXSummary
 *   POST /api/v1/chat                               -> ChatResponse
 *   POST /api/v1/plans/save                          -> UserTradePlan (signed)
 */

const BASE_URL = "http://127.0.0.1:8002";
const DEFAULT_TICKER = "AAPL";
// Backend's ChatContext.history caps at 30 entries; stay under that.
const MAX_HISTORY = 24;

const MONO =
  '[font-family:ui-monospace,"SF_Mono","JetBrains_Mono","IBM_Plex_Mono",Menlo,Consolas,monospace]';
const SCRIPT =
  '[font-family:"Segoe_Script","Bradley_Hand","Noteworthy","Comic_Sans_MS",cursive]';

const EXPIRIES = ["8/7", "8/14", "8/21", "9/18"];

// Illustrative-only trend — the backend has no historical GEX endpoint yet.
const DEMO_SPARK_VALUES = [1.8, 0.9, -0.3, -0.6, -1.4, -1.0, -2.1];

function daysUntil(monthDay) {
  const [month, day] = monthDay.split("/").map(Number);
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  let target = new Date(now.getFullYear(), month - 1, day);
  if (target < today) target = new Date(now.getFullYear() + 1, month - 1, day);
  return Math.round((target - today) / 86400000);
}

function fmtDollar(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return "$" + n.toLocaleString("en-US", { maximumFractionDigits: 2 });
}

function fmtCompactUsd(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  const sign = n < 0 ? "-" : "+";
  const abs = Math.abs(n);
  if (abs >= 1e9) return sign + (abs / 1e9).toFixed(2) + "B";
  if (abs >= 1e6) return sign + (abs / 1e6).toFixed(2) + "M";
  if (abs >= 1e3) return sign + (abs / 1e3).toFixed(1) + "K";
  return sign + abs.toFixed(0);
}

function fmtDateTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("zh-TW", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
}

async function parseErrorDetail(res) {
  try {
    const body = await res.json();
    return body.detail ? JSON.stringify(body.detail) : `HTTP ${res.status}`;
  } catch {
    return `HTTP ${res.status}`;
  }
}

/**
 * Detects an "A. ... / B. ... / C. ..." style option list inside an AI
 * reply so it can be rendered as quick-reply buttons. Heuristic and
 * best-effort — the LLM's output isn't a guaranteed format, so this only
 * fires when 2–6 lines form a clean, sequential A/B/C/... prefix.
 */
function extractQuickReplies(text) {
  if (!text) return [];
  const re = /^[\s>*-]*([A-Z])[.)、:：]\s*(.+)$/;
  const matches = [];
  for (const raw of text.split("\n")) {
    const m = raw.trim().match(re);
    if (m) {
      const clean = m[2].trim().replace(/\*+$/, "").trim();
      matches.push({ letter: m[1], label: `${m[1]}. ${clean}` });
    }
  }
  if (matches.length < 2 || matches.length > 6) return [];
  const isSequential = matches.every((m, i) => m.letter.charCodeAt(0) === "A".charCodeAt(0) + i);
  return isSequential ? matches : [];
}

/** Shared fixed-position tooltip box, cursor-anchored via clientX/clientY. */
function ChartTooltip({ point, title, lines }) {
  if (!point) return null;
  return (
    <div
      className={`pointer-events-none fixed z-50 px-2.5 py-2 rounded-md border border-[rgba(201,161,92,.4)] bg-[#0b0b0c] shadow-[0_8px_20px_rgba(0,0,0,.5)] text-[10.5px] leading-snug ${MONO}`}
      style={{ left: point.x + 14, top: point.y + 14, minWidth: 120 }}
    >
      <div className="text-[#c9a15c] font-bold mb-0.5">{title}</div>
      {lines.map((l, i) => (
        <div key={i} className="text-[#c7c7cc]">
          {l}
        </div>
      ))}
    </div>
  );
}

/**
 * The backend only returns three GEX levels (zero_gamma / call_wall / put_wall),
 * not a full per-strike chain, so this chart is an illustrative diverging shape
 * calibrated to those three real levels rather than a literal per-strike feed.
 * Hover/focus is honest about that: labeled rows show the real level, filler
 * rows are marked as interpolated, never a fabricated dollar-GEX number.
 */
function GexChart({ putWall, zeroGamma, callWall, gexStatus }) {
  const [hover, setHover] = useState(null); // { i, x, y }

  if (putWall == null || zeroGamma == null || callWall == null) {
    return <div className="text-[11px] text-[#57575c] py-6 text-center">— 無資料 —</div>;
  }
  const zeroFrac = gexStatus === "NEG_GAMMA" ? -0.12 : 0.12;
  const rows = [
    { k: putWall, label: "Put Wall", frac: -1, note: "最大 Put 方避險密集區" },
    { k: (putWall + zeroGamma) / 2, label: null, frac: -0.45, note: "內插估算，非個別檔位真實數據" },
    {
      k: zeroGamma,
      label: "Zero Γ",
      frac: zeroFrac,
      note: gexStatus === "NEG_GAMMA" ? "現貨低於此價，Gamma 翻負" : "現貨高於此價，Gamma 偏正",
    },
    { k: (zeroGamma + callWall) / 2, label: null, frac: 0.45, note: "內插估算，非個別檔位真實數據" },
    { k: callWall, label: "Call Wall", frac: 1, note: "最大 Call 方避險密集區" },
  ];
  const W = 300,
    rowH = 30,
    padTop = 6,
    padBottom = 6;
  const H = rows.length * rowH + padTop + padBottom;
  const centerX = 150;
  const maxHalf = 108;
  const hovered = hover ? rows[hover.i] : null;

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} role="img" aria-label="GEX wall structure">
        <line x1={centerX} y1={0} x2={centerX} y2={H} stroke="rgba(240,237,229,.16)" strokeWidth={1} />
        {rows.map((row, i) => {
          const y = padTop + i * rowH;
          const barW = Math.abs(row.frac) * maxHalf;
          const isBull = row.frac >= 0;
          const x = isBull ? centerX : centerX - barW;
          const color = isBull ? "#2fa37a" : "#d8622b";
          const labelX = x + barW / 2;
          const isHovered = hover?.i === i;

          const enter = (e) => setHover({ i, x: e.clientX, y: e.clientY });
          const move = (e) => setHover({ i, x: e.clientX, y: e.clientY });
          const leave = () => setHover(null);

          return (
            <g
              key={row.k}
              tabIndex={0}
              onMouseEnter={enter}
              onMouseMove={move}
              onMouseLeave={leave}
              onFocus={(e) => {
                const r = e.currentTarget.getBoundingClientRect();
                setHover({ i, x: r.right, y: r.top });
              }}
              onBlur={leave}
              aria-label={`${row.label || "內插值"} $${row.k.toFixed(2)}`}
              style={{ cursor: "pointer", outline: "none" }}
            >
              {/* full-row transparent hit area, bigger than the painted bar */}
              <rect x={0} y={y} width={W} height={rowH} fill="transparent" />
              {row.label === "Zero Γ" && (
                <line x1={0} y1={y} x2={W} y2={y} stroke="#c9a15c" strokeWidth={1} strokeDasharray="2,3" opacity={0.7} />
              )}
              <rect
                x={x}
                y={y + 4}
                width={barW}
                height={rowH - 12}
                rx={1.5}
                fill={color}
                opacity={isHovered ? 1 : row.label ? 1 : 0.55}
                stroke={isHovered ? "#f0ede5" : "none"}
                strokeWidth={isHovered ? 1 : 0}
              />
              <text x={8} y={y + rowH / 2 + 4} fontSize="9.5" fill="#57575c">
                ${row.k.toFixed(2)}
              </text>
              {row.label && (
                <text x={labelX} y={y + rowH / 2 - 6} fontSize="8" fill="#c9a15c" textAnchor="middle">
                  {row.label}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      {hovered && (
        <ChartTooltip
          point={hover}
          title={hovered.label || "內插值"}
          lines={[`Strike: $${hovered.k.toFixed(2)}`, hovered.note]}
        />
      )}
    </div>
  );
}

function Sparkline({ values }) {
  const [hover, setHover] = useState(null); // { i, x, y }
  const W = 300,
    H = 62,
    pad = 6;
  const max = Math.max(...values);
  const min = Math.min(...values);
  const range = max - min || 1;
  const stepX = (W - pad * 2) / (values.length - 1);
  const pts = values.map((v, i) => [pad + i * stepX, pad + (1 - (v - min) / range) * (H - pad * 2)]);
  const linePath = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
  const areaPath = `${linePath} L${pts[pts.length - 1][0].toFixed(1)},${H - pad} L${pts[0][0].toFixed(1)},${H - pad} Z`;
  const last = pts[pts.length - 1];
  const dayLabel = (i) => (i === values.length - 1 ? "今天" : `${values.length - 1 - i} 天前`);

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H}>
        <defs>
          <linearGradient id="sparkFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#c9a15c" stopOpacity={0.35} />
            <stop offset="100%" stopColor="#c9a15c" stopOpacity={0} />
          </linearGradient>
        </defs>
        <path d={areaPath} fill="url(#sparkFill)" />
        <path d={linePath} fill="none" stroke="#c9a15c" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round" />
        {hover != null && (
          <line x1={pts[hover.i][0]} y1={0} x2={pts[hover.i][0]} y2={H} stroke="rgba(240,237,229,.25)" strokeWidth={1} strokeDasharray="2,2" />
        )}
        <circle cx={last[0]} cy={last[1]} r={3} fill="#c9a15c" />
        {pts.map((p, i) => (
          <circle
            key={i}
            cx={p[0]}
            cy={p[1]}
            r={12}
            fill="transparent"
            tabIndex={0}
            style={{ cursor: "pointer", outline: "none" }}
            aria-label={`${dayLabel(i)}: ${values[i].toFixed(1)}`}
            onMouseEnter={(e) => setHover({ i, x: e.clientX, y: e.clientY })}
            onMouseMove={(e) => setHover({ i, x: e.clientX, y: e.clientY })}
            onMouseLeave={() => setHover(null)}
            onFocus={(e) => {
              const r = e.currentTarget.getBoundingClientRect();
              setHover({ i, x: r.right, y: r.top });
            }}
            onBlur={() => setHover(null)}
          />
        ))}
        {hover != null && (
          <circle cx={pts[hover.i][0]} cy={pts[hover.i][1]} r={3.5} fill="#f0ede5" stroke="#c9a15c" strokeWidth={1.5} />
        )}
      </svg>
      {hover != null && (
        <ChartTooltip
          point={hover}
          title={dayLabel(hover.i)}
          lines={[`Net GEX（估算）: ${values[hover.i] >= 0 ? "+" : ""}${values[hover.i].toFixed(1)}`]}
        />
      )}
    </div>
  );
}

export default function TradingTerminalNotebook() {
  const [ticker, setTicker] = useState(DEFAULT_TICKER);
  const [tickerInput, setTickerInput] = useState(DEFAULT_TICKER);

  const [health, setHealth] = useState({ state: "idle", mode: null, error: null });

  const [activeExpiry, setActiveExpiry] = useState("8/14");
  const [gexData, setGexData] = useState(null);
  const [gexLoading, setGexLoading] = useState(false);
  const [gexError, setGexError] = useState(null);

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);

  const [tradePlan, setTradePlan] = useState(null);
  const [signing, setSigning] = useState(false);
  const [signError, setSignError] = useState(null);
  const [journal, setJournal] = useState([]);

  const scrollRef = useRef(null);
  const isComposing = useRef(false);

  const [userId] = useState(() => {
    if (typeof window === "undefined") return "web-client";
    let id = window.localStorage.getItem("ttn_user_id");
    if (!id) {
      id = crypto.randomUUID();
      window.localStorage.setItem("ttn_user_id", id);
    }
    return id;
  });
  const [conversationId] = useState(() => crypto.randomUUID());

  const dte = daysUntil(activeExpiry);

  // ---------- Health check ----------
  async function checkHealth() {
    setHealth((h) => ({ ...h, state: "checking", error: null }));
    try {
      const res = await fetch(`${BASE_URL}/health`);
      if (!res.ok) throw new Error(await parseErrorDetail(res));
      const data = await res.json();
      setHealth({
        state: data.status === "ok" ? "ok" : "error",
        mode: data.market_data_mode,
        error: null,
      });
    } catch (err) {
      setHealth({ state: "error", mode: null, error: err.message || "無法連線" });
    }
  }

  useEffect(() => {
    checkHealth();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---------- GEX fetch (tab switch / ticker commit drives header + left panel) ----------
  useEffect(() => {
    let cancelled = false;
    setGexLoading(true);
    setGexError(null);
    fetch(`${BASE_URL}/api/v1/gex/${ticker}?days_to_expiration=${dte}`)
      .then(async (res) => {
        if (!res.ok) throw new Error(await parseErrorDetail(res));
        return res.json();
      })
      .then((data) => {
        if (cancelled) return;
        setGexData(data);
        setGexLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        setGexError(err.message || "無法連接後端");
        setGexLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeExpiry, ticker]);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages, chatLoading]);

  function commitTicker() {
    const next = tickerInput.trim().toUpperCase();
    if (next) setTicker(next);
    else setTickerInput(ticker);
  }

  // ---------- Chat ----------
  async function sendMessage(overrideText) {
    const text = (overrideText ?? input).trim();
    if (!text || chatLoading) return;
    const priorHistory = messages.slice(-MAX_HISTORY).map((m) => ({ role: m.role, content: m.content }));
    setMessages((m) => [...m, { role: "user", content: text }]);
    if (overrideText === undefined) setInput("");
    setChatLoading(true);
    try {
      const res = await fetch(`${BASE_URL}/api/v1/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_message: text,
          context: {
            user_id: userId,
            conversation_id: conversationId,
            ticker: ticker,
            days_to_expiration: dte,
            history: priorHistory,
          },
        }),
      });
      if (!res.ok) throw new Error(await parseErrorDetail(res));
      const data = await res.json();
      setMessages((m) => [...m, { role: "assistant", content: data.assistant_message }]);
      if (data.trade_plan_card) setTradePlan(data.trade_plan_card);
    } catch (err) {
      setMessages((m) => [
        ...m,
        { role: "assistant", content: `⚠️ 無法連接後端 (${BASE_URL})：${err.message || "未知錯誤"}` },
      ]);
    } finally {
      setChatLoading(false);
    }
  }

  // ---------- Sign & save ----------
  async function handleSign() {
    if (!tradePlan || signing || tradePlan.status === "SIGNED") return;
    setSigning(true);
    setSignError(null);
    try {
      const res = await fetch(`${BASE_URL}/api/v1/plans/save`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ plan: tradePlan }),
      });
      if (!res.ok) throw new Error(await parseErrorDetail(res));
      const signed = await res.json();
      setTradePlan(signed);
      setJournal((j) => [{ strategy: signed.strategy_type, time: fmtDateTime(signed.signed_at) }, ...j]);
    } catch (err) {
      setSignError(err.message || "簽署失敗");
    } finally {
      setSigning(false);
    }
  }

  function handleNewEntry() {
    setTradePlan(null);
    setSignError(null);
  }

  const netGexTone = gexData?.gex_status === "NEG_GAMMA" ? "bear" : "bull";

  return (
    <div className={`h-screen flex flex-col bg-[#121214] text-[#f0ede5] ${MONO} text-[13px] overflow-hidden`}>
      {/* HEADER */}
      <header className="flex items-center gap-7 px-6 py-3.5 border-b border-[rgba(240,237,229,.09)] bg-gradient-to-b from-[#1b1b1e] to-[#121214]">
        <div className="flex items-baseline gap-2.5">
          <input
            value={tickerInput}
            onChange={(e) => setTickerInput(e.target.value)}
            onBlur={commitTicker}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.nativeEvent.isComposing && e.keyCode !== 229) {
                e.currentTarget.blur();
              }
            }}
            spellCheck={false}
            className={`text-[22px] font-bold tracking-wide uppercase bg-transparent border-b border-dashed border-transparent focus:border-[#c9a15c] outline-none w-[100px] ${MONO}`}
            title="輸入代號後按 Enter 或點擊別處套用"
          />
          <span className="text-[11px] text-[#8d8d93]">Equity Options</span>
        </div>
        <div className="flex items-baseline gap-2">
          <span className="text-[22px] font-bold [font-variant-numeric:tabular-nums]">
            {fmtDollar(gexData?.stock_price)}
          </span>
        </div>
        <div className="flex gap-2.5 ml-1.5">
          <MetricChip label="Zero Γ" value={fmtDollar(gexData?.zero_gamma)} color="#c9a15c" />
          <MetricChip label="Call Wall" value={fmtDollar(gexData?.call_wall)} color="#2fa37a" />
          <MetricChip label="Put Wall" value={fmtDollar(gexData?.put_wall)} color="#d8622b" />
        </div>
        <div className="ml-auto flex items-center gap-3">
          {gexError && (
            <span className="text-[10px] text-[#d8622b]">GEX 請求失敗</span>
          )}
          <HealthBadge health={health} baseUrl={BASE_URL} />
          <button
            type="button"
            onClick={checkHealth}
            disabled={health.state === "checking"}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-full border border-[rgba(240,237,229,.16)] bg-[#0b0b0c] text-[#8d8d93] text-[11px] font-semibold hover:text-[#f0ede5] hover:border-[rgba(240,237,229,.28)] disabled:opacity-50 transition-colors"
            title={`GET ${BASE_URL}/health`}
          >
            <Wifi size={12} className={health.state === "checking" ? "animate-pulse" : ""} />
            {health.state === "checking" ? "測試中…" : "API 連線測試"}
          </button>
          <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-full bg-[rgba(201,161,92,.13)] border border-[rgba(201,161,92,.35)] text-[#c9a15c] text-[11px] font-semibold">
            <Calendar size={12} />
            Earnings 10/22
          </div>
        </div>
      </header>

      {/* BODY */}
      <div className="flex-1 min-h-0 grid grid-cols-[292px_minmax(380px,1fr)_352px] gap-px bg-[rgba(240,237,229,.09)]">
        {/* LEFT: GEX PANEL */}
        <section className="flex flex-col min-h-0 bg-[#121214] overflow-hidden">
          <PaneHeader icon={<Activity size={13} />} title="Gamma / GEX" />
          <div className="flex-1 min-h-0 overflow-y-auto px-4 pt-[18px] pb-5">
            <div className="flex gap-1 mb-3.5 bg-[#0b0b0c] p-[3px] rounded border border-[rgba(240,237,229,.09)]">
              {EXPIRIES.map((exp) => (
                <button
                  key={exp}
                  type="button"
                  onClick={() => setActiveExpiry(exp)}
                  className={`flex-1 py-1.5 text-[11.5px] font-semibold rounded-sm transition-colors ${
                    exp === activeExpiry ? "bg-[#c9a15c] text-[#1a1408]" : "text-[#8d8d93] hover:text-[#f0ede5]"
                  }`}
                >
                  {exp}
                </button>
              ))}
            </div>

            {gexError && (
              <div className="text-[10.5px] text-[#d8622b] mb-3 leading-snug">⚠ {gexError}</div>
            )}

            <div className="grid grid-cols-2 gap-2 mb-3">
              <Stat label="Net GEX" value={gexData ? fmtCompactUsd(gexData.net_gex) : "—"} tone={netGexTone} />
              <Stat label="IV Rank" value={gexData ? `${gexData.iv_rank.toFixed(0)}%` : "—"} />
            </div>

            <Card>
              <CardTitle title="GEX by Strike" tag={`${activeExpiry} exp · ${dte} DTE`} />
              <GexChart
                putWall={gexData?.put_wall}
                zeroGamma={gexData?.zero_gamma}
                callWall={gexData?.call_wall}
                gexStatus={gexData?.gex_status}
              />
              <div className="text-[9px] text-[#57575c] mt-1.5 mb-1">
                示意分佈，依真實 Zero Γ / Call Wall / Put Wall 校正
              </div>
              <div className="flex gap-3.5 text-[10px] text-[#8d8d93] mt-1.5">
                <span className="flex items-center gap-1.5">
                  <i className="w-2 h-2 rounded-sm inline-block bg-[#2fa37a]" />
                  Positive (Call)
                </span>
                <span className="flex items-center gap-1.5">
                  <i className="w-2 h-2 rounded-sm inline-block bg-[#d8622b]" />
                  Negative (Put)
                </span>
              </div>
            </Card>

            <Card>
              <CardTitle
                title="Net GEX Trend"
                tag="5D · Estimated"
                tagTitle="示意趨勢，非真實歷史資料 — 後端目前沒有 GEX 歷史時序端點"
              />
              <Sparkline values={DEMO_SPARK_VALUES} />
            </Card>
          </div>
        </section>

        {/* MIDDLE: AI CHAT */}
        <section className="flex flex-col min-h-0 bg-[#121214] overflow-hidden">
          <PaneHeader
            icon={<PenTool size={13} className="rotate-90" />}
            title="AI Copilot"
            trailing={
              <span className="text-[10px] text-[#2fa37a] flex items-center gap-1.5">
                <i className={`w-1.5 h-1.5 rounded-full bg-[#2fa37a] ${chatLoading ? "animate-pulse" : ""}`} />
                {chatLoading ? "thinking…" : "reading left panel"}
              </span>
            }
          />
          <div ref={scrollRef} className="flex-1 min-h-0 overflow-y-auto px-4 py-4 flex flex-col gap-3.5">
            {messages.length === 0 && (
              <div className="text-[11px] text-[#57575c] text-center mt-8 leading-relaxed">
                AI Copilot 已連接後端
                <br />
                輸入訊息開始分析（例如：「看好 {ticker} 跌破 Put Wall，想買 Put」）
              </div>
            )}
            {messages.map((m, i) => {
              const quickReplies =
                m.role === "assistant" && i === messages.length - 1 ? extractQuickReplies(m.content) : [];
              return (
                <div key={i} className={`max-w-[82%] flex flex-col gap-1 ${m.role === "user" ? "self-end items-end" : "self-start items-start"}`}>
                  <div className="text-[9.5px] tracking-wider uppercase text-[#57575c]">{m.role === "user" ? "YOU" : "AI COPILOT"}</div>
                  <div
                    className={`px-3.5 py-2.5 rounded-lg text-[12.5px] leading-relaxed whitespace-pre-wrap ${
                      m.role === "user"
                        ? "bg-[rgba(201,161,92,.13)] border border-[rgba(201,161,92,.3)] rounded-br-sm"
                        : "bg-[#1b1b1e] border border-[rgba(240,237,229,.09)] rounded-bl-sm"
                    }`}
                  >
                    {m.content}
                  </div>
                  {quickReplies.length > 0 && (
                    <div className="flex flex-wrap gap-1.5 w-full mt-0.5">
                      {quickReplies.map((qr) => (
                        <button
                          key={qr.letter}
                          type="button"
                          onClick={() => sendMessage(qr.letter)}
                          disabled={chatLoading}
                          className={`text-left leading-snug px-3 py-1.5 rounded-full border border-[rgba(201,161,92,.35)] bg-[rgba(201,161,92,.08)] text-[#c9a15c] text-[11.5px] font-semibold hover:bg-[rgba(201,161,92,.18)] hover:border-[rgba(201,161,92,.55)] active:scale-[.98] disabled:opacity-50 transition-colors ${MONO}`}
                        >
                          {qr.label}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
            {chatLoading && (
              <div className="self-start flex gap-1 px-3.5 py-2.5 bg-[#1b1b1e] border border-[rgba(240,237,229,.09)] rounded-lg rounded-bl-sm">
                {[0, 1, 2].map((i) => (
                  <i
                    key={i}
                    className="w-[5px] h-[5px] rounded-full bg-[#8d8d93] animate-bounce"
                    style={{ animationDelay: `${i * 0.15}s` }}
                  />
                ))}
              </div>
            )}
          </div>
          <div className="flex items-center gap-2 px-3.5 py-3 border-t border-[rgba(240,237,229,.09)] bg-[#1b1b1e]">
            <IconButton title="Voice input">
              <Mic size={16} />
            </IconButton>
            <IconButton title="Sticker">
              <Smile size={16} />
            </IconButton>
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onCompositionStart={() => (isComposing.current = true)}
              onCompositionEnd={() => (isComposing.current = false)}
              onKeyDown={(e) => {
                if (e.key !== "Enter") return;
                if (isComposing.current || e.nativeEvent.isComposing || e.keyCode === 229) return;
                sendMessage();
              }}
              type="text"
              disabled={chatLoading}
              placeholder="問問左側數據代表什麼…"
              autoComplete="off"
              className={`flex-1 bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded-md px-3 py-2 text-[#f0ede5] text-[12.5px] outline-none focus:border-[#c9a15c] disabled:opacity-50 ${MONO}`}
            />
            <button
              type="button"
              onClick={() => sendMessage()}
              disabled={chatLoading || !input.trim()}
              title="Send"
              className="w-[34px] h-[34px] flex items-center justify-center rounded-md bg-[#c9a15c] border border-[#c9a15c] text-[#1a1408] hover:bg-[#d8b06c] disabled:opacity-40 disabled:hover:bg-[#c9a15c]"
            >
              <Send size={16} />
            </button>
          </div>
        </section>

        {/* RIGHT: TRADE PLAN NOTEBOOK CARD */}
        <section className="flex flex-col min-h-0 bg-[#121214] overflow-hidden">
          <PaneHeader icon={<PenTool size={13} />} title="Trade Plan Notebook" />
          <div
            className="flex-1 min-h-0 overflow-y-auto px-4 pt-[18px] pb-5"
            style={{ background: "radial-gradient(circle at 15% 8%, rgba(201,161,92,.05), transparent 45%)" }}
          >
            <div
              key={tradePlan?.plan_id || "empty"}
              className={`relative rounded-sm px-5 pt-5 pb-[18px] pl-[34px] -rotate-[0.6deg] text-[#2b2620] motion-safe:animate-[cardIn_360ms_ease-out]`}
              style={{
                background: "#e9e2cf",
                boxShadow: "0 14px 30px -10px rgba(0,0,0,.55), 0 2px 0 rgba(0,0,0,.2)",
                backgroundImage:
                  "repeating-linear-gradient(to bottom, transparent 0px, transparent 25px, rgba(43,38,32,.09) 26px)",
              }}
            >
              <div className="absolute left-6 top-0 bottom-0 w-px bg-[rgba(163,58,58,.4)]" />
              <div className="absolute -top-2 left-5 w-14 h-5 -rotate-[5deg] bg-[rgba(240,237,229,.5)] border border-[rgba(43,38,32,.08)] shadow-sm" />
              <div className="absolute -top-2 right-3.5 w-14 h-5 rotate-[4deg] bg-[rgba(240,237,229,.5)] border border-[rgba(43,38,32,.08)] shadow-sm" />

              {!tradePlan ? (
                <div className="py-10 text-center">
                  <div className="text-[9.5px] tracking-[.12em] uppercase text-[#6b6252] mb-2">Options Strategy</div>
                  <div className="text-[12px] text-[#6b6252] leading-relaxed px-2">
                    與中間的 AI Copilot 討論策略，
                    <br />
                    達成進出場共識後計畫卡會自動出現在這裡
                  </div>
                </div>
              ) : (
                <>
                  <div className="mb-3.5">
                    <div className="text-[9.5px] tracking-[.12em] uppercase text-[#6b6252] mb-0.5">Options Strategy</div>
                    <div className="text-[17px] font-bold">{tradePlan.strategy_type}</div>
                    <div className="text-[11px] text-[#6b6252] mt-0.5">
                      {tradePlan.ticker} · drafted {fmtDateTime(tradePlan.created_at)}
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2.5 mb-3.5">
                    <Field icon={<Target size={10} />} label="Entry" value={fmtDollar(tradePlan.entry_price)} />
                    <Field icon={<ShieldAlert size={10} />} label="Stop Loss" value={fmtDollar(tradePlan.stop_loss)} valueClass="text-[#8a3d17]" />
                    <Field icon={<Target size={10} />} label="Take Profit" value={fmtDollar(tradePlan.target_price)} valueClass="text-[#1c5a41]" />
                    <Field icon={<Activity size={10} />} label="Max Loss" value={fmtDollar(tradePlan.max_loss_usd)} />
                  </div>

                  {tradePlan.theta_warning && (
                    <div className="flex gap-2 items-start px-2.5 py-2.5 mb-4 rounded-sm bg-[rgba(201,161,92,.18)] border border-[rgba(180,130,40,.4)] text-[11px]">
                      <AlertTriangle size={14} className="text-[#8a5a17] shrink-0 mt-px" />
                      <div>
                        <b className="block mb-px">Theta Decay Warning</b>
                        AI 已標記此計畫的時間價值損耗風險較高，請留意持倉天數
                      </div>
                    </div>
                  )}

                  <div className="relative border-t border-dashed border-[rgba(43,38,32,.22)] pt-3">
                    <div className="flex items-end justify-between mb-2.5">
                      <span className={`text-[22px] text-[rgba(43,38,32,.55)] ${tradePlan.status === "SIGNED" ? SCRIPT : MONO}`}>
                        {tradePlan.status === "SIGNED" ? "✕ Signed" : "✕_____________"}
                      </span>
                      <span className="text-[10px] text-[#6b6252]">
                        {tradePlan.status === "SIGNED" ? fmtDateTime(tradePlan.signed_at) : "未簽署"}
                      </span>
                    </div>

                    {tradePlan.status !== "SIGNED" ? (
                      <>
                        <button
                          type="button"
                          onClick={handleSign}
                          disabled={signing}
                          className="w-full py-2.5 rounded-sm border border-[#2b2620] bg-[#2b2620] text-[#e9e2cf] text-xs font-bold tracking-wider uppercase flex items-center justify-center gap-2 hover:bg-[#1b1712] active:scale-[.98] transition disabled:opacity-60"
                        >
                          <PenTool size={14} />
                          {signing ? "簽署中…" : "簽署存入日誌"}
                        </button>
                        {signError && <div className="text-[10.5px] text-[#8a3d17] mt-1.5 text-center">⚠ {signError}</div>}
                      </>
                    ) : (
                      <>
                        <div className="flex items-center justify-center gap-2 text-[11px] text-[#1c5a41] font-semibold py-1.5">
                          <CheckCircle2 size={14} />
                          已存入交易日誌
                        </div>
                        <button
                          type="button"
                          onClick={handleNewEntry}
                          className="block w-full text-center mt-2 text-[10.5px] text-[#6b6252] underline hover:text-[#2b2620]"
                        >
                          + 開立新計畫
                        </button>
                      </>
                    )}
                  </div>

                  {tradePlan.status === "SIGNED" && (
                    <div className="absolute top-2 -right-1.5 w-24 h-24 rounded-full border-[3px] border-[#d8622b] text-[#d8622b] flex flex-col items-center justify-center -rotate-[16deg] pointer-events-none [mix-blend-mode:multiply] motion-safe:animate-[stampDown_.28s_cubic-bezier(.2,1.6,.4,1)]">
                      <b className="text-sm tracking-wider font-extrabold">SIGNED</b>
                      <span className="text-[8px] mt-0.5">{fmtDateTime(tradePlan.signed_at)}</span>
                    </div>
                  )}
                </>
              )}
            </div>

            <div className="mt-4">
              <div className="text-[10px] tracking-wider uppercase text-[#8d8d93] mb-2.5">Trade Journal Log</div>
              {journal.length === 0 ? (
                <div className="text-[11px] text-[#57575c] px-0.5 py-1.5">尚無紀錄 — 簽署上方計畫卡後會出現在這裡</div>
              ) : (
                <div className="flex flex-col gap-1.5">
                  {journal.map((e, i) => (
                    <div key={i} className="flex items-center gap-2.5 px-2.5 py-2 bg-[#1b1b1e] border border-[rgba(240,237,229,.09)] rounded-sm text-[11px]">
                      <CheckCircle2 size={13} className="text-[#2fa37a] shrink-0" />
                      <div className="flex-1">
                        <b className="text-[#f0ede5] font-bold">{e.strategy}</b>
                        <span className="block text-[9.5px] text-[#57575c] mt-px">{e.time} 簽署存入</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </section>
      </div>

      <style>{`
        @keyframes stampDown { from { opacity:0; transform:rotate(-16deg) scale(2.4); } to { opacity:1; transform:rotate(-16deg) scale(1); } }
        @keyframes cardIn { from { opacity:0; transform: translateY(-14px) rotate(-0.6deg); } to { opacity:1; transform: translateY(0) rotate(-0.6deg); } }
      `}</style>
    </div>
  );
}

function PaneHeader({ icon, title, trailing }) {
  return (
    <div className="flex items-center justify-between px-4 py-3 border-b border-[rgba(240,237,229,.09)]">
      <h2 className="m-0 text-[11px] tracking-wider uppercase text-[#8d8d93] font-semibold flex items-center gap-1.5">
        <span className="text-[#57575c]">{icon}</span>
        {title}
      </h2>
      {trailing}
    </div>
  );
}

function HealthBadge({ health, baseUrl }) {
  let dot = "#57575c";
  let label = "尚未測試";
  if (health.state === "checking") {
    dot = "#8d8d93";
    label = "測試中…";
  } else if (health.state === "ok" && health.mode === "moomoo") {
    dot = "#2fa37a";
    label = "已連線至 Moomoo 後端";
  } else if (health.state === "ok") {
    dot = "#c9a15c";
    label = `已連線（${health.mode || "mock"} 模式）`;
  } else if (health.state === "error") {
    dot = "#d8622b";
    label = "連線失敗";
  }
  return (
    <span
      className="text-[10px] text-[#8d8d93] flex items-center gap-1.5"
      title={health.error ? `${baseUrl}: ${health.error}` : `${baseUrl}/health`}
    >
      <i className="w-1.5 h-1.5 rounded-full" style={{ background: dot, boxShadow: `0 0 0 2px ${dot}26` }} />
      {label}
    </span>
  );
}

function MetricChip({ label, value, color }) {
  return (
    <div className="flex flex-col gap-0.5 px-3 py-1.5 bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded-sm min-w-[88px]">
      <span className="text-[9.5px] tracking-wider uppercase text-[#8d8d93]">{label}</span>
      <span className="text-sm font-bold [font-variant-numeric:tabular-nums]" style={{ color }}>
        {value}
      </span>
    </div>
  );
}

function Stat({ label, value, tone }) {
  const color = tone === "bull" ? "#2fa37a" : tone === "bear" ? "#d8622b" : "#f0ede5";
  return (
    <div className="bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded-sm px-2.5 py-2">
      <div className="text-[9px] tracking-wider uppercase text-[#57575c] mb-0.5">{label}</div>
      <div className="text-sm font-bold [font-variant-numeric:tabular-nums]" style={{ color }}>
        {value}
      </div>
    </div>
  );
}

function Card({ children }) {
  return <div className="bg-[#1b1b1e] border border-[rgba(240,237,229,.09)] rounded px-3.5 py-3.5 mb-3">{children}</div>;
}

function CardTitle({ title, tag, tagTitle }) {
  return (
    <div className="text-[10px] tracking-wider uppercase text-[#8d8d93] mb-2.5 flex items-center justify-between">
      {title}
      <span
        className={`text-[9.5px] text-[#57575c] normal-case tracking-normal ${tagTitle ? "cursor-help border-b border-dotted border-[#57575c]" : ""}`}
        title={tagTitle}
      >
        {tag}
      </span>
    </div>
  );
}

function IconButton({ title, children }) {
  return (
    <button
      type="button"
      title={title}
      className="w-[34px] h-[34px] shrink-0 flex items-center justify-center rounded-md border border-[rgba(240,237,229,.09)] bg-[#0b0b0c] text-[#8d8d93] hover:text-[#f0ede5] hover:border-[rgba(240,237,229,.16)] transition-colors"
    >
      {children}
    </button>
  );
}

function Field({ icon, label, value, valueClass = "" }) {
  return (
    <div className="border-t border-dashed border-[rgba(43,38,32,.22)] pt-1.5">
      <div className="text-[9px] tracking-wider uppercase text-[#6b6252] mb-0.5 flex items-center gap-1">
        {icon}
        {label}
      </div>
      <div className={`text-[13px] font-bold [font-variant-numeric:tabular-nums] ${valueClass}`}>{value}</div>
    </div>
  );
}
