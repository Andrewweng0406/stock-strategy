import { useEffect, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BookOpen,
  CheckCircle2,
  History,
  Plus,
  PenTool,
  RefreshCw,
  Send,
  Settings,
  ShieldAlert,
  Target,
  Wifi,
  X,
} from "lucide-react";
import TradeJournalPanel from "./TradeJournal.jsx";
import { parseErrorDetail } from "./apiError.js";

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

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8002";
const DEFAULT_TICKER = "AAPL";
// Backend's ChatContext.history caps at 30 entries; stay under that.
const MAX_HISTORY = 24;

const MONO =
  '[font-family:ui-monospace,"SF_Mono","JetBrains_Mono","IBM_Plex_Mono",Menlo,Consolas,monospace]';
const SCRIPT =
  '[font-family:"Segoe_Script","Bradley_Hand","Noteworthy","Comic_Sans_MS",cursive]';

// How many of the (potentially dozens of) real expirations the backend
// returns get shown in the picker. Near-term contracts are what
// cross-period GEX judgment actually needs — LEAPS two years out would
// just clutter the list without adding signal.
const MAX_EXPIRATIONS_SHOWN = 10;

// Aggregate mode issues one option-chain fetch per expiration in a single
// request. Moomoo/Futu caps that specific call at 10/30s — keep this well
// under it (with headroom for whatever single-expiration lookups happen to
// land in the same window) rather than reliably tripping the rate limit.
const MAX_AGGREGATE_EXPIRATIONS = 5;

const EXPIRATION_TYPE_LABEL = {
  "0DTE": "0DTE",
  "1DTE": "1DTE",
  WEEKLY: "週",
  MONTHLY: "月",
};

function fmtExpirationDate(iso) {
  const d = new Date(iso + "T00:00:00");
  return `${d.getMonth() + 1}/${d.getDate()}`;
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
  // Include the year for anything outside the current one, so a conversation
  // or signed plan from last year can't read as if it were from today.
  const isThisYear = d.getFullYear() === new Date().getFullYear();
  return d.toLocaleString("zh-TW", {
    ...(isThisYear ? {} : { year: "numeric" }),
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
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

/**
 * Renders **bold** spans within a line of chat text as <strong> instead of
 * showing the literal asterisks. Plain string splitting, not
 * dangerouslySetInnerHTML — every fragment is text React escapes on its own.
 */
function renderInlineFormatting(line, keyPrefix) {
  const parts = line.split(/(\*\*[^*]+\*\*)/g).filter((p) => p !== "");
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return (
        <strong key={`${keyPrefix}-${i}`} className="font-semibold text-[#f0ede5]">
          {part.slice(2, -2)}
        </strong>
      );
    }
    return <span key={`${keyPrefix}-${i}`}>{part}</span>;
  });
}

/**
 * Minimal safe markdown for AI chat bubbles: "- "/"* " lines become a
 * bullet list, everything else stays a plain (bold-formatted) line. Covers
 * what the backend's replies actually use without pulling in a full
 * markdown dependency.
 */
function ChatMessageBody({ text }) {
  const lines = text.split("\n");
  const blocks = [];
  let currentList = null;
  for (const line of lines) {
    const bulletMatch = line.match(/^\s*[-*]\s+(.*)$/);
    if (bulletMatch) {
      if (!currentList) {
        currentList = [];
        blocks.push({ type: "list", items: currentList });
      }
      currentList.push(bulletMatch[1]);
    } else {
      currentList = null;
      blocks.push({ type: "line", text: line });
    }
  }
  return (
    <>
      {blocks.map((block, i) => {
        if (block.type === "list") {
          return (
            <ul key={i} className="list-disc pl-4 space-y-0.5 marker:text-[#57575c]">
              {block.items.map((item, j) => (
                <li key={j}>{renderInlineFormatting(item, `${i}-${j}`)}</li>
              ))}
            </ul>
          );
        }
        const rendered = renderInlineFormatting(block.text, `${i}`);
        return <div key={i}>{rendered.length > 0 ? rendered : " "}</div>;
      })}
    </>
  );
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

function GexChart({ putWall, zeroGamma, callWall, stockPrice, gexStatus }) {
  const [hover, setHover] = useState(null); // { i, x, y }

  if (putWall == null || zeroGamma == null || callWall == null) {
    return <div className="text-[11px] text-[#57575c] py-6 text-center">— 無資料 —</div>;
  }
  const levels = [
    { k: putWall, label: "Put Wall", color: "#d8622b", note: "後端回傳的 Put Wall" },
    {
      k: zeroGamma,
      label: "Zero Γ",
      color: "#c9a15c",
      note: gexStatus === "NEG_GAMMA" ? "現貨低於此價，Gamma 翻負" : "現貨高於此價，Gamma 偏正",
    },
    { k: callWall, label: "Call Wall", color: "#2fa37a", note: "後端回傳的 Call Wall" },
  ];
  if (stockPrice != null) {
    levels.push({ k: stockPrice, label: "Spot", color: "#f0ede5", note: "目前標的價格" });
  }
  levels.sort((a, b) => a.k - b.k);
  const W = 300,
    H = 112,
    padX = 24;
  const min = Math.min(...levels.map((level) => level.k));
  const max = Math.max(...levels.map((level) => level.k));
  const range = max - min || 1;
  const xFor = (k) => padX + ((k - min) / range) * (W - padX * 2);
  const hovered = hover ? levels[hover.i] : null;

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} role="img" aria-label="GEX levels">
        <line x1={padX} y1={56} x2={W - padX} y2={56} stroke="rgba(240,237,229,.2)" strokeWidth={1.5} />
        <text x={padX} y={92} fontSize="9" fill="#57575c" textAnchor="middle">
          ${min.toFixed(2)}
        </text>
        <text x={W - padX} y={92} fontSize="9" fill="#57575c" textAnchor="middle">
          ${max.toFixed(2)}
        </text>
        {levels.map((level, i) => {
          const x = xFor(level.k);
          const isHovered = hover?.i === i;

          const enter = (e) => setHover({ i, x: e.clientX, y: e.clientY });
          const move = (e) => setHover({ i, x: e.clientX, y: e.clientY });
          const leave = () => setHover(null);

          return (
            <g
              key={`${level.label}-${level.k}`}
              tabIndex={0}
              onMouseEnter={enter}
              onMouseMove={move}
              onMouseLeave={leave}
              onFocus={(e) => {
                const r = e.currentTarget.getBoundingClientRect();
                setHover({ i, x: r.right, y: r.top });
              }}
              onBlur={leave}
              aria-label={`${level.label} $${level.k.toFixed(2)}`}
              style={{ cursor: "pointer", outline: "none" }}
            >
              <rect x={x - 16} y={24} width={32} height={56} fill="transparent" />
              <line x1={x} y1={34} x2={x} y2={78} stroke={level.color} strokeWidth={isHovered ? 2.5 : 1.5} />
              <circle cx={x} cy={56} r={isHovered ? 5 : 4} fill={level.color} stroke="#121214" strokeWidth={1.5} />
              <text x={x} y={20} fontSize="8.5" fill={level.color} textAnchor="middle">
                {level.label}
              </text>
              <text x={x} y={104} fontSize="9" fill="#c7c7cc" textAnchor="middle">
                ${level.k.toFixed(2)}
              </text>
            </g>
          );
        })}
      </svg>
      {hovered && (
        <ChartTooltip
          point={hover}
          title={hovered.label}
          lines={[`Strike: $${hovered.k.toFixed(2)}`, hovered.note]}
        />
      )}
    </div>
  );
}

function Sparkline({ values, labels = [] }) {
  const [hover, setHover] = useState(null); // { i, x, y }
  if (!values || values.length < 2) {
    return (
      <div className="h-[62px] flex items-center justify-center text-[10.5px] text-[#57575c] border border-dashed border-[rgba(240,237,229,.12)] rounded">
        尚無足夠歷史快照
      </div>
    );
  }
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
  const dayLabel = (i) => labels[i] || (i === values.length - 1 ? "最新" : `${values.length - 1 - i} 筆前`);

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
          lines={[`Net GEX: ${fmtCompactUsd(values[hover.i])}`]}
        />
      )}
    </div>
  );
}

export default function TradingTerminalNotebook() {
  const [ticker, setTicker] = useState(DEFAULT_TICKER);
  const [tickerInput, setTickerInput] = useState(DEFAULT_TICKER);

  const [health, setHealth] = useState({ state: "idle", mode: null, error: null });

  const [expirations, setExpirations] = useState([]);
  // Which ticker the current `expirations` list actually belongs to. Without
  // this the GEX effect can fire the moment `ticker` changes, using the
  // PREVIOUS ticker's expiration dates / DTE.
  const [expirationsTicker, setExpirationsTicker] = useState(null);
  const [expirationsLoading, setExpirationsLoading] = useState(false);
  const [expirationsError, setExpirationsError] = useState(null);
  const [selectedDate, setSelectedDate] = useState(null);
  const [aggregateMode, setAggregateMode] = useState(false);

  const [gexData, setGexData] = useState(null);
  const [gexLoading, setGexLoading] = useState(false);
  const [gexError, setGexError] = useState(null);
  const [gexHistory, setGexHistory] = useState([]);
  const [gexHistoryError, setGexHistoryError] = useState(null);

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);

  const [tradePlan, setTradePlan] = useState(null);
  const [signing, setSigning] = useState(false);
  const [signError, setSignError] = useState(null);
  const [journal, setJournal] = useState([]);
  const [journalError, setJournalError] = useState(null);

  // Mobile-only: which of the three panels is showing (below the `lg` breakpoint).
  const [mobileTab, setMobileTab] = useState("chat");
  const [planBadge, setPlanBadge] = useState(false);

  // ---------- Phase 3: persistent memory ----------
  const [conversations, setConversations] = useState([]);
  const [historyError, setHistoryError] = useState(null);
  const [profileError, setProfileError] = useState(null);
  const [showHistory, setShowHistory] = useState(false);
  const [profile, setProfile] = useState({ risk_tolerance: null, preferred_strategy_types: [], notes: "" });
  const [profileDraft, setProfileDraft] = useState({ risk_tolerance: null, strategyTypesInput: "", notes: "" });
  const [profileSaving, setProfileSaving] = useState(false);
  const [showProfile, setShowProfile] = useState(false);
  const [showJournal, setShowJournal] = useState(false);

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
  const [conversationId, setConversationId] = useState(() => crypto.randomUUID());

  const selectedExpiration = expirations.find((e) => e.date === selectedDate) || null;
  const aggregateExpirations = expirations.slice(0, MAX_AGGREGATE_EXPIRATIONS);
  // null means "no expiration is actually resolved for the CURRENT ticker"
  // (mid-switch, or the expirations fetch failed). Consumers that would
  // otherwise present a placeholder 0 as a real 0DTE — notably the trade
  // journal's entry GEX snapshot — need to tell those two cases apart.
  const resolvedDte =
    expirationsTicker !== ticker
      ? null
      : aggregateMode
        ? aggregateExpirations.length
          ? Math.min(...aggregateExpirations.map((e) => e.days_to_expiration))
          : null
        : selectedExpiration?.days_to_expiration ?? null;
  // Backend endpoints that still require an integer get a numeric default, but
  // chat can send null so the copilot never treats "unknown right now" as 0DTE.
  const dte = resolvedDte ?? 0;

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

  // ---------- Expirations fetch (ticker change drives the real, dynamic list) ----------
  useEffect(() => {
    let cancelled = false;
    // A new ticker invalidates everything derived from the old one. Clearing
    // up-front is what keeps the previous symbol's expirations (and the DTE
    // derived from them, which also feeds the chat payload) from being used
    // under the new symbol's header.
    setExpirations([]);
    setExpirationsTicker(null);
    setSelectedDate(null);
    setGexData(null);
    setGexHistory([]);
    setGexHistoryError(null);
    setGexError(null);
    setGexLoading(true);
    setExpirationsLoading(true);
    setExpirationsError(null);
    fetch(`${BASE_URL}/api/v1/expirations/${ticker}`)
      .then(async (res) => {
        if (!res.ok) throw new Error(await parseErrorDetail(res));
        return res.json();
      })
      .then((data) => {
        if (cancelled) return;
        const list = (data.expirations || []).slice(0, MAX_EXPIRATIONS_SHOWN);
        setExpirations(list);
        setExpirationsTicker(ticker);
        setExpirationsLoading(false);
        const defaultPick =
          list.find((e) => e.expiration_type !== "0DTE" && e.expiration_type !== "1DTE") || list[0];
        setSelectedDate(defaultPick ? defaultPick.date : null);
      })
      .catch((err) => {
        if (cancelled) return;
        setExpirationsError(err.message || "無法取得到期日");
        setExpirationsLoading(false);
        // Nothing will fetch GEX now, so don't leave the panel spinning.
        setGexLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [ticker]);

  useEffect(() => {
    let cancelled = false;
    setGexHistory([]);
    setGexHistoryError(null);
    fetch(`${BASE_URL}/api/v1/gex/${ticker}/history?limit=30`)
      .then(async (res) => {
        if (!res.ok) throw new Error(await parseErrorDetail(res));
        return res.json();
      })
      .then((data) => {
        if (cancelled) return;
        setGexHistory(data.snapshots || []);
      })
      .catch((err) => {
        if (cancelled) return;
        setGexHistoryError(err.message || "無法取得 GEX 歷史");
      });
    return () => {
      cancelled = true;
    };
  }, [ticker]);

  // ---------- GEX fetch (expiration / aggregate toggle drives header + left panel) ----------
  useEffect(() => {
    // The expirations list in state may still be the previous ticker's — never
    // build a GEX request out of it.
    if (expirationsTicker !== ticker) return;
    if (aggregateMode ? expirations.length === 0 : !selectedDate) {
      setGexLoading(false);
      return;
    }
    let cancelled = false;
    setGexLoading(true);
    setGexError(null);
    const url = aggregateMode
      ? `${BASE_URL}/api/v1/gex/${ticker}/aggregate?` +
        expirations
          .slice(0, MAX_AGGREGATE_EXPIRATIONS)
          .map((e) => `expirations=${e.date}`)
          .join("&")
      : `${BASE_URL}/api/v1/gex/${ticker}?days_to_expiration=${dte}`;
    fetch(url)
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
        // Keeping the previous payload on screen would render last-known
        // numbers as if they were current — drop them and show the reason.
        setGexData(null);
        setGexError(err.message || "無法連接後端");
        setGexLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ticker, selectedDate, aggregateMode, expirations, expirationsTicker]);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages, chatLoading]);

  function commitTicker() {
    const next = tickerInput.trim().toUpperCase();
    if (next) setTicker(next);
    else setTickerInput(ticker);
  }

  function refreshConversations({ clearError = true } = {}) {
    return fetch(`${BASE_URL}/api/v1/conversations?user_id=${userId}`)
      .then(async (res) => {
        if (!res.ok) throw new Error(await parseErrorDetail(res));
        return res.json();
      })
      .then((data) => {
        setConversations(data.conversations || []);
        if (clearError) setHistoryError(null);
      })
      // A failed list used to be indistinguishable from "no history yet".
      .catch((err) => setHistoryError(err.message || "無法載入歷史對話"));
  }

  // ---------- Phase 3: load persisted history, plans, and profile once ----------
  function loadSignedPlans() {
    return fetch(`${BASE_URL}/api/v1/plans?user_id=${userId}`)
      .then(async (res) => {
        if (!res.ok) throw new Error(await parseErrorDetail(res));
        return res.json();
      })
      .then((data) => {
        const plans = (data.plans || []).map((p) => ({
          strategy: p.strategy_type,
          time: fmtDateTime(p.signed_at),
        }));
        if (plans.length) setJournal(plans);
        setJournalError(null);
      })
      // Otherwise a failed load reads as "you have never signed a plan".
      .catch((err) => setJournalError(err.message || "無法載入已簽署計畫"));
  }

  function loadProfile() {
    return fetch(`${BASE_URL}/api/v1/profile/${userId}`)
      .then(async (res) => {
        if (!res.ok) throw new Error(await parseErrorDetail(res));
        return res.json();
      })
      .then((data) => {
        if (!data) return;
        setProfile(data);
        setProfileDraft({
          risk_tolerance: data.risk_tolerance,
          strategyTypesInput: (data.preferred_strategy_types || []).join(", "),
          notes: data.notes || "",
        });
        setProfileError(null);
      })
      // A silent failure here shows an empty preferences form that looks
      // saved — the user would overwrite their real profile with blanks.
      .catch((err) => setProfileError(err.message || "無法載入交易偏好設定"));
  }

  useEffect(() => {
    refreshConversations();
    loadSignedPlans();
    loadProfile();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function resumeConversation(conversationSummary) {
    setHistoryError(null);
    try {
      const res = await fetch(
        `${BASE_URL}/api/v1/conversations/${conversationSummary.conversation_id}/messages?user_id=${userId}`
      );
      if (!res.ok) throw new Error(await parseErrorDetail(res));
      const data = await res.json();
      setMessages((data.messages || []).map((m) => ({ role: m.role, content: m.content })));
      setConversationId(conversationSummary.conversation_id);
      setTradePlan(null);
      setShowHistory(false);
    } catch (err) {
      // A failed resume used to look identical to a successful one that
      // happened to load nothing — keep the panel open and say why.
      setHistoryError(err.message || "無法載入這段對話");
    }
  }

  function startNewChat() {
    setShowHistory(false);
    setMessages([]);
    setConversationId(crypto.randomUUID());
    setTradePlan(null);
  }

  async function saveProfile() {
    setProfileSaving(true);
    setProfileError(null);
    try {
      const res = await fetch(`${BASE_URL}/api/v1/profile/${userId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          risk_tolerance: profileDraft.risk_tolerance,
          preferred_strategy_types: profileDraft.strategyTypesInput
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean),
          notes: profileDraft.notes,
        }),
      });
      if (!res.ok) throw new Error(await parseErrorDetail(res));
      const saved = await res.json();
      setProfile(saved);
      setProfileDraft({
        risk_tolerance: saved.risk_tolerance,
        strategyTypesInput: (saved.preferred_strategy_types || []).join(", "),
        notes: saved.notes || "",
      });
      setShowProfile(false);
    } catch (err) {
      // The draft stays open so the user can retry — but they need to know
      // the save didn't land, otherwise the panel silently closes on nothing.
      setProfileError(err.message || "儲存偏好失敗");
    } finally {
      setProfileSaving(false);
    }
  }

  // ---------- Chat ----------
  async function sendMessage(overrideText) {
    const text = (overrideText ?? input).trim();
    if (!text || chatLoading) return;
    const restoreInput = overrideText === undefined ? input : null;
    // `localOnly` bubbles are client-side error notices the model never
    // actually said — sending them back as history would tell it that it did.
    const priorHistory = messages
      .filter((m) => !m.localOnly)
      .slice(-MAX_HISTORY)
      .map((m) => ({ role: m.role, content: m.content }));
    // Tagged so a failure can take the optimistic bubble back out again —
    // otherwise a retry leaves the same user turn in the transcript twice,
    // with no assistant reply between them.
    const pendingId = crypto.randomUUID();
    setMessages((m) => [...m, { role: "user", content: text, pendingId }]);
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
            days_to_expiration: resolvedDte,
            // Match what the GEX panel actually shows: when Aggregate mode
            // is on, the copilot has to reason over the same combined
            // dataset, not a single nearest-expiration snapshot the user
            // isn't looking at.
            aggregate: aggregateMode,
            expiration_dates: aggregateMode
              ? aggregateExpirations.map((e) => e.date)
              : [],
            history: priorHistory,
          },
        }),
      });
      if (!res.ok) throw new Error(await parseErrorDetail(res));
      const data = await res.json();
      setMessages((m) => [
        // The turn is committed now — drop the pending marker.
        ...m.map((msg) =>
          msg.pendingId === pendingId ? { role: msg.role, content: msg.content } : msg
        ),
        { role: "assistant", content: data.assistant_message },
      ]);
      if (data.trade_plan_card) {
        setTradePlan(data.trade_plan_card);
        setPlanBadge(true);
      }
      refreshConversations({ clearError: false });
    } catch (err) {
      const detail = err.message || "未知錯誤";
      // Only a thrown fetch (TypeError) is an actual connectivity failure; a
      // 429 or other 4xx reached the backend just fine and deserves its own
      // message rather than "can't reach the backend".
      const isConnectivity = err instanceof TypeError;
      setMessages((m) => [
        // Roll the optimistic user bubble back out; the text goes back into
        // the input instead, so a retry sends exactly one copy of the turn.
        ...m.filter((msg) => msg.pendingId !== pendingId),
        {
          role: "assistant",
          content: isConnectivity ? `⚠️ 無法連接後端 (${BASE_URL})：${detail}` : `⚠️ ${detail}`,
          localOnly: true,
        },
      ]);
      // Don't make the user retype what they just sent.
      if (restoreInput !== null) setInput(restoreInput);
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
  const gexHistoryChronological = [...gexHistory].reverse();
  const gexHistoryValues = gexHistoryChronological.map((s) => s.net_gex);
  const gexHistoryLabels = gexHistoryChronological.map((s) => fmtDateTime(s.captured_at));

  return (
    <div className={`h-screen flex flex-col bg-[#121214] text-[#f0ede5] ${MONO} text-[13px] overflow-hidden`}>
      {/* HEADER */}
      <header className="flex items-center gap-4 lg:gap-7 px-4 lg:px-6 py-3.5 border-b border-[rgba(240,237,229,.09)] bg-gradient-to-b from-[#1b1b1e] to-[#121214] overflow-x-auto">
        <div className="flex items-baseline gap-2.5 shrink-0">
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
        <div className="flex items-baseline gap-2 shrink-0">
          {gexLoading ? (
            <span className="text-[13px] font-semibold text-[#8d8d93] animate-pulse">載入中…</span>
          ) : (
            <span className="text-[22px] font-bold [font-variant-numeric:tabular-nums]">
              {fmtDollar(gexData?.stock_price)}
            </span>
          )}
        </div>
        <div className="flex gap-2.5 ml-1.5 shrink-0">
          <MetricChip label="Zero Γ" value={fmtDollar(gexData?.zero_gamma)} color="#c9a15c" loading={gexLoading} />
          <MetricChip label="Call Wall" value={fmtDollar(gexData?.call_wall)} color="#2fa37a" loading={gexLoading} />
          <MetricChip label="Put Wall" value={fmtDollar(gexData?.put_wall)} color="#d8622b" loading={gexLoading} />
        </div>
        <div className="lg:ml-auto flex items-center gap-3 shrink-0">
          {gexError && !gexLoading && (
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
        </div>
      </header>

      {/* BODY */}
      <div className="flex-1 min-h-0 flex flex-col lg:grid lg:grid-cols-[292px_minmax(380px,1fr)_352px] gap-px bg-[rgba(240,237,229,.09)] pb-14 lg:pb-0">
        {/* LEFT: GEX PANEL */}
        <section className={`${mobileTab === "gex" ? "flex" : "hidden"} lg:flex flex-1 flex-col min-h-0 bg-[#121214] overflow-hidden`}>
          <PaneHeader icon={<Activity size={13} />} title="Gamma / GEX" />
          <div className="flex-1 min-h-0 overflow-y-auto px-4 pt-[18px] pb-5">
            <select
              value={aggregateMode ? "__aggregate__" : selectedDate || ""}
              onChange={(e) => {
                if (e.target.value === "__aggregate__") {
                  setAggregateMode(true);
                } else {
                  setAggregateMode(false);
                  setSelectedDate(e.target.value);
                }
              }}
              disabled={expirationsLoading || expirations.length === 0}
              className={`w-full mb-3.5 bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2.5 py-2 text-[11.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] disabled:opacity-50 ${MONO}`}
              title="選擇到期日，或切換至全到期日總合"
            >
              {expirationsLoading && <option>載入到期日中…</option>}
              {!expirationsLoading &&
                expirations.map((e) => (
                  <option key={e.date} value={e.date}>
                    {fmtExpirationDate(e.date)} · {EXPIRATION_TYPE_LABEL[e.expiration_type] || e.expiration_type} ·{" "}
                    {e.days_to_expiration} DTE
                  </option>
                ))}
              {!expirationsLoading && (
                <option value="__aggregate__">📊 Aggregate GEX（全到期日總合）</option>
              )}
            </select>

            {expirationsError && (
              <div className="text-[10.5px] text-[#d8622b] mb-3 leading-snug">⚠ {expirationsError}</div>
            )}

            {gexLoading ? (
              <GexPanelSkeleton ticker={ticker} />
            ) : gexError || !gexData ? (
              <div className="border border-[rgba(216,98,44,.35)] bg-[rgba(216,98,44,.08)] rounded px-3 py-3 text-[11px] leading-relaxed">
                <div className="text-[#d8622b] font-bold mb-1">
                  {gexError ? `${ticker} GEX 資料載入失敗` : `${ticker} 尚無 GEX 資料`}
                </div>
                <div className="text-[#8d8d93]">
                  {gexError || "請選擇到期日，或稍後再試。"}
                </div>
                <div className="text-[9.5px] text-[#57575c] mt-1.5">
                  為避免誤導，這裡不會保留上一次查詢的數字。
                </div>
              </div>
            ) : (
              <>
                <div className="grid grid-cols-2 gap-2 mb-3">
                  <Stat label="Net GEX" value={fmtCompactUsd(gexData.net_gex)} tone={netGexTone} />
                  <Stat label="IV Rank" value={`${gexData.iv_rank.toFixed(0)}%`} />
                </div>

                <PinningCard pinning={gexData.pinning} />

                <Card>
                  <CardTitle
                    title="GEX Levels"
                    tag={
                      aggregateMode
                        ? `Aggregate · ${aggregateExpirations.length} 檔到期日`
                        : selectedExpiration
                          ? `${fmtExpirationDate(selectedExpiration.date)} exp · ${selectedExpiration.days_to_expiration} DTE`
                          : "—"
                    }
                  />
                  <GexChart
                    putWall={gexData.put_wall}
                    zeroGamma={gexData.zero_gamma}
                    callWall={gexData.call_wall}
                    stockPrice={gexData.stock_price}
                    gexStatus={gexData.gex_status}
                  />
                  <div className="flex flex-wrap gap-3.5 text-[10px] text-[#8d8d93] mt-1.5">
                    <span className="flex items-center gap-1.5">
                      <i className="w-2 h-2 rounded-full inline-block bg-[#d8622b]" />
                      Put Wall
                    </span>
                    <span className="flex items-center gap-1.5">
                      <i className="w-2 h-2 rounded-full inline-block bg-[#c9a15c]" />
                      Zero Γ
                    </span>
                    <span className="flex items-center gap-1.5">
                      <i className="w-2 h-2 rounded-full inline-block bg-[#2fa37a]" />
                      Call Wall
                    </span>
                  </div>
                </Card>

                <Card>
                  <CardTitle
                    title="Net GEX Trend"
                    tag={gexHistoryValues.length ? `${gexHistoryValues.length} snapshots` : "No history"}
                    tagTitle={
                      gexHistoryError ||
                      "來自後端 gex_snapshots；沒有真實快照時不顯示假趨勢"
                    }
                  />
                  <Sparkline values={gexHistoryValues} labels={gexHistoryLabels} />
                  {gexHistoryError && (
                    <div className="text-[9.5px] text-[#d8622b] mt-1.5">
                      ⚠ {gexHistoryError}
                    </div>
                  )}
                </Card>
              </>
            )}
          </div>
        </section>

        {/* MIDDLE: AI CHAT */}
        <section className={`${mobileTab === "chat" ? "flex" : "hidden"} lg:flex flex-1 flex-col min-h-0 bg-[#121214] overflow-hidden relative`}>
          <PaneHeader
            icon={<PenTool size={13} className="rotate-90" />}
            title="AI Copilot"
            trailing={
              <div className="flex items-center gap-3">
                <span className="text-[10px] text-[#2fa37a] flex items-center gap-1.5">
                  <i className={`w-1.5 h-1.5 rounded-full bg-[#2fa37a] ${chatLoading ? "animate-pulse" : ""}`} />
                  {chatLoading ? "thinking…" : "reading left panel"}
                </span>
                <div className="flex items-center gap-1">
                  <button
                    type="button"
                    onClick={startNewChat}
                    title="開始新對話"
                    className="w-6 h-6 flex items-center justify-center rounded text-[#8d8d93] hover:text-[#f0ede5] hover:bg-[rgba(240,237,229,.08)] transition-colors"
                  >
                    <Plus size={13} />
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setShowHistory((v) => !v);
                      setShowProfile(false);
                      setShowJournal(false);
                      refreshConversations();
                    }}
                    title="歷史對話"
                    className={`w-6 h-6 flex items-center justify-center rounded transition-colors ${
                      showHistory ? "text-[#c9a15c] bg-[rgba(201,161,92,.13)]" : "text-[#8d8d93] hover:text-[#f0ede5] hover:bg-[rgba(240,237,229,.08)]"
                    }`}
                  >
                    <History size={13} />
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setShowProfile((v) => !v);
                      setShowHistory(false);
                      setShowJournal(false);
                    }}
                    title="交易偏好設定"
                    className={`w-6 h-6 flex items-center justify-center rounded transition-colors ${
                      showProfile ? "text-[#c9a15c] bg-[rgba(201,161,92,.13)]" : "text-[#8d8d93] hover:text-[#f0ede5] hover:bg-[rgba(240,237,229,.08)]"
                    }`}
                  >
                    <Settings size={13} />
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setShowJournal((v) => !v);
                      setShowHistory(false);
                      setShowProfile(false);
                    }}
                    title="交易日誌"
                    className={`w-6 h-6 flex items-center justify-center rounded transition-colors ${
                      showJournal ? "text-[#c9a15c] bg-[rgba(201,161,92,.13)]" : "text-[#8d8d93] hover:text-[#f0ede5] hover:bg-[rgba(240,237,229,.08)]"
                    }`}
                  >
                    <BookOpen size={13} />
                  </button>
                </div>
              </div>
            }
          />

          {showHistory && (
            <HistoryPanel
              conversations={conversations}
              activeConversationId={conversationId}
              error={historyError}
              onSelect={resumeConversation}
              onRetry={() => refreshConversations()}
              onClose={() => setShowHistory(false)}
            />
          )}

          {showProfile && (
            <ProfilePanel
              draft={profileDraft}
              setDraft={setProfileDraft}
              saving={profileSaving}
              error={profileError}
              onSave={saveProfile}
              onRetry={loadProfile}
              onClose={() => setShowProfile(false)}
            />
          )}

          {showJournal && (
            <TradeJournalPanel
              userId={userId}
              ticker={ticker}
              expirationDate={aggregateMode ? null : selectedExpiration?.date ?? null}
              onClose={() => setShowJournal(false)}
            />
          )}

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
                    <ChatMessageBody text={m.content} />
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
        <section className={`${mobileTab === "plan" ? "flex" : "hidden"} lg:flex flex-1 flex-col min-h-0 bg-[#121214] overflow-hidden`}>
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
              {/* Signed AI *plans*, not executed trades — the trade journal
                  overlay (BookOpen icon in the chat pane) is a different thing. */}
              <div className="text-[10px] tracking-wider uppercase text-[#8d8d93] mb-2.5">
                已簽署計畫 · AI 建議紀錄
              </div>
              {journalError ? (
                <div className="px-0.5 py-1.5">
                  <div className="text-[10.5px] text-[#d8622b] leading-snug">
                    ⚠ 無法載入已簽署計畫（{journalError}）— 這不代表沒有紀錄。
                  </div>
                  <button
                    type="button"
                    onClick={loadSignedPlans}
                    className="mt-1.5 inline-flex items-center gap-1 text-[10px] text-[#c9a15c] hover:text-[#d8b06c]"
                  >
                    <RefreshCw size={11} />
                    重試
                  </button>
                </div>
              ) : journal.length === 0 ? (
                <div className="text-[11px] text-[#57575c] px-0.5 py-1.5">尚無已簽署計畫 — 簽署上方計畫卡後會出現在這裡</div>
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

      {/* MOBILE BOTTOM NAV — hidden at lg: and up, where all three panes show at once */}
      <nav className="lg:hidden fixed bottom-0 inset-x-0 z-40 flex border-t border-[rgba(240,237,229,.09)] bg-[#1b1b1e] pb-[env(safe-area-inset-bottom)]">
        <BottomNavTab icon="📊" label="GEX 籌碼" active={mobileTab === "gex"} onClick={() => setMobileTab("gex")} />
        <BottomNavTab icon="💬" label="AI Copilot" active={mobileTab === "chat"} onClick={() => setMobileTab("chat")} />
        <BottomNavTab
          icon="📝"
          label="交易計畫"
          active={mobileTab === "plan"}
          badge={planBadge}
          onClick={() => {
            setMobileTab("plan");
            setPlanBadge(false);
          }}
        />
      </nav>

      <style>{`
        @keyframes stampDown { from { opacity:0; transform:rotate(-16deg) scale(2.4); } to { opacity:1; transform:rotate(-16deg) scale(1); } }
        @keyframes cardIn { from { opacity:0; transform: translateY(-14px) rotate(-0.6deg); } to { opacity:1; transform: translateY(0) rotate(-0.6deg); } }
      `}</style>
    </div>
  );
}

function BottomNavTab({ icon, label, active, badge, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex-1 flex flex-col items-center justify-center gap-0.5 py-2 text-[10px] font-semibold transition-colors ${
        active ? "text-[#c9a15c]" : "text-[#8d8d93]"
      }`}
    >
      <span className="relative">
        <span className="text-base leading-none">{icon}</span>
        {badge && <span className="absolute -top-1 -right-1.5 w-2 h-2 rounded-full bg-[#d8622b] ring-2 ring-[#1b1b1e]" />}
      </span>
      {label}
    </button>
  );
}

function HistoryPanel({ conversations, activeConversationId, error, onSelect, onRetry, onClose }) {
  return (
    <div className="absolute inset-x-0 top-11 bottom-0 z-30 bg-[#121214] border-t border-[rgba(240,237,229,.09)] flex flex-col">
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-[rgba(240,237,229,.09)]">
        <span className="text-[10px] tracking-wider uppercase text-[#8d8d93] font-semibold">歷史對話</span>
        <button type="button" onClick={onClose} className="text-[#8d8d93] hover:text-[#f0ede5]">
          <X size={14} />
        </button>
      </div>
      <div className="flex-1 overflow-y-auto px-3 py-2.5 flex flex-col gap-1.5">
        {error && (
          <div className="px-1 pb-1">
            <div className="text-[10.5px] text-[#d8622b] leading-snug">⚠ {error}</div>
            <button
              type="button"
              onClick={onRetry}
              className="mt-1 inline-flex items-center gap-1 text-[10px] text-[#c9a15c] hover:text-[#d8b06c]"
            >
              <RefreshCw size={11} />
              重試
            </button>
          </div>
        )}
        {conversations.length === 0 && (
          <div className="text-[11px] text-[#57575c] text-center mt-8">尚無歷史對話</div>
        )}
        {conversations.map((c) => (
          <button
            key={c.conversation_id}
            type="button"
            onClick={() => onSelect(c)}
            className={`text-left px-3 py-2.5 rounded-md border transition-colors ${
              c.conversation_id === activeConversationId
                ? "border-[rgba(201,161,92,.4)] bg-[rgba(201,161,92,.08)]"
                : "border-[rgba(240,237,229,.09)] bg-[#1b1b1e] hover:border-[rgba(240,237,229,.2)]"
            }`}
          >
            <div className="flex items-center justify-between mb-1">
              <span className="text-[11px] font-bold text-[#f0ede5]">{c.ticker || "—"}</span>
              <span className="text-[9.5px] text-[#57575c]">{fmtDateTime(c.last_message_at)}</span>
            </div>
            <div className="text-[10.5px] text-[#8d8d93] line-clamp-2 leading-snug">{c.last_message}</div>
            <div className="text-[9px] text-[#57575c] mt-1">{c.message_count} 則訊息</div>
          </button>
        ))}
      </div>
    </div>
  );
}

function ProfilePanel({ draft, setDraft, saving, error, onSave, onRetry, onClose }) {
  const options = [
    { value: "CONSERVATIVE", label: "保守" },
    { value: "BALANCED", label: "中性" },
    { value: "AGGRESSIVE", label: "激進" },
  ];
  return (
    <div className="absolute inset-x-0 top-11 bottom-0 z-30 bg-[#121214] border-t border-[rgba(240,237,229,.09)] flex flex-col">
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-[rgba(240,237,229,.09)]">
        <span className="text-[10px] tracking-wider uppercase text-[#8d8d93] font-semibold">交易偏好設定</span>
        <button type="button" onClick={onClose} className="text-[#8d8d93] hover:text-[#f0ede5]">
          <X size={14} />
        </button>
      </div>
      <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-4">
        <div>
          <div className="text-[10px] tracking-wider uppercase text-[#57575c] mb-2">風險偏好</div>
          <div className="flex gap-1.5">
            {options.map((opt) => (
              <button
                key={opt.value}
                type="button"
                onClick={() =>
                  setDraft((d) => ({
                    ...d,
                    risk_tolerance: d.risk_tolerance === opt.value ? null : opt.value,
                  }))
                }
                className={`flex-1 py-2 rounded-md text-[11.5px] font-semibold border transition-colors ${
                  draft.risk_tolerance === opt.value
                    ? "bg-[#c9a15c] text-[#1a1408] border-[#c9a15c]"
                    : "border-[rgba(240,237,229,.09)] text-[#8d8d93] hover:text-[#f0ede5]"
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
        <div>
          <div className="text-[10px] tracking-wider uppercase text-[#57575c] mb-2">常用策略（逗號分隔）</div>
          <input
            value={draft.strategyTypesInput}
            onChange={(e) => setDraft((d) => ({ ...d, strategyTypesInput: e.target.value }))}
            placeholder="例如：Put Bear Spread, Iron Condor"
            className={`w-full bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2.5 py-2 text-[11.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
          />
        </div>
        <div>
          <div className="text-[10px] tracking-wider uppercase text-[#57575c] mb-2">備註</div>
          <textarea
            value={draft.notes}
            onChange={(e) => setDraft((d) => ({ ...d, notes: e.target.value }))}
            rows={4}
            placeholder="任何想讓 AI 記住的偏好…"
            className={`w-full bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2.5 py-2 text-[11.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] resize-none ${MONO}`}
          />
        </div>
      </div>
      <div className="px-4 py-3 border-t border-[rgba(240,237,229,.09)]">
        {error && (
          <div className="mb-2">
            <div className="text-[10.5px] text-[#d8622b] leading-snug">⚠ {error}</div>
            <button
              type="button"
              onClick={onRetry}
              className="mt-1 inline-flex items-center gap-1 text-[10px] text-[#c9a15c] hover:text-[#d8b06c]"
            >
              <RefreshCw size={11} />
              重試
            </button>
          </div>
        )}
        <button
          type="button"
          onClick={onSave}
          disabled={saving}
          className="w-full py-2.5 rounded-md bg-[#c9a15c] text-[#1a1408] text-[11.5px] font-bold uppercase tracking-wide hover:bg-[#d8b06c] disabled:opacity-50 transition-colors"
        >
          {saving ? "儲存中…" : "儲存偏好"}
        </button>
      </div>
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
    label = "已連線至 Moomoo 後端（即時）";
  } else if (health.state === "ok" && health.mode === "yfinance") {
    dot = "#c9a15c";
    label = "已連線（Yahoo Finance · 約 15-20 分鐘延遲）";
  } else if (health.state === "ok" && health.mode === "unavailable") {
    dot = "#d8622b";
    label = "市場資料源未設定";
  } else if (health.state === "ok" && health.mode === "mock") {
    dot = "#d8622b";
    label = "已連線（Demo/Mock 模式 · 非真實資料）";
  } else if (health.state === "ok") {
    dot = "#8d8d93";
    label = `已連線（${health.mode || "unknown"} 模式 · 請確認資料來源）`;
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

function MetricChip({ label, value, color, loading }) {
  return (
    <div
      className={`flex flex-col gap-0.5 px-3 py-1.5 bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded-sm min-w-[88px] ${
        loading ? "opacity-40 animate-pulse" : ""
      }`}
    >
      <span className="text-[9.5px] tracking-wider uppercase text-[#8d8d93]">{label}</span>
      <span
        className="text-sm font-bold [font-variant-numeric:tabular-nums]"
        style={{ color: loading ? "#57575c" : color }}
      >
        {loading ? "···" : value}
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

const PINNING_REGIME_STYLE = {
  PINNING: { label: "Pinning · 磁吸區間", bg: "rgba(47,163,122,.16)", border: "rgba(47,163,122,.4)", fg: "#2fa37a" },
  BREAKOUT: { label: "Breakout · 突破區間", bg: "rgba(216,98,44,.16)", border: "rgba(216,98,44,.4)", fg: "#d8622b" },
  NEUTRAL: { label: "Neutral · 中性觀望", bg: "rgba(141,141,147,.14)", border: "rgba(141,141,147,.35)", fg: "#8d8d93" },
};

/**
 * 顯示 backend app/pinning_engine.py 算出的 Pinning 判定——跟 Zero Γ/Call
 * Wall/Put Wall 用的是同一份真實 Moomoo 期權鏈資料（後端在同一次抓取裡
 * 附加上去，不是另外打一次 API），所以這裡的數字永遠跟上方的 GEX 卡片
 * 一致，不會有兩邊對不上的問題。pinning 為 null 時（加分項計算失敗）
 * 顯示佔位文字，不讓整張卡片消失造成版面跳動。
 */
function PinningCard({ pinning }) {
  const style = pinning ? PINNING_REGIME_STYLE[pinning.regime] || PINNING_REGIME_STYLE.NEUTRAL : null;
  return (
    <Card>
      <CardTitle
        title="Pinning 判定"
        tag={pinning ? `${pinning.score}/100` : "—"}
        tagTitle="做市商磁吸/卡價效應分數（0~100，規則透明評分，非黑箱模型）"
      />
      {!pinning ? (
        <div className="text-[10.5px] text-[#57575c]">尚無 Pinning 資料</div>
      ) : (
        <>
          <div
            className="inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-bold mb-3"
            style={{ background: style.bg, border: `1px solid ${style.border}`, color: style.fg }}
          >
            {style.label}
          </div>
          <div className="grid grid-cols-2 gap-2">
            <Stat label="Pin Strike" value={fmtDollar(pinning.pin_strike)} />
            <Stat label="距離" value={`${pinning.distance_pct.toFixed(2)}%`} />
            <Stat label="OI 集中度" value={`${pinning.oi_concentration_pct.toFixed(1)}%`} />
            <Stat
              label="正 Gamma"
              value={pinning.in_positive_gamma ? "是" : "否"}
              tone={pinning.in_positive_gamma ? "bull" : "bear"}
            />
          </div>
        </>
      )}
    </Card>
  );
}

/**
 * Shown while a GEX request for the *current* ticker is in flight. The point
 * is that nothing here is a number: a stale payload dimmed out would still
 * read as data.
 */
function GexPanelSkeleton({ ticker }) {
  return (
    <div className="animate-pulse">
      <div className="text-[10.5px] text-[#c9a15c] mb-3">載入 {ticker} GEX 資料中…</div>
      <div className="grid grid-cols-2 gap-2 mb-3">
        {[0, 1].map((i) => (
          <div key={i} className="bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded-sm px-2.5 py-2">
            <div className="h-2 w-12 rounded bg-[rgba(240,237,229,.09)] mb-2" />
            <div className="h-3.5 w-16 rounded bg-[rgba(240,237,229,.06)]" />
          </div>
        ))}
      </div>
      {[110, 190, 90].map((h, i) => (
        <div
          key={i}
          className="bg-[#1b1b1e] border border-[rgba(240,237,229,.09)] rounded mb-3"
          style={{ height: h }}
        />
      ))}
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
