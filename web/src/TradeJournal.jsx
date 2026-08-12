import { useEffect, useRef, useState } from "react";
import { Plus, Star, X } from "lucide-react";
import { parseErrorDetail } from "./apiError.js";
import { apiUrl, pathSegment } from "./apiUrl.js";

const MONO =
  '[font-family:ui-monospace,"SF_Mono","JetBrains_Mono","IBM_Plex_Mono",Menlo,Consolas,monospace]';

const COMMON_STRATEGY_TYPES = [
  "WEEKLY_CSP",
  "DEFI_LP",
  "Long Call",
  "Long Put",
  "Bull Call Debit Spread",
  "Bear Put Debit Spread",
  "Bull Put Credit Spread",
  "Bear Call Credit Spread",
  "Covered Call",
  "Cash-Secured Put",
  "Defined-Risk Iron Condor",
];

const TRADE_DIRECTIONS = [
  { value: "LONG", label: "看多" },
  { value: "SHORT", label: "看空" },
  { value: "NEUTRAL", label: "中性" },
];

const CREDIT_DEBIT_TYPES = [
  { value: "DEBIT", label: "Debit" },
  { value: "CREDIT", label: "Credit" },
];

const OPTION_TYPES = [
  { value: "CALL", label: "Call" },
  { value: "PUT", label: "Put" },
  { value: "MULTI_LEG", label: "Multi-leg" },
];

const WHEEL_STAGES = [
  { value: "CSP", label: "CSP 賣 Put" },
  { value: "ASSIGNED_STOCK", label: "已指派接股" },
  { value: "COVERED_CALL", label: "Covered Call" },
  { value: "COMPLETED", label: "循環完成" },
];

// One options contract controls 100 shares. The backend uses the same
// multiplier when it derives pnl_pct (pnl / (entry_price * 100 * size) * 100),
// so the close form's derived figures have to match it exactly.
const CONTRACT_MULTIPLIER = 100;

// How far a hand-entered PnL may drift from the derived one (slippage,
// commissions, partial fills are all legitimate) before we flag it.
const PNL_MISMATCH_TOLERANCE = 0.2;

/** Finite and > 0, else null. For fields the backend declares `gt=0`. */
function parsePositiveNumber(raw) {
  const s = String(raw ?? "").trim();
  if (s === "") return null;
  const n = Number(s);
  return Number.isFinite(n) && n > 0 ? n : null;
}

/** Finite and >= 0, else null. For yield targets. */
function parseNonNegativeNumber(raw) {
  const s = String(raw ?? "").trim();
  if (s === "") return null;
  const n = Number(s);
  return Number.isFinite(n) && n >= 0 ? n : null;
}

/** Finite (negative allowed), else null. For PnL. */
function parseFiniteNumber(raw) {
  const s = String(raw ?? "").trim();
  if (s === "") return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}

/** Strictly a positive integer, else null. No silent coercion to 1. */
function parsePositiveInt(raw) {
  const s = String(raw ?? "").trim();
  if (!/^\d+$/.test(s)) return null;
  const n = Number(s);
  return n > 0 ? n : null;
}

/**
 * Credit structures invert the debit convention: `entry_price` is premium
 * COLLECTED and `exit_price` is what you pay to buy it back, so profit is
 * (entry − exit), not (exit − entry). Getting this backwards turns a winning
 * short put into a headline loss of the same magnitude.
 *
 * "Iron Condor" is included because the datalist's entry is the standard
 * defined-risk (credit) construction. A debit "reverse iron condor" would be
 * mis-inferred, which is why the applied convention is always labelled in the
 * UI and the manual PnL field stays authoritative.
 */
const CREDIT_STRATEGY_RE = /weekly_csp|credit|covered\s*call|cash[-\s]?secured|iron\s*condor/i;

function isCreditStrategy(strategyType) {
  return CREDIT_STRATEGY_RE.test(String(strategyType || ""));
}

function inferDirection(strategyType) {
  const text = String(strategyType || "").toLowerCase();
  if (/weekly_csp|cash[-\s]?secured/.test(text)) return "LONG";
  if (/(iron\s*condor|butterfly|calendar)/i.test(text)) return "NEUTRAL";
  // "bull" must be checked before the generic short/put match below — a
  // Bull Put Credit Spread is bullish despite containing "put".
  if (/bull/i.test(text)) return "LONG";
  if (/(bear|put|short|sell|covered\s*call|cash[-\s]?secured)/i.test(text)) return "SHORT";
  return "LONG";
}

function inferCreditDebit(strategyType) {
  return isCreditStrategy(strategyType) ? "CREDIT" : "DEBIT";
}

function inferOptionType(strategyType) {
  const text = String(strategyType || "").toLowerCase();
  if (/weekly_csp|cash[-\s]?secured/.test(text)) return "PUT";
  if (/(spread|condor|butterfly|calendar|straddle|strangle|ratio)/i.test(text)) {
    return "MULTI_LEG";
  }
  if (/\bput\b/i.test(text)) return "PUT";
  return "CALL";
}

function directionLabel(value) {
  return TRADE_DIRECTIONS.find((option) => option.value === value)?.label || value || "—";
}

function creditDebitLabel(value) {
  return CREDIT_DEBIT_TYPES.find((option) => option.value === value)?.label || value || "—";
}

function optionTypeLabel(value) {
  return OPTION_TYPES.find((option) => option.value === value)?.label || value || "—";
}

function wheelStageLabel(value) {
  return WHEEL_STAGES.find((option) => option.value === value)?.label || value || "—";
}

/**
 * The PnL/PnL% implied by the exit price, on the backend's own formula
 * (pnl_pct = pnl / (entry_price * 100 * size) * 100). `isCredit` only flips
 * the direction of the price difference — the cost basis stays the absolute
 * premium, exactly as the backend computes it.
 */
function deriveClosePnl(entryPrice, positionSize, exitPrice, isCredit = false) {
  if (!Number.isFinite(entryPrice) || !Number.isFinite(positionSize)) return null;
  if (!Number.isFinite(exitPrice)) return null;
  const cost = entryPrice * CONTRACT_MULTIPLIER * positionSize;
  if (cost === 0) return null;
  const diff = isCredit ? entryPrice - exitPrice : exitPrice - entryPrice;
  const pnl = diff * CONTRACT_MULTIPLIER * positionSize;
  return { pnl, pct: (pnl / cost) * 100, cost, isCredit };
}

function fmtDollar(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return "$" + n.toLocaleString("en-US", { maximumFractionDigits: 2 });
}

function fmtPct(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  const sign = n > 0 ? "+" : "";
  return `${sign}${n.toFixed(2)}%`;
}

function defaultLocalDateTimeInput() {
  const now = new Date();
  now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
  return now.toISOString().slice(0, 16);
}

function fmtDateTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  // Without the year, a trade from a previous year reads as if it were this
  // year. Only spend the extra characters when it isn't the current year.
  const isThisYear = d.getFullYear() === new Date().getFullYear();
  return d.toLocaleString("zh-TW", {
    ...(isThisYear ? {} : { year: "numeric" }),
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

function fmtExpirationDate(iso) {
  if (!iso) return "—";
  const d = new Date(`${iso}T00:00:00`);
  if (Number.isNaN(d.getTime())) return iso;
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

function contractSummary(trade) {
  if (trade.option_type === "MULTI_LEG") {
    const count = Array.isArray(trade.legs) ? trade.legs.length : 0;
    return count ? `${count} legs` : "Multi-leg";
  }
  const strike = trade.strike_price ? `$${trade.strike_price}` : "—";
  const type = optionTypeLabel(trade.option_type);
  return `${strike} ${type}${trade.contract_symbol ? ` · ${trade.contract_symbol}` : ""}`;
}

function tradeExpirationSummary(trade) {
  return trade.expiration_date ? `到期 ${fmtExpirationDate(trade.expiration_date)}` : "到期未記錄";
}

function defaultLeg(expirationDate = "") {
  return {
    side: "BUY",
    option_type: "CALL",
    strike_price: "",
    expiration_date: expirationDate || "",
    quantity: "1",
    price: "",
    contract_symbol: "",
  };
}

function defaultLegs(expirationDate = "") {
  return [
    defaultLeg(expirationDate),
    { ...defaultLeg(expirationDate), side: "SELL", strike_price: "" },
  ];
}

function buildDraft(ticker, expirationDate, initialDraft = null) {
  const base = {
    ticker: (ticker || "").toUpperCase(),
    strategyType: "",
    direction: "LONG",
    creditDebit: "DEBIT",
    expirationDate: expirationDate || "",
    optionType: "CALL",
    strikePrice: "",
    contractSymbol: "",
    legs: defaultLegs(expirationDate || ""),
    entryPrice: "",
    positionSize: "1",
    entryDate: defaultLocalDateTimeInput(),
    notes: "",
    sourcePlanId: "",
    wheelStage: "",
    lpRangeLower: "",
    lpRangeUpper: "",
    weeklyTargetYield: "",
  };
  if (!initialDraft) return base;
  const nextExpiration = initialDraft.expirationDate ?? base.expirationDate;
  return {
    ...base,
    ...initialDraft,
    ticker: (initialDraft.ticker || base.ticker).toUpperCase(),
    expirationDate: nextExpiration || "",
    legs: defaultLegs(nextExpiration || ""),
  };
}

function earliestIsoDate(values) {
  const dates = values
    .map((value) => String(value || "").trim())
    .filter(Boolean)
    .sort();
  return dates[0] || null;
}

export default function TradeJournalPanel({ userId, ticker, expirationDate, initialDraft, onClose }) {
  const [trades, setTrades] = useState([]);
  const [plans, setPlans] = useState([]);
  const [plansError, setPlansError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [formError, setFormError] = useState(null);
  const [closeError, setCloseError] = useState(null);

  const [draft, setDraft] = useState(() => buildDraft(ticker, expirationDate, initialDraft));
  const [creating, setCreating] = useState(false);

  const [closingId, setClosingId] = useState(null);
  const [closeDraft, setCloseDraft] = useState({ exitPrice: "", pnl: "" });
  const [closing, setClosing] = useState(false);

  const [reviews, setReviews] = useState({});
  const [reviewingId, setReviewingId] = useState(null);

  async function loadTrades() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(apiUrl("/api/v1/trades", { user_id: userId }));
      if (!res.ok) throw new Error(await parseErrorDetail(res));
      const data = await res.json();
      const fetchedTrades = data.trades || [];
      setTrades(fetchedTrades);
      await loadReviewsFor(fetchedTrades.filter((t) => t.status === "CLOSED"));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function loadReviewsFor(closedTrades) {
    const entries = await Promise.all(
      closedTrades.map(async (t) => {
        try {
          const res = await fetch(
            apiUrl(`/api/v1/trades/${pathSegment(t.id)}/review`, { user_id: userId })
          );
          if (!res.ok) return null;
          const review = await res.json();
          return review ? [t.id, review] : null;
        } catch {
          return null;
        }
      })
    );
    setReviews((prev) => {
      const next = { ...prev };
      for (const entry of entries) {
        if (entry) next[entry[0]] = entry[1];
      }
      return next;
    });
  }

  async function loadPlans() {
    setPlansError(null);
    try {
      const res = await fetch(apiUrl("/api/v1/plans", { user_id: userId }));
      if (!res.ok) throw new Error(await parseErrorDetail(res));
      const data = await res.json();
      setPlans(data.plans || []);
    } catch (err) {
      // Linking a plan is what supplies stop_loss/target_price to the AI
      // review's execution score — a silent failure here quietly degrades it
      // while looking identical to "you have no saved plans".
      setPlansError(err.message || "無法載入已儲存計畫");
    }
  }

  useEffect(() => {
    loadTrades();
    loadPlans();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId]);

  // Prefill from the terminal's current symbol, but leave it editable — a user
  // may well want to log a trade on something other than what's on screen.
  // Once they've typed in the field the panel stops touching it, otherwise a
  // ticker change underneath the open overlay silently reverts their edit.
  const tickerDirty = useRef(false);
  const expirationDirty = useRef(false);
  const initialDraftKey = useRef(initialDraft?.key || null);
  useEffect(() => {
    if (!initialDraft || initialDraftKey.current === initialDraft.key) return;
    initialDraftKey.current = initialDraft.key;
    tickerDirty.current = false;
    expirationDirty.current = false;
    setDraft(buildDraft(ticker, expirationDate, initialDraft));
  }, [initialDraft, ticker, expirationDate]);

  useEffect(() => {
    if (!ticker || tickerDirty.current) return;
    setDraft((d) => ({
      ...d,
      ticker: ticker.toUpperCase(),
      expirationDate: expirationDirty.current ? d.expirationDate : expirationDate || "",
      legs:
        expirationDirty.current || !expirationDate
          ? d.legs
          : d.legs.map((leg) => ({ ...leg, expiration_date: leg.expiration_date || expirationDate })),
    }));
  }, [ticker, expirationDate]);

  useEffect(() => {
    if (expirationDirty.current || tickerDirty.current) return;
    setDraft((d) => ({
      ...d,
      expirationDate: expirationDate || "",
      legs: d.legs.map((leg) => ({ ...leg, expiration_date: leg.expiration_date || expirationDate || "" })),
    }));
  }, [expirationDate]);

  // Recomputed each render so a panel left open overnight can't go stale.
  const maxEntryDate = defaultLocalDateTimeInput();

  // Editing any field clears a stale validation message — it used to sit there
  // until the *next* failed submit, long after the user had fixed the field.
  function updateDraft(patch) {
    setDraft((d) => ({ ...d, ...patch }));
    if (formError) setFormError(null);
  }

  function updateLeg(index, patch) {
    setDraft((d) => ({
      ...d,
      legs: d.legs.map((leg, i) => (i === index ? { ...leg, ...patch } : leg)),
    }));
    if (formError) setFormError(null);
  }

  function addLeg() {
    setDraft((d) => ({
      ...d,
      legs: [...d.legs, defaultLeg(d.expirationDate)],
    }));
    if (formError) setFormError(null);
  }

  function removeLeg(index) {
    setDraft((d) => ({
      ...d,
      legs: d.legs.filter((_, i) => i !== index),
    }));
    if (formError) setFormError(null);
  }

  function updateCloseDraft(patch) {
    setCloseDraft((d) => ({ ...d, ...patch }));
    if (closeError) setCloseError(null);
  }

  async function createTrade() {
    const tickerValue = draft.ticker.trim().toUpperCase();
    const strategyValue = draft.strategyType.trim();
    const entryPrice = parsePositiveNumber(draft.entryPrice);
    const positionSize = parsePositiveInt(draft.positionSize);
    const legExpirationDate =
      draft.optionType === "MULTI_LEG"
        ? earliestIsoDate(draft.legs.map((leg) => leg.expiration_date))
        : null;
    const tradeExpirationDate =
      draft.optionType === "MULTI_LEG"
        ? legExpirationDate
        : draft.expirationDate.trim() || null;
    const strikePrice = parsePositiveNumber(draft.strikePrice);
    const lpRangeLower = parsePositiveNumber(draft.lpRangeLower);
    const lpRangeUpper = parsePositiveNumber(draft.lpRangeUpper);
    const weeklyTargetYield = parseNonNegativeNumber(draft.weeklyTargetYield);
    let legs = [];
    if (draft.optionType === "MULTI_LEG") {
      legs = draft.legs.map((leg) => ({
        side: leg.side,
        option_type: leg.option_type,
        strike_price: parsePositiveNumber(leg.strike_price),
        expiration_date: leg.expiration_date,
        quantity: parsePositiveInt(leg.quantity),
        price: parsePositiveNumber(leg.price),
        contract_symbol: leg.contract_symbol.trim() || null,
      }));
    }

    // These used to be a silent `return` — the button simply did nothing.
    let validation = null;
    if (!tickerValue) validation = "請輸入股票代號";
    else if (!strategyValue) validation = "請輸入策略類型";
    else if (!tradeExpirationDate)
      validation =
        draft.optionType === "MULTI_LEG"
          ? "請填寫每腿到期日"
          : "請選擇到期日";
    else if (!draft.optionType) validation = "請選擇 Call/Put 或 Multi-leg";
    else if (draft.optionType !== "MULTI_LEG" && strikePrice === null)
      validation = "履約價必須是大於 0 的數字";
    else if (draft.optionType === "MULTI_LEG" && legs.length < 2)
      validation = "Multi-leg 至少需要兩腿";
    else if (
      draft.optionType === "MULTI_LEG" &&
      legs.some(
        (leg) =>
          leg.strike_price === null ||
          leg.quantity === null ||
          !leg.expiration_date ||
          leg.price === null
      )
    )
      validation = "請完整填寫每腿的到期日、履約價、口數與價格";
    else if (entryPrice === null) validation = "進場價必須是大於 0 的數字";
    else if (positionSize === null) validation = "口數必須是大於 0 的整數";
    else if (
      lpRangeLower !== null &&
      lpRangeUpper !== null &&
      lpRangeLower >= lpRangeUpper
    )
      validation = "LP/CSP 區間下限必須低於上限";
    else if (draft.entryDate && new Date(draft.entryDate).getTime() > Date.now() + 60_000)
      validation = "進場時間不能設定在未來";
    if (validation) {
      setFormError(validation);
      return;
    }

    setFormError(null);
    setCreating(true);
    setError(null);
    try {
      const res = await fetch(apiUrl("/api/v1/trades"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          ticker: tickerValue,
          strategy_type: strategyValue,
          direction: draft.direction,
          credit_debit: draft.creditDebit,
          expiration_date: tradeExpirationDate,
          option_type: draft.optionType,
          strike_price: draft.optionType === "MULTI_LEG" ? null : strikePrice,
          contract_symbol: draft.contractSymbol.trim() || null,
          legs: draft.optionType === "MULTI_LEG" ? legs : [],
          entry_price: entryPrice,
          position_size: positionSize,
          entry_date: draft.entryDate
            ? new Date(draft.entryDate).toISOString()
            : null,
          notes: draft.notes.trim() || null,
          source_plan_id: draft.sourcePlanId || null,
          wheel_stage: draft.wheelStage || null,
          lp_range_lower: lpRangeLower,
          lp_range_upper: lpRangeUpper,
          weekly_target_yield: weeklyTargetYield,
          // expiration_date alone drives the entry GEX snapshot's DTE now —
          // the backend derives it, so the snapshot always matches the
          // expiration actually being recorded, even if the user edited it
          // away from whatever the terminal happened to be showing.
        }),
      });
      if (!res.ok) throw new Error(await parseErrorDetail(res));
      const trade = await res.json();
      setTrades((t) => [trade, ...t]);
      // Fresh form — the prefill is welcome to take over the ticker again.
      tickerDirty.current = false;
      expirationDirty.current = false;
      setDraft(buildDraft(ticker, expirationDate));
    } catch (err) {
      setError(err.message);
    } finally {
      setCreating(false);
    }
  }

  async function closeTrade(tradeId) {
    const exitPrice = parsePositiveNumber(closeDraft.exitPrice);
    const pnl = parseFiniteNumber(closeDraft.pnl);

    let validation = null;
    if (exitPrice === null) validation = "出場價必須是大於 0 的數字";
    else if (pnl === null) validation = "損益必須是數字（可為負數）";
    if (validation) {
      setCloseError(validation);
      return;
    }

    setCloseError(null);
    setClosing(true);
    setError(null);
    try {
      const res = await fetch(
        apiUrl(`/api/v1/trades/${pathSegment(tradeId)}`, { user_id: userId }),
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            exit_price: exitPrice,
            exit_date: new Date().toISOString(),
            // Submitted as typed — real fills carry slippage and commissions
            // the derived figure can't know about.
            pnl,
          }),
        }
      );
      if (!res.ok) throw new Error(await parseErrorDetail(res));
      const updated = await res.json();
      setTrades((ts) => ts.map((t) => (t.id === updated.id ? updated : t)));
      setClosingId(null);
      setCloseDraft({ exitPrice: "", pnl: "" });
    } catch (err) {
      setError(err.message);
    } finally {
      setClosing(false);
    }
  }

  // `force` re-runs the (paid) AI analysis. The first-time trigger leaves it
  // off so a double-click or a retry after a network hiccup replays the
  // stored review instead of buying a second completion; only the explicit
  // 「重新分析」 button asks for a genuinely fresh one.
  async function triggerReview(tradeId, force = false) {
    setReviewingId(tradeId);
    setError(null);
    try {
      const res = await fetch(
        apiUrl(`/api/v1/trades/${pathSegment(tradeId)}/review`, {
          user_id: userId,
          force: force ? "true" : null,
        }),
        { method: "POST" }
      );
      if (!res.ok) throw new Error(await parseErrorDetail(res));
      const review = await res.json();
      setReviews((r) => ({ ...r, [tradeId]: review }));
    } catch (err) {
      setError(err.message);
    } finally {
      setReviewingId(null);
    }
  }

  const openTrades = trades.filter((t) => t.status === "OPEN");
  const closedTrades = trades.filter((t) => t.status === "CLOSED");

  // Disabled only for *visibly* incomplete forms. Filled-but-invalid input
  // (e.g. "abc" in a price) stays clickable so the click can explain itself.
  const legExpirationDate =
    draft.optionType === "MULTI_LEG"
      ? earliestIsoDate(draft.legs.map((leg) => leg.expiration_date))
      : null;
  const createDisabled =
    !draft.ticker.trim() ||
    !draft.strategyType.trim() ||
    (draft.optionType === "MULTI_LEG"
      ? !legExpirationDate
      : !draft.expirationDate.trim()) ||
    (draft.optionType !== "MULTI_LEG" && !draft.strikePrice.trim()) ||
    (draft.optionType === "MULTI_LEG" &&
      (draft.legs.length < 2 ||
        draft.legs.some(
          (leg) =>
            !leg.expiration_date.trim() ||
            !leg.strike_price.trim() ||
            !leg.quantity.trim() ||
            !leg.price.trim()
        ))) ||
    !draft.entryPrice.trim() ||
    !draft.positionSize.trim();
  const closeDisabled = !closeDraft.exitPrice.trim() || !closeDraft.pnl.trim();
  // The DTE always comes from the terminal, even when the symbol was typed by
  // hand — the hint has to say so rather than implying both came from it.
  const tickerOverridden =
    !!draft.ticker.trim() &&
    draft.ticker.trim().toUpperCase() !== (ticker || "").toUpperCase();
  const effectiveExpirationDate =
    draft.optionType === "MULTI_LEG"
      ? legExpirationDate
      : draft.expirationDate.trim() || null;
  const showIncomeFields =
    /weekly_csp|defi_lp|cash[-\s]?secured|covered\s*call/i.test(draft.strategyType) ||
    draft.wheelStage ||
    draft.lpRangeLower ||
    draft.lpRangeUpper ||
    draft.weeklyTargetYield;

  // Live math for the close form: the same formula the backend uses for
  // pnl_pct, so the ×100 contract multiplier stops being invisible.
  const closingTrade = openTrades.find((t) => t.id === closingId) || null;
  const closeExitPrice = parsePositiveNumber(closeDraft.exitPrice);
  // Direction comes from the *trade being closed*, not the create form's
  // strategy field — that field belongs to a different (possibly empty,
  // possibly unrelated) draft.
  const closingIsCredit = closingTrade
    ? (closingTrade.credit_debit || inferCreditDebit(closingTrade.strategy_type)) === "CREDIT"
    : false;
  const derivedClose = closingTrade
    ? deriveClosePnl(
        closingTrade.entry_price,
        closingTrade.position_size,
        closeExitPrice,
        closingIsCredit
      )
    : null;
  const manualPnl = parseFiniteNumber(closeDraft.pnl);
  // Floor the tolerance on the position's cost basis rather than a flat $1,
  // otherwise a near-breakeven trade flags on any realistic commission.
  const pnlMismatch =
    derivedClose !== null &&
    manualPnl !== null &&
    Math.abs(manualPnl - derivedClose.pnl) >
      Math.max(
        Math.abs(derivedClose.pnl) * PNL_MISMATCH_TOLERANCE,
        derivedClose.cost * 0.01,
        5
      );

  return (
    <div className="absolute inset-x-0 top-11 bottom-0 z-30 bg-[#121214] border-t border-[rgba(240,237,229,.09)] flex flex-col">
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-[rgba(240,237,229,.09)]">
        <span className="text-[10px] tracking-wider uppercase text-[#8d8d93] font-semibold">
          交易日誌
        </span>
        <button type="button" onClick={onClose} className="text-[#8d8d93] hover:text-[#f0ede5]">
          <X size={14} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-4">
        {error && <div className="text-[11px] text-[#d8622b]">{error}</div>}

        {/* Quick-add card */}
        <div className="border border-[rgba(240,237,229,.09)] rounded-md p-3 flex flex-col gap-2">
          <div className="text-[10px] tracking-wider uppercase text-[#57575c]">新增交易</div>
          <div className="flex gap-2">
            <input
              value={draft.ticker}
              onChange={(e) => {
                const nextTicker = e.target.value.toUpperCase();
                tickerDirty.current = true;
                updateDraft({
                  ticker: e.target.value,
                  expirationDate:
                    !expirationDirty.current && nextTicker !== (ticker || "").toUpperCase()
                      ? ""
                      : draft.expirationDate,
                });
              }}
              placeholder="代號"
              className={`w-20 bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
            />
            <input
              value={draft.strategyType}
              onChange={(e) => {
                const strategyType = e.target.value;
                updateDraft({
                  strategyType,
                  direction: inferDirection(strategyType),
                  creditDebit: inferCreditDebit(strategyType),
                  optionType: inferOptionType(strategyType),
                });
              }}
              placeholder="策略類型（可挑選或自行輸入）"
              list="trade-journal-strategy-types"
              className={`flex-1 bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
            />
            <datalist id="trade-journal-strategy-types">
              {COMMON_STRATEGY_TYPES.map((s) => (
                <option key={s} value={s} />
              ))}
            </datalist>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <select
              value={draft.direction}
              onChange={(e) => updateDraft({ direction: e.target.value })}
              className={`bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
            >
              {TRADE_DIRECTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  方向 · {option.label}
                </option>
              ))}
            </select>
            <select
              value={draft.creditDebit}
              onChange={(e) => updateDraft({ creditDebit: e.target.value })}
              className={`bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
            >
              {CREDIT_DEBIT_TYPES.map((option) => (
                <option key={option.value} value={option.value}>
                  資流 · {option.label}
                </option>
              ))}
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-[9.5px] text-[#57575c]">到期日（請確認買入的合約到期日）</span>
            <input
              type="date"
              value={draft.expirationDate}
              onChange={(e) => {
                expirationDirty.current = true;
                const nextExpiration = e.target.value;
                updateDraft({
                  expirationDate: nextExpiration,
                  legs: draft.legs.map((leg) => ({
                    ...leg,
                    expiration_date: leg.expiration_date || nextExpiration,
                  })),
                });
              }}
              className={`bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
            />
          </div>
          <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)] gap-2">
            <select
              value={draft.optionType}
              onChange={(e) =>
                updateDraft({
                  optionType: e.target.value,
                  strikePrice: e.target.value === "MULTI_LEG" ? "" : draft.strikePrice,
                  legs:
                    e.target.value === "MULTI_LEG" && draft.legs.length < 2
                      ? defaultLegs(draft.expirationDate)
                      : draft.legs,
                })
              }
              className={`bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
            >
              {OPTION_TYPES.map((option) => (
                <option key={option.value} value={option.value}>
                  合約 · {option.label}
                </option>
              ))}
            </select>
            {draft.optionType === "MULTI_LEG" ? (
              <input
                value={draft.contractSymbol}
                onChange={(e) => updateDraft({ contractSymbol: e.target.value })}
                placeholder="組合代碼（選填）"
                className={`bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
              />
            ) : (
              <input
                value={draft.strikePrice}
                onChange={(e) => updateDraft({ strikePrice: e.target.value })}
                placeholder="履約價"
                inputMode="decimal"
                className={`bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
              />
            )}
          </div>
          {draft.optionType !== "MULTI_LEG" && (
            <input
              value={draft.contractSymbol}
              onChange={(e) => updateDraft({ contractSymbol: e.target.value })}
              placeholder="合約代碼（選填）"
              className={`bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
            />
          )}
          {draft.optionType === "MULTI_LEG" && (
            <div className="flex flex-col gap-1.5">
              <div className="flex items-center justify-between">
                <span className="text-[9.5px] text-[#57575c]">策略腿</span>
                <button
                  type="button"
                  onClick={addLeg}
                  className="text-[9.5px] text-[#c9a15c] hover:text-[#d8b06c]"
                >
                  + 新增腿
                </button>
              </div>
              {draft.legs.map((leg, index) => (
                <div
                  key={index}
                  className="grid grid-cols-[64px_64px_minmax(0,1fr)_82px_54px_72px_24px] gap-1.5"
                >
                  <select
                    value={leg.side}
                    onChange={(e) => updateLeg(index, { side: e.target.value })}
                    className={`bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-1.5 py-1.5 text-[10.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
                  >
                    <option value="BUY">Buy</option>
                    <option value="SELL">Sell</option>
                  </select>
                  <select
                    value={leg.option_type}
                    onChange={(e) => updateLeg(index, { option_type: e.target.value })}
                    className={`bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-1.5 py-1.5 text-[10.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
                  >
                    <option value="CALL">Call</option>
                    <option value="PUT">Put</option>
                  </select>
                  <input
                    type="date"
                    value={leg.expiration_date}
                    onChange={(e) => updateLeg(index, { expiration_date: e.target.value })}
                    className={`min-w-0 bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-1.5 py-1.5 text-[10.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
                  />
                  <input
                    value={leg.strike_price}
                    onChange={(e) => updateLeg(index, { strike_price: e.target.value })}
                    placeholder="Strike"
                    inputMode="decimal"
                    className={`bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-1.5 py-1.5 text-[10.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
                  />
                  <input
                    value={leg.quantity}
                    onChange={(e) => updateLeg(index, { quantity: e.target.value })}
                    placeholder="Qty"
                    inputMode="numeric"
                    className={`bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-1.5 py-1.5 text-[10.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
                  />
                  <input
                    value={leg.price}
                    onChange={(e) => updateLeg(index, { price: e.target.value })}
                    placeholder="Price"
                    inputMode="decimal"
                    className={`bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-1.5 py-1.5 text-[10.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
                  />
                  <button
                    type="button"
                    onClick={() => removeLeg(index)}
                    disabled={draft.legs.length <= 2}
                    title="刪除此腿"
                    className="rounded border border-[rgba(240,237,229,.09)] text-[#8d8d93] hover:text-[#f0ede5] disabled:opacity-30"
                  >
                    ×
                  </button>
                </div>
              ))}
              <input
                value={draft.legs.map((leg) => leg.contract_symbol).filter(Boolean).join(", ")}
                onChange={(e) => {
                  const symbols = e.target.value.split(",").map((s) => s.trim());
                  setDraft((d) => ({
                    ...d,
                    legs: d.legs.map((leg, index) => ({
                      ...leg,
                      contract_symbol: symbols[index] || "",
                    })),
                  }));
                  if (formError) setFormError(null);
                }}
                placeholder="每腿合約代碼（選填，用逗號分隔）"
                className={`bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[10.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
              />
            </div>
          )}
          <div className="flex gap-2">
            <input
              value={draft.entryPrice}
              onChange={(e) => updateDraft({ entryPrice: e.target.value })}
              placeholder="進場價"
              inputMode="decimal"
              className={`flex-1 bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
            />
            <input
              value={draft.positionSize}
              onChange={(e) => updateDraft({ positionSize: e.target.value })}
              placeholder="口數"
              inputMode="numeric"
              className={`w-16 bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
            />
          </div>
          {showIncomeFields && (
            <div className="grid grid-cols-4 gap-2">
              <select
                value={draft.wheelStage}
                onChange={(e) => updateDraft({ wheelStage: e.target.value })}
                className={`bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[10.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
              >
                <option value="">Wheel 階段</option>
                {WHEEL_STAGES.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              <input
                value={draft.lpRangeLower}
                onChange={(e) => updateDraft({ lpRangeLower: e.target.value })}
                placeholder="區間下限"
                inputMode="decimal"
                className={`bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[10.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
              />
              <input
                value={draft.lpRangeUpper}
                onChange={(e) => updateDraft({ lpRangeUpper: e.target.value })}
                placeholder="區間上限"
                inputMode="decimal"
                className={`bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[10.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
              />
              <input
                value={draft.weeklyTargetYield}
                onChange={(e) => updateDraft({ weeklyTargetYield: e.target.value })}
                placeholder="週目標%"
                inputMode="decimal"
                className={`bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[10.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
              />
            </div>
          )}
          <div className="flex flex-col gap-1">
            <span className="text-[9.5px] text-[#57575c]">進場時間（預設現在，可自行調整）</span>
            <input
              type="datetime-local"
              value={draft.entryDate}
              max={maxEntryDate}
              onChange={(e) => updateDraft({ entryDate: e.target.value })}
              className={`bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
            />
          </div>
          {plansError && (
            <div className="text-[10px] text-[#d8622b] leading-snug">
              ⚠ 無法載入已儲存計畫（{plansError}）— 未連結計畫時，AI 覆盤缺少停損/目標價，執行評分會較不準確。
              <button
                type="button"
                onClick={loadPlans}
                className="ml-1 underline hover:text-[#c9a15c]"
              >
                重試
              </button>
            </div>
          )}
          {plans.length > 0 && (
            <select
              value={draft.sourcePlanId}
              onChange={(e) => updateDraft({ sourcePlanId: e.target.value })}
              className={`bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
            >
              <option value="">（可選）連結已儲存的交易計畫</option>
              {plans.map((p) => (
                <option key={p.plan_id} value={p.plan_id}>
                  {p.ticker} {p.strategy_type} E{p.entry_price}
                </option>
              ))}
            </select>
          )}
          <textarea
            value={draft.notes}
            onChange={(e) => updateDraft({ notes: e.target.value })}
            placeholder="備註（選填）"
            rows={2}
            className={`bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] resize-none ${MONO}`}
          />
          {formError && (
            <div className="text-[10.5px] text-[#d8622b] leading-snug">⚠ {formError}</div>
          )}
          <div className="text-[9px] text-[#57575c] leading-snug">
            {tickerOverridden ? (
              <>
                代號 {draft.ticker}（手動輸入）· 將以
                {draft.optionType === "MULTI_LEG" ? "最早腿到期日" : "到期日"}{" "}
                {fmtExpirationDate(effectiveExpirationDate)} 擷取進場 GEX 快照（與終端機目前的{" "}
                {(ticker || "—").toUpperCase()} 無關）
              </>
            ) : (
              <>
                將以{draft.optionType === "MULTI_LEG" ? "最早腿到期日" : "到期日"}{" "}
                {fmtExpirationDate(effectiveExpirationDate)} 擷取 {draft.ticker || "—"} 的進場 GEX 快照
              </>
            )}
          </div>
          <button
            type="button"
            onClick={createTrade}
            disabled={creating || createDisabled}
            title={createDisabled ? "請先填寫代號、策略類型、到期日、合約資訊、進場價與口數" : undefined}
            className="flex items-center justify-center gap-1.5 py-2 rounded-md bg-[#c9a15c] text-[#1a1408] text-[11.5px] font-bold uppercase tracking-wide hover:bg-[#d8b06c] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Plus size={13} />
            {creating ? "建立中…" : "新增交易"}
          </button>
        </div>

        {loading && <div className="text-[11px] text-[#57575c] text-center">載入中…</div>}

        {/* Open trades */}
        <div>
          <div className="text-[10px] tracking-wider uppercase text-[#57575c] mb-2">
            持倉中（{openTrades.length}）
          </div>
          <div className="flex flex-col gap-2">
            {!loading && openTrades.length === 0 && (
              <div className="text-[11px] text-[#57575c]">尚無持倉中交易</div>
            )}
            {openTrades.map((t) => (
              <div
                key={t.id}
                className="border border-[rgba(240,237,229,.09)] rounded-md bg-[#1b1b1e] p-3"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[11.5px] font-bold text-[#f0ede5]">
                    {t.ticker} · {t.strategy_type}
                  </span>
                  <span className="text-[9.5px] text-[#57575c]">{fmtDateTime(t.entry_date)}</span>
                </div>
                <div
                  className="text-[10.5px] text-[#8d8d93] mb-2"
                  title={`每口 ${CONTRACT_MULTIPLIER} 股，成本 ${fmtDollar(
                    t.entry_price * CONTRACT_MULTIPLIER * t.position_size
                  )}`}
                >
                  進場 {fmtDollar(t.entry_price)}/股 × {t.position_size} 口 ×{" "}
                  {CONTRACT_MULTIPLIER} = {fmtDollar(t.entry_price * CONTRACT_MULTIPLIER * t.position_size)}
                </div>
                <div className="text-[9.5px] text-[#57575c] mb-2">
                  {tradeExpirationSummary(t)} · {contractSummary(t)} · 方向 {directionLabel(t.direction)} · 資流 {creditDebitLabel(t.credit_debit)}
                </div>
                {(t.wheel_stage || t.lp_range_lower || t.lp_range_upper || t.weekly_target_yield) && (
                  <div className="text-[9.5px] text-[#8d8d93] mb-2">
                    Wheel {wheelStageLabel(t.wheel_stage)} · 區間{" "}
                    {t.lp_range_lower ? fmtDollar(t.lp_range_lower) : "—"}–{t.lp_range_upper ? fmtDollar(t.lp_range_upper) : "—"} · 週目標{" "}
                    {t.weekly_target_yield ?? "—"}%
                  </div>
                )}
                {closingId === t.id ? (
                  <div className="flex flex-col gap-1.5">
                    <div className="flex gap-1.5">
                      <input
                        value={closeDraft.exitPrice}
                        onChange={(e) =>
                          updateCloseDraft({ exitPrice: e.target.value })
                        }
                        placeholder="出場價"
                        inputMode="decimal"
                        className={`flex-1 bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
                      />
                      <input
                        value={closeDraft.pnl}
                        onChange={(e) => updateCloseDraft({ pnl: e.target.value })}
                        placeholder="損益（$）"
                        inputMode="decimal"
                        className={`flex-1 bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
                      />
                    </div>

                    {derivedClose && (
                      <div className="rounded border border-[rgba(240,237,229,.09)] bg-[#0b0b0c] px-2 py-1.5 leading-snug">
                        <div className="text-[10.5px] text-[#8d8d93]">
                          依出場價推算：
                          <span
                            className="font-bold ml-1"
                            style={{ color: derivedClose.pnl >= 0 ? "#2fa37a" : "#d8622b" }}
                          >
                            {fmtDollar(derivedClose.pnl)}（{fmtPct(derivedClose.pct)}）
                          </span>
                          <button
                            type="button"
                            onClick={() =>
                              updateCloseDraft({ pnl: derivedClose.pnl.toFixed(2) })
                            }
                            className="ml-1.5 text-[9.5px] text-[#c9a15c] underline hover:text-[#d8b06c]"
                          >
                            套用
                          </button>
                        </div>
                        <div className="text-[9px] text-[#57575c] mt-0.5">
                          {derivedClose.isCredit ? (
                            <>
                              ({fmtDollar(t.entry_price)} 收取 − {fmtDollar(closeExitPrice)} 買回) ×{" "}
                              {CONTRACT_MULTIPLIER} 股/口 × {t.position_size} 口
                            </>
                          ) : (
                            <>
                              ({fmtDollar(closeExitPrice)} − {fmtDollar(t.entry_price)}) ×{" "}
                              {CONTRACT_MULTIPLIER} 股/口 × {t.position_size} 口
                            </>
                          )}
                        </div>
                        <div className="text-[9px] text-[#57575c] mt-0.5">
                          依「{derivedClose.isCredit ? "信用策略（收取權利金）" : "借記策略（買方）"}
                          」計算；若實際成交不同，請直接以手動損益欄為準。
                        </div>
                      </div>
                    )}
                    {pnlMismatch && (
                      <div className="text-[10px] text-[#c9a15c] leading-snug">
                        ⚠ 手動輸入的損益與推算值相差超過 {Math.round(PNL_MISMATCH_TOLERANCE * 100)}%，
                        請確認是否有滑價/手續費因素（仍可送出）。
                      </div>
                    )}
                    {closeError && (
                      <div className="text-[10px] text-[#d8622b] leading-snug">⚠ {closeError}</div>
                    )}

                    <div className="flex gap-1.5">
                      <button
                        type="button"
                        onClick={() => closeTrade(t.id)}
                        disabled={closing || closeDisabled}
                        title={closeDisabled ? "請先填寫出場價與損益" : undefined}
                        className="flex-1 py-1.5 rounded bg-[#c9a15c] text-[#1a1408] text-[10.5px] font-bold disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {closing ? "平倉中…" : "確認平倉"}
                      </button>
                      <button
                        type="button"
                        onClick={() => setClosingId(null)}
                        className="px-3 py-1.5 rounded border border-[rgba(240,237,229,.09)] text-[#8d8d93] text-[10.5px]"
                      >
                        取消
                      </button>
                    </div>
                  </div>
                ) : (
                  <button
                    type="button"
                    onClick={() => {
                      setClosingId(t.id);
                      setCloseDraft({ exitPrice: "", pnl: "" });
                      setCloseError(null);
                    }}
                    className="w-full py-1.5 rounded border border-[rgba(240,237,229,.16)] text-[#8d8d93] text-[10.5px] font-semibold hover:text-[#f0ede5] hover:border-[rgba(240,237,229,.28)] transition-colors"
                  >
                    平倉
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Closed trades */}
        <div>
          <div className="text-[10px] tracking-wider uppercase text-[#57575c] mb-2">
            已平倉（{closedTrades.length}）
          </div>
          <div className="flex flex-col gap-2">
            {!loading && closedTrades.length === 0 && (
              <div className="text-[11px] text-[#57575c]">尚無已平倉交易</div>
            )}
            {closedTrades.map((t) => {
              const review = reviews[t.id];
              const isPositive = (t.pnl_pct ?? 0) >= 0;
              return (
                <div
                  key={t.id}
                  className="border border-[rgba(240,237,229,.09)] rounded-md bg-[#1b1b1e] p-3"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[11.5px] font-bold text-[#f0ede5]">
                      {t.ticker} · {t.strategy_type}
                    </span>
                    <span
                      className={`text-[11px] font-bold ${MONO}`}
                      style={{ color: isPositive ? "#2fa37a" : "#d8622b" }}
                    >
                      {fmtPct(t.pnl_pct)}
                    </span>
                  </div>
                  <div className="text-[10.5px] text-[#8d8d93] mb-2">
                    {fmtDollar(t.entry_price)} → {fmtDollar(t.exit_price)} · 損益{" "}
                    {fmtDollar(t.pnl)}
                  </div>
                  <div className="text-[9.5px] text-[#57575c] mb-2">
                    {tradeExpirationSummary(t)} · {contractSummary(t)} · 方向 {directionLabel(t.direction)} · 資流 {creditDebitLabel(t.credit_debit)}
                  </div>
                  {(t.wheel_stage || t.lp_range_lower || t.lp_range_upper || t.weekly_target_yield) && (
                    <div className="text-[9.5px] text-[#8d8d93] mb-2">
                      Wheel {wheelStageLabel(t.wheel_stage)} · 區間{" "}
                      {t.lp_range_lower ? fmtDollar(t.lp_range_lower) : "—"}–{t.lp_range_upper ? fmtDollar(t.lp_range_upper) : "—"} · 週目標{" "}
                      {t.weekly_target_yield ?? "—"}%
                    </div>
                  )}

                  {review ? (
                    <div className="border-t border-[rgba(240,237,229,.09)] pt-2 mt-1 flex flex-col gap-1.5">
                      <div className="flex items-center gap-0.5">
                        {Array.from({ length: 5 }).map((_, i) => (
                          <Star
                            key={i}
                            size={12}
                            fill={i < review.execution_score ? "#c9a15c" : "none"}
                            color="#c9a15c"
                          />
                        ))}
                      </div>
                      <div className="text-[11px] text-[#f0ede5] leading-relaxed">
                        {review.ai_feedback}
                      </div>
                      <ul className="list-disc pl-4 flex flex-col gap-0.5">
                        {review.key_takeaways.map((k, i) => (
                          <li key={i} className="text-[10.5px] text-[#8d8d93]">
                            {k}
                          </li>
                        ))}
                      </ul>
                      <button
                        type="button"
                        onClick={() => triggerReview(t.id, true)}
                        disabled={reviewingId === t.id}
                        className="self-start text-[10px] text-[#57575c] hover:text-[#c9a15c] disabled:opacity-50 transition-colors"
                      >
                        {reviewingId === t.id ? "分析中…" : "🔄 重新分析"}
                      </button>
                    </div>
                  ) : (
                    <button
                      type="button"
                      onClick={() => triggerReview(t.id)}
                      disabled={reviewingId === t.id}
                      className="w-full py-1.5 rounded border border-[rgba(201,161,92,.35)] bg-[rgba(201,161,92,.08)] text-[#c9a15c] text-[10.5px] font-semibold hover:bg-[rgba(201,161,92,.18)] disabled:opacity-50 transition-colors"
                    >
                      {reviewingId === t.id ? "分析中…" : "🤖 觸發 AI 覆盤分析"}
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
