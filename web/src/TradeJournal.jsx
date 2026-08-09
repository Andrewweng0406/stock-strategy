import { useEffect, useState } from "react";
import { Plus, Star, X } from "lucide-react";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8002";
const MONO =
  '[font-family:ui-monospace,"SF_Mono","JetBrains_Mono","IBM_Plex_Mono",Menlo,Consolas,monospace]';

const COMMON_STRATEGY_TYPES = [
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

async function parseErrorDetail(res) {
  try {
    const body = await res.json();
    return body.detail ? JSON.stringify(body.detail) : `HTTP ${res.status}`;
  } catch {
    return `HTTP ${res.status}`;
  }
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

function fmtDateTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("zh-TW", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

export default function TradeJournalPanel({ userId, onClose }) {
  const [trades, setTrades] = useState([]);
  const [plans, setPlans] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [draft, setDraft] = useState({
    ticker: "",
    strategyType: "",
    entryPrice: "",
    positionSize: "1",
    notes: "",
    sourcePlanId: "",
  });
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
      const res = await fetch(`${BASE_URL}/api/v1/trades?user_id=${userId}`);
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
            `${BASE_URL}/api/v1/trades/${t.id}/review?user_id=${userId}`
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
    try {
      const res = await fetch(`${BASE_URL}/api/v1/plans?user_id=${userId}`);
      if (!res.ok) return;
      const data = await res.json();
      setPlans(data.plans || []);
    } catch {
      // Optional dropdown data — a failed fetch just leaves it empty.
    }
  }

  useEffect(() => {
    loadTrades();
    loadPlans();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId]);

  async function createTrade() {
    if (!draft.ticker.trim() || !draft.strategyType.trim() || !draft.entryPrice) return;
    setCreating(true);
    setError(null);
    try {
      const res = await fetch(`${BASE_URL}/api/v1/trades`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          ticker: draft.ticker.trim().toUpperCase(),
          strategy_type: draft.strategyType.trim(),
          entry_price: Number(draft.entryPrice),
          position_size: Number(draft.positionSize) || 1,
          notes: draft.notes.trim() || null,
          source_plan_id: draft.sourcePlanId || null,
          days_to_expiration: 30,
        }),
      });
      if (!res.ok) throw new Error(await parseErrorDetail(res));
      const trade = await res.json();
      setTrades((t) => [trade, ...t]);
      setDraft({
        ticker: "",
        strategyType: "",
        entryPrice: "",
        positionSize: "1",
        notes: "",
        sourcePlanId: "",
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setCreating(false);
    }
  }

  async function closeTrade(tradeId) {
    if (!closeDraft.exitPrice || closeDraft.pnl === "") return;
    setClosing(true);
    setError(null);
    try {
      const res = await fetch(
        `${BASE_URL}/api/v1/trades/${tradeId}?user_id=${userId}`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            exit_price: Number(closeDraft.exitPrice),
            exit_date: new Date().toISOString(),
            pnl: Number(closeDraft.pnl),
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

  async function triggerReview(tradeId) {
    setReviewingId(tradeId);
    setError(null);
    try {
      const res = await fetch(
        `${BASE_URL}/api/v1/trades/${tradeId}/review?user_id=${userId}`,
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
              onChange={(e) => setDraft((d) => ({ ...d, ticker: e.target.value }))}
              placeholder="代號"
              className={`w-20 bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
            />
            <input
              value={draft.strategyType}
              onChange={(e) => setDraft((d) => ({ ...d, strategyType: e.target.value }))}
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
          <div className="flex gap-2">
            <input
              value={draft.entryPrice}
              onChange={(e) => setDraft((d) => ({ ...d, entryPrice: e.target.value }))}
              placeholder="進場價"
              inputMode="decimal"
              className={`flex-1 bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
            />
            <input
              value={draft.positionSize}
              onChange={(e) => setDraft((d) => ({ ...d, positionSize: e.target.value }))}
              placeholder="口數"
              inputMode="numeric"
              className={`w-16 bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
            />
          </div>
          {plans.length > 0 && (
            <select
              value={draft.sourcePlanId}
              onChange={(e) => setDraft((d) => ({ ...d, sourcePlanId: e.target.value }))}
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
            onChange={(e) => setDraft((d) => ({ ...d, notes: e.target.value }))}
            placeholder="備註（選填）"
            rows={2}
            className={`bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11.5px] text-[#f0ede5] outline-none focus:border-[#c9a15c] resize-none ${MONO}`}
          />
          <button
            type="button"
            onClick={createTrade}
            disabled={creating}
            className="flex items-center justify-center gap-1.5 py-2 rounded-md bg-[#c9a15c] text-[#1a1408] text-[11.5px] font-bold uppercase tracking-wide hover:bg-[#d8b06c] disabled:opacity-50 transition-colors"
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
            {openTrades.length === 0 && (
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
                <div className="text-[10.5px] text-[#8d8d93] mb-2">
                  進場 {fmtDollar(t.entry_price)} × {t.position_size}
                </div>
                {closingId === t.id ? (
                  <div className="flex flex-col gap-1.5">
                    <div className="flex gap-1.5">
                      <input
                        value={closeDraft.exitPrice}
                        onChange={(e) =>
                          setCloseDraft((d) => ({ ...d, exitPrice: e.target.value }))
                        }
                        placeholder="出場價"
                        inputMode="decimal"
                        className={`flex-1 bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
                      />
                      <input
                        value={closeDraft.pnl}
                        onChange={(e) => setCloseDraft((d) => ({ ...d, pnl: e.target.value }))}
                        placeholder="損益（$）"
                        inputMode="decimal"
                        className={`flex-1 bg-[#0b0b0c] border border-[rgba(240,237,229,.09)] rounded px-2 py-1.5 text-[11px] text-[#f0ede5] outline-none focus:border-[#c9a15c] ${MONO}`}
                      />
                    </div>
                    <div className="flex gap-1.5">
                      <button
                        type="button"
                        onClick={() => closeTrade(t.id)}
                        disabled={closing}
                        className="flex-1 py-1.5 rounded bg-[#c9a15c] text-[#1a1408] text-[10.5px] font-bold disabled:opacity-50"
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
            {closedTrades.length === 0 && (
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
                        onClick={() => triggerReview(t.id)}
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
