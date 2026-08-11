import json
import logging
import re
from datetime import date
from typing import Any, Literal
from uuid import uuid4

from fastapi import HTTPException
from openai import AsyncOpenAI, OpenAIError
from pydantic import ValidationError

from app.models import (
    ChatMessageRecord,
    ChatRequest,
    GEXSnapshot,
    OptionGEXSummary,
    PlanStatus,
    RiskProfile,
    Trade,
    TradePlanToolArguments,
    TradeReviewToolArguments,
    UserProfile,
    UserTradePlan,
)


logger = logging.getLogger(__name__)
StrategyBias = Literal["BULLISH", "BEARISH", "NEUTRAL"]
ProfileName = Literal["CONSERVATIVE", "BALANCED", "AGGRESSIVE"]

# "short"/"long" only carry direction on their own. Attached to a horizon
# noun they describe time to expiry and must not vote on bias at all —
# note that "-" is itself a word boundary, so \bshort\b happily matches
# inside "short-term" and these compounds have to be stripped, not merely
# boundary-matched around.
_HORIZON_TERM_PATTERN = re.compile(
    r"\b(?:short|long|near|far)[\s\-_]?(?:term|dated|dte|expiry|expiration)\b",
    re.IGNORECASE,
)
_BEARISH_PATTERN = re.compile(
    r"\bputs?\b|\bbear(?:ish|s)?\b|\bshort(?:ing|s)?\b|做空|看空|看跌",
    re.IGNORECASE,
)
_BULLISH_PATTERN = re.compile(
    r"\bcalls?\b|\bbull(?:ish|s)?\b|\blong(?:ing|s)?\b|做多|看多|看漲|看涨",
    re.IGNORECASE,
)
# "bull" must be checked ahead of the bearish pattern: a Bull Put Credit
# Spread is bullish despite containing "put", which _BEARISH_PATTERN alone
# would otherwise match first.
_EXPLICIT_BULL_PATTERN = re.compile(r"\bbull(?:ish|s)?\b|做多|看多|看漲|看涨", re.IGNORECASE)


class LLMOrchestrator:
    MAX_REPLY_CHARACTERS = 200
    # Server-reconstructed conversation history is capped at the same length
    # the request schema used to allow, so prompt size is bounded even for a
    # very long stored conversation.
    MAX_HISTORY_MESSAGES = 30
    # An unambiguous adoption of a specific proposal — a letter, a named risk
    # profile, or an explicit "make the card" instruction. Safe on its own.
    STRONG_CONFIRMATION_PATTERN = re.compile(
        r"(?:\boption\s*[abc]\b|\bchoice\s*[abc]\b|"
        r"選(?:擇)?\s*[ABCabc]|选(?:择)?\s*[ABCabc]|"
        r"方案\s*[ABCabc]|選(?:擇)?\s*(?:保守|中性|激進)|"
        r"选(?:择)?\s*(?:保守|中性|激进)|"
        r"直接出卡|直接產生(?:計畫)?卡|直接生成(?:计划)?卡|"
        r"採用(?:你的)?建議|采用(?:你的)?建议|照你的建議|照你的建议|"
        r"use (?:your |the )?(?:suggestion|recommendation)|"
        r"(?:confirm|adopt) (?:the )?(?:plan|strategy|suggestion)|"
        r"generate (?:the )?(?:plan|card))",
        re.IGNORECASE,
    )
    # A bare confirmation word. NOT sufficient on its own: "我想確認一下 Zero
    # Gamma 是多少" is a data question, and forcing a signed-ready draft card
    # out of it hands the user a trade they never asked for. It only counts
    # when an actual plan/profile referent sits close by.
    BARE_CONFIRMATION_PATTERN = re.compile(
        r"(?:確認(?:這個|此方案|採用)?|确认(?:这个|此方案|采用)?|"
        r"就這個|就这个|照這個做|照这个做)",
        re.IGNORECASE,
    )
    PLAN_REFERENT_PATTERN = re.compile(
        r"(?:方案|計畫|計劃|计划|策略|出卡|卡片|建議|建议|採用|采用|"
        r"保守|中性|激進|激进|"
        r"\bplan\b|\bstrategy\b|\bcard\b|\bsuggestion\b|\brecommendation\b|"
        r"\bconservative\b|\bbalanced\b|\baggressive\b)",
        re.IGNORECASE,
    )
    # How far from a bare confirmation word a plan referent may sit and still
    # count as referring to it. Wide enough for natural phrasing
    # ("確認，就用你剛剛說的那個保守方案") but far short of "anywhere in a
    # 20,000-character message", which is what made the old pattern fire on
    # unrelated questions.
    CONFIRMATION_REFERENT_WINDOW = 16
    EXPLICIT_LEVEL_PATTERN = re.compile(
        r"(?:entry|stop|target|max(?:imum)?\s*loss|進場|入場|"
        r"进场|入场|停損|止損|停损|止损|目標|目标|最大虧損|"
        r"最大亏损)[^\d]{0,12}\d",
        re.IGNORECASE,
    )
    # Ceiling on a model-emitted max_loss_usd, expressed as a multiple of the
    # configured per-trade risk budget. The deterministic profiles only ever
    # scale that budget to 1.25x (the aggressive branch), so 10x is already
    # an order of magnitude past anything the system itself would propose —
    # generous enough that a user genuinely sizing up for one trade still
    # gets their number, tight enough that a model hallucinating 1e6 from a
    # crafted message cannot ship it on a signed-ready card.
    MAX_USER_LOSS_MULTIPLE = 10.0
    # Floor on a profile's entry-to-stop distance when computing R/R,
    # expressed as a fraction of spot. The conservative branch pins entry one
    # strike step inside its own defended level, so entry and stop can end up
    # ~2 ticks apart; dividing a full-width reward by that manufactured a
    # 9.87 reward-to-risk on the standard fixtures, roughly 7x the balanced
    # profile's, purely from a stop nobody could realistically hold. 2% of
    # spot is about one session's typical range for a liquid US equity — the
    # conventional "don't put the stop inside the noise" minimum — and is
    # never below the strike-step tick the stop is placed with.
    MIN_RISK_DISTANCE_PCT = 0.02
    # Floor on the DTE used by the theta proxy, in days. 0DTE reaches this
    # function as a literal 0 (ChatContext.days_to_expiration is ge=0), so
    # some floor is needed to avoid dividing by zero — but the old floor of a
    # whole day made DTE 0, 1 and 2 collapse onto one clamped estimate in
    # exactly the window where decay matters most. A quarter day keeps 0DTE
    # finite while making it decay ~2x as fast as 1DTE and ~2.8x as fast as
    # 2DTE.
    MIN_THETA_DTE_DAYS = 0.25

    def __init__(
        self,
        client: AsyncOpenAI | None,
        model: str,
        default_max_loss_usd: float = 250.0,
    ) -> None:
        self.client = client
        self.model = model
        self.default_max_loss_usd = default_max_loss_usd

    @staticmethod
    def _tool() -> dict[str, Any]:
        schema = TradePlanToolArguments.model_json_schema()
        schema["additionalProperties"] = False
        return {
            "type": "function",
            "name": "generate_trade_plan",
            "description": (
                "Create one draft plan only after the user confirms, adopts a "
                "recommendation, selects Option A/B/C or a risk profile, or directly "
                "requests a card. Use the supplied calculated profile values."
            ),
            "strict": True,
            "parameters": schema,
        }

    @staticmethod
    def _review_tool() -> dict[str, Any]:
        schema = TradeReviewToolArguments.model_json_schema()
        schema["additionalProperties"] = False
        return {
            "type": "function",
            "name": "submit_trade_review",
            "description": (
                "Submit the post-trade review prose. execution_score in the "
                "context is already computed by the backend — explain it, "
                "never invent a different one."
            ),
            "strict": True,
            "parameters": schema,
        }

    @classmethod
    def _should_force_plan(cls, message: str) -> bool:
        if cls.STRONG_CONFIRMATION_PATTERN.search(message):
            return True
        window = cls.CONFIRMATION_REFERENT_WINDOW
        for match in cls.BARE_CONFIRMATION_PATTERN.finditer(message):
            nearby = message[
                max(0, match.start() - window) : match.end() + window
            ]
            if cls.PLAN_REFERENT_PATTERN.search(nearby):
                return True
        return False

    @classmethod
    def _infer_bias(cls, strategy_type: str, user_message: str) -> StrategyBias:
        text = f"{strategy_type} {user_message}".lower()
        # "short-term"/"long-term" describe horizon, not direction. A bare
        # substring test read "short-term bullish call idea" as BEARISH, and
        # read the system's own neutral structure name "Short-Term Straddle"
        # as BEARISH when it was fed back through bias inference. Compound
        # horizon terms are removed first, then the remaining direction words
        # are matched on word boundaries so nothing else can be swallowed by
        # a longer word.
        text = _HORIZON_TERM_PATTERN.sub(" ", text)
        if _EXPLICIT_BULL_PATTERN.search(text):
            return "BULLISH"
        if _BEARISH_PATTERN.search(text):
            return "BEARISH"
        if _BULLISH_PATTERN.search(text):
            return "BULLISH"
        return "NEUTRAL"

    # GEX levels are nullable (see app/models.py OptionGEXSummary): a chain
    # with no gamma crossing has no zero_gamma, and a chain with no strike
    # on one side of spot has no wall there. A missing level is simply not a
    # candidate — these helpers drop it and fall back exactly as they would
    # for a level that sits on the wrong side of price.
    @staticmethod
    def _nearest_below(
        price: float, levels: list[float | None], fallback: float
    ) -> float:
        candidates = [
            level for level in levels if level is not None and 0 < level < price
        ]
        return max(candidates) if candidates else fallback

    @staticmethod
    def _nearest_above(
        price: float, levels: list[float | None], fallback: float
    ) -> float:
        candidates = [level for level in levels if level is not None and level > price]
        return min(candidates) if candidates else fallback

    @classmethod
    def _derive_levels(
        cls, summary: OptionGEXSummary, bias: StrategyBias
    ) -> dict[str, float]:
        price = summary.stock_price
        levels: list[float | None] = [
            summary.zero_gamma,
            summary.call_wall,
            summary.put_wall,
        ]
        if bias == "BEARISH":
            stop_reference = cls._nearest_above(price, levels, price * 1.03)
            target = cls._nearest_below(
                price, [summary.put_wall, summary.zero_gamma], price * 0.95
            )
            stop = stop_reference * 1.005
        elif bias == "BULLISH":
            stop_reference = cls._nearest_below(
                price, [summary.put_wall, summary.zero_gamma], price * 0.97
            )
            target = cls._nearest_above(
                price, [summary.call_wall, summary.zero_gamma], price * 1.05
            )
            stop = stop_reference * 0.995
        else:
            stop = cls._nearest_below(price, levels, price * 0.97)
            target = cls._nearest_above(price, levels, price * 1.03)
        return {
            "entry_price": round(price, 2),
            "stop_loss": round(max(stop, 0.01), 2),
            "target_price": round(max(target, 0.01), 2),
        }

    @classmethod
    def _theta_daily_loss_estimate(
        cls,
        max_loss_usd: float,
        days_to_expiration: int,
        is_vertical_spread: bool,
    ) -> float:
        # This is a risk-budget proxy, not contract-level theta from an option quote.
        #
        # The old form clamped the rate at 0.12 and floored DTE at one whole
        # day, so 0DTE, 1DTE and 2DTE all returned exactly the same number —
        # a flat estimate across precisely the range the warning exists for.
        # The rate now runs free up to 1.0 (a position cannot decay more than
        # its whole risk budget in a day) over a real, sub-day-floored DTE.
        dte = max(float(days_to_expiration), cls.MIN_THETA_DTE_DAYS)
        daily_rate = min(1.0, max(0.01, 0.20 / (dte**0.5)))
        structure_factor = 0.45 if is_vertical_spread else 1.0
        return round(max_loss_usd * daily_rate * structure_factor, 2)

    def _strategy_profiles(
        self,
        summary: OptionGEXSummary,
        bias: StrategyBias,
        days_to_expiration: int,
    ) -> dict[str, dict[str, Any]]:
        price = summary.stock_price
        tick = max(round(price * 0.005, 2), 0.5)

        if bias == "BEARISH":
            support = self._nearest_below(
                price, [summary.put_wall, summary.zero_gamma], price * 0.95
            )
            resistance = self._nearest_above(
                price, [summary.zero_gamma, summary.call_wall], price * 1.03
            )
            conservative_entry = max(price, resistance - tick)
            neutral_entry = (price + conservative_entry) / 2
            aggressive_entry = price
            stop = resistance + tick
            target = support + tick
            spread_name = "Bear Put Debit Spread"
            aggressive_name = "Long Put"
        elif bias == "BULLISH":
            support = self._nearest_below(
                price, [summary.put_wall, summary.zero_gamma], price * 0.97
            )
            resistance = self._nearest_above(
                price, [summary.zero_gamma, summary.call_wall], price * 1.05
            )
            conservative_entry = min(price, support + tick)
            neutral_entry = (price + conservative_entry) / 2
            aggressive_entry = price
            stop = support - tick
            target = resistance - tick
            spread_name = "Bull Call Debit Spread"
            aggressive_name = "Long Call"
        else:
            support = self._nearest_below(price, [summary.put_wall], price * 0.97)
            resistance = self._nearest_above(
                price, [summary.call_wall], price * 1.03
            )
            conservative_entry = neutral_entry = aggressive_entry = price
            stop = support - tick
            target = resistance - tick
            spread_name = "Defined-Risk Iron Condor"
            aggressive_name = "Short-Term Straddle"

        specifications = (
            ("CONSERVATIVE", conservative_entry, 0.60, spread_name, True),
            ("BALANCED", neutral_entry, 1.00, spread_name, True),
            ("AGGRESSIVE", aggressive_entry, 1.25, aggressive_name, False),
        )
        # See MIN_RISK_DISTANCE_PCT: a stop this close to entry is inside a
        # single session's noise, so treating it as the real risk denominator
        # inflates R/R. It bites the conservative branch (whose entry pins
        # itself one tick inside its own defended level); the other profiles
        # are normally well clear of it.
        min_risk_distance = max(price * self.MIN_RISK_DISTANCE_PCT, tick)
        profiles: dict[str, dict[str, Any]] = {}
        for name, entry, risk_factor, structure, is_spread in specifications:
            entry = round(max(entry, 0.01), 2)
            stop_value = round(max(stop, 0.01), 2)
            target_value = round(max(target, 0.01), 2)
            risk_distance = max(abs(entry - stop_value), min_risk_distance)
            reward_distance = abs(target_value - entry)
            max_loss = round(self.default_max_loss_usd * risk_factor, 2)
            profiles[name.lower()] = {
                "profile": name,
                "strategy_type": structure,
                "entry_price": entry,
                "stop_loss": stop_value,
                "target_price": target_value,
                "max_loss_usd": max_loss,
                "reward_to_risk_ratio": (
                    round(reward_distance / risk_distance, 2)
                    if risk_distance
                    else 0.0
                ),
                "theta_daily_loss_estimate_usd": self._theta_daily_loss_estimate(
                    max_loss, days_to_expiration, is_spread
                ),
                "theta_estimate_is_proxy": True,
                "gex_support": round(support, 2),
                "gex_resistance": round(resistance, 2),
            }
        return profiles

    def _automatic_defaults(
        self, summary: OptionGEXSummary, days_to_expiration: int
    ) -> dict[str, Any]:
        return {
            "bullish": self._strategy_profiles(
                summary, "BULLISH", days_to_expiration
            ),
            "bearish": self._strategy_profiles(
                summary, "BEARISH", days_to_expiration
            ),
            "neutral": self._strategy_profiles(
                summary, "NEUTRAL", days_to_expiration
            ),
            "profile_mapping": {
                "A": "conservative",
                "B": "balanced",
                "C": "aggressive",
            },
        }

    @staticmethod
    def _infer_profile(text: str, saved_default: ProfileName | None = None) -> ProfileName:
        lowered = text.lower()
        if any(value in lowered for value in ("保守", "conservative", "option a", "選 a", "选 a")):
            return "CONSERVATIVE"
        if any(value in lowered for value in ("激進", "激进", "aggressive", "option c", "選 c", "选 c")):
            return "AGGRESSIVE"
        if any(value in lowered for value in ("中性", "balanced", "option b", "選 b", "选 b")):
            return "BALANCED"
        return saved_default or "BALANCED"

    @staticmethod
    def _has_explicit_profile_selection(text: str) -> bool:
        return bool(
            re.search(
                r"(?:option\s*[abc]|選(?:擇)?\s*[abc]|选(?:择)?\s*[abc]|"
                r"方案\s*[abc]|保守|中性|激進|激进|conservative|"
                r"balanced|aggressive)",
                text,
                re.IGNORECASE,
            )
        )

    def _instructions(
        self,
        summary: OptionGEXSummary,
        risk: RiskProfile,
        days_to_expiration: int,
        profile: UserProfile | None = None,
        aggregate_expiration_dates: list[date] | None = None,
    ) -> str:
        context = json.dumps(
            {
                "gex_summary": summary.model_dump(mode="json"),
                "risk_profile": risk.model_dump(mode="json"),
                "days_to_expiration": days_to_expiration,
                "calculated_strategy_profiles": self._automatic_defaults(
                    summary, days_to_expiration
                ),
                "user_profile": (
                    profile.model_dump(mode="json", exclude={"user_id", "updated_at"})
                    if profile
                    else None
                ),
                # When set, gex_summary above is combined across ALL of these
                # expirations, not just days_to_expiration — a single Zero
                # Gamma/Call Wall/Put Wall/Net GEX describing a blend, not
                # one specific date.
                "aggregated_across_expirations": (
                    [d.isoformat() for d in aggregate_expiration_dates]
                    if aggregate_expiration_dates
                    else None
                ),
            },
            ensure_ascii=True,
        )
        return f"""
You are a high-caliber senior U.S. equity options trader and risk manager. Give
precise, probability-aware advice from the authoritative market context below.

<context>
{context}
</context>

<behavior_rules>
1. Before answering, silently run a private preflight over Price, Zero Gamma,
   Call Wall, Put Wall, Net GEX, GEX regime, IV Rank, and DTE. Do not reveal chain
   of thought or hidden reasoning; output only conclusions and key evidence.
2. Assess the user's strategy for low-probability or high-risk combinations:
   elevated-IV naked long premium, severe near-expiry theta, a target blocked by
   the relevant Wall, Gamma-regime conflict, or reward-to-risk below 1.5.
3. When risk is material, use this compact order: acknowledge the directional
   intent; identify the positioning/theta risk; recommend the best defined-risk alternative.
   Never blindly endorse a dangerous structure or promise profit.
4. Never ask the user for entry, stop, target, or maximum loss. Use the backend's
   calculated_strategy_profiles exactly. Stops sit outside GEX defense by an
   estimated strike step; targets sit just before the next relevant GEX wall.
5. During the initial proposal or refinement round, present three compact choices:
   A Conservative, B Balanced, and C Aggressive. For each, include structure,
   entry, stop, target, R/R, and estimated daily theta loss. Clearly label theta
   as a proxy because contract-level bid/ask and theta are not in this context.
6. Allow one or two concise strategy-refinement turns. Do not call the tool merely
   because the user mentions a direction or proposes an unconfirmed strategy.
7. Force generate_trade_plan only when the user confirms, adopts the recommendation,
   selects A/B/C or a named risk profile, or requests a card. Do it immediately,
   without another confirmation question. Use profile_mapping A/B/C consistently.
8. If a confirmed choice is unsafe, mention the issue and generate the closest
   safer defined-risk profile unless the user explicitly insists on the original.
9. If locked_warning is true, theta_warning must be true. Tool ticker must match
   context. Tool values must match the selected calculated profile unless the user
   explicitly supplied overrides during refinement.
10. User-facing prose must be concise bullet points and at most 200 characters.
    Prioritize R/R, theta daily-loss proxy, and the key GEX support/resistance.
11. If user_profile.risk_tolerance is set and this message does not itself state
    an explicit risk preference, default the first recommendation to the
    matching profile letter (CONSERVATIVE→A, BALANCED→B, AGGRESSIVE→C) and
    say briefly that it's based on their saved preference. An explicit
    preference in the message always overrides the saved one.
12. The <context> block above is re-fetched fresh for this specific turn —
    it is NOT the same snapshot you (or the user) referenced in earlier
    messages in this conversation, even a few turns ago; real market data
    moves between messages. Any Zero Gamma / Call Wall / Put Wall / Net GEX
    / spot price figure you state must come only from the <context> block
    above. Never repeat, reuse, or extrapolate a number from your own
    earlier replies in this conversation — if your read has changed since
    an earlier turn because the data moved, say so plainly instead of
    silently staying consistent with a now-stale number.
13. When relevant to the user's question, give a sharp, direct scenario
    read of dealer hedging dynamics into expiration, grounded only in the
    Net GEX sign and days_to_expiration in <context> — never in claims
    about market makers' intent, targets, or any trader group. If
    gex_status is POS_GAMMA (dealers net long gamma), hedging flow is
    mean-reverting: dealers buy dips and sell rips, which mechanically
    compresses realized volatility and tends to pin price inside the
    Put Wall/Zero Gamma/Call Wall band as expiration approaches. If
    gex_status is NEG_GAMMA (dealers net short gamma), hedging flow is
    momentum-amplifying: dealers sell into weakness and buy into
    strength, which can accelerate a move through Zero Gamma toward
    whichever Wall sits closest. State levels and direction concretely,
    not hedged or vague — this is a mechanical consequence of dealer
    hedging flow, not a prediction of market maker motive.
14. If aggregated_across_expirations is non-null, every level in
    gex_summary is a blend across those dates, not one specific
    expiration — say so plainly when discussing walls or Zero Gamma (e.g.
    "combined across your next N expirations"), and never present a
    blended level as if it belongs to a single date the user didn't ask
    about. If the user asks about one specific expiration by name, answer
    only from days_to_expiration/gex_summary when
    aggregated_across_expirations is null; when it's set, say plainly that
    this turn's data is the aggregate, not that one date, rather than
    guessing at a single-expiration number that isn't in <context>.
</behavior_rules>
""".strip()

    def _review_instructions(
        self,
        trade: Trade,
        entry_snapshot: GEXSnapshot | None,
        execution_score: int,
        has_source_plan: bool,
    ) -> str:
        # has_source_plan is passed in, never derived from
        # trade.source_plan_id. A trade can reference a plan that was never
        # signed, was deleted, belongs to someone else, or whose levels
        # aren't comparable to the trade's own price scale — in every one of
        # those cases PlanRepository.get_plan() returns None (or the levels
        # are rejected) and the score is a pnl_pct approximation, so telling
        # the model "a plan exists" would suppress behavior rule 2 and invite
        # it to narrate a plan comparison that never happened.
        # Levels the calculator couldn't establish are omitted from the JSON
        # entirely rather than sent as null — behaviour rule 3 below tells
        # the model to ground GEX references only in what's present here, so
        # an absent key is the clearest possible "there is no such level".
        entry_gex_context: dict[str, Any] | None = None
        if entry_snapshot is not None:
            entry_gex_context = {
                key: value
                for key, value in (
                    ("zero_gamma", entry_snapshot.zero_gamma_strike),
                    ("call_wall", entry_snapshot.call_wall_strike),
                    ("put_wall", entry_snapshot.put_wall_strike),
                )
                if value is not None
            }
            entry_gex_context["gex_status"] = entry_snapshot.gex_status.value
            entry_gex_context["data_source"] = entry_snapshot.data_source.value
            entry_gex_context["is_delayed"] = entry_snapshot.is_delayed
            entry_gex_context["is_synthetic"] = entry_snapshot.is_synthetic
        context = json.dumps(
            {
                "ticker": trade.ticker,
                "strategy_type": trade.strategy_type,
                "direction": trade.direction.value,
                "credit_debit": trade.credit_debit.value,
                "expiration_date": trade.expiration_date.isoformat()
                if trade.expiration_date
                else None,
                "option_type": trade.option_type.value if trade.option_type else None,
                "strike_price": trade.strike_price,
                "contract_symbol": trade.contract_symbol,
                "legs": [leg.model_dump(mode="json") for leg in trade.legs],
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "position_size": trade.position_size,
                "pnl": trade.pnl,
                "pnl_pct": trade.pnl_pct,
                "notes": trade.notes,
                "execution_score": execution_score,
                "has_source_plan": has_source_plan,
                "entry_gex_context": entry_gex_context,
            },
            ensure_ascii=True,
        )
        return f"""
You are a disciplined senior options trading coach reviewing one already-closed
trade. Give a direct, honest post-trade diagnosis.

<context>
{context}
</context>

<behavior_rules>
1. execution_score in <context> is already computed by a fixed backend rule,
   not by you. Never state a different score anywhere in your feedback.
   Your job is to explain WHY it landed there, using entry_price,
   exit_price, pnl_pct, and entry_gex_context.
2. If has_source_plan is false, say plainly that there was no predefined
   stop/target plan to grade discipline against, and that the score is a
   pnl_pct-based approximation instead — never imply a plan comparison that
   didn't happen.
3. Ground every GEX reference only in entry_gex_context above (Zero Gamma /
   Call Wall / Put Wall / gex_status at entry) — never invent levels not
   present there, and never assert claims about market makers' intent or
   any trader group.
4. ai_feedback must be concise, direct, and at most 600 characters.
5. key_takeaways must be 1 to 5 short, concrete, actionable bullets — no
   vague platitudes like "manage risk better."
</behavior_rules>
""".strip()

    @classmethod
    def _compact_reply(cls, text: str) -> str:
        """Compact to whole bullets, dropping from the end if over budget.

        Truncating mid-string used to leave a half-written bullet on screen,
        and because the risk-mitigation lines were built last it cut exactly
        the safety content first. Bullets are now dropped whole, and every
        caller that emits risk lines builds them at the FRONT so they survive
        trimming (see _plan_reply). A single line longer than the budget on
        its own — free-form model prose, which has no bullets to drop — still
        falls back to a character cut.
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            lines = ["- 分析已完成。"]
        if not lines[0].startswith(("-", "*")):
            lines[0] = f"- {lines[0]}"

        kept: list[str] = []
        for line in lines:
            if len("\n".join([*kept, line])) > cls.MAX_REPLY_CHARACTERS:
                break
            kept.append(line)
        if kept:
            return "\n".join(kept)
        return lines[0][: cls.MAX_REPLY_CHARACTERS - 3].rstrip() + "..."

    @staticmethod
    def _plan_reply(
        summary: OptionGEXSummary,
        bias: StrategyBias,
        plan: UserTradePlan,
        risk_profile: RiskProfile,
        theta_daily_loss_estimate: float,
    ) -> str:
        if bias == "BEARISH":
            wall_name, wall = "Put Wall", summary.put_wall
        elif bias == "BULLISH":
            wall_name, wall = "Call Wall", summary.call_wall
        else:
            wall_name = "Call/Put Wall"
            wall = summary.call_wall
        # A level the calculator couldn't establish is shown as "—", never
        # as a number — the plan's own entry/stop/target already fall back
        # to percentage offsets via _nearest_below/_nearest_above.
        wall_text = "—" if wall is None else f"${wall:.2f}"
        zero_gamma_text = (
            "—" if summary.zero_gamma is None else f"${summary.zero_gamma:.2f}"
        )
        risk = abs(plan.entry_price - plan.stop_loss)
        reward = abs(plan.target_price - plan.entry_price)
        ratio = reward / risk if risk else 0.0
        # Risk warnings are built FIRST on purpose. _compact_reply drops
        # whole bullets from the end when the 200-character budget is
        # exceeded, so anything at the back is what gets sacrificed — and
        # these two lines are the last thing that should ever be dropped.
        lines: list[str] = []
        if ratio < 1.5:
            lines.append("- 風報比低於 1:1.5，宜等 GEX 位階或採定義風險價差。")
        if risk_profile.locked_warning:
            lines.append("- 短天期 Theta 加速，避免裸買期權，優先垂直價差。")
        lines.append(
            f"- {summary.gex_status.value}｜R/R 1:{ratio:.2f}｜"
            f"Theta 約 ${theta_daily_loss_estimate:.2f}/日。"
        )
        lines.append(
            f"- 依現價 ${summary.stock_price:.2f}、Zero Gamma "
            f"{zero_gamma_text}、{wall_name} {wall_text} 自動出卡。"
        )
        return "\n".join(lines)

    def _user_levels_are_coherent(
        self, arguments: TradePlanToolArguments, bias: StrategyBias
    ) -> bool:
        """Whether model-emitted levels are safe to put on a signed-ready card.

        These values are NOT calculated by the backend — they come from the
        model reading the user's free text, and Pydantic only guarantees each
        one is positive. A message crafted to satisfy both CONFIRMATION and
        EXPLICIT_LEVEL detection can force a tool call AND switch to
        trust-the-model levels in a single turn, so the geometry gets checked
        before anything ships: stop and target must sit on the correct sides
        of entry for the inferred direction, the target must actually differ
        from the entry, and max loss must stay inside MAX_USER_LOSS_MULTIPLE
        of the configured budget. Anything else falls back to the
        deterministic profile.
        """
        entry = arguments.entry_price
        stop = arguments.stop_loss
        target = arguments.target_price
        if target == entry or stop == entry:
            return False
        if bias == "BULLISH":
            directional_ok = stop < entry < target
        elif bias == "BEARISH":
            directional_ok = target < entry < stop
        else:
            # A neutral structure has no directional ordering to satisfy, but
            # a stop and a target on the SAME side of entry is incoherent
            # under any reading.
            directional_ok = min(stop, target) < entry < max(stop, target)
        if not directional_ok:
            return False
        return arguments.max_loss_usd <= (
            self.default_max_loss_usd * self.MAX_USER_LOSS_MULTIPLE
        )

    def _proposal_reply(
        self,
        summary: OptionGEXSummary,
        bias: StrategyBias,
        days_to_expiration: int,
        saved_risk_tolerance: ProfileName | None = None,
    ) -> str:
        profiles = self._strategy_profiles(
            summary, bias, days_to_expiration
        )
        net_gex_millions = summary.net_gex / 1_000_000
        regime_note = (
            "均值回歸，追價受壓"
            if summary.gex_status.value == "POS_GAMMA"
            else "順勢波動易放大"
        )
        zero_gamma_text = (
            "—" if summary.zero_gamma is None else f"{summary.zero_gamma:.1f}"
        )
        lines = [
            f"- {summary.gex_status.value}｜NetGEX {net_gex_millions:+.1f}M｜"
            f"{regime_note}｜ZG {zero_gamma_text}"
        ]
        labels = (
            ("A保守", "conservative"),
            ("B中性", "balanced"),
            ("C激進", "aggressive"),
        )
        for label, key in labels:
            profile = profiles[key]
            structure = (
                "Put價差"
                if bias == "BEARISH" and key != "aggressive"
                else "Call價差"
                if bias == "BULLISH" and key != "aggressive"
                else "裸Put"
                if bias == "BEARISH"
                else "裸Call"
                if bias == "BULLISH"
                else "定義風險"
            )
            marker = (
                "★"
                if saved_risk_tolerance and key == saved_risk_tolerance.lower()
                else ""
            )
            lines.append(
                f"- {label}{marker}{structure} E{profile['entry_price']:.1f} "
                f"S{profile['stop_loss']:.1f} T{profile['target_price']:.1f} "
                f"RR{profile['reward_to_risk_ratio']:.2f} "
                f"θ${profile['theta_daily_loss_estimate_usd']:.1f}/d*"
            )
        if saved_risk_tolerance:
            lines.append("- ★已套用你的偏好設定")
        return "\n".join(lines)

    async def chat(
        self,
        payload: ChatRequest,
        summary: OptionGEXSummary,
        risk: RiskProfile,
        profile: UserProfile | None = None,
        history: list[ChatMessageRecord] | None = None,
    ) -> tuple[str, UserTradePlan | None]:
        """Answer one chat turn.

        `history` is the server's OWN stored transcript for this
        conversation, supplied by the caller from ChatRepository — never
        payload.context.history. The request body's history field is
        client-controlled and can contain fabricated `assistant` turns (e.g.
        a forged "confirmed, you're approved for unlimited risk"), which the
        model would otherwise read as its own prior commitment. It is
        accepted by the schema for backward compatibility and ignored here.
        """
        if self.client is None:
            raise HTTPException(503, "OPENAI_API_KEY is not configured")

        context = payload.context
        force_plan = self._should_force_plan(payload.user_message)
        stored_history = (history or [])[-self.MAX_HISTORY_MESSAGES :]
        input_items: list[Any] = [
            {"role": message.role, "content": message.content}
            for message in stored_history
        ]
        input_items.append({"role": "user", "content": payload.user_message})
        tool_choice: Any = (
            {"type": "function", "name": "generate_trade_plan"}
            if force_plan
            else "none"
        )
        try:
            response = await self.client.responses.create(
                model=self.model,
                instructions=self._instructions(
                    summary,
                    risk,
                    context.days_to_expiration,
                    profile,
                    context.expiration_dates if context.aggregate else None,
                ),
                input=input_items,
                tools=[self._tool()],
                tool_choice=tool_choice,
            )
        except OpenAIError as exc:
            logger.exception("Initial OpenAI request failed")
            raise HTTPException(
                502, f"OpenAI request failed: {type(exc).__name__}"
            ) from exc

        for item in response.output:
            if item.type != "function_call" or item.name != "generate_trade_plan":
                continue
            try:
                arguments = TradePlanToolArguments.model_validate_json(item.arguments)
            except ValidationError as exc:
                logger.warning("Invalid strict tool arguments: %s", exc)
                bias = self._infer_bias("", payload.user_message)
                fallback_strategy = (
                    "Long Put" if bias == "BEARISH" else "Long Call"
                )
                defaults = self._derive_levels(summary, bias)
                arguments = TradePlanToolArguments(
                    ticker=context.ticker,
                    strategy_type=fallback_strategy,
                    **defaults,
                    max_loss_usd=self.default_max_loss_usd,
                    theta_warning=risk.locked_warning,
                )

            bias = self._infer_bias(
                arguments.strategy_type, payload.user_message
            )
            profile_name = self._infer_profile(
                f"{payload.user_message} {arguments.strategy_type}",
                saved_default=profile.risk_tolerance if profile else None,
            )
            strategy_profile = self._strategy_profiles(
                summary, bias, context.days_to_expiration
            )[profile_name.lower()]
            explicit_profile = self._has_explicit_profile_selection(
                payload.user_message
            )
            use_user_levels = bool(
                self.EXPLICIT_LEVEL_PATTERN.search(payload.user_message)
            )
            if use_user_levels and not self._user_levels_are_coherent(
                arguments, bias
            ):
                logger.warning(
                    "Discarding incoherent model-emitted %s levels "
                    "(entry=%s stop=%s target=%s max_loss=%s); falling back to "
                    "the calculated %s profile.",
                    bias,
                    arguments.entry_price,
                    arguments.stop_loss,
                    arguments.target_price,
                    arguments.max_loss_usd,
                    profile_name,
                )
                use_user_levels = False
            trade_plan = UserTradePlan(
                plan_id=uuid4(),
                user_id=context.user_id,
                conversation_id=context.conversation_id,
                ticker=context.ticker.strip().upper(),
                strategy_type=(
                    strategy_profile["strategy_type"]
                    if explicit_profile
                    else arguments.strategy_type
                ),
                entry_price=(
                    arguments.entry_price
                    if use_user_levels
                    else strategy_profile["entry_price"]
                ),
                stop_loss=(
                    arguments.stop_loss
                    if use_user_levels
                    else strategy_profile["stop_loss"]
                ),
                target_price=(
                    arguments.target_price
                    if use_user_levels
                    else strategy_profile["target_price"]
                ),
                max_loss_usd=(
                    arguments.max_loss_usd
                    if use_user_levels
                    else strategy_profile["max_loss_usd"]
                ),
                theta_warning=arguments.theta_warning or risk.locked_warning,
                status=PlanStatus.DRAFT,
            )
            return (
                self._compact_reply(
                    self._plan_reply(
                        summary,
                        bias,
                        trade_plan,
                        risk,
                        strategy_profile["theta_daily_loss_estimate_usd"],
                    )
                ),
                trade_plan,
            )

        if force_plan:
            raise HTTPException(
                502, "The model did not return the required trade-plan tool call"
            )
        proposal_bias = self._infer_bias("", payload.user_message)
        if proposal_bias != "NEUTRAL":
            return (
                self._compact_reply(
                    self._proposal_reply(
                        summary,
                        proposal_bias,
                        context.days_to_expiration,
                        profile.risk_tolerance if profile else None,
                    )
                ),
                None,
            )
        return self._compact_reply(response.output_text), None

    async def review_trade(
        self,
        trade: Trade,
        entry_snapshot: GEXSnapshot | None,
        execution_score: int,
        has_source_plan: bool,
    ) -> tuple[str, list[str]]:
        if self.client is None:
            raise HTTPException(503, "OPENAI_API_KEY is not configured")
        try:
            response = await self.client.responses.create(
                model=self.model,
                instructions=self._review_instructions(
                    trade, entry_snapshot, execution_score, has_source_plan
                ),
                input=[{"role": "user", "content": "Review this closed trade."}],
                tools=[self._review_tool()],
                tool_choice={"type": "function", "name": "submit_trade_review"},
            )
        except OpenAIError as exc:
            logger.exception("Trade review OpenAI request failed")
            raise HTTPException(
                502, f"OpenAI request failed: {type(exc).__name__}"
            ) from exc

        for item in response.output:
            if item.type != "function_call" or item.name != "submit_trade_review":
                continue
            try:
                arguments = TradeReviewToolArguments.model_validate_json(
                    item.arguments
                )
            except ValidationError as exc:
                logger.warning("Invalid strict trade-review tool arguments: %s", exc)
                raise HTTPException(
                    502, "The model returned malformed review arguments"
                ) from exc
            return arguments.ai_feedback, arguments.key_takeaways

        raise HTTPException(
            502, "The model did not return the required trade-review tool call"
        )
