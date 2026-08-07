from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.models import (
    ChatContext,
    ChatRequest,
    GEXStatus,
    OptionGEXSummary,
    RiskProfile,
    UserProfile,
)
from app.services.openai_orchestrator import LLMOrchestrator


def gex_summary() -> OptionGEXSummary:
    return OptionGEXSummary(
        ticker="AAPL",
        stock_price=311,
        zero_gamma=333.7,
        call_wall=350,
        put_wall=300,
        iv_rank=55,
        net_gex=-1_000_000,
        gex_status=GEXStatus.NEG_GAMMA,
        calculated_at=datetime.now(timezone.utc),
    )


def test_only_confirmation_or_selection_forces_plan() -> None:
    assert LLMOrchestrator._should_force_plan("選 B 買 Put")
    assert LLMOrchestrator._should_force_plan("直接出卡")
    assert LLMOrchestrator._should_force_plan("採用你的建議")
    assert not LLMOrchestrator._should_force_plan("I want to buy put")
    assert not LLMOrchestrator._should_force_plan("我想做多")


def test_bearish_defaults_use_zero_gamma_and_put_wall() -> None:
    levels = LLMOrchestrator._derive_levels(gex_summary(), "BEARISH")
    assert levels == {
        "entry_price": 311,
        "stop_loss": 335.37,
        "target_price": 300,
    }


def test_three_profiles_scale_risk_and_theta() -> None:
    orchestrator = LLMOrchestrator(None, "test-model", 250)
    profiles = orchestrator._strategy_profiles(
        gex_summary(), "BEARISH", 5
    )
    assert set(profiles) == {"conservative", "balanced", "aggressive"}
    assert profiles["conservative"]["max_loss_usd"] == 150
    assert profiles["balanced"]["reward_to_risk_ratio"] == 1.46
    assert profiles["aggressive"]["theta_daily_loss_estimate_usd"] == 27.95
    assert profiles["aggressive"]["theta_estimate_is_proxy"] is True


def test_proposal_reply_contains_all_three_profiles_under_limit() -> None:
    orchestrator = LLMOrchestrator(None, "test-model", 250)
    reply = orchestrator._compact_reply(
        orchestrator._proposal_reply(gex_summary(), "BEARISH", 5)
    )
    assert "A保守" in reply
    assert "B中性" in reply
    assert "C激進" in reply
    assert "NetGEX" in reply
    assert "RR" in reply
    assert "θ$" in reply
    assert len(reply) <= 200


def test_plan_reply_is_bulleted_and_below_limit() -> None:
    plan = SimpleNamespace(
        entry_price=311,
        stop_loss=335.37,
        target_price=300,
    )
    risk = RiskProfile(
        gex_status=GEXStatus.NEG_GAMMA,
        volatility_regime="HIGH_VOL_TRENDING",
        risk_level="NORMAL",
        warnings=[],
        locked_warning=False,
    )
    reply = LLMOrchestrator._plan_reply(
        gex_summary(), "BEARISH", plan, risk, 8.5
    )
    compact = LLMOrchestrator._compact_reply(reply)
    assert compact.startswith("-")
    assert "Put Wall $300.00" in compact
    assert "NEG_GAMMA" in compact
    assert "R/R" in compact
    assert "Theta 約 $8.50/日" in compact
    assert "低於 1:1.5" in compact
    assert len(compact) <= 200


def test_infer_profile_prefers_explicit_mention_over_saved_default() -> None:
    assert (
        LLMOrchestrator._infer_profile("選 A 保守一點", saved_default="AGGRESSIVE")
        == "CONSERVATIVE"
    )


def test_infer_profile_falls_back_to_saved_default_when_unstated() -> None:
    assert (
        LLMOrchestrator._infer_profile("這檔怎麼看", saved_default="AGGRESSIVE")
        == "AGGRESSIVE"
    )


def test_infer_profile_defaults_to_balanced_with_no_saved_preference() -> None:
    assert LLMOrchestrator._infer_profile("這檔怎麼看", saved_default=None) == "BALANCED"


def test_proposal_reply_marks_saved_preference_profile() -> None:
    orchestrator = LLMOrchestrator(None, "test-model", 250)
    reply = orchestrator._proposal_reply(
        gex_summary(), "BEARISH", 5, saved_risk_tolerance="AGGRESSIVE"
    )
    assert "C激進★" in reply
    assert "A保守★" not in reply
    assert "B中性★" not in reply
    assert "已套用你的偏好設定" in reply


def test_proposal_reply_has_no_marker_without_saved_preference() -> None:
    orchestrator = LLMOrchestrator(None, "test-model", 250)
    reply = orchestrator._proposal_reply(gex_summary(), "BEARISH", 5)
    assert "★" not in reply


def test_instructions_embed_user_profile_when_present() -> None:
    orchestrator = LLMOrchestrator(None, "test-model", 250)
    risk = RiskProfile(
        gex_status=GEXStatus.NEG_GAMMA,
        volatility_regime="HIGH_VOL_TRENDING",
        risk_level="NORMAL",
        warnings=[],
        locked_warning=False,
    )
    profile = UserProfile(user_id="user-1", risk_tolerance="AGGRESSIVE")
    prompt = orchestrator._instructions(gex_summary(), risk, 5, profile)
    assert '"risk_tolerance": "AGGRESSIVE"' in prompt
    assert "user_id" not in prompt

    prompt_without_profile = orchestrator._instructions(gex_summary(), risk, 5, None)
    assert '"user_profile": null' in prompt_without_profile


def test_system_prompt_contains_trader_risk_framework() -> None:
    orchestrator = LLMOrchestrator(None, "test-model", 250)
    risk = RiskProfile(
        gex_status=GEXStatus.NEG_GAMMA,
        volatility_regime="HIGH_VOL_TRENDING",
        risk_level="HIGH",
        warnings=["High risk/high volatility; accelerated theta decay."],
        locked_warning=True,
    )
    prompt = orchestrator._instructions(gex_summary(), risk, 5)
    assert "senior U.S. equity options trader" in prompt
    assert "reward-to-risk below 1.5" in prompt
    assert "defined-risk alternative" in prompt
    assert "silently run a private preflight" in prompt
    assert "Do not reveal chain" in prompt
    assert "present three compact choices" in prompt
    assert "estimated daily theta loss" in prompt


@pytest.mark.asyncio
async def test_selected_put_forces_tool_and_builds_card() -> None:
    class FakeResponses:
        def __init__(self) -> None:
            self.request: dict = {}

        async def create(self, **kwargs):
            self.request = kwargs
            return SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="function_call",
                        name="generate_trade_plan",
                        arguments=(
                            '{"ticker":"AAPL","strategy_type":"Long Put",'
                            '"entry_price":1,"stop_loss":1,"target_price":1,'
                            '"max_loss_usd":1,"theta_warning":false}'
                        ),
                        call_id="call-1",
                    )
                ],
                output_text="",
            )

    fake_responses = FakeResponses()
    fake_client = SimpleNamespace(responses=fake_responses)
    orchestrator = LLMOrchestrator(fake_client, "test-model", 250)
    payload = ChatRequest(
        user_message="選 B 買 Put",
        context=ChatContext(
            user_id="user-1",
            conversation_id="conversation-1",
            ticker="AAPL",
            days_to_expiration=5,
        ),
    )
    risk = RiskProfile(
        gex_status=GEXStatus.NEG_GAMMA,
        volatility_regime="HIGH_VOL_TRENDING",
        risk_level="HIGH",
        warnings=["High risk/high volatility; accelerated theta decay."],
        locked_warning=True,
    )

    reply, plan = await orchestrator.chat(payload, gex_summary(), risk)

    assert fake_responses.request["tool_choice"] == {
        "type": "function",
        "name": "generate_trade_plan",
    }
    assert plan is not None
    assert plan.strategy_type == "Bear Put Debit Spread"
    assert plan.entry_price == 321.57
    assert plan.stop_loss == 335.25
    assert plan.target_price == 301.55
    assert plan.max_loss_usd == 250
    assert plan.theta_warning is True
    assert len(reply) <= 200
