from datetime import datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.database import (
    Base,
    ChatRepository,
    GEXSnapshotRepository,
    ProfileRepository,
)
from app.models import GEXStatus, OptionGEXSummary, UserProfileUpdate


async def _session_factory() -> async_sessionmaker[AsyncSession]:
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


def gex_summary(stock_price: float = 100.0) -> OptionGEXSummary:
    return OptionGEXSummary(
        ticker="AAPL",
        stock_price=stock_price,
        zero_gamma=95.0,
        call_wall=110.0,
        put_wall=90.0,
        iv_rank=40.0,
        net_gex=1_000_000.0,
        gex_status=GEXStatus.POS_GAMMA,
    )


@pytest.mark.asyncio
async def test_chat_repository_groups_messages_into_conversations() -> None:
    repo = ChatRepository(await _session_factory())
    await repo.save_message("conv-1", "user-1", "AAPL", "user", "hi")
    await repo.save_message("conv-1", "user-1", "AAPL", "assistant", "hello")
    await repo.save_message("conv-2", "user-1", "TSLA", "user", "other convo")

    conversations = await repo.list_conversations("user-1")
    assert len(conversations) == 2
    by_id = {c.conversation_id: c for c in conversations}
    assert by_id["conv-1"].message_count == 2
    assert by_id["conv-1"].last_message == "hello"
    assert by_id["conv-2"].ticker == "TSLA"

    messages = await repo.get_messages("conv-1", "user-1")
    assert [m.role for m in messages.messages] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_chat_repository_scopes_conversations_by_user() -> None:
    repo = ChatRepository(await _session_factory())
    await repo.save_message("conv-1", "user-1", "AAPL", "user", "hi")

    other_user_messages = await repo.get_messages("conv-1", "user-2")
    assert other_user_messages.messages == []


@pytest.mark.asyncio
async def test_profile_repository_returns_none_until_upserted() -> None:
    repo = ProfileRepository(await _session_factory())
    assert await repo.get_profile("user-1") is None

    saved = await repo.upsert_profile(
        "user-1",
        UserProfileUpdate(
            risk_tolerance="AGGRESSIVE",
            preferred_strategy_types=["Long Put", "Bear Put Spread"],
            notes="likes short-dated plays",
        ),
    )
    assert saved.risk_tolerance == "AGGRESSIVE"
    assert saved.preferred_strategy_types == ["Long Put", "Bear Put Spread"]

    fetched = await repo.get_profile("user-1")
    assert fetched == saved


@pytest.mark.asyncio
async def test_profile_repository_upsert_overwrites_existing() -> None:
    repo = ProfileRepository(await _session_factory())
    await repo.upsert_profile(
        "user-1", UserProfileUpdate(risk_tolerance="CONSERVATIVE")
    )
    updated = await repo.upsert_profile(
        "user-1", UserProfileUpdate(risk_tolerance="BALANCED")
    )
    assert updated.risk_tolerance == "BALANCED"
    assert await repo.get_profile("user-1") == updated


@pytest.mark.asyncio
async def test_gex_snapshot_repository_save_and_list() -> None:
    repo = GEXSnapshotRepository(await _session_factory())
    assert await repo.last_snapshot_time("AAPL") is None

    await repo.save_snapshot("AAPL", 30, gex_summary(stock_price=100.0))
    await repo.save_snapshot("AAPL", 30, gex_summary(stock_price=101.0))
    await repo.save_snapshot("TSLA", 30, gex_summary(stock_price=250.0))

    snapshots = await repo.list_snapshots("AAPL")
    assert len(snapshots) == 2
    # Most recent first.
    assert snapshots[0].underlying_price == 101.0
    assert snapshots[1].underlying_price == 100.0
    assert all(s.ticker == "AAPL" for s in snapshots)

    last_time = await repo.last_snapshot_time("AAPL")
    assert isinstance(last_time, datetime)
    assert last_time.tzinfo is not None


@pytest.mark.asyncio
async def test_gex_snapshot_repository_list_respects_limit() -> None:
    repo = GEXSnapshotRepository(await _session_factory())
    for price in (100.0, 101.0, 102.0):
        await repo.save_snapshot("AAPL", 30, gex_summary(stock_price=price))

    snapshots = await repo.list_snapshots("AAPL", limit=1)
    assert len(snapshots) == 1
    assert snapshots[0].underlying_price == 102.0
