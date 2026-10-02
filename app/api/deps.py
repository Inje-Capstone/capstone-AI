"""서비스 조립. 어댑터 선택은 여기 한 곳에서만 일어난다."""

from functools import lru_cache
from typing import Any, Optional

from fastapi import Query

from app.adapters.diagnostic.seed import load_diagnostic
from app.adapters.glossary.seed import load_glossary
from app.adapters.llm import build_llm_client
from app.adapters.relay.fixture import FixtureRelaySource
from app.adapters.stats.fixture import FixtureStatsSource
from app.adapters.video.notes import VideoNotes
from app.config import get_settings
from app.domain.models import ExplainProfile
from app.domain.profile import profile_from_onboarding
from app.services.card_service import CardService
from app.services.chat_service import ChatService
from app.services.glossary_service import GlossaryService
from app.services.live_service import LiveRelaySource
from app.services.matchup_service import MatchupService
from app.services.moment_service import MomentService
from app.services.quiz_service import QuizService


@lru_cache(maxsize=1)
def get_live_source() -> LiveRelaySource:
    """실시간 경기 + 파일 경기. 실시간 경기도 일반 경기 엔드포인트로 그대로 보인다."""
    return LiveRelaySource(FixtureRelaySource(get_settings().fixture_dir))


def get_relay_source() -> LiveRelaySource:
    return get_live_source()


@lru_cache(maxsize=1)
def get_stats_source() -> FixtureStatsSource:
    return FixtureStatsSource()


@lru_cache(maxsize=1)
def get_video_notes() -> VideoNotes:
    return VideoNotes(get_settings().video_dir)


@lru_cache(maxsize=1)
def get_card_service() -> CardService:
    settings = get_settings()
    return CardService(
        relay=get_relay_source(),
        llm=build_llm_client(),
        snapshot_dir=settings.snapshot_dir,
    )


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    return ChatService(
        relay=get_relay_source(), llm=build_llm_client(), video=get_video_notes()
    )


@lru_cache(maxsize=1)
def get_matchup_service() -> MatchupService:
    return MatchupService(
        relay=get_relay_source(), stats=get_stats_source(), llm=build_llm_client()
    )


@lru_cache(maxsize=1)
def get_glossary_service() -> GlossaryService:
    return GlossaryService(load_glossary())


@lru_cache(maxsize=1)
def get_quiz_service() -> QuizService:
    return QuizService(load_glossary(), build_llm_client())


@lru_cache(maxsize=1)
def get_diagnostic_questions() -> list[dict[str, Any]]:
    """온보딩 수준 진단 문제은행. 고정 콘텐츠라 LLM도 경기 데이터도 쓰지 않는다."""
    return load_diagnostic()


@lru_cache(maxsize=1)
def get_moment_service() -> MomentService:
    settings = get_settings()
    return MomentService(
        relay=get_relay_source(),
        llm=build_llm_client(settings.summary_model),
        snapshot_dir=settings.snapshot_dir,
    )


def reset_services() -> None:
    """테스트에서 백엔드를 바꿔 끼울 때 캐시를 비운다."""
    for fn in (
        get_live_source,
        get_stats_source,
        get_card_service,
        get_chat_service,
        get_matchup_service,
        get_glossary_service,
        get_video_notes,
        get_quiz_service,
        get_moment_service,
        get_diagnostic_questions,
    ):
        fn.cache_clear()


def profile_params(
    level: Optional[str] = Query(
        default=None, description="온보딩 1단계: 입문 | 초보 | 익숙"
    ),
    category: Optional[list[str]] = Query(
        default=None,
        description="온보딩 2단계 (복수). 예: category=기본 룰&category=구종 · 투구",
    ),
    threshold: float = Query(default=0.5, ge=0.0, le=2.0),
) -> ExplainProfile:
    """유저 프로필은 요청 파라미터로 받는다 — 유저 DB는 백엔드 팀 소유다."""
    return profile_from_onboarding(level=level, categories=category, threshold=threshold)
