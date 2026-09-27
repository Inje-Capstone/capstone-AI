"""오늘 본 룰 + 퀴즈. 포인트·출제 기록·팝업 표시 조건은 백엔드 소유다."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from app.adapters.relay.base import RelaySourceError
from app.api.deps import get_card_service, get_quiz_service, get_relay_source, profile_params
from app.api.schemas import QuizIn, QuizItemOut, QuizOut, TodayRuleOut, TodayRulesOut
from app.domain.models import CATEGORY_LABELS, ExplainProfile
from app.domain.profile import profile_from_onboarding, select
from app.domain.timeline import Timeline

router = APIRouter(tags=["quiz"])


@router.get(
    "/api/games/{game_id}/today-rules",
    response_model=TodayRulesOut,
    summary="이 경기에서 카드로 본 룰 (시청 종료 팝업)",
)
def today_rules(
    game_id: str = Path(...),
    t: Optional[int] = Query(default=None, ge=0, description="어디까지 봤나(영상 초). 생략=끝까지"),
    profile: ExplainProfile = Depends(profile_params),
):
    """카드와 같은 선택 규칙(프로필 가중치)을 그대로 쓴다 — 화면에 뜬 카드 = 퀴즈 재료.

    LLM을 부르지 않는다. 카드 생성이 죽어 있어도 목록은 나온다.
    """
    try:
        feed = get_relay_source().load(game_id)
    except RelaySourceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    timeline = Timeline(feed.meta.relay_video_offset_sec)
    seen = select(get_card_service().situations(feed.meta.id, t), profile)

    by_rule: dict[str, TodayRuleOut] = {}
    for s in seen:
        row = by_rule.get(s.rule_id)
        if row is None:
            by_rule[s.rule_id] = TodayRuleOut(
                rule_id=s.rule_id, term_id=s.term_id, label=s.label, category=s.category,
                category_label=CATEGORY_LABELS.get(s.category, s.category),
                first_t=timeline.to_video(s.t), count=1,
            )
        else:
            row.count += 1
    rules = sorted(by_rule.values(), key=lambda r: r.first_t)
    term_ids = list(dict.fromkeys(r.term_id for r in rules))
    return TodayRulesOut(
        game_id=feed.meta.id, t=t, level=profile.level, rules=rules, term_ids=term_ids
    )


@router.post("/api/quiz", response_model=QuizOut, summary="오늘 본 룰 퀴즈 (하루 5문제)")
def quiz(payload: QuizIn):
    level = profile_from_onboarding(level=payload.level).level
    items = get_quiz_service().build(
        payload.term_ids, level=level, count=payload.count, seed=payload.seed
    )
    return QuizOut(level=level, items=[QuizItemOut(**i.model_dump()) for i in items])
