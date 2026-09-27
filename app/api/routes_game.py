"""경기 시청 화면(S4) + 홈(S3) 엔드포인트."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from app.adapters.relay.base import RelaySourceError
from app.adapters.video.notes import evidence_line, near
from app.api.deps import (
    get_card_service,
    get_matchup_service,
    get_moment_service,
    get_relay_source,
    get_video_notes,
    profile_params,
)
from app.api.schemas import CardOut, CardsOut, GameOut, MatchupOut, MomentOut, StateOut
from app.domain.game_state import replay
from app.domain.models import Card, ExplainProfile
from app.domain.timeline import Timeline

router = APIRouter(prefix="/api/games", tags=["game"])

_T_QUERY = Query(default=None, ge=0, description="영상 타임코드(초). 생략하면 경기 전체.")


@router.get("", response_model=list[GameOut], summary="지난 경기 목록 (S3 홈)")
def list_games():
    return [GameOut.of(meta) for meta in get_relay_source().list_games()]


@router.get("/{game_id}/state", response_model=StateOut, summary="스코어보드 (S4 좌하단)")
def game_state(game_id: str = Path(...), t: Optional[int] = _T_QUERY):
    feed = _load(game_id)
    timeline = Timeline(feed.meta.relay_video_offset_sec)
    relay_t = timeline.to_relay(int(t)) if t is not None else None
    state = replay(feed.events, feed.meta.away_team, feed.meta.home_team, until_t=relay_t)
    return StateOut.of(state)


@router.get("/{game_id}/cards", response_model=CardsOut, summary="AI 설명 카드 (S4 우상단)")
def game_cards(
    game_id: str = Path(...),
    t: Optional[int] = _T_QUERY,
    profile: ExplainProfile = Depends(profile_params),
):
    feed = _load(game_id)
    timeline = Timeline(feed.meta.relay_video_offset_sec)
    service = get_card_service()
    cards = service.cards(feed.meta.id, t, profile)
    # 감지된 상황은 있는데 카드가 하나도 안 나왔다면 생성이 죽은 것이다.
    detected = service.situations(feed.meta.id, t)
    degraded = bool(detected) and not cards
    return CardsOut(
        game_id=feed.meta.id,
        t=t,
        level=profile.level,
        cards=[_card_out(c, feed.meta.id, timeline) for c in cards],
        degraded=degraded,
    )


@router.post(
    "/{game_id}/cards/{card_id:path}/simplify",
    response_model=CardOut,
    summary='"더 쉽게 설명해줘" (S4 카드 버튼)',
)
def simplify_card(
    game_id: str = Path(...),
    card_id: str = Path(...),
    profile: ExplainProfile = Depends(profile_params),
):
    feed = _load(game_id)
    card = get_card_service().simplify(feed.meta.id, card_id, profile)
    if card is None:
        raise HTTPException(
            status_code=409,
            detail="더 쉽게 설명할 수 없는 카드입니다 (이미 가장 쉬운 단계이거나 존재하지 않음)",
        )
    return _card_out(card, feed.meta.id, Timeline(feed.meta.relay_video_offset_sec))


@router.get(
    "/{game_id}/moment", response_model=MomentOut, summary="방금 장면 한 줄 요약 (S4)"
)
def game_moment(
    game_id: str = Path(...),
    t: Optional[int] = _T_QUERY,
    profile: ExplainProfile = Depends(profile_params),
):
    """t 직전의 결과성 장면 하나. LLM이 없거나 죽어도 중계 원문 조립으로 항상 응답한다."""
    feed = _load(game_id)
    moment = get_moment_service().moment(feed.meta.id, t, profile.level)
    return MomentOut(**moment.model_dump())


@router.get(
    "/{game_id}/matchup", response_model=MatchupOut, summary="선수·매치업 분석 (S4 우하단)"
)
def game_matchup(game_id: str = Path(...), t: Optional[int] = _T_QUERY):
    feed = _load(game_id)
    return MatchupOut.of(get_matchup_service().matchup(feed.meta.id, t))


def _load(game_id: str):
    try:
        return get_relay_source().load(game_id)
    except RelaySourceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _card_out(card: Card, game_id: str, timeline: Timeline) -> CardOut:
    """카드 t는 내부에선 중계 시각이다 — 프론트가 시킹할 수 있게 영상 타임코드로 바꿔 내보낸다.

    영상 분석 스냅샷이 있으면 그 장면 전후의 VLM 관찰을 근거(reasons)에 덧붙인다.
    판정은 바꾸지 않는다 — 카드가 뜬 이유는 여전히 중계 + 규칙이다.
    """
    out = CardOut.of(card)
    out.t = timeline.to_video(card.t)
    notes = near(get_video_notes().events(game_id), out.t)
    if notes:
        out.reasons = out.reasons + [evidence_line(n) for n in notes[:2]]
    return out
