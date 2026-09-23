"""챗봇 엔드포인트 (S4 하단 입력바 → 확장 패널)."""

from fastapi import APIRouter, HTTPException, Path

from app.adapters.relay.base import RelaySourceError
from app.api.deps import get_chat_service, get_relay_source
from app.api.schemas import ChatIn, ChatOut
from app.domain.profile import profile_from_onboarding

router = APIRouter(prefix="/api/games", tags=["chat"])


@router.post("/{game_id}/chat", response_model=ChatOut, summary="경기 맥락 질의응답")
def chat(payload: ChatIn, game_id: str = Path(...)):
    try:
        feed = get_relay_source().load(game_id)
    except RelaySourceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    profile = profile_from_onboarding(
        level=payload.level, categories=payload.categories
    )
    answer = get_chat_service().answer(
        game_id=feed.meta.id,
        question=payload.question,
        video_t=payload.t,
        profile=profile,
    )
    # 생성 실패도 200으로 돌려준다 — 화면은 안내 문구와 재시도 버튼으로 살아야 한다.
    return ChatOut(
        answer=answer.text,
        ok=answer.ok,
        source=answer.source,
        context=answer.context_summary,
        used_event_ids=answer.used_event_ids,
    )
