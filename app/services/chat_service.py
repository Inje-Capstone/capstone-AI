"""경기 맥락 챗봇.

"방금 저게 왜 아웃이에요?"가 성립하려면 질문 시점의 경기 상태와 직전 중계가
컨텍스트로 들어가야 한다. 그 조립이 이 모듈의 전부다.

카드와 달리 사전 생성이 불가능한(사용자가 언제 뭘 물을지 모른다) 유일한 실시간 경로다.
"""

import logging
from typing import Optional

from pydantic import BaseModel

from app.adapters.llm.base import LLMClient, LLMError
from app.adapters.relay.base import RelaySource
from app.adapters.video.notes import VideoNotes, clock, recent
from app.domain.game_state import replay
from app.domain.models import ExplainProfile, RelayEvent
from app.domain.timeline import Timeline
from app.prompt_templates import chat_user_prompt, system_prompt

log = logging.getLogger(__name__)

RECENT_EVENT_COUNT = 6
FAILURE_MESSAGE = "지금은 답변을 만들지 못했어요. 잠시 후 다시 시도해 주세요."


class ChatAnswer(BaseModel):
    text: str
    ok: bool = True
    source: str = "llm"
    context_summary: str = ""
    used_event_ids: list[str] = []


class ChatService:
    def __init__(
        self, relay: RelaySource, llm: LLMClient, video: Optional[VideoNotes] = None
    ) -> None:
        self.relay = relay
        self.llm = llm
        self.video = video

    def answer(
        self,
        game_id: str,
        question: str,
        video_t: Optional[int],
        profile: ExplainProfile,
    ) -> ChatAnswer:
        feed = self.relay.load(game_id)
        timeline = Timeline(feed.meta.relay_video_offset_sec)
        relay_t = timeline.to_relay(int(video_t)) if video_t is not None else None

        state = replay(feed.events, feed.meta.away_team, feed.meta.home_team, until_t=relay_t)
        recent_events = _recent_events(feed.events, relay_t)
        context_summary = f"{state.scoreboard_text()} · {state.runners_text()}"
        video_lines = []
        if self.video is not None and video_t is not None:
            video_lines = [
                f"({clock(e.t_start)}) {e.description or e.event_type}"
                for e in recent(self.video.events(feed.meta.id), video_t)[-3:]
            ]

        try:
            result = self.llm.complete(
                system=system_prompt("chat_system"),
                user=chat_user_prompt(
                    question, state, profile.level, recent_events, video_lines=video_lines
                ),
                max_tokens=500,
                effort="medium",
                thinking=True,  # 경기 상황 추론이 필요하다
            )
        except LLMError as exc:
            log.warning("챗봇 응답 실패 (%s): %s", game_id, exc)
            return ChatAnswer(
                text=FAILURE_MESSAGE,
                ok=False,
                source="none",
                context_summary=context_summary,
                used_event_ids=[e.id for e in recent_events],
            )

        return ChatAnswer(
            text=result.text,
            ok=True,
            source=result.source,
            context_summary=context_summary,
            used_event_ids=[e.id for e in recent_events],
        )


def _recent_events(
    events: list[RelayEvent], relay_t: Optional[int]
) -> list[RelayEvent]:
    """질문 시점 직전의 중계 몇 줄. 텍스트가 있는 것만 골라 맥락 밀도를 높인다."""
    if relay_t is None:
        candidates = events
    else:
        candidates = [e for e in events if e.t <= relay_t]
    meaningful = [e for e in candidates if e.text]
    return meaningful[-RECENT_EVENT_COUNT:]
