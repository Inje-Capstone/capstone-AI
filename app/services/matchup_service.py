"""선수·매치업 분석 패널 (와이어프레임 S4 우측 하단).

기록은 **조회한 값**이고 AI가 하는 일은 그 위에 한 줄 해석을 얹는 것뿐이다.
해석 생성이 실패해도 기록은 그대로 보여준다. 기록 조회 자체가 실패하면 패널만
"기록을 불러올 수 없어요"로 내려가고 나머지 화면은 산다 (user-flow §5).
"""

import logging
from typing import Optional

from app.adapters.llm.base import LLMClient, LLMError
from app.adapters.relay.base import RelaySource
from app.adapters.stats.base import StatsSource, StatsSourceError
from app.domain.game_state import replay
from app.domain.models import Matchup
from app.domain.timeline import Timeline
from app.prompt_templates import matchup_user_prompt, system_prompt

log = logging.getLogger(__name__)

UNAVAILABLE_TEXT = "기록을 불러올 수 없어요"


class MatchupService:
    def __init__(self, relay: RelaySource, stats: StatsSource, llm: LLMClient) -> None:
        self.relay = relay
        self.stats = stats
        self.llm = llm

    def matchup(
        self, game_id: str, video_t: Optional[int], with_ai_note: bool = True
    ) -> Matchup:
        feed = self.relay.load(game_id)
        timeline = Timeline(feed.meta.relay_video_offset_sec)
        relay_t = timeline.to_relay(int(video_t)) if video_t is not None else None
        state = replay(feed.events, feed.meta.away_team, feed.meta.home_team, until_t=relay_t)

        if not state.batter:
            return _unavailable("아직 타석이 시작되지 않았어요")

        try:
            facts = self.stats.batter_facts(
                state.batter, state.pitcher, state.batting_team
            )
        except StatsSourceError as exc:
            log.warning("기록 조회 실패 (%s): %s", state.batter, exc)
            return _unavailable(UNAVAILABLE_TEXT)

        matchup = Matchup(
            batter=facts.batter,
            batter_line=f"시즌 타율 {facts.avg}" if facts.avg else "기록 없음",
            recent_form=facts.recent or "기록 없음",
            vs_pitcher=facts.vs_pitcher or "상대 전적 없음",
            team_form=facts.team_form or "기록 없음",
        )

        if with_ai_note:
            lines = facts.as_lines()
            if lines:
                try:
                    result = self.llm.complete(
                        system=system_prompt("matchup_system"),
                        user=matchup_user_prompt(state, lines),
                        max_tokens=150,
                        effort="low",
                        thinking=False,
                    )
                    matchup.ai_note = result.text.strip() or None
                except LLMError as exc:
                    # 해석은 '선택적'이다 — 없으면 기록만 보여준다.
                    log.info("한 줄 해석 생략 (%s): %s", state.batter, exc)

        return matchup


def _unavailable(reason: str) -> Matchup:
    return Matchup(
        batter="",
        batter_line="",
        recent_form="",
        vs_pitcher="",
        team_form="",
        available=False,
        unavailable_reason=reason,
    )
