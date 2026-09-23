"""중계 이벤트 → 경기 상태 머신. 순수 로직(I/O 없음).

문자중계는 '결과'를 알려주지 주루 시뮬레이션을 요구하지 않는다. 그래서 이 머신은
주루를 추론하지 않고 이벤트가 실어 보낸 `bases_after`/`runs`/`outs_made`를 **적용하고
검증한다**. 스스로 계산하는 것은 볼카운트·이닝 전환·경기 종료뿐 — 중계에 안 나오는 것들이다.
"""

from collections.abc import Iterable, Sequence
from typing import Optional

from app.domain.models import (
    KIND_ATBAT,
    KIND_CALL,
    KIND_PITCH,
    KIND_RESULT,
    KIND_STEAL,
    GameState,
    RelayEvent,
)

KIND_SUB = "sub"

# 볼카운트에 영향을 주는 투구 결과
_BALL_RESULTS = {"ball"}
_STRIKE_RESULTS = {"strike", "swing_strike", "called_strike"}
_FOUL_RESULTS = {"foul"}

REGULATION_INNINGS = 9


class GameStateError(ValueError):
    """fixture나 중계 데이터가 물리적으로 불가능한 상태를 만들 때."""


def bases_from_list(nums: Sequence[int]) -> tuple[bool, bool, bool]:
    """`[1, 3]` → `(True, False, True)`"""
    slots = [False, False, False]
    for n in nums:
        if n not in (1, 2, 3):
            raise GameStateError(f"주자 위치는 1·2·3루만 가능하다: {n}")
        slots[n - 1] = True
    return (slots[0], slots[1], slots[2])


class GameSim:
    """이벤트를 순서대로 먹여서 상태를 굴린다."""

    def __init__(self, away_team: str, home_team: str) -> None:
        self.state = GameState(away_team=away_team, home_team=home_team)

    # ── 공개 API ────────────────────────────────────────────────────────
    def apply(self, event: RelayEvent) -> GameState:
        if self.state.game_over:
            # 경기가 끝난 뒤의 이벤트는 상태를 바꾸지 않는다(세리머니·인터뷰 등).
            return self.state

        self._sync_half(event)

        if event.kind == KIND_ATBAT:
            self._start_atbat(event)
        elif event.kind == KIND_PITCH:
            self._apply_pitch(event)
        elif event.kind == KIND_SUB:
            self._apply_sub(event)
        elif event.kind in (KIND_CALL, KIND_STEAL):
            self._apply_delta(event)
        elif event.kind == KIND_RESULT:
            self._apply_delta(event)
            self._reset_count()
            self._check_half_end()
        # 알 수 없는 kind는 무시한다 — 어댑터가 새 이벤트를 흘려도 엔진이 죽지 않게.

        self.state.last_event_id = event.id
        return self.state

    # ── 내부 ────────────────────────────────────────────────────────────
    def _sync_half(self, event: RelayEvent) -> None:
        """이벤트가 명시한 이닝/초말을 신뢰해 맞춘다.

        `_check_half_end`가 3아웃으로 이미 넘겨놨으면 보통 일치한다. 어긋나면
        중계 쪽이 옳다고 보고 따라가되, 카운트·주자는 초기화한다.
        """
        if event.inning == self.state.inning and event.half == self.state.half:
            return
        self.state.inning = event.inning
        self.state.half = event.half
        self.state.outs = 0
        self.state.bases = (False, False, False)
        self._reset_count()

    def _start_atbat(self, event: RelayEvent) -> None:
        self.state.batter = event.batter or self.state.batter
        self.state.pitcher = event.pitcher or self.state.pitcher
        self._reset_count()

    def _apply_sub(self, event: RelayEvent) -> None:
        pitcher = event.detail.get("pitcher") or event.pitcher
        if pitcher:
            self.state.pitcher = pitcher

    def _apply_pitch(self, event: RelayEvent) -> None:
        if event.batter:
            self.state.batter = event.batter
        if event.pitcher:
            self.state.pitcher = event.pitcher

        result = event.detail.get("result")
        if result in _BALL_RESULTS:
            self.state.balls += 1
        elif result in _STRIKE_RESULTS:
            self.state.strikes += 1
        elif result in _FOUL_RESULTS:
            # 파울은 투스트라이크 이후엔 카운트가 늘지 않는다.
            if self.state.strikes < 2:
                self.state.strikes += 1

    def _apply_delta(self, event: RelayEvent) -> None:
        """`outs_made` / `runs` / `bases_after`를 적용한다. call·steal·result 공통."""
        detail = event.detail
        if event.batter:
            self.state.batter = event.batter
        if event.pitcher:
            self.state.pitcher = event.pitcher

        outs_made = int(detail.get("outs_made", 0))
        runs = int(detail.get("runs", 0))
        if outs_made < 0 or runs < 0:
            raise GameStateError(f"{event.id}: 아웃/득점은 음수가 될 수 없다")

        if self.state.outs + outs_made > 3:
            raise GameStateError(
                f"{event.id}: 아웃 카운트가 3을 넘는다 "
                f"({self.state.outs} + {outs_made})"
            )
        self.state.outs += outs_made

        if runs:
            if self.state.half == "top":
                self.state.away_score += runs
            else:
                self.state.home_score += runs

        if "bases_after" in detail:
            self.state.bases = bases_from_list(detail["bases_after"])

        self._check_walk_off()

    def _reset_count(self) -> None:
        self.state.balls = 0
        self.state.strikes = 0

    def _check_walk_off(self) -> None:
        """9회말 이후 홈팀이 역전하면 그 순간 경기가 끝난다(끝내기)."""
        if (
            self.state.inning >= REGULATION_INNINGS
            and self.state.half == "bot"
            and self.state.home_score > self.state.away_score
        ):
            self.state.game_over = True

    def _check_half_end(self) -> None:
        if self.state.outs < 3:
            return

        finished_inning = self.state.inning
        finished_half = self.state.half

        # 9회초 종료 시점에 홈팀이 앞서면 9회말은 하지 않는다.
        if (
            finished_half == "top"
            and finished_inning >= REGULATION_INNINGS
            and self.state.home_score > self.state.away_score
        ):
            self.state.game_over = True
            return
        # 9회말 종료 시점에 동점이 아니면 경기 종료.
        if (
            finished_half == "bot"
            and finished_inning >= REGULATION_INNINGS
            and self.state.home_score != self.state.away_score
        ):
            self.state.game_over = True
            return

        if finished_half == "top":
            self.state.half = "bot"
        else:
            self.state.half = "top"
            self.state.inning += 1
        self.state.outs = 0
        self.state.bases = (False, False, False)
        self._reset_count()


def replay(
    events: Iterable[RelayEvent],
    away_team: str,
    home_team: str,
    until_t: Optional[int] = None,
) -> GameState:
    """`until_t`(포함)까지 이벤트를 재생한 상태를 돌려준다.

    시킹·배속이 그냥 되는 이유가 여기다 — 상태는 타임코드의 순수 함수다.
    """
    sim = GameSim(away_team, home_team)
    for event in events:
        if until_t is not None and event.t > until_t:
            break
        sim.apply(event)
    return sim.state


def states_by_event(
    events: Sequence[RelayEvent], away_team: str, home_team: str
) -> list[GameState]:
    """각 이벤트 **직후**의 상태 목록. 감지기가 이벤트별 문맥을 볼 때 쓴다."""
    sim = GameSim(away_team, home_team)
    out: list[GameState] = []
    for event in events:
        sim.apply(event)
        out.append(sim.state.model_copy(deep=True))
    return out
