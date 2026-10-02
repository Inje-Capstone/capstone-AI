"""중계 화면 점수판 판독 → 상태 전이. 순수 로직(I/O 없음).

영상만으로 판정할 때 점수판이 상태의 뼈대다. VLM이 화면 구석의 점수판(이닝·볼카운트·
아웃·주자·점수)을 읽어 오면, 여기서 튀는 판독을 걸러 내고 "무엇이 바뀌었나"를 계산한다.
무슨 플레이였는지(보크인지 도루인지)는 video_judge가 장면 단서와 합쳐 정한다.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

Bases = tuple[bool, bool, bool]


@dataclass(frozen=True)
class Reading:
    """한 시점의 점수판. `t`는 영상 기준 초."""

    t: float
    inning: int
    half: str  # top | bot
    balls: int
    strikes: int
    outs: int
    bases: Bases
    away: int
    home: int

    def key(self) -> tuple:
        """시각을 뺀 상태 — 같은 화면이 반복되는지 비교할 때."""
        return (self.inning, self.half, self.balls, self.strikes, self.outs, self.bases,
                self.away, self.home)

    def runners(self) -> int:
        return sum(self.bases)


@dataclass(frozen=True)
class Transition:
    """두 안정 상태 사이의 변화. `t`는 새 상태가 처음 보인 시각."""

    t: float
    before: Reading
    after: Reading
    outs_made: int
    runs: int
    half_change: bool
    count_reset: bool  # 새 타자 (카운트가 0-0으로 돌아감)

    @property
    def runners_before(self) -> int:
        return self.before.runners()

    @property
    def runners_after(self) -> int:
        return self.after.runners()


def _plausible(r: Reading) -> bool:
    return (
        r.inning >= 1 and r.half in ("top", "bot")
        and 0 <= r.balls <= 3 and 0 <= r.strikes <= 2 and 0 <= r.outs <= 3
        and r.away >= 0 and r.home >= 0
    )


def stabilize(readings: Sequence[Reading], min_repeat: int = 2) -> list[Reading]:
    """튀는 판독을 걸러 낸 안정 상태 열. 각 상태의 첫 등장 판독을 돌려준다.

    VLM은 한두 프레임을 잘못 읽는다. 같은 상태가 `min_repeat`번 연속 보여야 인정한다.
    물리적으로 불가능한 값(4볼·3스트라이크 표시, 음수)은 먼저 버린다.
    점수는 줄어들 수 없다 — 점수가 줄어든 판독은 오독으로 본다.
    """
    clean = [r for r in sorted(readings, key=lambda r: r.t) if _plausible(r)]
    stable: list[Reading] = []
    i = 0
    while i < len(clean):
        j = i
        while j + 1 < len(clean) and clean[j + 1].key() == clean[i].key():
            j += 1
        run = j - i + 1
        candidate = clean[i]
        accepted = run >= min_repeat
        if accepted and stable:
            prev = stable[-1]
            if candidate.key() == prev.key():
                accepted = False
            elif candidate.away < prev.away or candidate.home < prev.home:
                accepted = False
            elif _half_order(candidate) < _half_order(prev):  # 이닝은 되돌아가지 않는다
                accepted = False
        if accepted:
            stable.append(candidate)
        i = j + 1
    return stable


def _half_order(r: Reading) -> tuple[int, int]:
    return (r.inning, 0 if r.half == "top" else 1)


def transitions(stable: Sequence[Reading]) -> list[Transition]:
    """안정 상태 사이의 전이. 볼·스트라이크 증가처럼 플레이가 아닌 변화도 포함한다."""
    out = []
    for before, after in zip(stable, stable[1:]):
        half_change = _half_order(after) != _half_order(before)
        if half_change:
            outs_made = max(0, 3 - before.outs)
        else:
            outs_made = max(0, after.outs - before.outs)
        runs = (after.away - before.away) + (after.home - before.home)
        count_reset = (after.balls, after.strikes) == (0, 0) and bool(
            (before.balls, before.strikes) != (0, 0) or half_change or outs_made
            or runs or after.bases != before.bases
        )
        out.append(Transition(
            t=after.t, before=before, after=after, outs_made=outs_made, runs=max(0, runs),
            half_change=half_change, count_reset=count_reset,
        ))
    return out


def is_count_only(tr: Transition) -> bool:
    """볼·스트라이크만 바뀐 전이 — 플레이가 아니라 투구 하나."""
    return (
        not tr.half_change and tr.outs_made == 0 and tr.runs == 0
        and tr.after.bases == tr.before.bases and not tr.count_reset
    )


def advanced_all_one_base(before: Bases, after: Bases) -> bool:
    """모든 주자가 정확히 한 베이스씩 (3루 주자는 홈으로). 보크의 점수판 모양."""
    if not any(before):
        return False
    expected = [False, False, False]
    for i, on in enumerate(before):
        if on and i < 2:
            expected[i + 1] = True
    return tuple(expected) == tuple(after)


def single_runner_advance(before: Bases, after: Bases) -> Optional[int]:
    """주자 한 명만 다음 베이스로 갔으면 도착 베이스(2·3·4=홈), 아니면 None."""
    for i in range(3):
        if not before[i]:
            continue
        moved = list(before)
        moved[i] = False
        if i < 2:
            if before[i + 1]:
                continue
            moved[i + 1] = True
        if tuple(moved) == tuple(after):
            return i + 2
    return None


def runner_removed(before: Bases, after: Bases) -> bool:
    """주자 한 명이 사라지고 나머지는 그대로 (도루 실패·견제사의 모양)."""
    return sum(before) - sum(after) == 1 and all(not a or b for a, b in zip(after, before))
