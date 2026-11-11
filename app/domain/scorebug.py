"""중계 화면 점수판 판독 → 상태 전이. 순수 로직(I/O 없음).

영상만으로 판정할 때 점수판이 상태의 뼈대다. VLM이 화면 구석의 점수판(이닝·볼카운트·
아웃·주자·점수)을 읽어 오면, 여기서 튀는 판독을 걸러 내고 "무엇이 바뀌었나"를 계산한다.
무슨 플레이였는지(보크인지 도루인지)는 video_judge가 장면 단서와 합쳐 정한다.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
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


def possible_step(a: Reading, b: Reading) -> bool:
    """점수판이 a에서 b로 한 번에 바뀔 수 있나 — 주자·아웃·득점 수지가 맞나.

    같은 반 이닝에서 아웃은 줄지 않고, 주자는 아웃·득점 없이 사라지지 않으며, 늘어도 타자
    한 명까지다. 볼카운트가 줄었으면(새 타자) 타석 하나가 끝난 것 — 타자 한 명이 주자·아웃·
    득점 중 하나가 됐어야 한다. 반 이닝이 바뀌면 순서만 본다.
    """
    if b.away < a.away or b.home < a.home:
        return False
    if _half_order(b) != _half_order(a):
        return _half_order(b) > _half_order(a)
    if b.outs < a.outs:
        return False
    after = b.runners() + (b.outs - a.outs) + (b.away - a.away) + (b.home - a.home)
    if b.balls < a.balls or b.strikes < a.strikes:
        return after == a.runners() + 1
    return a.runners() <= after <= a.runners() + 1


def stabilize(readings: Sequence[Reading], min_repeat: int = 2) -> list[Reading]:
    """튀는 판독을 걸러 낸 안정 상태 열. 각 상태의 첫 등장 판독을 돌려준다.

    VLM은 한두 프레임을 잘못 읽는다. 같은 상태가 `min_repeat`번 연속 보여야 인정한다.
    물리적으로 불가능한 값(4볼·3스트라이크 표시, 음수)은 먼저 버린다.
    점수는 줄어들 수 없다 — 점수가 줄어든 판독은 오독으로 본다.
    같은 반 이닝에서 아웃도 줄 수 없다 — 받아들이면 다음 판독에서 아웃을 또 세어 경기 상태가
    3아웃을 넘는다(시뮬레이션 실측).

    이 세 규칙은 앞에서부터 하나씩 적용하지 않는다 — 회차 오독 한 장('4.강백호'의 타순 4를
    4회로 읽음)을 먼저 받으면 그 뒤 진짜 2·3회 판독이 전부 '되돌아감'으로 버려졌다(2026-11-07
    하이라이트 실측: 36칸 중 31칸 놓침). 서로 모순 없는 판독 덩어리 중 판독이 가장 많은 사슬을
    고른다(_consistent_chain).

    `min_repeat` 1(점수판 전용 판독 — 상태마다 한 장)은 반복으로 거를 수 없어, 앞뒤 판독과
    수지가 안 맞는 한 장을 버린다: 앞→뒤는 가능한데 앞→이것 또는 이것→뒤가 불가능할 때.
    """
    clean = [r for r in sorted(readings, key=lambda r: r.t) if _plausible(r)]
    stable = _stabilize_clean(clean, min_repeat)
    fixed = _fix_count_as_score(clean, stable)
    if fixed is not None:
        clean, stable = fixed, _stabilize_clean(fixed, min_repeat)
    repaired = _repair_innings(clean, stable)
    return stable if repaired is None else _stabilize_clean(repaired, min_repeat)


def _fix_count_as_score(clean: list[Reading], stable: list[Reading]) -> Optional[list[Reading]]:
    """볼카운트를 점수로도 읽은 판독의 점수를 앞뒤 안정 상태의 점수로 되돌린다. 고칠 게 없으면 None.

    가운데 'X-Y'로 볼카운트를 그리는 판(MBC 포스트시즌 상자)에서 VLM이 그 숫자를 점수에도 넣었다
    (한국시리즈 5차전 '0-1 카운트 · 0:1', '3-0 카운트 · 3:0'). 점수가 카운트와 같고 앞뒤 안정 상태의
    점수가 서로 같은데 이 판독만 다르면 — 점수는 줄지 않으니 이 판독의 점수가 틀렸다. 주자·아웃
    (화소)과 회차는 그대로 쓴다. 버리면 마지막 아웃 앞 한 장 같은 상태를 잃는다(2사 삼진을 놓침).
    사슬에 없는 상태를 더할 때만 고친다. 날 판독 이웃이 아니라 안정 상태와 견준다 —
    이웃도 오독이면(투수 모음 판독이 0:0과 진짜 점수를
    오감) 맞는 판독을 틀린 점수로 끌어당겼다. 0-0 카운트와 0:0 점수는 경기 초반에 늘 같아 근거가
    못 된다 — PO 2차전 '2사 1·3루 0:0'을 1:0으로 고쳐 희생플라이를 놓쳤다.
    """
    out, changed = list(clean), False
    for i, r in enumerate(clean):
        score = (r.away, r.home)
        if score != (r.balls, r.strikes) or score == (0, 0):
            continue
        prev = [s for s in stable if s.t < r.t]
        nxt = [s for s in stable if s.t > r.t]
        if not prev or not nxt:
            continue
        a, c = prev[-1], nxt[0]
        if (a.away, a.home) == (c.away, c.home) != score:
            fixed = replace(r, away=a.away, home=a.home)
            # 고친 판독이 앞뒤 안정 상태와 같으면 그 상태가 처음 보인 시각만 당긴다 —
            # 틀린 점수를 읽은
            # 판독으로 시각을 옮기지 않는다(PO 4차전: 2사 상태가 0.2초 당겨져 앞 투구 칸으로 감)
            if fixed.key() not in (a.key(), c.key()):
                out[i], changed = fixed, True
    return out if changed else None


def _stabilize_clean(clean: list[Reading], min_repeat: int) -> list[Reading]:
    runs: list[tuple[Reading, int]] = []
    for r in clean:
        if runs and runs[-1][0].key() == r.key():
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
        else:
            runs.append((r, 1))
    if min_repeat <= 1:
        runs = _drop_spikes(runs)
    stable: list[Reading] = []
    for candidate, _run in _consistent_chain([rn for rn in runs if rn[1] >= min_repeat]):
        if not stable or candidate.key() != stable[-1].key():
            stable.append(candidate)
    return stable


def _repair_innings(clean: list[Reading], stable: list[Reading]) -> Optional[list[Reading]]:
    """회차만 거꾸로 간 판독을 앞뒤 안정 상태 사이에서 고친다. 고칠 게 없으면 None.

    VLM이 회차 숫자를 잘못 읽는다 — KS3 하이라이트 7회초~8회말 판독 35장을 '1회'로 읽었다(점수판에는
    7▲·8▲·8▼). 회차는 거꾸로 가지 않고 점수도 줄지 않으니, 앞 안정 상태보다 회차가 앞인데 점수는
    앞뒤 사이인 판독은 회차만 틀렸다. 초·말(화소)과 아웃 순서를 따라 회차를 다시 매긴다 — 같은
    초·말인데 아웃이 줄면 다음 회차(다른 반 이닝을 건너뜀), 초 → 말은 같은 회차, 말 → 초는
    다음 회차.
    다시 매긴 회차가 뒤 안정 상태보다 앞일 때만 받는다. 안정 상태(가장 긴 사슬) 사이만 보므로 한 장
    회차 오독이 진짜 판독을 끌어올리지 않는다('4.강백호'). 같은 초·말에 아웃이 같거나 많으면 같은
    회차로 본다 — 회차를 통째로 건너뛰었어도 판독만으로는 가를 수 없다(독립 리뷰, 기록만).
    """
    if len(stable) < 2:
        return None
    out = list(clean)
    changed = False
    for a, b in zip(stable, stable[1:]):
        cur = a
        fixed: dict[int, Reading] = {}
        for i, r in enumerate(out):
            if not a.t < r.t < b.t:
                continue
            score_ok = (a.away <= r.away <= b.away) and (a.home <= r.home <= b.home)
            if _half_order(cur) <= _half_order(r) <= _half_order(b) or not score_ok:
                if _half_order(cur) <= _half_order(r) <= _half_order(b):
                    cur = r
                continue
            if _half_order(r) >= _half_order(cur):
                continue
            if r.half == cur.half:
                inning = cur.inning if r.outs >= cur.outs else cur.inning + 1
                # 아웃 오독 한 장이 회차를 올리지 않게 — 다음 회차로 올릴 땐 같은 판독이 두 번 이상
                # 이어 읽혔어야 한다(독립 리뷰)
                twice = any(0 <= k < len(out) and out[k].key() == r.key() for k in (i - 1, i + 1))
                if inning > cur.inning and not twice:
                    continue
            else:
                inning = cur.inning if cur.half == "top" else cur.inning + 1
            new = replace(r, inning=inning)
            if _half_order(new) > _half_order(b):
                fixed = {}
                break
            fixed[i] = new
            cur = new
        for i, r in fixed.items():
            out[i] = r
            changed = True
    return out if changed else None


def _can_follow(a: Reading, b: Reading) -> bool:
    """a 다음에 b가 올 수 있나 — 점수는 줄지 않고, 이닝은 되돌아가지 않고, 같은 반 이닝에서
    아웃도 줄지 않는다."""
    if b.away < a.away or b.home < a.home:
        return False
    if _half_order(b) < _half_order(a):
        return False
    return not (_half_order(b) == _half_order(a) and b.outs < a.outs)


def _consistent_chain(runs: list[tuple[Reading, int]]) -> list[tuple[Reading, int]]:
    """시간순 판독 덩어리 중 서로 모순 없는 사슬 가운데 판독 수가 가장 많은 것."""
    best = [n for _r, n in runs]
    back = [-1] * len(runs)
    for j, (rj, nj) in enumerate(runs):
        for i in range(j):
            if best[i] + nj > best[j] and _can_follow(runs[i][0], rj):
                best[j], back[j] = best[i] + nj, i
    if not runs:
        return []
    j = max(range(len(runs)), key=lambda k: (best[k], k))
    chain = []
    while j >= 0:
        chain.append(runs[j])
        j = back[j]
    return chain[::-1]


def _drop_spikes(runs: list[tuple[Reading, int]]) -> list[tuple[Reading, int]]:
    out: list[tuple[Reading, int]] = []
    for i, (r, n) in enumerate(runs):
        if out and i + 1 < len(runs):
            a, c = out[-1][0], runs[i + 1][0]
            # 두 번 이상 읽혔고 앞뒤 모두 타석 여러 개를 건너뛴 것으로 설명되면 남긴다(하이라이트)
            if n >= 2 and _skippable(a, r) and _skippable(r, c):
                out.append((r, n))
                continue
            spike = possible_step(a, c) and not (possible_step(a, r) and possible_step(r, c))
            if spike and not _next_is_spike(runs, i, a) and not (
                    n == 1 and (_starts_next_half(a, r, c) or _ends_half(a, r, c))):
                continue
        out.append((r, n))
    return out


def _next_is_spike(runs: list[tuple[Reading, int]], i: int, a: Reading) -> bool:
    """여러 번 읽힌 상태 `runs[i]`와 안 맞는 다음 판독이 한 번만 읽힌 것이고, 그 다음 상태는
    이 상태에서 한 번에 올 수 있으면 — 튄 것은 다음 판독이다(KS1: 5번 읽힌 '1사 2루 2점'이
    한 번 읽힌 '2사 주자 없음'(태그업 주자가 3루에 그려지기 전) 때문에 버려져 2타점 2루타를 놓침).
    """
    r, n = runs[i]
    if n < 2 or i + 2 >= len(runs) or runs[i + 1][1] != 1:
        return False
    if not possible_step(a, r):
        return False
    if possible_step(r, runs[i + 2][0]):
        return True
    # 그 다음도 한 번만 읽힌 오독이면(점수 오독 한 장 — KS3) 두 번 이상 읽힌 다음 상태와 맞춘다
    later = [rn for rn in runs[i + 3:i + 5] if rn[1] >= 2]
    # 그 상태가 다른 반 이닝이면 '가능'은 늘 참이라 근거가 못 된다(독립 리뷰)
    return (runs[i + 2][1] == 1 and bool(later) and _half_order(later[0][0]) == _half_order(r)
            and possible_step(r, later[0][0]))


def _starts_next_half(a: Reading, r: Reading, c: Reading) -> bool:
    """한 번 읽힌 `r`이 a 뒤 새 반 이닝의 앞 상태이고 c까지 그 반 이닝에서 이어지면 튄 판독이
    아니다.

    a → c가 반 이닝을 넘으면 늘 '가능'이라, 새 반 이닝 첫 판독(KS3 '5회초 무사')이 c('2사')와
    한 번에 안 이어진다는 이유로 버려졌다 — 하이라이트는 타석을 건너뛴다(삼진 6K를 놓침).
    """
    return _half_order(a) < _half_order(r) == _half_order(c) and _skippable(r, c)


def _ends_half(a: Reading, r: Reading, c: Reading) -> bool:
    """한 번 읽힌 `r`이 a와 같은 반 이닝의 마지막 상태이고 c부터 다음 반 이닝이면 튄 판독이 아니다.

    하이라이트는 3아웃 직전에 상태를 한 번만 비춘다. 한국시리즈 5차전 1회초 '무사 2-2 → 2사 2-2'
    (삼진 + 못 본 아웃, 다음 타자도 2-2)는 카운트가 그대로라 한 번에 바뀔 수 없어 버려졌고,
    무사에서 바로 2회말로 넘어가 삼진을 놓쳤다. 타석 여럿을 건너뛴 것으로 설명되면 남긴다.
    아웃만 늘었을 때만 — 주자·점수까지 바뀐 한 장은 못 본 플레이를 판정으로 만든다(투수 모음
    '1사 → 2사 2루' 한 장이 공 없는 칸에 '아웃'을 냈다). 3아웃 직전(2사)만 — 아웃 점 오독 한 장이
    없는 아웃을 만들지 않게(독립 리뷰). 바로 다음 반 이닝일 필요는 없다(KS5 1회초 → 2회말).
    """
    return (_half_order(a) == _half_order(r) < _half_order(c)
            and r.outs == 2 > a.outs and r.bases == a.bases
            and (r.away, r.home) == (a.away, a.home)
            and _skippable(r, c))


def _skippable(a: Reading, b: Reading) -> bool:
    """a 뒤에 타석 여러 개를 건너뛰고 b가 보일 수 있나.

    `possible_step`(한 번에 바뀔 수 있나)과 달리 타자가 여럿 지나갈 수 있다 — 하이라이트는
    다음 장면으로 건너뛴다. 반 이닝이 바뀌면 순서만, 같은 반 이닝이면 점수·아웃이 줄지 않고
    주자는 아웃·득점 없이 사라지지 않으며, 카운트가 새로 시작했으면 한 명 이상 늘었어야 한다.
    3아웃 판독은 남기지 않는다 — 다음 반 이닝 전이가 아웃 없는 '출루'가 된다(독립 리뷰).
    """
    if b.away < a.away or b.home < a.home or b.outs >= 3:
        return False
    if _half_order(a) != _half_order(b):
        return _half_order(b) > _half_order(a)
    if b.outs < a.outs:
        return False
    after = b.runners() + (b.outs - a.outs) + (b.away - a.away) + (b.home - a.home)
    if b.balls < a.balls or b.strikes < a.strikes:  # 새 타자 — 타석이 하나 이상 끝났다
        return after >= a.runners() + 1
    return after >= a.runners()


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
        # 한 타석 안에서 볼·스트라이크는 줄지 않는다 — 줄었고 아웃·주자·득점도 바뀌었으면 새 타자다.
        # 하이라이트는 새 타자의 초구 뒤에야 점수판이 다시 보여 0-0이 안 읽힌다(2026-11-07 실측:
        # 안타가 '주자 진루'로). 카운트만 줄었으면 오독일 수 있어 새 타자로 보지 않는다
        dropped = after.balls < before.balls or after.strikes < before.strikes
        moved = half_change or outs_made or runs or after.bases != before.bases
        # 주자 수 + 득점이 늘었으면 타자가 나갔다 — 한 타석 안의 진루(도루·폭투·보크)로는 늘 수
        # 없다. 다음 타자의 첫 판독이 이미 2-1이면 카운트로는 새 타자를 모른다(2타점 2루타 실측)
        new_runner = sum(after.bases) + max(0, runs) > sum(before.bases)
        count_reset = count_reset or bool(dropped and moved) or (new_runner and not half_change)
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


def forced_to_first(before: Bases, after: Bases, runs: int) -> bool:
    """볼넷·몸에 맞는 공의 모양 — 타자가 1루, 밀려난 주자만 한 베이스씩(5.05(b)(1)).

    1루부터 이어진 주자만 움직이고, 득점은 만루에서 하나뿐이다. 2·3루 주자가 1루 빈 채로
    들어왔으면 볼넷이 아니다(KS1: 2·3루 2타점 적시타가 3볼에서 안 맞은 것으로 보여 '볼넷').
    """
    first, second, third = before
    forced = (True, first or second, (first and second) or third)
    return tuple(after) == forced and runs == (1 if all(before) else 0)


def runner_removed(before: Bases, after: Bases) -> bool:
    """주자 한 명이 사라지고 나머지는 그대로 (도루 실패·견제사의 모양)."""
    return sum(before) - sum(after) == 1 and all(not a or b for a, b in zip(after, before))
