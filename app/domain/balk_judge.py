"""보크 판정 — 해설 + 투수 동작, 점수판은 보일 때 검증. 순수 로직(I/O 없음).

사용자 결정(2026-10-26): 보크는 해설과 동작으로 판정한다. 점수판이 보이면 좋고, 보이면
'모든 주자가 한 베이스씩 진루'로 검증한다. 점수판이 없는 직캠·해설 영상도 판정 대상이다.

확신도
- 해설이 '보크'라고 했고 그 앞뒤에 동작 의심이 있다 → 0.8 (종류는 동작이 준다)
- 해설만 → 0.5 (종류는 해설 말에서 고른다)
- 동작만 → 0.35 — 화면에 띄우지 않는 후보. 와인드업(주자 없음)도 '정지 없음'처럼 보인다
- 점수판 검증: 주자가 한 베이스씩 → +0.15 / 주자가 그대로인 채 투구가 이어짐 → -0.2
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

from app.domain.pitch_motion import PitchMotion
from app.domain.scorebug import Reading, advanced_all_one_base

# Whisper는 '보크'를 '복구·보컬·버크'로 적기도 한다(실측 2026-10-26)
BALK_WORDS = re.compile(r"보크|복구|보컬|복크|버크|볼크")
SPEECH_TYPES = (
    ("double_motion", re.compile(r"이중|두 번|투 ?모션|멈췄다가|멈춤 동작이 있")),
    ("no_stop", re.compile(r"정지|멈추지|안 멈|멈춤 ?동작이 없|스톱이 없")),
    ("foot_off", re.compile(r"발을 빼|발이 빠|발 빼|다리를 빼|다리를 풀")),
    ("upper_body", re.compile(r"상체|어깨|고개")),
    ("unoccupied_base", re.compile(r"빈 베이스|비어 있")),
)
TYPE_LABELS = {
    "no_stop": "세트 정지 없이 투구 (6.02(a)(13))",
    "double_motion": "이중 동작 — 투구 동작 중단 (6.02(a)(1))",
    "kick_aborted": "다리를 들다 멈춤 — 투구 동작 중단 (6.02(a)(1))",
    "foot_off": "세트 후 자유발 옮김 — 투구 동작 중단 (6.02(a)(1))",
    "hands_separated": "세트 후 손 분리 (6.02(a)(10))",
    "pickoff_no_step": "발 안 내딛고 견제 (6.02(a)(3))",
    "upper_body": "투구 전 상체 움직임",
    "unoccupied_base": "빈 베이스로 송구 (6.02(a)(4))",
    "unknown": "종류 미상",
}
BEFORE, AFTER = 20.0, 10.0  # 해설 앞뒤로 동작을 찾는 범위(초)
MERGE = 20.0  # 이 안의 후보는 같은 보크(다시보기·해설 반복)


@dataclass
class BalkCall:
    t: float
    kind: str
    confidence: float
    evidence: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return TYPE_LABELS.get(self.kind, self.kind)


def speech_type(text: str) -> str:
    for kind, pattern in SPEECH_TYPES:
        if pattern.search(text):
            return kind
    return "unknown"


def _verify(call: BalkCall, readings: Sequence[Reading]) -> None:
    """점수판으로 검증 — 성립 조건(주자 있음)과 결과(모든 주자가 한 베이스씩)."""
    near = sorted((r for r in readings if call.t - 10 <= r.t <= call.t + 40), key=lambda r: r.t)
    before = [r for r in near if r.t <= call.t]
    if before and not any(before[-1].bases):
        call.confidence = max(0.0, call.confidence - 0.4)
        call.evidence.append("점수판: 주자 없음 — 보크가 성립하지 않는다(와인드업일 수 있음)")
        return
    for a, b in zip(near, near[1:]):
        if not any(a.bases) or (a.inning, a.half) != (b.inning, b.half):
            continue
        if advanced_all_one_base(a.bases, b.bases) and b.outs == a.outs:
            call.confidence = min(1.0, call.confidence + 0.15)
            call.evidence.append(f"점수판: 주자 한 베이스씩 진루 ({a.t:.0f}s→{b.t:.0f}s)")
            return
    if len(near) >= 2 and any(near[0].bases) and all(r.bases == near[0].bases for r in near):
        call.confidence = max(0.0, call.confidence - 0.2)
        call.evidence.append("점수판: 주자 진루 없음")


def judge_balk(
    motions: Sequence[PitchMotion],
    speech: Sequence[tuple[float, str]],
    readings: Sequence[Reading] = (),
    speech_span: float = 10.0,
) -> list[BalkCall]:
    """동작 의심·해설·점수판 → 보크 판정 목록(시간순). `speech`는 (조각 시작 초, 받아쓴 말)."""
    suspects = [m for m in motions if m.suspect]
    calls: list[BalkCall] = []
    used: set[int] = set()
    for t, text in sorted(speech):
        if not BALK_WORDS.search(text):
            continue
        near = [
            k
            for k, m in enumerate(suspects)
            if t - BEFORE <= (m.t_move or m.t_set) <= t + speech_span + AFTER
        ]
        said = speech_type(text)
        quote = f"해설({t:.0f}s): '{text[:60]}'"
        if near:
            m = suspects[near[0]]
            used.update(near)
            calls.append(
                BalkCall(m.t_move or m.t_set, m.suspect or said, 0.8, [quote] + m.evidence)
            )
        else:
            calls.append(BalkCall(t, said, 0.5, [quote]))
    for k, m in enumerate(suspects):
        if k not in used:
            calls.append(BalkCall(m.t_move or m.t_set, m.suspect or "unknown", 0.35, m.evidence))
    merged: list[BalkCall] = []
    for c in sorted(calls, key=lambda c: (-c.confidence, c.t)):
        if all(abs(c.t - o.t) > MERGE for o in merged):
            merged.append(c)
    for c in merged:
        _verify(c, readings)
    return sorted(merged, key=lambda c: c.t)


def best(calls: Sequence[BalkCall], threshold: float = 0.5) -> Optional[BalkCall]:
    shown = [c for c in calls if c.confidence >= threshold]
    return max(shown, key=lambda c: c.confidence) if shown else None
