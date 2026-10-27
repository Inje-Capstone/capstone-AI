"""보크 판정 — 해설 + 투수 동작, 점수판은 보일 때 검증. 순수 로직(I/O 없음).

사용자 결정(2026-10-26): 보크는 해설과 동작으로 판정한다. 점수판이 보이면 좋고, 보이면
'모든 주자가 한 베이스씩 진루'로 검증한다. 점수판이 없는 직캠·해설 영상도 판정 대상이다.

확신도
- 해설이 '보크'라고 했고(또는 1·3루 견제 시늉을 설명했고) 그 앞뒤에 동작 의심이 있다 → 0.8
  (종류는 해설이 동작을 설명했으면 해설, 아니면 동작)
- 해설만 → 0.5 (종류는 해설 말에서 고른다)

종류 실측(2026-10-27 보크 클립 18개): 해설이 종류를 말한 19건 중 말 분류가 맞은 것 약 15건,
같은 때 자세가 고른 종류는 해설과 9건 중 2건만 같았다 — 자세는 보크 뒤 손을 푸는 동작을
'손 분리'로 잡거나, 뒤로 빼는 발(카메라 쪽 깊이 이동)을 못 잰다. 그래서 해설이 우선이다.
- 동작만 → 0.35 — 화면에 띄우지 않는 후보. 와인드업(주자 없음)도 '정지 없음'처럼 보인다
- 점수판 검증: 주자가 한 베이스씩 → +0.15 / 주자가 그대로인 채 투구가 이어짐 → -0.2
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

from app.domain.pitch_motion import PitchMotion
from app.domain.scorebug import Reading, advanced_all_one_base

# Whisper는 '보크'를 '복구·보컬·버크·보쿠'로 적기도 한다(실측 2026-10-26)
BALK_WORDS = re.compile(r"보크|복구|보컬|복크|버크|볼크|보쿠")
# '보크'라는 말 없이 반칙 동작을 설명하는 해설 — 1·3루 견제 시늉(6.02(a)(2), 2루 시늉은 합법)
FAKE_THROW = re.compile(r"(1루|3루|일루|삼루|견제).{0,25}(던지지 않|안 던지|던지는 척|시늉)")
SPEECH_TYPES = (
    # '발이 빠지기 전에 상체가 먼저 움직였다' — 발 이야기가 섞여도 반칙은 상체(실측)
    ("upper_body", re.compile(r"상체[가를]? 먼저|빠지기 전에 상체")),
    # '두 번째 장면'은 순서일 뿐 이중 동작이 아니다(실측)
    ("double_motion", re.compile(r"이중|두 번(?!째)|투 ?모션|멈췄다가|멈춤 동작이 있")),
    ("no_stop", re.compile(r"정지|멈추지|안 멈|멈춤 ?동작이 없|스톱이 없")),
    ("foot_off", re.compile(r"발을 빼|발이 빠|발 빼|다리를 빼|다리를 풀")),
    ("upper_body", re.compile(r"상체|어깨|고개")),
    ("unoccupied_base", re.compile(r"빈 베이스|비어 있")),
    ("fake_throw", FAKE_THROW),
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
    "fake_throw": "1·3루 견제 시늉만 (6.02(a)(2))",
    "unknown": "종류 미상",
}
# 입문자에게 '어떤 동작이 반칙이었나'를 한 줄로 — 카드 근거에 그대로 들어간다
TYPE_PLAIN = {
    "no_stop": "주자가 있으면 손을 모으고 완전히 멈춘 뒤 던져야 하는데, 멈추지 않고 바로 던졌다",
    "double_motion": "세트 자세로 들어가다 한 번 멈췄다가 다시 움직였다 — 주자를 속이는 동작이다",
    "kick_aborted": "다리를 들어 던지려다 중간에 멈췄다 — 던지는 척으로 본다",
    "foot_off": "손을 모은 뒤 멈추려면 투수판을 밟은 발(축발)부터 뒤로 빼야 하는데 다른 발을 뺐다",
    "hands_separated": "손을 모아 멈춘 뒤 던지지도 견제하지도 않고 손을 풀었다",
    "pickoff_no_step": "견제하면서 공을 던질 베이스 쪽으로 발을 내딛지 않았다",
    "upper_body": "던지기 전에 상체·어깨를 움직였다가 멈췄다 — 던지는 척으로 본다",
    "unoccupied_base": "주자가 없는 베이스로 공을 던졌다",
    "fake_throw": "1루나 3루로 던지는 척만 하고 공을 던지지 않았다",
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
        if not (BALK_WORDS.search(text) or FAKE_THROW.search(text)):
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
            kind = said if said != "unknown" else (m.suspect or said)  # 해설이 우선(모듈 설명)
            calls.append(BalkCall(m.t_move or m.t_set, kind, 0.8, [quote] + m.evidence))
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
