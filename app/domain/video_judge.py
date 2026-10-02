"""영상만으로 플레이 판정. 순수 로직(I/O 없음).

"VLM이 보고, 규칙 엔진이 판정한다." 입력은 두 가지다.
- 점수판 전이(scorebug.transitions): 아웃·주자·점수·카운트가 어떻게 바뀌었나 — 상태의 뼈대
- 장면 단서(Cue): VLM이 본 동작(슬라이딩·포수 놓침·담장 넘김…)과 들은 해설 키워드("보크")

전이의 모양으로 후보를 좁히고, 단서로 종류를 가른다. 단서가 없으면 확신도를 낮추고,
모양만으로 정해지는 것(병살·볼넷)은 단서 없이도 판정한다. 결과는 문자중계와 같은
RelayEvent 형식이라 GameSim·detectors·카드가 그대로 받는다.

판정 규칙표는 docs/exec-plans의 07-video-judge 계획과 1:1이다.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

from app.domain.models import RelayEvent
from app.domain.scorebug import (
    Reading,
    Transition,
    advanced_all_one_base,
    is_count_only,
    runner_removed,
    single_runner_advance,
    stabilize,
    transitions,
)

# 점수판은 플레이보다 늦게 바뀐다. 전이 시각 기준 앞 BEFORE초 ~ 뒤 AFTER초의 단서를 본다.
CUE_BEFORE = 20.0
CUE_AFTER = 6.0
MIN_CONFIDENCE = 0.5
SUB_LOOKAHEAD = 10.0

# 단서 종류 (VLM 프롬프트 v2와 같은 어휘)
CUE_KINDS = (
    "pitch", "swing_miss", "contact", "slide", "pickoff_throw", "pitcher_stops",
    "catcher_miss", "infield_popup", "deep_fly", "ball_over_fence", "hit_by_pitch",
    "pitching_change", "mound_visit", "replay_review", "celebration", "crowd_cheer",
    "speech",  # 해설 키워드 — text에 "보크", "도루" 등
)


@dataclass(frozen=True)
class Cue:
    t: float
    kind: str
    text: str = ""


@dataclass
class Judgment:
    t: float
    kind: str  # result | call | steal | sub
    code: str  # walk, balk, steal …
    confidence: float
    evidence: list[str] = field(default_factory=list)
    transition: Optional[Transition] = None
    detail: dict = field(default_factory=dict)


def _near(cues: Sequence[Cue], t: float) -> list[Cue]:
    return [c for c in cues if t - CUE_BEFORE <= c.t <= t + CUE_AFTER]


def _has(cues: Sequence[Cue], kind: str, words: Sequence[str] = ()) -> Optional[Cue]:
    for c in cues:
        if c.kind == kind:
            return c
        if words and c.kind == "speech" and any(w in c.text for w in words):
            return c
    return None


def _speech(cues: Sequence[Cue], *words: str) -> Optional[Cue]:
    for c in cues:
        if c.kind == "speech" and any(w in c.text for w in words):
            return c
    return None


def _ev(c: Cue) -> str:
    return f"해설 '{c.text}' ({c.t:.0f}s)" if c.kind == "speech" else f"장면 {c.kind} ({c.t:.0f}s)"


def _confidence(base: float, supports: Sequence[Optional[Cue]]) -> tuple[float, list[str]]:
    found = [c for c in supports if c is not None]
    return min(1.0, base + 0.25 * len(found)), [_ev(c) for c in found]


def judge_transition(tr: Transition, cues: Sequence[Cue]) -> Optional[Judgment]:
    """점수판 전이 하나를 판정한다. 플레이가 아니면(투구 하나) None."""
    if is_count_only(tr):
        return None
    near = _near(cues, tr.t)
    b, a = tr.before.bases, tr.after.bases
    def bits(bases: tuple) -> str:
        return "".join("1" if x else "0" for x in bases)

    shape = f"점수판 아웃+{tr.outs_made} 득점+{tr.runs} 주자 {bits(b)}→{bits(a)}"

    def make(kind: str, code: str, prior: float, supports=(), **detail) -> Judgment:
        conf, ev = _confidence(prior, supports)
        return Judgment(tr.t, kind, code, conf, [shape] + ev, tr, detail)

    # 같은 타석 안의 주자 움직임 (카운트 유지 — 0-0 카운트면 리셋처럼 보여도 단서로 가른다)
    mid_pa = not tr.half_change and (
        not tr.count_reset or (tr.before.balls, tr.before.strikes) == (0, 0)
    ) and not _has(near, "contact")
    if mid_pa and tr.outs_made == 0 and b != a:
        balk_cue = _speech(near, "보크") or _has(near, "pitcher_stops")
        if advanced_all_one_base(b, a) and balk_cue:
            return make("call", "balk", 0.5, [balk_cue, _speech(near, "보크")])
        steal_cue = _has(near, "slide") or _speech(near, "도루")
        if single_runner_advance(b, a) and steal_cue and not _speech(near, "폭투", "포일"):
            return make("steal", "steal", 0.5, [steal_cue, _speech(near, "도루")],
                        base=single_runner_advance(b, a))
        wp = _speech(near, "폭투")
        if wp:
            return make("call", "wild_pitch", 0.5, [wp])
        pb = _speech(near, "포일")
        if pb:
            return make("call", "passed_ball", 0.5, [pb])
        if advanced_all_one_base(b, a):  # 모양은 보크인데 단서가 없다
            return make("call", "balk", 0.3)
        if not tr.count_reset:
            return make("call", "runner", 0.4)
    if mid_pa and tr.outs_made == 1 and runner_removed(b, a):
        caught = _has(near, "slide") or _speech(near, "도루", "견제")
        if caught:
            return make("steal", "caught_stealing", 0.5, [caught], caught=True)

    # 타석 결과
    if tr.outs_made == 2:
        k = _speech(near, "삼진") or _has(near, "swing_miss")
        if tr.before.strikes == 2 and k and not _speech(near, "병살"):
            # 삼진 + 도루 실패(스트라이크 아웃 스로 아웃) — 병살타 카드를 띄우면 틀린 설명이 된다
            return make("result", "strikeout", 0.5, [k])
        return make("result", "double_play", 0.8, [_speech(near, "병살")])
    if tr.outs_made == 3:  # 판독 누락으로도 생긴다 — 해설이 있을 때만 삼중살
        tp = _speech(near, "삼중살")
        if tp:
            return make("result", "triple_play", 0.6, [tp])
    hr = _has(near, "ball_over_fence") or _speech(near, "홈런")
    if tr.runs >= 1 and tr.runs == tr.runners_before + 1 and not any(a) and hr:
        return make("result", "homerun", 0.5, [hr, _speech(near, "홈런")])
    if tr.outs_made == 1 and tr.runs >= 1:
        sf = _has(near, "deep_fly") or _speech(near, "희생플라이", "희생 플라이")
        if sf:
            return make("result", "sac_fly", 0.5, [sf])
    if tr.outs_made == 1 and b == a and not tr.half_change:
        iff = _speech(near, "인필드플라이", "인필드 플라이") or _has(near, "infield_popup")
        if iff and any(b[:2]):
            return make("call", "infield_fly", 0.4, [iff, _speech(near, "인필드")])
    if tr.outs_made == 0 and tr.count_reset and a[0]:
        hbp = _has(near, "hit_by_pitch") or _speech(near, "몸에 맞", "사구", "데드볼")
        if hbp:
            return make("result", "hbp", 0.5, [hbp])
        dts = _has(near, "catcher_miss") or _speech(near, "낫아웃", "낫 아웃")
        if dts:  # 2스트라이크가 점수판에 보였으면 확신도를 더 준다(놓친 투구 판독 대비)
            prior = 0.5 if tr.before.strikes == 2 else 0.3
            return make("call", "dropped_third_strike", prior,
                        [dts, _speech(near, "낫아웃", "낫 아웃")])
        walk_speech = _speech(near, "볼넷", "포볼", "고의")
        if tr.before.balls == 3 and not _has(near, "contact"):
            return make("result", "walk", 0.6, [walk_speech])
        if walk_speech:
            return make("result", "walk", 0.5, [walk_speech])
    if tr.outs_made == 1 and tr.before.strikes == 2:
        dts = _has(near, "catcher_miss") or _speech(near, "낫아웃", "낫 아웃")
        if dts:  # 낫아웃 상황이었지만 1루에서 아웃
            return make("call", "dropped_third_strike", 0.5, [dts], out=True)
        k = _has(near, "swing_miss") or _speech(near, "삼진")
        if k:
            return make("result", "strikeout", 0.5, [k])
    # 나머지 타석 결과는 종류를 모른다 — 상태는 반영하되 카드는 띄우지 않는다
    if tr.count_reset or tr.half_change or tr.outs_made or tr.runs:
        return make("result", "out" if tr.outs_made else "reach", 0.5)
    return None


def judge(readings: Sequence[Reading], cues: Sequence[Cue]) -> list[Judgment]:
    """점수판 판독 열 + 단서 → 판정 목록 (시간순). 투수 교체는 단서만으로 판정한다."""
    stable = stabilize(readings)
    out = []
    for tr in transitions(stable):
        j = judge_transition(tr, cues)
        if j is not None:
            out.append(j)
    seen_change: list[float] = []
    for c in sorted(cues, key=lambda c: c.t):
        is_change = c.kind == "pitching_change" or (
            c.kind == "speech" and "투수" in c.text and "교체" in c.text
        )
        if is_change and all(abs(c.t - s) > 60 for s in seen_change):
            seen_change.append(c.t)
            # 교체는 보통 이닝 사이에 한다 — 점수판이 다음 이닝으로 넘어가는 몇 초를 기다린다
            at = [r for r in stable if r.t <= c.t + SUB_LOOKAHEAD]
            where = at[-1] if at else (stable[0] if stable else None)
            half = {"inning": where.inning, "half": where.half} if where else {}
            out.append(Judgment(c.t, "sub", "pitcher", 0.6, [_ev(c)], None, half))
    return sorted(out, key=lambda j: j.t)


# ── 분석 스냅샷(analyze_video.py) → 판독·단서 ───────────────────────────
def from_snapshot(snap: dict) -> tuple[list[Reading], list[Cue]]:
    """`data/video/{id}.json` → (점수판 판독, 장면·해설 단서)."""
    readings = []
    for r in snap.get("scoreboard") or []:
        bases = tuple(n in (r.get("bases") or []) for n in (1, 2, 3))
        readings.append(Reading(float(r["t"]), int(r["inning"]), r["half"], int(r["balls"]),
                                int(r["strikes"]), int(r["outs"]), bases,  # type: ignore[arg-type]
                                int(r["away"]), int(r["home"])))
    cues = [Cue(float(e["t_start"]), e["event_type"], e.get("description", ""))
            for e in snap.get("events") or [] if e.get("event_type") in CUE_KINDS]
    cues += [Cue(float(s["t"]), "speech", s["text"]) for s in snap.get("speech") or []]
    return readings, sorted(cues, key=lambda c: c.t)


# ── RelayEvent 변환 (기존 엔진으로 넘기기) ────────────────────────────────
_CODE_TEXT = {
    "balk": "보크", "steal": "도루", "caught_stealing": "도루 실패", "wild_pitch": "폭투",
    "passed_ball": "포일", "runner": "주자 진루", "double_play": "병살",
    "triple_play": "삼중살", "homerun": "홈런", "sac_fly": "희생플라이",
    "infield_fly": "인필드플라이", "hbp": "몸에 맞는 공",
    "dropped_third_strike": "낫아웃", "walk": "볼넷", "strikeout": "삼진", "out": "아웃",
    "reach": "출루", "pitcher": "투수 교체",
}


def _bases_list(bases: tuple) -> list[int]:
    return [n for n, on in zip((1, 2, 3), bases) if on]


def to_relay_events(
    judgments: Sequence[Judgment], min_confidence: float = MIN_CONFIDENCE
) -> list[RelayEvent]:
    """판정 → RelayEvent. 확신도가 모자란 판정은 '상태만 반영하는 일반 결과'로 낮춘다.

    상태(아웃·주자·점수)는 확신도와 무관하게 점수판 값 그대로 넘긴다 — 점수판이 뼈대다.
    종류(보크인지)만 확신이 있을 때 붙인다. 그래서 확신 없는 판정은 카드가 안 뜬다.
    """
    events: list[RelayEvent] = []
    for n, j in enumerate(judgments):
        if j.kind == "sub":
            events.append(RelayEvent(
                id=f"v{n}", t=int(j.t), inning=j.detail.get("inning", 1),
                half=j.detail.get("half", "top"), kind="sub",
                text="투수 교체 (영상)", detail={"sub_type": "pitcher", "evidence": j.evidence,
                                             "confidence": j.confidence},
            ))
            continue
        tr = j.transition
        assert tr is not None
        confident = j.confidence >= min_confidence
        code = j.code if confident else ("out" if tr.outs_made else "reach")
        kind = j.kind if confident else "result"
        delta = {
            "outs_made": tr.outs_made, "runs": tr.runs,
            "bases_after": _bases_list(tr.after.bases),
            "evidence": j.evidence, "confidence": round(j.confidence, 2), "source": "video",
        }
        base = dict(inning=tr.before.inning, half=tr.before.half)
        label = _CODE_TEXT.get(code, code)
        text = f"{label} (영상 판정, 확신도 {j.confidence:.2f})"
        if kind == "call" and code == "dropped_third_strike":
            events.append(RelayEvent(id=f"v{n}c", t=int(j.t), kind="call", text=text,
                                     detail={"call": code}, **base))
            result = "strikeout" if j.detail.get("out") else "dropped_third_strike_safe"
            events.append(RelayEvent(id=f"v{n}", t=int(j.t), kind="result", text=text,
                                     detail={"result": result, **delta}, **base))
        elif kind == "call" and code == "infield_fly":
            events.append(RelayEvent(id=f"v{n}c", t=int(j.t), kind="call", text=text,
                                     detail={"call": code}, **base))
            events.append(RelayEvent(id=f"v{n}", t=int(j.t), kind="result", text=text,
                                     detail={"result": "infield_fly", **delta}, **base))
        elif kind == "call":
            events.append(RelayEvent(id=f"v{n}", t=int(j.t), kind="call", text=text,
                                     detail={"call": code, **delta}, **base))
        elif kind == "steal":
            extra = {"caught": True} if j.detail.get("caught") else {"base": j.detail.get("base")}
            events.append(RelayEvent(id=f"v{n}", t=int(j.t), kind="steal", text=text,
                                     detail={**extra, **delta}, **base))
        else:
            events.append(RelayEvent(id=f"v{n}", t=int(j.t), kind="result", text=text,
                                     detail={"result": code, **delta}, **base))
    return events
