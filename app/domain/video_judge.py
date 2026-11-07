"""영상만으로 플레이 판정. 순수 로직(I/O 없음).

"VLM이 보고, 규칙 엔진이 판정한다." 입력은 두 가지다.
- 점수판 전이(scorebug.transitions): 아웃·주자·점수·카운트가 어떻게 바뀌었나 — 상태의 뼈대
- 장면 단서(Cue): VLM이 본 동작(슬라이딩·포수 놓침·담장 넘김…)과 들은 해설 키워드("보크")

전이의 모양으로 후보를 좁히고, 단서로 종류를 가른다. 단서가 없으면 확신도를 낮추고,
모양만으로 정해지는 것(병살·볼넷)은 단서 없이도 판정한다. 결과는 문자중계와 같은
RelayEvent 형식이라 GameSim·detectors·카드가 그대로 받는다.

순서는 영상 먼저, 해설은 검사다(`judge_checked`): 해설 없이 판정한 뒤 해설과 맞는지 본다.

공식 결과는 심판 판정이다(점수판이 뼈대). 영상에서 본 동작이 그 판정과 갈려도 판정을 뒤집지 않고
`detail["observed"]`에 남긴다 — 관절 신호에도 오차가 있고 중계 각도로는 접촉 순간을 다 못 본다.

판정 규칙표는 docs/exec-plans의 07-video-judge 계획과 1:1이다.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

from app.domain.balk_judge import BALK_WORDS, TYPE_LABELS, speech_type
from app.domain.models import KIND_STATE, RelayEvent
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
# 보크가 '어떤 동작'이었는지는 해설이 다시보기 때 설명한다 — 판정 뒤로 더 길게(balk_judge와 같은 폭)
BALK_TALK = (20.0, 30.0)

# 해설에서 들어야 할 말 (Omni 모델은 음성을 같이 듣는다) — VLM 지시문(adapters/video/base)이 쓴다
SPEECH_KEYWORDS = (
    "보크", "도루", "견제", "폭투", "포일", "낫아웃", "인필드플라이", "희생플라이", "홈런",
    "볼넷", "몸에 맞는 공", "삼진", "병살", "삼중살", "투수 교체",
)

# 볼넷(5.05(b)(1)) — '고의사구'는 고의4구(四球), 그냥 '사구'는 몸에 맞는 공(死球)
WALK_WORDS = (
    "볼넷", "포볼", "밀어내기", "고의사구", "고의 사구", "고의4구", "고의 4구", "자동 고의",
)
# 파울팁(정의: 배트에 스친 공을 포수가 바로 잡음) — Whisper가 '파울TV'로 받아 적는다(실측)
FOUL_TIP_WORDS = ("파울팁", "파울 팁", "파울tv", "파울 tv")
TRIPLE_WORDS = ("삼중살", "트리플 플레이", "트리플플레이")
HBP_WORDS = ("몸에 맞", "사구", "데드볼", "헤드샷")  # 5.05(b)(2)

# 관절로 잰 동작 단서의 설명 머리말 — 카드 '영상 근거'에 그대로 나간다
POSE_TAG = "[관절] "
POSE_TEXT = {"slide_tag": "미끄러져 들어온 주자에게 수비 손이 닿음(태그)",
             "collision": "두 선수가 빠르게 부딪혀 한 명이 넘어짐",
             "swing": "타자가 배트를 휘두름",
             "dive": "수비가 잔디 위로 몸을 던짐",
             "no_swing": "타자가 공을 그냥 보냄"}
# 영상 관측이 공식 판정과 갈린 장면 — 판정은 그대로 두고 기록만 한다(오심 검토 후보)
OBSERVED_TAG_ON_SAFE = "tag_on_safe_runner"
OBSERVED_SWING_ON_BALL_FOUR = "swing_on_ball_four"
# 몸 아래 잔디 비율 — 넘으면 수비 다이빙. 실측(2026-11-03): 호수비 모음 다이빙 35건 중
# 28건이 70% 이상, KBO 정답 클립 주자 슬라이딩 24건은 최대 52%
DIVE_GRASS = 0.7
DIVE_RESULT = {"dive_out": "out", "dive_hit": "reach"}  # 다이빙 판정 뒤 타석 결과(일반)
SWING_GAP = 8.0  # 투구 간격보다 짧다 — 이 안의 스윙은 같은 공(다시보기·이어진 화면)이다
LAST_PITCH_SEC = 12.0  # 점수판 전이보다 이만큼 앞선 공은 그 플레이의 공으로 보지 않는다

# 단서 종류 (VLM 프롬프트 v2와 같은 어휘)
CUE_KINDS = (
    "pitch", "swing_miss", "contact", "slide", "pickoff_throw", "pitcher_stops",
    "catcher_miss", "infield_popup", "deep_fly", "ball_over_fence", "hit_by_pitch",
    "pitching_change", "mound_visit", "replay_review", "celebration", "crowd_cheer",
    "strikeout", "ball_four", "out_at_base", "runner_scores", "fielder_drops_ball",
    "umpire_signal",
    "swing", "no_swing",  # 관절로 잰 타자 손 움직임(domain/swing.py)
    "dive",  # 관절로 찾은 슬라이딩 중 몸 아래가 잔디 — 주자가 아니라 수비
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


def _near(cues: Sequence[Cue], t: float, before: float = CUE_BEFORE,
          after: float = CUE_AFTER) -> list[Cue]:
    return [c for c in cues if t - before <= c.t <= t + after]


def _has(cues: Sequence[Cue], kind: str, words: Sequence[str] = ()) -> Optional[Cue]:
    for c in cues:
        if c.kind == kind:
            return c
        if words and c.kind == "speech" and any(w in c.text for w in words):
            return c
    return None


# 해설이 영어(MLB 중계)이거나 VLM이 영어로 적을 때 — 한국어 키워드마다 같은 뜻의 영어.
# "walk"는 "walk-off"와 겹쳐 쓰지 않는다.
_EN_WORDS = {
    "보크": ("balk",), "도루": ("steal", "stole"), "폭투": ("wild pitch",),
    "포일": ("passed ball",), "삼진": ("strikeout", "struck out", "strike three"),
    "병살": ("double play",), "삼중살": ("triple play",), "홈런": ("home run", "homer"),
    "희생플라이": ("sacrifice fly", "sac fly"), "인필드": ("infield fly",),
    "몸에 맞": ("hit by pitch", "hit by a pitch", "plunk"),
    "볼넷": ("ball four", "base on balls", "draws a walk", "drew a walk"),
    "낫아웃": ("dropped third", "dropped strike"), "견제": ("pickoff", "picked off"),
}


# 받아쓰기(Whisper)가 숫자로 적는 말 — '삼진'을 '3진'으로(2026-11-07 하이라이트 실측: "4타자 연속
# 3진"). '3진루' 같은 말은 해설에 없다
_ASR_WORDS = {"삼진": ("3진",)}


# 같은 글자의 다른 말 — '사구(死球)'는 몸에 맞는 공인데 '고의사구(四球)'는 고의4구다.
# 낱말을 품은 다른 말(복합어)을 지우고 찾는다
_HOMONYMS = {
    "사구": ("고의사구", "고의 사구"),
    "홈런": ("홈런스타", "홈런더비", "홈런 더비", "홈런왕"),  # 광고·행사 자막(실측 '크보 홈런스타')
}


def _without_homonyms(text: str, word: str) -> str:
    for other in _HOMONYMS.get(word, ()):
        text = text.replace(other, "")
    return text


def _says(text: str, word: str) -> bool:
    return word in _without_homonyms(text, word)


def _speech(cues: Sequence[Cue], *words: str) -> Optional[Cue]:
    wanted = [w.lower() for w in words]
    for w in words:
        wanted += [e for k, en in _EN_WORDS.items() if k in w for e in en]
        wanted += [e for k, alt in _ASR_WORDS.items() if k in w for e in alt]
    for c in cues:
        if c.kind == "speech" and any(_says(c.text.lower(), w) for w in wanted):
            return c
    return None


def _ev(c: Cue) -> str:
    if c.kind == "speech":
        return f"해설 '{c.text}' ({c.t:.0f}s)"
    if c.text.startswith(POSE_TAG):  # 관절로 잰 동작 — 무엇을 봤는지 그대로
        return f"{c.text} ({c.t:.0f}s)"
    return f"장면 {c.kind} ({c.t:.0f}s)"


def _confidence(base: float, supports: Sequence[Optional[Cue]]) -> tuple[float, list[str]]:
    found = [c for c in supports if c is not None]
    return min(1.0, base + 0.25 * len(found)), [_ev(c) for c in found]


def judge_transition(
    tr: Transition, cues: Sequence[Cue], window: tuple[float, float] = (CUE_BEFORE, CUE_AFTER)
) -> Optional[Judgment]:
    """점수판 전이 하나를 판정한다. 플레이가 아니면(투구 하나) None."""
    if is_count_only(tr):
        return None
    near = _near(cues, tr.t, *window)
    b, a = tr.before.bases, tr.after.bases
    def bits(bases: tuple) -> str:
        return "".join("1" if x else "0" for x in bases)

    shape = f"점수판 아웃+{tr.outs_made} 득점+{tr.runs} 주자 {bits(b)}→{bits(a)}"

    def make(kind: str, code: str, prior: float, supports=(), **detail) -> Judgment:
        conf, ev = _confidence(prior, supports)
        return Judgment(tr.t, kind, code, conf, [shape] + ev, tr, detail)

    # 끝내기: 경기가 끝나면 점수판이 주자를 지우거나 그대로 멈춰 주자 모양을 믿을 수 없다.
    # 결승점이 들어왔다는 사실만 믿고 종류는 단서로 가른다 (2026-10-25 하이라이트 클립 실측).
    if is_walk_off(tr):
        found = _walk_off_kind(near)
        if found:
            code, cue = found
            kind = "result" if code in ("homerun", "walk", "hbp", "sac_fly") else "call"
            return make(kind, code, 0.5, [cue], walk_off=True)

    # 한 플레이로 설명이 안 되면(여러 플레이가 한 전이로 합쳐짐) 종류를 말하지 않는다
    if not one_play(tr):
        return make("result", "out" if tr.outs_made else "reach", 0.5, merged=True)

    # 같은 타석 안의 주자 움직임 (카운트 유지 — 0-0 카운트면 리셋처럼 보여도 단서로 가른다)
    mid_pa = not tr.half_change and (
        not tr.count_reset or (tr.before.balls, tr.before.strikes) == (0, 0)
    ) and not _has(near, "contact")
    if mid_pa and tr.outs_made == 0 and b != a:
        # 보크(6.02(a))는 투구 없이 모든 주자가 한 베이스씩 — 볼카운트가 그대로다.
        # 폭투·포일·더블스틸도 같은 진루 모양이지만 투구가 있어 카운트가 바뀐다.
        same_count = (tr.before.balls, tr.before.strikes) == (tr.after.balls, tr.after.strikes)
        balk_cue = _speech(near, "보크") or _has(near, "pitcher_stops")
        if advanced_all_one_base(b, a) and balk_cue and (same_count or _speech(near, "보크")):
            return make("call", "balk", 0.5, [balk_cue, _speech(near, "보크")])
        steal_cue = _has(near, "slide") or _speech(near, "도루")
        if single_runner_advance(b, a) and steal_cue and not _speech(near, "폭투", "포일"):
            # 세이프인데 태그가 닿아 보였다 — 베이스가 먼저였을 수도 있다. 관측만 남긴다
            seen = {"observed": OBSERVED_TAG_ON_SAFE} if _has(near, "slide_tag") else {}
            return make("steal", "steal", 0.5, [steal_cue, _speech(near, "도루")],
                        base=single_runner_advance(b, a), **seen)
        double = advanced_all_one_base(b, a) and not same_count and sum(b) >= 2
        if double and steal_cue and not _speech(near, "폭투", "포일"):  # 더블스틸 — 투구가 있었다
            return make("steal", "steal", 0.5, [steal_cue, _speech(near, "도루")], double=True)
        # 폭투·포일(9.13)은 투구가 있었다 — 카운트가 그대로면(점수판이 늦게 바뀐 경우) 확신을 낮춘다
        prior = 0.4 if same_count else 0.5
        wp = _speech(near, "폭투")
        if wp:
            return make("call", "wild_pitch", prior, [wp])
        pb = _speech(near, "포일")
        if pb:
            return make("call", "passed_ball", prior, [pb])
        if advanced_all_one_base(b, a) and same_count:  # 모양은 보크인데 단서가 없다
            return make("call", "balk", 0.3)
        if not tr.count_reset:
            return make("call", "runner", 0.4)
    if mid_pa and tr.outs_made == 1 and runner_removed(b, a):
        caught = _has(near, "slide") or _speech(near, "도루", "견제")
        if caught:
            return make("steal", "caught_stealing", 0.5, [caught], caught=True)

    # 타석 결과
    if tr.outs_made == 2:
        k = _speech(near, "삼진") or _has(near, "swing_miss") or _has(near, "strikeout")
        if tr.before.strikes == 2 and k and not _speech(near, "병살"):
            # 삼진 + 도루 실패(스트라이크 아웃 스로 아웃) — 병살타 카드를 띄우면 틀린 설명이 된다
            return make("result", "strikeout", 0.5, [k])
        return make("result", "double_play", 0.8, [_speech(near, "병살")])
    if tr.outs_made == 3:  # 판독 누락으로도 생긴다 — 해설이 있을 때만 삼중살
        tp = _speech(near, *TRIPLE_WORDS)
        if tp:
            return make("result", "triple_play", 0.6, [tp])
    hr = _has(near, "ball_over_fence") or _speech(near, "홈런")
    if tr.runs >= 1 and tr.runs == tr.runners_before + 1 and not any(a) and hr:
        return make("result", "homerun", 0.5, [hr, _speech(near, "홈런")])
    if tr.outs_made == 1 and tr.runs >= 1 and tr.before.outs <= 1:  # 9.08(d) 2아웃 전
        sf = _has(near, "deep_fly") or _speech(near, "희생플라이", "희생 플라이")
        if sf:
            return make("result", "sac_fly", 0.5, [sf])
    if tr.outs_made == 1 and b == a and not tr.half_change:
        iff = _speech(near, "인필드플라이", "인필드 플라이") or _has(near, "infield_popup")
        if iff and b[0] and b[1] and tr.before.outs <= 1:  # 5.09(a)(5) 무사·1사 1·2루/만루
            return make("call", "infield_fly", 0.4, [iff, _speech(near, "인필드")])
    if tr.outs_made == 0 and tr.count_reset and a[0]:
        hbp = _has(near, "hit_by_pitch") or _speech(near, *HBP_WORDS)
        if hbp:
            return make("result", "hbp", 0.5, [hbp])
        dts = _strike_three_miss(near) or _speech(near, "낫아웃", "낫 아웃")
        # 낫아웃은 1루가 비었거나 2아웃일 때만(5.09(a)(3)). 2스트라이크가 보였으면 확신도를 더 준다
        if dts and dts_possible(tr.before):
            prior = 0.5 if tr.before.strikes == 2 else 0.3
            return make("call", "dropped_third_strike", prior,
                        [dts, _speech(near, "낫아웃", "낫 아웃")])
        walk_speech = _speech(near, *WALK_WORDS) or _has(near, "ball_four")
        swung = _swung_last_pitch(near, tr.t)
        batted = _has(near, "contact") or swung
        if tr.before.balls == 3 and not batted:
            return make("result", "walk", 0.6, [walk_speech, _took_last_pitch(near, tr.t)])
        # 3볼에서 휘둘렀는데 1루 — 안타거나, 심판이 체크스윙을 '안 돌았다'로 본 볼넷이다
        seen = {"observed": OBSERVED_SWING_ON_BALL_FOUR} if tr.before.balls == 3 and swung else {}
        if walk_speech:
            return make("result", "walk", 0.5, [walk_speech], **seen)
        if seen:
            return make("result", "reach", 0.5, [swung], **seen)
    if tr.outs_made == 1 and tr.before.strikes == 2:
        dts = _strike_three_miss(near) or _speech(near, "낫아웃", "낫 아웃")
        if dts and dts_possible(tr.before):  # 낫아웃 상황이었지만 1루에서 아웃
            return make("call", "dropped_third_strike", 0.5, [dts], out=True)
        k = _has(near, "swing_miss") or _has(near, "strikeout") or _speech(near, "삼진")
        if k:
            return make("result", "strikeout", 0.5, [k])
        took = _took_last_pitch(near, tr.t)
        # 안 쳤는데 아웃 — 3스트라이크뿐이다. 주자가 움직였으면 다른 플레이가 섞였다(홈스틸 실측)
        if took and (tr.count_reset or tr.half_change) and b == a and not tr.runs:
            return make("result", "strikeout", 0.5, [took], looking=True)
    dive = _has(near, "dive")
    if dive and not tr.runs and not b[0]:
        # 1루가 비어 있어야 아웃이 타자의 것이다 — 1루 주자가 있으면 '주자 포스아웃 + 타자 1루'와
        # '타자 아웃 + 주자 그대로'가 점수판에서 같은 모양이다
        batter_out = (b == a and not tr.half_change) or (tr.half_change and not any(b))
        if tr.outs_made == 1 and batter_out:  # 몸을 던져 타자를 잡았다
            return make("call", "dive_out", 0.5, [dive])
        reached = a[0] and tr.runners_after == tr.runners_before + 1
        if tr.outs_made == 0 and reached and not tr.half_change:  # 몸을 던졌는데 타자가 살았다
            return make("call", "dive_hit", 0.5, [dive])
    # 나머지 타석 결과는 종류를 모른다 — 상태는 반영하되 카드는 띄우지 않는다
    if tr.count_reset or tr.half_change or tr.outs_made or tr.runs:
        return make("result", "out" if tr.outs_made else "reach", 0.5)
    return None


def _mentions(c: Cue, code: str) -> int:
    """해설 한 조각이 그 판정을 몇 번 말했나 — 판정 낱말(별칭 포함)이 나온 횟수, 최소 1."""
    words = next((w for k, _kind, w in _CUE_ONLY if k == code), ())
    alts = [*words]
    for w in words:
        alts += [e for k, en in _EN_WORDS.items() if k in w for e in en]
        alts += [e for k, alt in _ASR_WORDS.items() if k in w for e in alt]
    text = c.text.lower()
    return max(1, sum(_without_homonyms(text, w.lower()).count(w.lower()) for w in alts))


def _judge_spending_words(
    tr: Transition, cues: Sequence[Cue], window: tuple[float, float], spent: Counter
) -> Optional[Judgment]:
    """해설 한 조각의 '삼진' 한 마디는 삼진 하나만 설명한다.

    받아쓰기는 30초 조각이라 한 조각 안에 앞 타자 삼진과 다음 타자 땅볼이 같이 든다. 앞 전이가
    그 말로 삼진을 냈으면 다음 전이는 같은 말로 다시 삼진을 낼 수 없다(2026-11-07 하이라이트
    실측: 땅볼 아웃 두 개가 앞 타자의 '삼진'으로 삼진 카드가 됐다). 다른 종류(홈런 등)에는 쓴다.
    """
    pool = list(cues)
    while True:
        j = judge_transition(tr, pool, window)
        if j is None:
            return None
        said = [c for c in pool if c.kind == "speech" and _ev(c) in j.evidence]
        over = [c for c in said if spent[(c, j.code)] >= _mentions(c, j.code)]
        if not over:
            for c in said:
                spent[(c, j.code)] += 1
            return j
        pool = [c for c in pool if c not in over]


def _last_pitch(cues: Sequence[Cue], t: float) -> list[Cue]:
    """전이 시각 `t` 직전의 공에 속한 스윙 단서들. 전이 뒤의 것은 다시보기다."""
    seen = [c for c in cues if c.kind in ("swing", "no_swing")
            and t - LAST_PITCH_SEC <= c.t <= t]
    if not seen:
        return []
    last = max(c.t for c in seen)
    return [c for c in seen if last - c.t <= SWING_GAP]


def _took_last_pitch(cues: Sequence[Cue], t: float) -> Optional[Cue]:
    """마지막 공을 그냥 보냈나. 같은 공에 스윙이 하나라도 보이면 아니다 — 느린 다시보기는 손이
    느려 '안 침'으로 재진다."""
    pitch = _last_pitch(cues, t)
    if not pitch or any(c.kind == "swing" for c in pitch):
        return None
    return pitch[-1]


def _swung_last_pitch(cues: Sequence[Cue], t: float) -> Optional[Cue]:
    return next((c for c in _last_pitch(cues, t) if c.kind == "swing"), None)


def _strike_three_miss(cues: Sequence[Cue]) -> Optional[Cue]:
    """낫아웃(5.05(a)(2))의 '포수가 3스트라이크를 놓침' — 타구(contact) 직후 3초 안의 놓침은
    인플레이 타구를 처리하다 놓친 것이라 낫아웃이 아니다(실측: 끝내기 뜬공 직후 '포수 놓침')."""
    hits = [c.t for c in cues if c.kind == "contact"]
    for c in cues:
        if c.kind == "catcher_miss" and not any(0 <= c.t - h <= 3 for h in hits):
            return c
    return None


def one_play(tr: Transition) -> bool:
    """한 플레이로 설명되나 — 직전 주자와 타자는 아웃·득점·베이스 중 하나로 끝난다(사람 수 보존).
    아웃 + 득점 + 남은 주자가 직전 주자 + 타자 한 명보다 많으면 타자가 둘 이상 지나간 것이다
    (실측: 2분짜리 클립의 처음·끝 점수판 '무사 1·2루 → 2사 1·2루 3득점'을 병살로 냈다).
    이닝이 바뀌면 남은 주자를 알 수 없어 따지지 않는다."""
    if tr.half_change:
        return True
    return tr.outs_made + tr.runs + tr.runners_after <= tr.runners_before + 1


def dts_possible(before: Reading) -> bool:
    """낫아웃 성립 조건(5.09(a)(3)): 1루가 비었거나 2아웃. 아니면 공을 놓쳐도 타자는 아웃이다."""
    return not before.bases[0] or before.outs == 2


def is_walk_off(tr: Transition) -> bool:
    """9회 이후 말 공격에서 홈팀이 뒤지거나 비기다 앞서게 된 전이."""
    b, a = tr.before, tr.after
    return (a.half == "bot" and a.inning >= 9 and not tr.half_change
            and b.home <= b.away and a.home > a.away)


def _walk_off_kind(near: Sequence[Cue]) -> Optional[tuple[str, Cue]]:
    """끝내기 결승점의 종류 — 단서가 분명한 것만. 순서는 오인 비용이 큰 것부터."""
    checks = (
        ("homerun", _has(near, "ball_over_fence") or _speech(near, "홈런")),
        ("balk", _speech(near, "보크") or _has(near, "pitcher_stops")),
        ("wild_pitch", _speech(near, "폭투")),
        ("passed_ball", _speech(near, "포일")),
        ("hbp", _has(near, "hit_by_pitch") or _speech(near, *HBP_WORDS)),
        ("dropped_third_strike", _speech(near, "낫아웃", "낫 아웃")),
        ("walk", _speech(near, *WALK_WORDS) or _has(near, "ball_four")),
        ("sac_fly", _speech(near, "희생플라이", "희생 플라이") or _has(near, "deep_fly")),
    )
    for code, cue in checks:
        if cue is not None:
            return code, cue
    return None


def judge(
    readings: Sequence[Reading],
    cues: Sequence[Cue],
    min_repeat: int = 2,
    window: tuple[float, float] = (CUE_BEFORE, CUE_AFTER),
) -> list[Judgment]:
    """점수판 판독 열 + 단서 → 판정 목록 (시간순). 투수 교체는 단서만으로 판정한다.

    `min_repeat`: 같은 점수판이 몇 번 연속 보여야 믿나. 경기 전체는 2(오독 거르기),
    점수판이 몇 번밖에 안 나오는 하이라이트 클립은 1로 쓴다.
    `window`: 전이 앞뒤로 단서를 볼 범위(초). 하이라이트는 느린 화면 뒤에야 점수판이 다시
    보여 단서와 멀리 떨어진다 — 클립 평가에선 클립 전체를 본다.
    """
    stable = stabilize(readings, min_repeat=min_repeat)
    out = []
    spent: Counter = Counter()
    for tr in transitions(stable):
        j = _judge_spending_words(tr, cues, window, spent)
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
    out += judge_calls(cues, stable)
    for j in out:
        _balk_kind(j, cues)
    return sorted(out, key=lambda j: j.t)


# ── 영상 먼저, 해설은 검사 ────────────────────────────────────────────────
# 영상 판정은 해설을 보지 않고 끝낸다. 해설은 그 판정이 맞는지 되묻는 데만 쓴다.
CHECK_CONFIRMED = "confirmed"  # 영상 판정을 해설이 뒷받침
CHECK_VIDEO_ONLY = "video_only"  # 해설이 말이 없다 — 영상 판정 그대로
CHECK_REFINED = "refined"  # 영상은 일반 결과까지, 종류는 해설이 댔다 — 서로 어긋나지 않는다
# (아래 둘은 영상 신호가 늘면 줄어야 할 몫 — 지금은 카드를 띄운다. 해설뿐인 판정을 막으면
#  클립 평가에서 정답이 그만큼 빠진다: data/clip_eval/ABLATION.md)
CHECK_SPEECH_ONLY = "speech_only"  # 영상 판정 없이 해설 낱말로만 — 영상으로 옮겨야 할 몫
CHECK_CONFLICT = "conflict"  # 영상과 해설이 다른 판정 — 카드를 보류한다
HOLD_CONFIDENCE = 0.4  # 보류 — MIN_CONFIDENCE 아래라 상태만 반영하고 카드는 안 뜬다
CHECK_WINDOW = (10.0, CUE_AFTER)  # 해설을 대조할 범위 — 지난 일 언급을 덜 받게 판정보다 좁다
# 이름은 달라도 같은 판정: (영상, 해설)이 이 짝이면 어긋난 것이 아니다
SAME_KIND = frozenset({("strikeout", "dropped_third_strike"), ("caught_stealing", "steal")})


def split_cues(cues: Sequence[Cue]) -> tuple[list[Cue], list[Cue]]:
    """(화면에서 본 단서, 귀로 들은 해설). 자막은 화면에 있는 글자라 본 쪽이다."""
    def heard(c: Cue) -> bool:
        return c.kind == "speech" and not c.text.startswith("[자막]")

    return [c for c in cues if not heard(c)], [c for c in cues if heard(c)]


def _same_play(j: Judgment, video: Sequence[Judgment]) -> Optional[Judgment]:
    if j.transition is not None:
        return next((v for v in video if v.transition is not None and v.t == j.t), None)
    return next((v for v in video if v.transition is None and v.code == j.code
                 and abs(v.t - j.t) <= CALL_GAP), None)


def _commentary_says(t: float, heard: Sequence[Cue], window: tuple[float, float]) -> set[str]:
    """그 장면 앞뒤로 해설이 말한 판정 코드."""
    near = _near(heard, t, *window)
    return {code for code, _, words in _CUE_ONLY if _speech(near, *words)}


def _agrees(code: str, said: set[str]) -> bool:
    return code in said or any((code, s) in SAME_KIND or (s, code) in SAME_KIND for s in said)


def _check_play(j: Judgment, v: Judgment, said: set[str]) -> Judgment:
    """점수판 전이 판정 하나의 대조 — 영상이 종류까지 낸 판정은 해설이 말한 것과 맞춘다."""
    if v.code in GENERIC_CODES or v.confidence < MIN_CONFIDENCE:
        if v.code == j.code:
            j.detail["check"] = CHECK_VIDEO_ONLY
        else:
            j.detail.update(check=CHECK_REFINED, video_code=v.code)
    elif _agrees(v.code, said):
        j.detail["check"] = CHECK_CONFIRMED
    elif said:
        j.detail.update(check=CHECK_CONFLICT, video_code=v.code)
        j.evidence.append(f"영상은 {_CODE_TEXT.get(v.code, v.code)} — 해설과 달라 보류")
        j.confidence = min(j.confidence, HOLD_CONFIDENCE)
    else:  # 대조 범위 밖의 말이 판정을 바꿨다 — 영상 판정을 그대로 둔다
        v.detail["check"] = CHECK_VIDEO_ONLY
        return v
    return j


def judge_checked(
    readings: Sequence[Reading],
    cues: Sequence[Cue],
    min_repeat: int = 2,
    window: tuple[float, float] = (CUE_BEFORE, CUE_AFTER),
    check_window: tuple[float, float] = CHECK_WINDOW,
) -> list[Judgment]:
    """영상 판정 → 해설 대조. 판정마다 `detail["check"]`에 대조 결과를 단다.

    영상 판정은 해설 없이 끝낸 뒤, 그 장면 앞뒤로 해설이 말한 판정과 맞춘다. 해설이 다른 판정을
    말하면 보류한다(점수판 상태는 그대로 반영하고 카드만 안 뜬다). 해설까지 넣은 판정(merged)은
    영상이 일반 결과까지만 냈을 때 종류를 대는 데 쓴다.
    """
    seen, heard = split_cues(cues)
    video = judge(readings, seen, min_repeat, window)
    out = []
    for j in judge(readings, cues, min_repeat, window):
        v = _same_play(j, video)
        if v is None:
            j.detail["check"] = CHECK_SPEECH_ONLY
        elif j.transition is not None:
            j = _check_play(j, v, _commentary_says(j.t, heard, check_window))
        else:  # 교체·심판 선언 — 함께 일어날 수 있어 서로 어긋난다고 보지 않는다
            backed = set(j.evidence) - set(v.evidence)
            j.detail["check"] = CHECK_CONFIRMED if backed else CHECK_VIDEO_ONLY
        out.append(j)
    return out


def _balk_kind(j: Judgment, cues: Sequence[Cue]) -> None:
    """보크 판정에 '어떤 동작이 반칙이었나'를 붙인다 — 해설이 동작을 설명했을 때만.

    자세로 고른 종류는 해설과 9건 중 2건만 같아(balk_judge 실측) 쓰지 않는다. '보크'라는 말이
    같이 나온 설명만 — VLM 장면 글 '투수가 몸을 정지시킵니다'가 '정지 없음'으로 읽혔다(실측).
    """
    if j.code != "balk" or "balk_kind" in j.detail:
        return
    talk = [(c, speech_type(c.text)) for c in _near(cues, j.t, *BALK_TALK)
            if c.kind == "speech" and BALK_WORDS.search(c.text)]
    talk = [(c, k) for c, k in talk if k != "unknown"]
    if not talk:
        return
    c, kind = min(talk, key=lambda ck: abs(ck[0].t - j.t))
    j.detail["balk_kind"] = kind
    j.evidence.append(f"해설이 말한 동작: {TYPE_LABELS[kind]} ({c.t:.0f}s)")


# ── 심판 선언·특수 판정 — 해설·자막 낱말 + 확인 (점수판 전이와 따로) ─────────────
# (코드, 낱말, 확인 장면). 낱말은 무슨 판정인지를, 확인은 실제로 일어났는지를 준다.
# 해설은 '타격방해 아니었나요?', '스퀴즈 가능성도 있죠'처럼 일어나지 않은 일도 말한다 —
# 낱말 앞뒤 CALL_SPAN초에 확인 장면(심판 동작·판독 화면 등)이 있거나, 자막에 나왔거나,
# 60초 안에 두 번 넘게 말했을 때만 낸다. 흔한 말과 겹치는 낱말은 넣지 않는다
# ('퇴장'은 박수 받으며 퇴장, '블로킹'은 포수가 원바운드를 막는 것과 겹친다).
_UMPIRE = ("umpire_signal", "replay_review")
CALL_TABLE = (
    ("catcher_interference", ("타격방해", "타격 방해", "캐처 인터페어", "포수 인터페어"), _UMPIRE),
    ("obstruction", ("주루방해", "주루 방해"), _UMPIRE + ("collision",)),
    ("interference", ("수비방해", "수비 방해"), _UMPIRE + ("collision",)),
    ("video_review", ("비디오 판독", "비디오판독"), _UMPIRE),
    ("check_swing", ("체크스윙", "체크 스윙"), _UMPIRE),
    ("ejection",
     ("퇴장 명령", "퇴장 조치", "퇴장을 당", "퇴장당", "퇴장 선언", "퇴장을 선언", "퇴장이 됐",
      "퇴장 사유"), _UMPIRE),
    ("bench_clearing", ("벤치클리어링", "벤치 클리어링"), _UMPIRE),
    ("ground_rule_double", ("인정 2루타", "인정2루타"), _UMPIRE + ("ball_over_fence",)),
    ("foreign_substance", ("이물질",), _UMPIRE),
    ("pitch_clock", ("피치클락 위반", "피치 클락 위반", "피치클락 바이올레이션"), _UMPIRE),
    ("squeeze", ("스퀴즈", "스쿠즈"), ("runner_scores",)),  # Whisper '스쿠즈'(실측)
    ("missed_base", ("누의 공과", "누의공과"), _UMPIRE),
    # 관절로 잰 충돌·미끄러져 닿음(contact.py)도 '실제로 부딪혔다'는 확인이 된다
    ("home_collision", ("홈 충돌", "충돌 방지 규정", "홈플레이트 충돌"),
     _UMPIRE + ("collision", "slide_tag")),
    ("slide_rule", ("슬라이딩 규정", "슬라이딩 방해", "병살 방해 슬라이딩"),
     _UMPIRE + ("slide_tag", "collision")),
    ("runner_lane", ("3피트", "쓰리피트", "스리피트"), _UMPIRE),
    ("fan_interference", ("관중 방해", "팬 방해", "관중이 공을 건드"), _UMPIRE),
    ("intentional_drop", ("고의낙구", "고의 낙구"), _UMPIRE),
    ("passing_runner", ("주자 추월", "앞 주자를 추월", "추월 아웃"), _UMPIRE),
    ("called_game", ("콜드게임", "콜드 게임", "강우 콜드", "노게임", "서스펜디드"), _UMPIRE),
)
CALL_SPAN = 20.0
CALL_GAP = 60.0  # 같은 판정을 다시 내지 않는 간격(다시보기·해설 반복)


def _confirmation(c: Cue, cues: Sequence[Cue], words: Sequence[str],
                  kinds: Sequence[str]) -> Optional[tuple[float, str]]:
    """(확신도, 근거) — 그 말이 실제로 일어난 일이라는 확인. 없으면 None."""
    if c.text.startswith("[자막]"):
        return 0.75, "중계 자막"
    seen = next((x for x in cues if x.kind in kinds and abs(x.t - c.t) <= CALL_SPAN), None)
    if seen is not None:
        return 0.75, _ev(seen)
    again = [x for x in cues if x is not c and x.kind == "speech" and 0 < x.t - c.t <= CALL_GAP
             and any(_says(x.text.lower(), w) for w in words)]
    if again:
        return 0.5, f"해설이 다시 말함 ({again[0].t:.0f}s)"
    return None


def judge_calls(cues: Sequence[Cue], stable: Sequence[Reading] = ()) -> list[Judgment]:
    """심판 선언·특수 판정 — 점수판 모양이 없거나 한 가지로 정해지지 않는 규칙들."""
    out: list[Judgment] = []
    speech = sorted((c for c in cues if c.kind == "speech"), key=lambda c: c.t)
    for code, words, kinds in CALL_TABLE:
        last = -1e9
        for c in speech:
            if c.t - last < CALL_GAP or not any(_says(c.text.lower(), w) for w in words):
                continue
            found = _confirmation(c, cues, words, kinds)
            if found is None:
                continue
            last = c.t
            at = [r for r in stable if r.t <= c.t]
            where = at[-1] if at else (stable[0] if stable else None)
            half = {"inning": where.inning, "half": where.half} if where else {}
            out.append(Judgment(c.t, "call", code, found[0], [_ev(c), found[1]], None, half))
    return out


# 점수판 없이 단서만으로 (직캠·느린 화면 위주 하이라이트). 드문 것부터 — 흔한 말("삼진")이
# 드문 판정("낫아웃")을 덮지 않게.
_CUE_ONLY = (
    ("triple_play", "call", TRIPLE_WORDS),
    ("balk", "call", ("보크",)),
    ("dropped_third_strike", "call", ("낫아웃", "낫 아웃")),
    ("infield_fly", "call", ("인필드 플라이", "인필드플라이")),
    ("passed_ball", "call", ("포일",)),
    ("wild_pitch", "call", ("폭투",)),
    ("sac_fly", "result", ("희생플라이", "희생 플라이")),
    ("hbp", "result", HBP_WORDS),
    ("steal", "steal", ("홈스틸", "더블스틸", "더블 스틸", "도루")),
    ("double_play", "result", ("병살", "더블플레이", "더블 플레이")),
    ("homerun", "result", ("홈런",)),
    ("walk", "result", WALK_WORDS),
    ("strikeout", "result", ("삼진",)),
)
GENERIC_CODES = frozenset({"out", "reach", "runner", "pitcher"})


def _keyword_judgment(cues: Sequence[Cue], conf: float, label: str) -> Optional[Judgment]:
    for code, kind, words in _CUE_ONLY:
        hit = _speech(cues, *words)
        if hit is not None:
            return Judgment(hit.t, kind, code, conf, [f"단서만({label}): " + _ev(hit)])
    return None


def _cluster(cues: Sequence[Cue], kind: str, span: float) -> list[Cue]:
    """`span`초 안에 몰린 같은 종류 단서 중 가장 큰 묶음."""
    hits = sorted((c for c in cues if c.kind == kind), key=lambda c: c.t)
    best: list[Cue] = []
    for i, c in enumerate(hits):
        group = [h for h in hits[i:] if h.t - c.t <= span]
        if len(group) > len(best):
            best = group
    return best


def _event_judgment(cues: Sequence[Cue]) -> Optional[Judgment]:
    """장면 단서의 순서로 — 해설을 못 알아들어도 동작은 보인다 (2026-10-26 KBO 클립 실측)."""
    def make(code: str, kind: str, cue: Cue, why: str) -> Judgment:
        return Judgment(cue.t, kind, code, 0.5, [f"장면 순서: {why}"])

    outs = _cluster(cues, "out_at_base", 15.0)
    if len(outs) > 3:  # 아웃은 반 이닝에 셋뿐(5.09(e)) — 넘으면 같은 장면을 되풀이해 지어낸 것
        outs = []
    if len(outs) == 3:
        return make("triple_play", "result", outs[0], f"베이스 아웃 {len(outs)}번이 15초 안에")
    over = _has(cues, "ball_over_fence")
    if over:
        return make("homerun", "result", over, "타구가 담장을 넘어감")
    hbp = _has(cues, "hit_by_pitch")
    if hbp:
        return make("hbp", "result", hbp, "투구가 타자 몸에 맞음")
    scores = [c for c in cues if c.kind == "runner_scores"]
    for score in scores:
        before = [c for c in cues if score.t - 15 <= c.t <= score.t]
        caught_fly = next((c for c in before if c.kind == "deep_fly" and "잡" in c.text), None)
        if caught_fly:
            return make("sac_fly", "result", score, "외야 뜬공을 잡은 뒤 득점")
        slide = next((c for c in before if c.kind == "slide" and score.t - c.t <= 5), None)
        # 득점 전 15초 안에 타구가 있었으면 홈스틸이 아니다(2루타에 홈 슬라이딩)
        contact = any(c.kind == "contact" for c in before)
        missed = any(c.kind in ("catcher_miss", "fielder_drops_ball") for c in before)
        if slide and not contact and not missed:  # 포수가 놓쳤으면 폭투·포일 득점
            return make("steal", "steal", score, "타격 없이 슬라이딩 득점(홈스틸)")
    if len(outs) == 2:
        # 병살은 한 플레이 — 직전 10초에 타구·헛스윙이 없으면 견제·도루 아웃이 따로 둘이다
        batted = any(c.kind in ("contact", "swing_miss", "strikeout")
                     and 0 <= outs[0].t - c.t <= 10 for c in cues)
        if batted:
            return make("double_play", "result", outs[0], "베이스 아웃 2번이 15초 안에")
        lead = _has(cues, "pickoff_throw") or _has(cues, "slide") or _speech(cues, "견제", "도루")
        if lead:
            return make("caught_stealing", "steal", outs[0], "타구 없이 베이스 아웃(견제·도루)")
    k = _has(cues, "strikeout") or _has(cues, "swing_miss")
    miss = _has(cues, "catcher_miss") or _has(cues, "fielder_drops_ball")
    if k and miss and abs(k.t - miss.t) <= 5:
        tip = _speech(cues, *FOUL_TIP_WORDS)
        if tip:  # 파울팁은 포수가 잡은 것 — 놓친 게 아니라 저글링(5.09(a)(2) 삼진)
            return make("strikeout", "result", tip, "헛스윙·포수 저글링 + 해설 '파울팁'")
        return make("dropped_third_strike", "call", miss, "헛스윙 직후 포수가 공을 놓침")
    return None


def judge_from_cues(cues: Sequence[Cue]) -> Optional[Judgment]:
    """점수판 전이로 종류를 못 가렸을 때 단서만으로 하나 고른다 (하이라이트 클립 전용).

    믿을 만한 순서: 화면 자막 → 장면 동작의 순서 → 해설. 한국어 해설 받아쓰기는 실측상
    지어낸 문장이 많아 마지막에 둔다. 경기 전체 판정에는 쓰지 않는다 — 해설은
    '도루 조심해야죠'처럼 일어나지 않은 일도 말한다.
    """
    captions = [c for c in cues if c.kind == "speech" and c.text.startswith("[자막]")]
    found = _keyword_judgment(captions, 0.6, "자막")
    if found is None:
        found = _event_judgment(cues)
        tp = _speech(cues, *TRIPLE_WORDS) if found and found.code == "double_play" else None
        if found and tp:  # 삼중살 안에 아웃 두 개가 들어 있다 — 장면이 하나를 놓쳤다
            found = Judgment(found.t, "result", "triple_play", 0.5,
                             found.evidence + [f"해설이 더 구체적: {_ev(tp)}"])
    found = found or _keyword_judgment([c for c in cues if c not in captions], 0.5, "해설·설명")
    if found is not None:
        _balk_kind(found, cues)
    return found


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
    cues += [Cue(float(s["t"]), "speech", s["text"]) for s in snap.get("speech") or []
             if not is_echoed_keyword(s.get("text", ""))]
    return readings, sorted(cues + pose_cues(snap.get("pose") or {}), key=lambda c: c.t)


def observe(snap: dict) -> tuple[list[Reading], list[Cue], int]:
    """스냅샷 → (점수판 판독, 단서, 안정화 기준). 클립 평가와 경기 분석이 같은 입력을 쓰게 한다.

    점수판 전용 판독(scorebug_reader)이 있으면 영상 통째 판독 대신 쓴다 — 더 정확하다.
    하이라이트 방식(바뀔 때마다 한 장)은 상태마다 한 장이라 안정화 1(앞뒤와 수지가 안 맞는 한 장은
    stabilize가 버린다), 경기 영상 방식(`every`초마다)은 같은 상태를 여러 번 읽어 안정화 2.
    Whisper 받아쓰기가 있으면 VLM 받아쓰기 대신 쓴다 — VLM은 한국어 해설을 지어냈다(실측).
    화면 자막은 남긴다.
    """
    readings, cues = from_snapshot(snap)
    asr = [s for s in (snap.get("asr") or {}).get("segments") or [] if s.get("text")]
    if asr:
        cues = [c for c in cues if c.kind != "speech" or c.text.startswith("[자막]")]
        cues = sorted(
            cues + [Cue(float(s["t"]), "speech", s["text"]) for s in asr], key=lambda c: c.t
        )
    crop = (snap.get("scorebug") or {}).get("scoreboard") or []
    if crop:
        readings, _ = from_snapshot({"scoreboard": crop})
        # 하이라이트를 일정 간격으로 읽으면 한 상태가 한 번만 읽히는 일이 많다 — 분석할 때
        # 정한 기준(analyze_video --min-repeat)이 있으면 그것을 쓴다
        fixed = snap["scorebug"].get("min_repeat")
        return readings, cues, int(fixed) if fixed else (2 if snap["scorebug"].get("every") else 1)
    return readings, cues, 2


def _slide_text(s: dict) -> str:
    far = s.get("distance")
    if far is None:
        how = ""
    elif s.get("thrown"):
        how = f" — 달려와 몸을 던져 몸길이 {far:.1f}배 이동"
    else:
        how = f" — 누운 채 몸길이 {far:.1f}배 미끄러짐"
    return f"슬라이딩·다이빙{how}, {s['t0']:.1f}~{s['t1']:.1f}초"


def _slide_kind(s: dict) -> str:
    """잔디 위면 수비 다이빙, 아니면 주자 슬라이딩. 색을 못 쟀으면 지금처럼 슬라이딩."""
    grass = s.get("grass")
    return "dive" if grass is not None and grass >= DIVE_GRASS else "slide"


def pose_cues(pose: dict) -> list[Cue]:
    """관절 분석(contact.py·swing.py) → 장면 단서. 슬라이딩은 VLM 'slide'와 같은 종류."""
    cues = [Cue(float(s["t0"]), _slide_kind(s), POSE_TAG + _slide_text(s))
            for s in pose.get("slides") or []]
    cues += [Cue(float(c["t0"]), c["kind"], POSE_TAG + POSE_TEXT[c["kind"]])
             for c in pose.get("contacts") or [] if c.get("kind") in POSE_TEXT]
    for s in pose.get("swings") or []:
        if s.get("clear", True):  # 경계에 걸친 값은 어느 쪽 단서로도 쓰지 않는다
            kind = "swing" if s["swing"] else "no_swing"
            cues.append(Cue(float(s["t"]), kind, POSE_TAG + POSE_TEXT[kind]))
    return cues


def is_echoed_keyword(text: str) -> bool:
    """VLM이 해설을 받아 적지 않고 지시문의 키워드 목록을 따라 적은 것.

    실측(2026-10-26): 해설이 없는 클립에서도 '보크', '포일', '몸에 맞는 공', '투수 교체'를
    한 단어씩 '들었다'고 내놓았다. 실제 해설은 문장이라 키워드 한 단어로만 오지 않는다.
    """
    bare = text.strip().strip(".!?~ ").replace(" ", "")
    return bare in {k.replace(" ", "") for k in SPEECH_KEYWORDS}


# ── RelayEvent 변환 (기존 엔진으로 넘기기) ────────────────────────────────
_CODE_TEXT = {
    "balk": "보크", "steal": "도루", "caught_stealing": "도루 실패", "wild_pitch": "폭투",
    "passed_ball": "포일", "runner": "주자 진루", "double_play": "병살",
    "triple_play": "삼중살", "homerun": "홈런", "sac_fly": "희생플라이",
    "infield_fly": "인필드플라이", "hbp": "몸에 맞는 공",
    "dropped_third_strike": "낫아웃", "walk": "볼넷", "strikeout": "삼진", "out": "아웃",
    "reach": "출루", "pitcher": "투수 교체",
    "catcher_interference": "타격방해", "obstruction": "주루방해", "interference": "수비방해",
    "video_review": "비디오 판독", "check_swing": "체크스윙", "ejection": "퇴장",
    "bench_clearing": "벤치클리어링", "ground_rule_double": "인정 2루타",
    "foreign_substance": "이물질", "pitch_clock": "피치클락 위반", "squeeze": "스퀴즈",
    "missed_base": "누의 공과", "home_collision": "홈 충돌", "slide_rule": "슬라이딩 규정",
    "runner_lane": "3피트 라인", "fan_interference": "관중 방해", "intentional_drop": "고의낙구",
    "passing_runner": "주자 추월", "called_game": "콜드게임·노게임",
    "dive_out": "몸을 던진 수비로 아웃", "dive_hit": "몸을 던졌지만 안타",
}


def _call_extra(j: Judgment) -> dict:
    """카드 설명에 쓰는 판정 세부 — 지금은 보크 동작 종류만."""
    return {"balk_kind": j.detail["balk_kind"]} if "balk_kind" in j.detail else {}


def _observed_extra(j: Judgment) -> dict:
    """영상 관측이 공식 판정과 갈렸다는 기록 — 판정·카드는 바꾸지 않는다."""
    return {"observed": j.detail["observed"]} if "observed" in j.detail else {}


def _check_extra(j: Judgment) -> dict:
    """영상↔해설 대조 결과 — judge_checked를 거친 판정에만 있다."""
    return {"check": j.detail["check"]} if "check" in j.detail else {}


def _bases_list(bases: tuple) -> list[int]:
    return [n for n, on in zip((1, 2, 3), bases) if on]


def state_events(stable: Sequence[Reading]) -> list[RelayEvent]:
    """안정된 점수판 상태마다 그대로 — 화면 스코어보드가 중계 점수판과 같게(점수판이 뼈대다).

    판정 이벤트는 변화량만 싣는다. 이것이 없으면 경기 중간에서 시작한 녹화본의 점수·아웃·주자가
    0에서 시작하고, 볼카운트는 늘 0-0이었다. 판정이 아니라 카드·장면 요약에는 쓰이지 않는다.
    같은 시각이면 판정 이벤트 뒤에 와야 한다 — `merge_states`.
    """
    out, used = [], set()
    for r in stable:
        n = f"st{int(round(r.t * 10))}"
        while n in used:
            n += "b"
        used.add(n)
        out.append(RelayEvent(
            id=n, t=int(r.t), inning=r.inning, half=r.half, kind=KIND_STATE, text="점수판 (영상)",
            detail={"balls": r.balls, "strikes": r.strikes, "outs": r.outs,
                    "bases": _bases_list(r.bases), "away": r.away, "home": r.home},
        ))
    return out


def merge_states(events: Sequence[RelayEvent], states: Sequence[RelayEvent]) -> list[RelayEvent]:
    """판정 이벤트 + 점수판 상태 — 시각순, 같은 시각이면 상태가 뒤(판정 변화량 다음 절댓값)."""
    return sorted([*events, *states], key=lambda e: (e.t, e.kind == KIND_STATE))


def to_relay_events(
    judgments: Sequence[Judgment], min_confidence: float = MIN_CONFIDENCE
) -> list[RelayEvent]:
    """판정 → RelayEvent. 확신도가 모자란 판정은 '상태만 반영하는 일반 결과'로 낮춘다.

    상태(아웃·주자·점수)는 확신도와 무관하게 점수판 값 그대로 넘긴다 — 점수판이 뼈대다.
    종류(보크인지)만 확신이 있을 때 붙인다. 그래서 확신 없는 판정은 카드가 안 뜬다.
    """
    events: list[RelayEvent] = []
    used: set[str] = set()
    for j in judgments:
        # id는 판정 시각에서 만든다 — 실시간 모드에서 판정이 늘어나도
        # 기존 이벤트 id(=카드 캐시 키)가 바뀌지 않게. 같은 0.1초에 둘이면 글자를 붙인다.
        n = ("s" if j.kind == "sub" else "") + str(int(round(j.t * 10)))
        while n in used:
            n += "b"
        used.add(n)
        if j.kind == "sub":
            events.append(RelayEvent(
                id=f"v{n}", t=int(j.t), inning=j.detail.get("inning", 1),
                half=j.detail.get("half", "top"), kind="sub",
                text="투수 교체 (영상)", detail={"sub_type": "pitcher", "evidence": j.evidence,
                                             "confidence": j.confidence, **_check_extra(j)},
            ))
            continue
        tr = j.transition
        if tr is None:  # 심판 선언 — 점수판 상태는 바꾸지 않는다
            if j.confidence >= min_confidence:
                events.append(RelayEvent(
                    id=f"v{n}", t=int(j.t), inning=j.detail.get("inning", 1),
                    half=j.detail.get("half", "top"), kind="call",
                    text=f"{_CODE_TEXT.get(j.code, j.code)} (영상 판정, 확신도 {j.confidence:.2f})",
                    detail={"call": j.code, "evidence": j.evidence,
                            "confidence": round(j.confidence, 2), "source": "video",
                            **_call_extra(j), **_check_extra(j)},
                ))
            continue
        confident = (j.confidence >= min_confidence
                     and j.detail.get("check") != CHECK_CONFLICT)
        code = j.code if confident else ("out" if tr.outs_made else "reach")
        kind = j.kind if confident else "result"
        delta = {
            "outs_made": tr.outs_made, "runs": tr.runs,
            "bases_after": _bases_list(tr.after.bases),
            "evidence": j.evidence, "confidence": round(j.confidence, 2), "source": "video",
            **_check_extra(j), **_observed_extra(j),
        }
        base = dict(inning=tr.before.inning, half=tr.before.half)
        label = _CODE_TEXT.get(code, code)
        text = f"{label} (영상 판정, 확신도 {j.confidence:.2f})"
        if kind == "call" and code == "dropped_third_strike":
            events.append(RelayEvent(id=f"v{n}c", t=int(j.t), kind="call", text=text,
                                     detail={"call": code, **_check_extra(j)}, **base))
            result = "strikeout" if j.detail.get("out") else "dropped_third_strike_safe"
            events.append(RelayEvent(id=f"v{n}", t=int(j.t), kind="result", text=text,
                                     detail={"result": result, **delta}, **base))
        elif kind == "call" and code in DIVE_RESULT:  # 카드는 call, 타석 끝은 result로
            events.append(RelayEvent(id=f"v{n}c", t=int(j.t), kind="call", text=text,
                                     detail={"call": code, **_check_extra(j)}, **base))
            events.append(RelayEvent(id=f"v{n}", t=int(j.t), kind="result", text=text,
                                     detail={"result": DIVE_RESULT[code], **delta}, **base))
        elif kind == "call" and code == "infield_fly":
            events.append(RelayEvent(id=f"v{n}c", t=int(j.t), kind="call", text=text,
                                     detail={"call": code, **_check_extra(j)}, **base))
            events.append(RelayEvent(id=f"v{n}", t=int(j.t), kind="result", text=text,
                                     detail={"result": "infield_fly", **delta}, **base))
        elif kind == "call":
            events.append(RelayEvent(id=f"v{n}", t=int(j.t), kind="call", text=text,
                                     detail={"call": code, **delta, **_call_extra(j)}, **base))
        elif kind == "steal":
            extra = {"caught": True} if j.detail.get("caught") else {"base": j.detail.get("base")}
            events.append(RelayEvent(id=f"v{n}", t=int(j.t), kind="steal", text=text,
                                     detail={**extra, **delta}, **base))
        else:
            events.append(RelayEvent(id=f"v{n}", t=int(j.t), kind="result", text=text,
                                     detail={"result": code, **delta}, **base))
    return events
