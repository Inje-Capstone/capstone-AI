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

# 단서 종류 (VLM 프롬프트 v2와 같은 어휘)
CUE_KINDS = (
    "pitch", "swing_miss", "contact", "slide", "pickoff_throw", "pitcher_stops",
    "catcher_miss", "infield_popup", "deep_fly", "ball_over_fence", "hit_by_pitch",
    "pitching_change", "mound_visit", "replay_review", "celebration", "crowd_cheer",
    "strikeout", "ball_four", "out_at_base", "runner_scores", "fielder_drops_ball",
    "umpire_signal",
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


# 같은 글자의 다른 말 — '사구(死球)'는 몸에 맞는 공인데 '고의사구(四球)'는 고의4구다
_HOMONYMS = {"사구": ("고의사구", "고의 사구")}


def _says(text: str, word: str) -> bool:
    for other in _HOMONYMS.get(word, ()):
        text = text.replace(other, "")
    return word in text


def _speech(cues: Sequence[Cue], *words: str) -> Optional[Cue]:
    wanted = [w.lower() for w in words]
    for w in words:
        wanted += [e for k, en in _EN_WORDS.items() if k in w for e in en]
    for c in cues:
        if c.kind == "speech" and any(_says(c.text.lower(), w) for w in wanted):
            return c
    return None


def _ev(c: Cue) -> str:
    return f"해설 '{c.text}' ({c.t:.0f}s)" if c.kind == "speech" else f"장면 {c.kind} ({c.t:.0f}s)"


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
            return make("steal", "steal", 0.5, [steal_cue, _speech(near, "도루")],
                        base=single_runner_advance(b, a))
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
        dts = _has(near, "catcher_miss") or _speech(near, "낫아웃", "낫 아웃")
        # 낫아웃은 1루가 비었거나 2아웃일 때만(5.09(a)(3)). 2스트라이크가 보였으면 확신도를 더 준다
        if dts and dts_possible(tr.before):
            prior = 0.5 if tr.before.strikes == 2 else 0.3
            return make("call", "dropped_third_strike", prior,
                        [dts, _speech(near, "낫아웃", "낫 아웃")])
        walk_speech = _speech(near, *WALK_WORDS) or _has(near, "ball_four")
        if tr.before.balls == 3 and not _has(near, "contact"):
            return make("result", "walk", 0.6, [walk_speech])
        if walk_speech:
            return make("result", "walk", 0.5, [walk_speech])
    if tr.outs_made == 1 and tr.before.strikes == 2:
        dts = _has(near, "catcher_miss") or _speech(near, "낫아웃", "낫 아웃")
        if dts and dts_possible(tr.before):  # 낫아웃 상황이었지만 1루에서 아웃
            return make("call", "dropped_third_strike", 0.5, [dts], out=True)
        k = _has(near, "swing_miss") or _has(near, "strikeout") or _speech(near, "삼진")
        if k:
            return make("result", "strikeout", 0.5, [k])
    # 나머지 타석 결과는 종류를 모른다 — 상태는 반영하되 카드는 띄우지 않는다
    if tr.count_reset or tr.half_change or tr.outs_made or tr.runs:
        return make("result", "out" if tr.outs_made else "reach", 0.5)
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
    for tr in transitions(stable):
        j = judge_transition(tr, cues, window)
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
        contact = any(c.kind == "contact" and score.t - c.t <= 10 for c in before)
        missed = any(c.kind in ("catcher_miss", "fielder_drops_ball") for c in before)
        if slide and not contact and not missed:  # 포수가 놓쳤으면 폭투·포일 득점
            return make("steal", "steal", score, "타격 없이 슬라이딩 득점(홈스틸)")
    if len(outs) == 2:
        return make("double_play", "result", outs[0], "베이스 아웃 2번이 15초 안에")
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
    return found or _keyword_judgment([c for c in cues if c not in captions], 0.5, "해설·설명")


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
    return readings, sorted(cues, key=lambda c: c.t)


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
