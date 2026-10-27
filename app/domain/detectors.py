"""특이 상황 감지. 순수 로직(I/O 없음).

규칙은 **데이터 테이블**(`RULES`)이다. 새 상황을 추가할 때 if문 사슬을 늘리지 않고
행 하나를 붙인다. 각 행은 "무엇을 보고" "왜 설명할 만한지"를 함께 들고 있어서,
감지 결과(`Situation.reasons`)가 그대로 근거가 된다 (core-belief 4).

여기서는 **누구에게 보여줄지 판단하지 않는다**. 그건 profile.py의 몫이다.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Optional

from app.domain.balk_judge import TYPE_PLAIN
from app.domain.game_state import KIND_SUB, states_by_event
from app.domain.models import (
    CATEGORY_BASIC,
    CATEGORY_CULTURE,
    CATEGORY_PITCHING,
    CATEGORY_TACTICS,
    KIND_ATBAT,
    KIND_CALL,
    KIND_PITCH,
    KIND_RESULT,
    KIND_STEAL,
    GameFeed,
    GameState,
    RelayEvent,
    Situation,
)

Predicate = Callable[[RelayEvent, GameState], bool]
Explainer = Callable[[RelayEvent, GameState], str]
OnceKey = Callable[[RelayEvent, GameState], str]

# 떨어지거나 휘는 공. 직구는 제외 — "구종 설명"의 대상이 아니다.
OFFSPEED_PITCHES = frozenset({"포크볼", "슬라이더", "체인지업", "커브", "스플리터", "너클볼"})


@dataclass(frozen=True)
class Rule:
    id: str
    term_id: str  # 용어 사전(S5) 딥링크 키
    category: str
    label: str
    priority: float
    matches: Predicate
    why: Explainer
    # 있으면 같은 키로는 한 경기에 한 번만 감지한다. 볼넷처럼 반복되는 상황이
    # 카드를 도배하지 않게 — 입문자에게 필요한 건 첫 설명 한 번이다.
    once: Optional[OnceKey] = None


# ── 술어 조합기 ─────────────────────────────────────────────────────────
def _call_is(name: str) -> Predicate:
    def _p(event: RelayEvent, _state: GameState) -> bool:
        return event.kind == KIND_CALL and event.detail.get("call") == name

    return _p


def _result_is(*names: str) -> Predicate:
    wanted = frozenset(names)

    def _p(event: RelayEvent, _state: GameState) -> bool:
        return event.kind == KIND_RESULT and event.detail.get("result") in wanted

    return _p


def _is_decisive_offspeed(event: RelayEvent, _state: GameState) -> bool:
    """타석을 끝낸 변화구 한 개. 매 투구마다 카드를 띄우지 않기 위한 좁은 조건."""
    if event.kind != KIND_PITCH:
        return False
    detail = event.detail
    return bool(detail.get("decisive")) and detail.get("pitch_type") in OFFSPEED_PITCHES


def _is_team_first_atbat(event: RelayEvent, _state: GameState) -> bool:
    """1회 타석 시작. 팀별 첫 번째만 남기는 건 `once`가 맡는다."""
    return event.kind == KIND_ATBAT and event.inning == 1


def _once_per_game(rule_id: str) -> OnceKey:
    return lambda _e, _s: rule_id


def _is_pitching_change(event: RelayEvent, _state: GameState) -> bool:
    return event.kind == KIND_SUB and event.detail.get("sub_type") == "pitcher"


# 영상에서 알아본 심판 선언(video_judge.CALL_TABLE) → 카드. (코드, 용어 키, 이름, 왜 설명할 만한가)
CALL_CARDS = (
    ("catcher_interference", "catcher_interference", "타격방해",
     "포수 미트가 배트에 닿아 타자가 그냥 1루로 나간다 — 결과가 아니라 반칙으로 출루한 장면이다"),
    ("obstruction", "obstruction", "주루방해",
     "공을 갖지 않은 수비수가 주자의 길을 막아 주자가 베이스를 받는 장면이다"),
    ("interference", "interference", "수비방해",
     "공격 쪽 선수가 수비를 방해해 아웃이 선언되는 장면이다"),
    ("video_review", "video_review", "비디오 판독",
     "감독이 판정에 이의를 걸어 화면으로 다시 본다 — 판정이 바뀔 수 있다"),
    ("check_swing", "check_swing", "체크스윙",
     "방망이를 멈췄는지 돌렸는지로 볼과 스트라이크가 갈린 장면이다"),
    ("ejection", "ejection", "퇴장",
     "심판이 선수·감독을 경기에서 내보낸 장면이다"),
    ("bench_clearing", "bench_clearing", "벤치클리어링",
     "양 팀 선수들이 그라운드로 몰려나온 장면 — 왜 일어나는지 맥락이 필요하다"),
    ("ground_rule_double", "ground_rule_double", "인정 2루타",
     "타구가 담장 밖으로 튀어 나가 주자와 타자가 정해진 두 베이스만 간다"),
    ("foreign_substance", "foreign_substance", "이물질",
     "투수가 공에 이물질을 썼는지 검사·제재한 장면이다"),
    ("pitch_clock", "pitch_clock", "피치클락 위반",
     "정해진 시간 안에 준비하지 않아 공을 안 던졌는데 볼·스트라이크가 올라간다"),
    ("squeeze", "squeeze", "스퀴즈",
     "번트와 동시에 3루 주자가 홈으로 뛰어 점수를 짜내는 작전이다"),
    ("missed_base", "missed_base", "누의 공과",
     "베이스를 밟지 않고 지나가 어필로 아웃될 수 있는 장면이다"),
    ("home_collision", "home_collision", "홈 충돌",
     "홈에서 주자와 포수가 부딪쳐 충돌 방지 규정을 따진다"),
    ("slide_rule", "slide_rule", "슬라이딩 규정",
     "병살을 막으려 수비수에게 슬라이딩해 규정 위반을 따진다"),
    ("runner_lane", "runner_lane", "3피트 라인",
     "타자주자가 1루 가는 길 안쪽으로 뛰어 송구를 방해했는지 따진다"),
    ("fan_interference", "fan_interference", "관중 방해",
     "관중이 경기 중인 공을 건드려 심판이 결과를 정한다"),
    ("intentional_drop", "intentional_drop", "고의낙구",
     "수비수가 병살을 노려 일부러 공을 떨어뜨렸다고 본 장면이다"),
    ("passing_runner", "passing_runner", "주자 추월",
     "뒤 주자가 앞 주자를 앞질러 아웃되는 드문 장면이다"),
    ("called_game", "called_game", "콜드게임·노게임",
     "경기가 9회를 다 하지 않고 끝나거나 무효가 된다"),
)


def _balk_why(event: RelayEvent, state: GameState) -> str:
    """입문자는 '보크'라는 말만으로 어떤 동작이 반칙인지 모른다 — 해설이 설명한 동작을 붙인다."""
    why = f"심판이 보크를 선언했고, {state.fielding_team} 투수의 반칙으로 주자가 그냥 진루했다"
    plain = TYPE_PLAIN.get(event.detail.get("balk_kind") or "")
    return f"{why} — 반칙 동작: {plain}" if plain else why


def _call_card(code: str, term: str, label: str, why: str) -> "Rule":
    return Rule(id=code, term_id=term, category=CATEGORY_BASIC, label=label, priority=0.9,
                matches=_call_is(code), why=lambda e, s, w=why: w)


# ── 규칙 테이블 ─────────────────────────────────────────────────────────
RULES: Sequence[Rule] = (
    Rule(
        id="balk",
        term_id="balk",
        category=CATEGORY_BASIC,
        label="보크",
        priority=1.0,
        matches=_call_is("balk"),
        why=_balk_why,
    ),
    Rule(
        id="dropped_third_strike",
        term_id="dropped_third_strike",
        category=CATEGORY_BASIC,
        label="낫아웃",
        priority=1.0,
        matches=_call_is("dropped_third_strike"),
        why=lambda e, s: "삼진인데 타자가 아웃되지 않는, 입문자가 가장 헷갈리는 장면이다",
    ),
    Rule(
        id="infield_fly",
        term_id="infield_fly",
        category=CATEGORY_BASIC,
        label="인필드플라이",
        priority=1.0,
        matches=_call_is("infield_fly"),
        why=lambda e, s: "공을 잡지 않았는데 타자가 아웃되는 선언이라 설명 없이는 이해할 수 없다",
    ),
    Rule(
        id="sac_fly",
        term_id="sacrifice_fly",
        category=CATEGORY_BASIC,
        label="희생플라이",
        priority=0.8,
        matches=_result_is("sac_fly"),
        why=lambda e, s: "타자가 아웃됐는데 점수가 올라가는 상황이다",
    ),
    Rule(
        id="walk",
        term_id="walk",
        category=CATEGORY_BASIC,
        label="볼넷",
        priority=0.5,
        matches=_result_is("walk", "intentional_walk"),
        why=lambda e, s: "공을 치지 않았는데 타자가 1루로 걸어 나가는 장면이다",
        once=_once_per_game("walk"),
    ),
    Rule(
        id="hit_by_pitch",
        term_id="hit_by_pitch",
        category=CATEGORY_BASIC,
        label="몸에 맞는 공",
        priority=0.7,
        matches=_result_is("hbp"),
        why=lambda e, s: "투구가 타자 몸에 맞아 타자가 그냥 1루로 나갔다",
        once=_once_per_game("hit_by_pitch"),
    ),
    Rule(
        id="homerun",
        term_id="homerun",
        category=CATEGORY_BASIC,
        label="홈런",
        priority=0.7,
        matches=_result_is("homerun"),
        why=lambda e, s: f"{s.batting_team}가 홈런으로 득점했다",
    ),
    Rule(
        id="double_play",
        term_id="double_play",
        category=CATEGORY_TACTICS,
        label="병살타",
        priority=0.6,
        matches=_result_is("double_play"),
        why=lambda e, s: "한 번의 타구로 아웃 두 개가 나와 공수 흐름이 크게 바뀌었다",
    ),
    Rule(
        id="steal",
        term_id="steal",
        category=CATEGORY_TACTICS,
        label="도루",
        priority=0.6,
        matches=lambda e, s: e.kind == KIND_STEAL and not e.detail.get("caught"),
        why=lambda e, s: "주자가 투수·포수의 빈틈을 노려 다음 베이스를 훔쳤다",
    ),
    Rule(
        id="decisive_offspeed",
        term_id="offspeed_pitch",
        category=CATEGORY_PITCHING,
        label="결정구 변화구",
        priority=0.9,
        matches=_is_decisive_offspeed,
        why=lambda e, s: (
            f"{e.detail.get('pitch_type')} {e.detail.get('speed')}km/h로 타석을 끝냈다 — "
            "구종 차이를 체감할 수 있는 장면이다"
        ),
        # 실경기엔 삼진 결정구가 한 경기 수십 개다. 구종마다 첫 장면만 설명한다.
        once=lambda e, s: f"offspeed:{e.detail.get('pitch_type')}",
    ),
    Rule(
        id="pitching_change",
        term_id="pitching_change",
        category=CATEGORY_TACTICS,
        label="투수 교체",
        priority=0.4,
        matches=_is_pitching_change,
        # 실경기엔 한 경기 교체가 6~11번이다. 팀마다 첫 교체에서 불펜 운용을 한 번 설명한다.
        once=lambda e, s: f"pitching_change:{s.fielding_team}",
        why=lambda e, s: "감독이 승부처로 판단해 투수를 바꿨다",
    ),
    Rule(
        id="bases_loaded",
        term_id="bases_loaded",
        category=CATEGORY_TACTICS,
        label="만루",
        priority=0.5,
        matches=lambda e, s: e.kind == KIND_RESULT and s.bases == (True, True, True),
        # 만루가 이어지는 동안 타석마다 뜨지 않게 — 반 이닝에 한 번.
        once=lambda e, s: f"bases_loaded:{s.inning}{s.half}",
        why=lambda e, s: "루상이 꽉 차 한 방이면 점수가 크게 움직이는 상황이다",
    ),
    Rule(
        id="cheer_song",
        term_id="cheer_song",
        category=CATEGORY_CULTURE,
        label="선수별 응원가",
        priority=0.6,
        matches=_is_team_first_atbat,
        why=lambda e, s: f"{s.batting_team}의 첫 공격 — 응원석이 타자마다 응원가를 부르기 시작한다",
        once=lambda e, s: f"cheer_song:{e.half}",
    ),
    Rule(
        id="homerun_cheer",
        term_id="homerun_celebration",
        category=CATEGORY_CULTURE,
        label="홈런 응원",
        priority=0.6,
        matches=_result_is("homerun"),
        why=lambda e, s: f"{s.batting_team} 응원석이 홈런 세리머니로 가장 크게 들썩이는 순간이다",
        # 홈런 자체는 매번 카드가 뜬다. 응원 문화 설명은 팀마다 첫 홈런에서 한 번.
        once=lambda e, s: f"homerun_cheer:{s.batting_team}",
    ),
    *(_call_card(*row) for row in CALL_CARDS),
)

RULES_BY_ID = {rule.id: rule for rule in RULES}


def detect(feed: GameFeed, until_t: Optional[int] = None) -> list[Situation]:
    """경기 전체(또는 `until_t`까지)에서 설명할 만한 상황을 모두 찾는다.

    프로필과 무관하다 — 여기서 거르지 않고 profile.select()가 거른다. 그래야
    같은 감지 결과를 여러 사용자에게 재사용할 수 있고, 배치 생성이 한 번으로 끝난다.
    """
    events = feed.events
    states = states_by_event(events, feed.meta.away_team, feed.meta.home_team)

    situations: list[Situation] = []
    seen_once: set[str] = set()
    for event, state in zip(events, states):
        if until_t is not None and event.t > until_t:
            break
        for rule in RULES:
            if not rule.matches(event, state):
                continue
            if rule.once is not None:
                key = rule.once(event, state)
                if key in seen_once:
                    continue
                seen_once.add(key)
            situations.append(
                Situation(
                    id=f"{feed.meta.id}:{event.id}:{rule.id}",
                    t=event.t,
                    rule_id=rule.id,
                    term_id=rule.term_id,
                    category=rule.category,
                    label=rule.label,
                    trigger_text=event.text,
                    event_ids=[event.id],
                    state=state,
                    priority=rule.priority,
                    reasons=[
                        f"감지: {rule.label} (규칙 {rule.id}, "
                        f"{'영상 판정' if event.detail.get('source') == 'video' else '중계'} "
                        f"{event.id})",
                        f"근거: {rule.why(event, state)}",
                        f"경기 상황: {state.scoreboard_text()} · {state.runners_text()}",
                    ] + [
                        # 영상 판정이면 무엇을 보고 판정했는지(점수판 변화·장면·해설)를 남긴다
                        f"영상 근거: {line}" for line in event.detail.get("evidence", [])
                    ],
                )
            )
    return situations
