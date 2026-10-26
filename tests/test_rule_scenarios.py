# ruff: noqa: E501 — 장면 표라 한 줄에 한 장면
"""규칙 → 판정 재검수: 카탈로그에서 영상 판정 코드가 있는 규칙마다 규칙대로의 가상 장면을 만들어
엔진이 그 코드를 내는지, 성립 조건이 안 맞으면 내지 않는지 확인한다.

새 규칙에 영상 코드를 달면 여기 장면이 없다는 이유로 테스트가 실패한다.
"""

import json
from pathlib import Path

import pytest

from app.domain.scorebug import Reading, stabilize, transitions
from app.domain.video_judge import (
    CALL_TABLE,
    Cue,
    judge,
    judge_calls,
    judge_from_cues,
    judge_transition,
)

ROOT = Path(__file__).resolve().parents[1]
E = (False, False, False)
B1, B2, B3 = (True, False, False), (False, True, False), (False, False, True)
B12, B13, B123 = (True, True, False), (True, False, True), (True, True, True)


def R(t, outs=0, bases=E, balls=0, strikes=0, away=0, home=0, inning=3, half="top"):
    return Reading(float(t), inning, half, balls, strikes, outs, bases, away, home)


def run(before, after, cues=()):
    """두 점수판(각각 두 번 읽힘) 사이의 판정 코드."""
    readings = []
    for r in (before, after):
        readings += [r, Reading(r.t + 0.5, *r.key())]
    trs = transitions(stabilize(readings))
    assert len(trs) == 1
    j = judge_transition(trs[0], cues)
    return j.code if j is not None and j.confidence >= 0.5 else None


def say(t, text):
    return Cue(float(t), "speech", text)


def ev(t, kind):
    return Cue(float(t), kind)


# (카탈로그 id들, 직전 점수판, 직후 점수판, 단서, 기대 코드) — 규칙 본문대로의 장면
SCENARIOS = [
    (("strikeout",), R(0, strikes=2), R(10, outs=1), [ev(9, "swing_miss")], "strikeout"),
    (("walk_basic",), R(0, balls=3), R(10, bases=B1), [], "walk"),
    (("auto_ibb",), R(0, bases=B2), R(10, bases=B12), [say(8, "자동 고의4구로 내보냅니다")], "walk"),
    (("homerun_basic", "b2b_hr"), R(0, bases=B1), R(10, away=2), [ev(8, "ball_over_fence")], "homerun"),
    (("grand_slam",), R(0, bases=B123), R(10, away=4), [say(9, "만루 홈런")], "homerun"),
    (("dts_reach",), R(0, strikes=2), R(10, bases=B1), [ev(8, "catcher_miss")], "dropped_third_strike"),
    (("hbp_basic",), R(0), R(10, bases=B1), [ev(8, "hit_by_pitch")], "hbp"),
    (("headshot",), R(0), R(10, bases=B1), [say(9, "헤드샷입니다, 퇴장")], "hbp"),
    (("steal_basic",), R(0, bases=B1, balls=1), R(10, bases=B2, balls=1), [ev(9, "slide")], "steal"),
    (("home_steal",), R(0, bases=B3, balls=1), R(10, away=1, balls=1), [ev(9, "slide")], "steal"),
    (("caught_stealing",), R(0, bases=B1), R(10, outs=1), [ev(9, "slide")], "caught_stealing"),
    (("pickoff_out",), R(0, bases=B1), R(10, outs=1), [say(9, "견제에 걸렸어요")], "caught_stealing"),
    (("balk_advance", "disengage", "balk_1_stop", "balk_2_fake1b", "balk_2_fake3b", "balk_3_step",
      "balk_4_empty", "balk_5_illegal", "balk_6_not_facing", "balk_7_off_rubber", "balk_8_delay",
      "balk_9_no_ball", "balk_10_hands", "balk_11_drop", "balk_12_catcher", "balk_13_no_stop"),
     R(0, bases=B13, balls=1), R(10, bases=B2, away=1, balls=1), [say(9, "보크가 선언됩니다")], "balk"),
    (("wp_advance",), R(0, bases=B1, balls=1), R(10, bases=B2, balls=2), [say(9, "폭투")], "wild_pitch"),
    (("pb_advance",), R(0, bases=B1, balls=1), R(10, bases=B2, balls=2), [say(9, "포일이네요")], "passed_ball"),
    (("double_play_r",), R(0, bases=B1), R(10, outs=2), [say(9, "병살")], "double_play"),
    (("line_drive_dp",), R(0, bases=B1), R(10, outs=2), [say(9, "직선타 병살")], "double_play"),
    (("triple_play_r", "unassisted_tp"), R(0, bases=B12), R(10, outs=3, half="bot"), [say(9, "삼중살!")], "triple_play"),
    (("infield_fly_r", "infield_fly_drop"), R(0, bases=B12), R(10, outs=1, bases=B12), [ev(8, "infield_popup")], "infield_fly"),
    (("sac_fly_r", "tag_up_score"), R(0, bases=B3), R(10, outs=1, away=1), [ev(8, "deep_fly")], "sac_fly"),
    (("walkoff_walk",), R(0, bases=B123, balls=3, inning=9, half="bot"),
     R(10, bases=B123, home=1, inning=9, half="bot"), [ev(9, "ball_four")], "walk"),
    (("walkoff_sf",), R(0, bases=B3, inning=10, half="bot"), R(10, outs=1, home=1, inning=10, half="bot"),
     [ev(8, "deep_fly")], "sac_fly"),
    (("walkoff_wp",), R(0, bases=B3, inning=9, half="bot"), R(10, home=1, inning=9, half="bot"),
     [say(9, "폭투로 끝납니다")], "wild_pitch"),
    (("balk_walkoff",), R(0, bases=B3, inning=9, half="bot"), R(10, home=1, inning=9, half="bot"),
     [say(9, "끝내기 보크")], "balk"),
]

# 성립 조건이 안 맞으면 그 판정을 내지 않는다 (카탈로그 규칙 본문의 전제)
NOT_THIS = [
    # 5.09(a)(5) 인필드 플라이는 1·2루 또는 만루 + 2아웃 전
    ("infield_fly_r", R(0, bases=B1), R(10, outs=1, bases=B1), [ev(8, "infield_popup")], "infield_fly"),
    ("infield_fly_r", R(0, outs=2, bases=B12), R(10, half="bot"), [ev(8, "infield_popup")], "infield_fly"),
    # 9.08(d) 희생플라이는 2아웃 전
    ("sac_fly_r", R(0, outs=2, bases=B3), R(10, half="bot", away=1), [ev(8, "deep_fly")], "sac_fly"),
    # 5.09(a)(3) 1루 주자 있고 2아웃 전이면 낫아웃 불성립
    ("dts_out", R(0, strikes=2, bases=B1), R(10, outs=1, bases=B1), [ev(8, "catcher_miss")], "dropped_third_strike"),
    # 고의사구(四球)는 몸에 맞는 공(死球)이 아니다
    ("auto_ibb", R(0, bases=B2), R(10, bases=B12), [say(8, "고의사구로 1루를 채웁니다")], "hbp"),
    # 삼진 + 도루 실패는 병살타가 아니다
    ("strikeout", R(0, bases=B1, strikes=2), R(10, outs=2), [ev(9, "swing_miss")], "double_play"),
    # 아웃 3개는 판독 누락으로도 생긴다 — 단서 없이 삼중살 아님
    ("triple_play_r", R(0, bases=B12), R(10, outs=3, half="bot"), [], "triple_play"),
    # 6.02(a) 보크는 투구가 없다 — 카운트가 바뀌고 해설도 없으면 보크 아님
    ("balk_advance", R(0, bases=B1, balls=1), R(10, bases=B2, balls=2), [ev(9, "pitcher_stops")], "balk"),
    # 폭투 해설이 있으면 도루로 내지 않는다
    ("steal_basic", R(0, bases=B1, balls=1), R(10, bases=B2, balls=2), [ev(9, "slide"), say(9, "폭투")], "steal"),
    # 9.07 홈스틸 — 포수가 놓친 장면(폭투·포일) 뒤 득점은 홈스틸이 아니다
    ("home_steal", R(0, bases=B3, balls=1), R(10, away=1, balls=2), [ev(9, "slide"), say(9, "포일")], "steal"),
    # 5.05(a)(2) 낫아웃은 3스트라이크를 놓친 것 — 타구 직후 포수가 놓친 건 수비 플레이
    ("dts_out", R(0, strikes=2, outs=1), R(10, outs=2), [ev(1, "contact"), ev(3, "catcher_miss")], "dropped_third_strike"),
    # 광고 자막 '홈런스타'는 홈런이 아니다
    ("homerun_basic", R(0), R(10, outs=1), [say(8, "[자막] 크보 홈런스타를 찾아라!")], "homerun"),
    # 병살은 한 플레이 — 아웃+득점+남은 주자가 직전 주자+타자보다 많으면 여러 플레이가 합쳐진 것
    ("double_play_r", R(0, bases=B12), R(130, outs=2, bases=B12, away=3), [], "double_play"),
    # 홈런은 득점 = 주자 수 + 1
    ("homerun_basic", R(0, bases=B1), R(10, away=1, bases=B3), [say(8, "홈런성 타구")], "homerun"),
]


def _rows():
    return json.loads((ROOT / "data" / "rules_catalog.json").read_text(encoding="utf-8"))["rules"]


@pytest.mark.parametrize("ids,before,after,cues,code", SCENARIOS,
                         ids=[s[0][0] for s in SCENARIOS])
def test_rule_scenario_gives_its_code(ids, before, after, cues, code):
    by_id = {r["id"]: r for r in _rows()}
    for rid in ids:
        assert by_id[rid]["video"] == code, f"{rid}: 카탈로그 영상 코드와 장면 기대 코드가 다르다"
    assert run(before, after, cues) == code


@pytest.mark.parametrize("rid,before,after,cues,code", NOT_THIS, ids=[n[0] for n in NOT_THIS])
def test_rule_precondition_blocks_the_call(rid, before, after, cues, code):
    assert run(before, after, cues) != code


def test_scene_double_play_with_triple_play_commentary_is_a_triple_play():
    """장면이 아웃 하나를 놓쳐도 해설 '트리플 플레이'는 병살을 품는 더 구체적인 판정이다."""
    cues = [ev(34, "contact"), ev(36, "out_at_base"), ev(40, "out_at_base"),
            say(45, "트리플 플레이를 처음 봤다고요?")]
    assert judge_from_cues(cues).code == "triple_play"
    assert judge_from_cues(cues[:3]).code == "double_play"


def test_more_than_three_outs_in_a_burst_is_not_a_triple_play():
    """반 이닝 아웃은 셋뿐(5.09(e)) — 15초에 아웃 여섯은 장면을 되풀이해 지어낸 관찰이다."""
    assert judge_from_cues([ev(140 + 5 * i, "out_at_base") for i in range(6)]) is None
    assert judge_from_cues([ev(140 + 5 * i, "out_at_base") for i in range(3)]).code == "triple_play"


def test_two_outs_without_a_batted_ball_are_not_a_double_play():
    """견제사 두 번(타구 없음)은 병살이 아니다 — 병살은 한 플레이에 아웃 둘."""
    pickoffs = [ev(20, "slide"), ev(40, "out_at_base"), ev(48, "out_at_base"), say(50, "견제로 잡아 냅니다")]
    assert judge_from_cues(pickoffs).code == "caught_stealing"
    batted = [ev(34, "contact"), ev(36, "out_at_base"), ev(40, "out_at_base")]
    assert judge_from_cues(batted).code == "double_play"


def test_juggled_foul_tip_is_a_strikeout_not_a_dropped_third_strike():
    """포수가 저글링해 '놓친' 것처럼 보여도 해설이 파울팁이면 잡은 것이다."""
    cues = [ev(1, "swing_miss"), ev(2, "catcher_miss"), say(3, "끝내기 파울TV입니다")]
    assert judge_from_cues(cues).code == "strikeout"
    assert judge_from_cues(cues[:2]).code == "dropped_third_strike"


def test_substitution_rules_come_from_cues():
    readings = [R(0), R(0.5), R(30, half="bot"), R(30.5, half="bot")]
    out = judge(readings, [ev(28, "pitching_change"), say(200, "투수 교체입니다")])
    assert [(j.kind, j.code) for j in out if j.kind == "sub"] == [("sub", "pitcher")] * 2


# 심판 선언 — (코드, 해설 한 마디, 확인 장면). 확인이 있으면 낸다
CALL_SCENARIOS = [
    ("catcher_interference", "포수 미트에 배트가 맞았네요, 타격방해입니다", "umpire_signal"),
    ("obstruction", "주루방해가 선언됩니다", "umpire_signal"),
    ("interference", "주자의 수비방해로 아웃입니다", "umpire_signal"),
    ("video_review", "비디오 판독 결과 원심 유지", "replay_review"),
    ("check_swing", "체크스윙 돌았다는 판정이에요", "umpire_signal"),
    ("ejection", "감독이 퇴장 명령을 받습니다", "umpire_signal"),
    ("bench_clearing", "양 팀 벤치클리어링이 일어났어요", "umpire_signal"),
    ("ground_rule_double", "펜스를 넘어가 인정 2루타", "ball_over_fence"),
    ("foreign_substance", "글러브에서 이물질이 발견됐습니다", "umpire_signal"),
    ("pitch_clock", "피치클락 위반으로 볼이 하나 올라갑니다", "umpire_signal"),
    ("squeeze", "스퀴즈! 3루 주자 들어옵니다", "runner_scores"),
    ("missed_base", "누의 공과로 어필 아웃", "umpire_signal"),
    ("home_collision", "홈 충돌 방지 규정 위반 여부를 봅니다", "replay_review"),
    ("slide_rule", "슬라이딩 규정 위반으로 타자까지 아웃", "replay_review"),
    ("runner_lane", "3피트 라인 위반이 선언됐어요", "umpire_signal"),
    ("fan_interference", "관중 방해가 인정됩니다", "replay_review"),
    ("intentional_drop", "고의낙구가 선언됩니다", "umpire_signal"),
    ("passing_runner", "앞 주자를 추월해서 아웃이에요", "umpire_signal"),
    ("called_game", "결국 강우 콜드게임으로 끝납니다", "umpire_signal"),
]


@pytest.mark.parametrize("code,said,seen", CALL_SCENARIOS, ids=[c[0] for c in CALL_SCENARIOS])
def test_umpire_call_needs_its_word_and_a_confirmation(code, said, seen):
    found = judge_calls([say(100, said), ev(105, seen)])
    assert [(j.code, j.confidence) for j in found] == [(code, 0.75)]
    assert judge_calls([say(100, said)]) == []  # 말만 한 번 — 일어났다는 확인이 없다
    assert [j.code for j in judge_calls([say(100, "[자막] " + said)])] == [code]


def test_umpire_call_table_and_scenarios_match():
    assert {c[0] for c in CALL_SCENARIOS} == {code for code, _w, _k in CALL_TABLE}


def test_common_phrases_are_not_umpire_calls():
    """'박수 받으며 퇴장', 원바운드 '블로킹', '스퀴즈 가능성'(득점 없음)은 판정이 아니다."""
    cues = [say(10, "박수를 받으며 퇴장합니다"), say(30, "포수 블로킹 좋네요"), ev(31, "umpire_signal"),
            say(200, "스퀴즈 작전 가능성도 있죠"), ev(205, "umpire_signal")]
    assert judge_calls(cues) == []


def test_ejection_described_after_the_fact():
    """'퇴장이 됐습니다'(실측 ABS 항의 클립) — 지난 일을 말해도 심판 동작이 근처면 퇴장이다."""
    cues = [say(60, "헬멧을 집어던져버렸거든요. 퇴장 사유가 되면서 퇴장이 됐습니다"), ev(75, "umpire_signal")]
    assert [j.code for j in judge_calls(cues)] == ["ejection"]


def test_repeated_commentary_confirms_a_call_with_low_confidence():
    found = judge_calls([say(100, "주루방해 아닌가요"), say(130, "네, 주루방해가 맞습니다")])
    assert [(j.code, j.confidence) for j in found] == [("obstruction", 0.5)]


def test_every_rule_with_a_video_code_has_a_scenario():
    covered = {rid for s in SCENARIOS for rid in s[0]}
    called = {c[0] for c in CALL_SCENARIOS}
    covered |= {r["id"] for r in _rows() if r["video"] in called}
    subs = {"pitching_change_r", "position_player_pitch"}  # 단서로 판정 — 위 교체 테스트
    need = {r["id"] for r in _rows() if r["video"]}
    assert need - covered - subs == set()
