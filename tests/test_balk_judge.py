"""보크 판정 — 해설 + 동작 + 점수판 검증."""

from app.domain.balk_judge import judge_balk, speech_type
from app.domain.pitch_motion import PitchMotion
from app.domain.scorebug import Reading

E = (False, False, False)


def _motion(t, suspect):
    return PitchMotion(
        t_set=t - 1,
        stop_sec=0.0,
        t_move=t,
        outcome="throw",
        suspect=suspect,
        evidence=[f"동작 {t}s"],
    )


def _r(t, bases, outs=1):
    return Reading(float(t), 7, "top", 0, 0, outs, bases, 0, 0)


def test_speech_and_motion_together_are_confident_and_motion_gives_the_kind():
    calls = judge_balk([_motion(12, "no_stop")], [(20.0, "아 지금 보크 선언입니다")])
    assert [(c.kind, c.confidence) for c in calls] == [("no_stop", 0.8)]


def test_speech_only_uses_words_and_motion_only_stays_below_threshold():
    calls = judge_balk(
        [_motion(200, "double_motion")],
        [(30.0, "팔이 한 번 멈췄다가 들어가죠 그럼 피쳐 복구입니다")],
    )
    kinds = {(c.kind, c.confidence) for c in calls}
    assert ("double_motion", 0.5) in kinds and ("double_motion", 0.35) in kinds


def test_scoreboard_confirms_one_base_advance_or_doubts_no_advance():
    up = judge_balk(
        [], [(10.0, "보크!")], [_r(5, (True, False, True)), _r(30, (False, True, False))]
    )
    assert up[0].confidence == 0.65 and "한 베이스씩" in up[0].evidence[-1]
    stay = judge_balk(
        [], [(10.0, "보크!")], [_r(5, (True, False, False)), _r(30, (True, False, False))]
    )
    assert stay[0].confidence == 0.3


def test_speech_type_keywords():
    assert speech_type("왼발이 갑자기 뒤로 빠졌어요 발이 빠지면") == "foot_off"
    assert speech_type("정지 동작이 딱 보여야 하는데") == "no_stop"
    assert speech_type("그냥 보크네요") == "unknown"


def test_no_runner_on_scoreboard_means_no_balk():
    calls = judge_balk([_motion(12, "no_stop")], [(15.0, "보크 같은데요")], [_r(5, E), _r(30, E)])
    assert calls[0].confidence < 0.5 and "주자 없음" in calls[0].evidence[-1]


def test_commentary_describing_a_fake_pickoff_without_saying_balk():
    """'1루 쪽으로 던져야 되는데 던지지 않고' — 보크라는 말이 없어도 6.02(a)(2) 견제 시늉이다."""
    said = "1루쪽으로 공을 던져야 됩니다 던지지 않고 끄고 가는 모습이 보였잖아요"
    calls = judge_balk([], [(40.0, said)])
    assert [(c.kind, c.confidence) for c in calls] == [("fake_throw", 0.5)]
    assert judge_balk([], [(40.0, "2루 쪽으로 던지는 척만 했어요")]) == []  # 2루 시늉은 합법


def test_whisper_mishearing_of_balk():
    assert judge_balk([], [(80.0, "보쿠예요? 보쿠는 부정 투구 동작을 말하는 건데요")])
