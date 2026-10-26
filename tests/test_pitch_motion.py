"""투수 동작 분석 — 합성 관절 좌표로 정지·정지 없음·이중 동작 갈래를 확인한다."""

from app.adapters.video.pose import pick_pitcher
from app.domain.pitch_motion import PoseFrame, analyze_motions

FPS = 10


def _pose(t, arm=0.0, knee_up=0.0):
    """몸통 길이 100px 투수. `arm`만큼 팔꿈치·어깨가 아래로 내려온다(세트 들어가기)."""
    p = [(0.0, 0.0, 0.0)] * 17
    p[0] = (200, 90, 0.9)
    p[5], p[6] = (180, 100 + arm * 0.2, 0.9), (220, 100 + arm * 0.2, 0.9)  # 어깨
    p[7], p[8] = (170, 140 + arm, 0.9), (230, 140 + arm, 0.9)  # 팔꿈치
    p[9], p[10] = (195, 160 + arm, 0.5), (205, 160 + arm, 0.5)  # 손목(가려져 신뢰도 낮음)
    p[11], p[12] = (185, 200, 0.9), (215, 200, 0.9)  # 엉덩이
    p[13], p[14] = (185, 260 - knee_up, 0.9), (215, 260, 0.9)  # 무릎
    p[15], p[16] = (185, 320, 0.9), (215, 320, 0.9)  # 발목
    return PoseFrame(t, tuple(p))


def _seq(*parts):
    """parts: (프레임 수, arm(k), knee(k)) — k는 구간 안 번호."""
    frames, t = [], 0.0
    for count, arm, knee in parts:
        for k in range(count):
            frames.append(_pose(round(t, 2), arm(k), knee(k)))
            t += 1 / FPS
    return frames


def still(n, at):
    return (n, lambda k: at, lambda k: 0)


def move(n, a, b):
    return (n, lambda k: a + (b - a) * (k + 1) / n, lambda k: 0)


def kick(n, at):
    return (n, lambda k: at, lambda k: 20 * (k + 1))


def test_sign_set_stop_then_leg_kick_is_legal():
    motions = analyze_motions(_seq(still(8, 0), move(4, 0, 40), still(8, 40), kick(5, 40)))
    assert [m.suspect for m in motions] == [None] and motions[0].stop_sec >= 0.3


def test_leg_kick_straight_out_of_the_set_motion_is_no_stop():
    motions = analyze_motions(_seq(still(8, 0), move(6, 0, 40), kick(5, 40)))
    assert [m.suspect for m in motions] == ["no_stop"]


def test_set_entered_in_two_steps_is_double_motion():
    frames = _seq(
        still(8, 0), move(3, 0, 20), still(2, 20), move(3, 20, 40), still(8, 40), kick(5, 40)
    )
    (m,) = analyze_motions(frames)
    assert m.suspect == "double_motion"


def test_pick_pitcher_prefers_big_lower_center_person_in_center_field_view():
    def person(cx, cy, h):
        return [(cx + (i % 3 - 1) * h * 0.1, cy + (i / 17 - 0.5) * h, 0.9) for i in range(17)]

    plate = [person(600, 150, 100), person(700, 160, 90), person(650, 170, 80)]
    pitcher = person(640, 450, 300)
    assert pick_pitcher(plate + [pitcher], 1280, 720) == 3
    assert pick_pitcher([pitcher], 1280, 720) is None  # 위쪽에 타자·포수가 없으면 다른 화면


def test_arms_drop_after_set_without_pitch_is_hands_separated():
    frames = _seq(still(8, 0), move(4, 0, 40), still(8, 40), move(4, 40, 0), still(30, 0))
    assert [m.suspect for m in analyze_motions(frames)] == ["hands_separated"]
