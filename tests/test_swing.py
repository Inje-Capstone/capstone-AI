"""타자 스윙 — 손 이동량과 투구 화면 찾기."""

import pytest

from app.domain.swing import (
    NO_SWING_MOVE,
    SWING_MOVE,
    hand_positions,
    measure,
    pitch_moment,
)

JOINTS = 17


def person(hand=(0.0, -40.0), score=0.9, hand_score=0.9, hip=(100.0, 100.0)):
    """키 100짜리 사람. 머리 y=50, 발 y=150. hand는 엉덩이 기준."""
    pts = [(hip[0], hip[1] - 50 + i * 100 / (JOINTS - 1), score) for i in range(JOINTS)]
    pts[11] = pts[12] = (*hip, score)
    pts[9] = pts[10] = (hip[0] + hand[0], hip[1] + hand[1], hand_score)
    return pts


def swing_frames(reach=60.0, count=30):
    """가운데 6프레임에 손이 reach만큼 움직인다."""
    out = [person() for _ in range(12)]
    out += [person(hand=(reach * i / 6, -40)) for i in range(1, 7)]
    out += [person(hand=(reach, -40)) for _ in range(count - 18)]
    return out


def test_fast_hands_are_a_swing():
    found = measure(swing_frames(reach=60))
    assert found.swing
    assert found.moved == pytest.approx(0.6, abs=0.03)
    assert 8 <= found.frame <= 16


def test_still_hands_are_not():
    found = measure([person() for _ in range(30)])
    assert not found.swing
    assert found.moved == 0


def test_between_the_two_thresholds_is_unclear():
    assert measure(swing_frames(reach=SWING_MOVE * 100 + 3)).swing
    middle = measure(swing_frames(reach=(SWING_MOVE + NO_SWING_MOVE) * 50))
    assert (middle.swing, middle.clear) == (False, False)
    still = measure(swing_frames(reach=NO_SWING_MOVE * 100 - 5))
    assert (still.swing, still.clear) == (False, True)


def test_lowering_the_bat_is_not_a_swing():
    frames = [person() for _ in range(12)]
    frames += [person(hand=(0, -40 + 70 * i / 6)) for i in range(1, 7)]
    frames += [person(hand=(0, 30)) for _ in range(12)]
    assert not measure(frames).swing


def test_body_walking_with_hands_is_not_a_swing():
    frames = [person(hip=(100.0 + i * 8, 100.0)) for i in range(30)]
    assert not measure(frames).swing


def test_one_blurred_hand_still_counts():
    frames = swing_frames(reach=60)
    for f in frames:
        f[9] = (f[9][0], f[9][1], 0.1)
    assert measure(frames).swing


def test_short_gap_is_filled_and_long_gap_is_not():
    frames = swing_frames(reach=60)
    for i in (13, 14, 15):
        frames[i][9] = frames[i][10] = (0.0, 0.0, 0.0)
    assert measure(frames).swing
    for i in range(6, 26):
        frames[i][9] = frames[i][10] = (0.0, 0.0, 0.0)
    assert not measure(frames).swing


def test_single_frame_joint_jump_is_ignored():
    frames = [person() for _ in range(30)]
    frames[15] = person(hand=(90, 40))
    assert not measure(frames).swing


def test_no_person_or_too_few_joints_gives_no_swing():
    assert measure([]).moved == 0
    faint = [person(score=0.1, hand_score=0.1) for _ in range(30)]
    assert hand_positions(faint) == [None] * 30
    assert measure(faint).frame is None


def test_hands_we_could_not_see_are_never_a_clear_take():
    assert not measure([]).clear
    assert not measure([person(hand_score=0.1) for _ in range(30)]).clear
    frames = [person() for _ in range(30)]
    for i in range(10, 20):  # 흐려서 놓친 구간 — 그 사이에 휘둘렀을 수 있다
        frames[i][9] = frames[i][10] = (0.0, 0.0, 0.0)
    found = measure(frames)
    assert (found.swing, found.clear) == (False, False)


def test_short_blur_in_a_still_shot_is_still_a_clear_take():
    frames = [person() for _ in range(30)]
    for i in (14, 15):
        frames[i][9] = frames[i][10] = (0.0, 0.0, 0.0)
    assert measure(frames).clear


def kicking(step=20.0):
    frames = [person() for _ in range(20)]
    for i in (9, 10, 11):
        frames[i][15] = (100.0, 150.0 - step * (3 - abs(i - 10)), 0.9)
    return frames


def test_pitch_moment_is_the_rear_foot_kick():
    found = pitch_moment(kicking())
    assert found.frame == 10
    assert found.lift > 0.4


def test_standing_pitcher_is_no_pitch():
    assert pitch_moment([person() for _ in range(20)]) is None
    assert pitch_moment([]) is None


def test_ankle_with_a_low_score_is_ignored():
    frames = kicking()
    for f in frames:
        f[15] = (f[15][0], f[15][1], 0.1)
    assert pitch_moment(frames) is None
