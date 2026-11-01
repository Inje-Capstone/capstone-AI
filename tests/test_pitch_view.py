"""투구 화면 찾기 — 작은 그림 묶기와 포수 기준 역할 찾기."""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from app.adapters.video.pitch_view import (  # noqa: E402
    Shot,
    crop_square,
    find_roles,
    pitch_groups,
    split_shots,
)


def body(x, top, height, crouch=False):
    """사람 하나. 어깨·엉덩이·무릎 높이로 웅크림을 표현한다."""
    shoulder, hip = top + height * 0.2, top + height * 0.5
    knee = hip + (height * 0.08 if crouch else height * 0.25)
    pts = [(x, top + height * i / 16, 0.9) for i in range(17)]
    pts[5], pts[6] = (x - 5, shoulder, 0.9), (x + 5, shoulder, 0.9)
    pts[11], pts[12] = (x - 4, hip, 0.9), (x + 4, hip, 0.9)
    pts[13], pts[14] = (x - 4, knee, 0.9), (x + 4, knee, 0.9)
    pts[9], pts[10] = (x - height * 0.3, hip, 0.9), (x + height * 0.3, hip, 0.9)  # 몸 폭
    return pts


PITCH_VIEW = [
    body(640, 500, 300),  # 투수 — 앞쪽이라 크다
    body(600, 300, 100, crouch=True),  # 포수
    body(690, 180, 230),  # 타자
    body(610, 270, 120),  # 심판 — 포수 뒤, 발이 더 높다
]


def test_roles_anchor_on_the_crouching_catcher():
    roles = find_roles(PITCH_VIEW)
    assert roles.pitcher[1] == 500
    assert roles.catcher[1] == 300
    assert roles.batter[1] == 180


def test_no_crouching_person_means_not_a_pitch_view():
    assert find_roles([body(640, 500, 300), body(600, 300, 100), body(690, 180, 230)]) is None
    assert find_roles([body(640, 500, 300)]) is None


def test_crop_stays_inside_the_frame():
    assert crop_square([0, 0, 50, 100], 2.0, (1920, 1080)) == (0, 0, 125, 150)


def test_pitch_groups_are_the_stillest_long_ones():
    shots = [Shot(0, 30, 1, 4.0), Shot(30, 60, 2, 12.0), Shot(60, 61, 3, 1.0), Shot(61, 90, 4, 5.0)]
    assert pitch_groups(shots) == {1, 4}
    assert pitch_groups([]) == set()


def test_catcher_look_alike_without_a_batter_is_skipped():
    far = body(900, 100, 110, crouch=True)  # 멀리 앉은 사람 — 옆에 타자가 없다
    assert find_roles([*PITCH_VIEW, far]).catcher[1] == 300


def test_umpire_behind_the_catcher_is_not_the_batter():
    umpire = body(610, 200, 190)  # 키는 타자만큼 크지만 발이 포수보다 뒤에 있다
    assert find_roles([body(640, 500, 300), body(600, 300, 100, crouch=True), umpire]) is None


def test_small_far_player_is_not_the_batter():
    people = [body(640, 500, 300), body(600, 300, 100, crouch=True), body(690, 280, 120)]
    assert find_roles(people) is None


def test_split_shots_cuts_on_a_brightness_jump():
    tiny = np.zeros((8, 36, 64, 3), np.uint8)
    tiny[4:] = 200
    shots = split_shots(tiny, [i / 4 for i in range(8)])
    assert [(s.t0, s.t1) for s in shots] == [(0.0, 0.75), (1.0, 1.75)]
    assert split_shots(tiny[:1], [0.0]) == []
