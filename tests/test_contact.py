"""선수 접촉·충돌·슬라이딩 — 합성 관절로 충돌·태그·깊이 다름·정지 겹침·슬라이딩·패닝 갈래."""

from app.domain.contact import (
    PeopleFrame,
    analyze,
    bodies,
    body,
    contacts,
    duplicate,
    same_depth,
    slides,
    touching,
    track,
)

FPS = 10
H = 200  # 사람 키(px)

# 서 있는 사람 관절 — (가운데에서 x 비율, 발에서 위로 y 비율)
STAND = (
    (0.0, 0.93), (-0.02, 0.95), (0.02, 0.95), (-0.05, 0.94), (0.05, 0.94),  # 얼굴
    (-0.15, 0.8), (0.15, 0.8), (-0.2, 0.62), (0.2, 0.62), (-0.2, 0.45), (0.2, 0.45),  # 팔
    (-0.1, 0.5), (0.1, 0.5), (-0.1, 0.25), (0.1, 0.25), (-0.1, 0.0), (0.1, 0.0),  # 다리
)


def person(cx, ground=500.0, h=H, lying=False, score=0.9):
    """서 있거나(lying=False) 머리를 오른쪽으로 두고 누운 사람. ground = 발(누우면 몸 아래) 높이."""
    if lying:
        return tuple((cx + (up - 0.475) * h, ground - (0.2 + dx) * h, score) for dx, up in STAND)
    return tuple((cx + dx * h, ground - up * h, score) for dx, up in STAND)


def frames(*tracks, n, t0=0.0, pan=0.0):
    """tracks: 프레임 번호 → 사람(없으면 None). n 프레임.

    pan: 카메라가 프레임마다 화면 내용을 이만큼(px) 오른쪽으로 민다 — 사람 좌표에 더하고
    cam에 적는다.
    """
    out = []
    for k in range(n):
        shift = pan * k
        people = tuple(
            tuple((x + shift, y, s) for x, y, s in p)
            for p in (f(k) for f in tracks) if p is not None
        )
        out.append(PeopleFrame(round(t0 + k / FPS, 2), people, cam=(1.0, 0.0, shift, 0.0)))
    return out


def test_body_needs_six_confident_joints():
    assert body(person(100)) is not None
    weak = tuple((x, y, 0.1) for x, y, _ in person(100))
    assert body(weak) is None


def test_skeleton_mixing_two_people_is_dropped():
    """머리·팔은 한 사람, 발목은 화면 반대편 다른 사람 — 뼈 길이가 말이 안 된다."""
    p = list(person(100))
    p[15], p[16] = (700, 500, 0.9), (720, 500, 0.9)
    assert body(tuple(p)) is None


def test_lying_needs_torso_seen():
    """가로로 넓어도 어깨·엉덩이를 못 봤으면 누웠는지 모른다."""
    p = [(x, y, 0.0 if k in (5, 6, 11, 12) else s)
         for k, (x, y, s) in enumerate(person(300, lying=True))]
    b = body(tuple(p))
    assert b is not None and not b.lying


def test_lying_person_keeps_height_as_scale():
    stand, lie = body(person(100)), body(person(100, lying=True))
    assert not stand.lying and lie.lying
    assert abs(stand.scale - lie.scale) < 1


def test_same_size_same_ground_is_same_depth_and_far_or_small_is_not():
    a = body(person(100))
    assert same_depth(a, body(person(300)))
    assert not same_depth(a, body(person(110, ground=350, h=80)))  # 멀리 작게 — 2D로만 겹침
    assert not same_depth(a, body(person(110, ground=420)))  # 같은 키라도 발 높이가 다르다


def test_touching_needs_joints_close():
    a = body(person(100))
    assert touching(a, body(person(100 + 0.4 * H + 10)))  # 팔 끝 사이 10px
    assert not touching(a, body(person(100 + 0.4 * H + 60)))


def _collision():
    """A는 오른쪽, B는 왼쪽으로 달려와 0.5초에 부딪히고, B가 0.8초에 뒤로 넘어진다."""
    def a(k):
        return person(100 + 30 * min(k, 5))

    def b(k):
        if k < 8:
            return person(500 - 30 * min(k, 5))
        return person(500 - 150 + 60, lying=True)

    return frames(a, b, n=20)


def test_fast_approach_then_fall_is_collision():
    [c] = contacts(_collision())
    assert c.kind == "collision" and c.t0 == 0.5
    assert c.closing >= 1.5 and c.fell and not c.slid
    assert "넘어짐 예" in c.why


def test_falling_person_keeps_track_id():
    ids = track(_collision())
    assert ids[0] == [0, 1] and ids[-1] == [0, 1]


def test_slow_touch_without_fall_is_plain_contact():
    """수비수는 서 있고 주자가 천천히 다가와 1초 닿았다 떨어진다(태그·하이파이브)."""
    def a(k):
        return person(100)

    def b(k):
        x = 400 - 10 * min(k, 20)  # 0.5키/초
        return person(x if k <= 30 else 400)

    [c] = contacts(frames(a, b, n=35))
    assert c.kind == "contact" and not c.fell and c.closing < 1.5


def test_overlap_at_different_depth_is_not_contact():
    """멀리 작은 사람이 가까운 사람 앞을 2D로 가로지른다."""
    def near(k):
        return person(300)

    def far(k):
        return person(150 + 30 * k, ground=350, h=80)

    assert contacts(frames(near, far, n=12)) == []


def test_standing_together_for_seconds_is_not_contact():
    """포수 뒤 심판처럼 몇 초씩 붙어 선 건 접촉이 아니다."""
    def a(k):
        return person(100)

    def b(k):
        return person(100 + 0.4 * H + 10)

    assert contacts(frames(a, b, n=40)) == []


def test_first_frame_contact_has_unknown_closing():
    def a(k):
        return person(100)

    def b(k):
        return person(100 + 0.4 * H + 10 + (300 if k > 5 else 0))

    [c] = contacts(frames(a, b, n=10))
    assert c.closing is None and c.kind == "contact" and "못 봄" in c.why


def _slide_into_tag():
    """주자가 누운 채 미끄러져(1.6키/초) 0.9초에 수비수 무릎에 닿고, 그대로 1.4초 누워 있는다."""
    def runner(k):
        return person(100 + 30 * min(k, 9), lying=True)

    def fielder(k):
        return person(500)

    return frames(runner, fielder, n=24)


def test_sliding_runner_touching_fielder_is_slide_tag():
    [c] = contacts(_slide_into_tag())
    assert c.kind == "slide_tag" and c.slid and c.t0 == 0.9


def test_slide_is_found_even_when_runner_lies_still_after():
    [s] = slides(_slide_into_tag())
    assert s.t0 == 0.0 and s.t1 == 2.3 and s.distance >= 1.0


def test_slow_motion_slide_is_still_slide():
    """느린 화면 다시보기 — 0.2키/초로 2초 동안 미끄러진다. 속도가 아니라 거리로 가른다."""
    def runner(k):
        return person(100 + 4 * k, lying=True)

    [s] = slides(frames(runner, n=21))
    assert 0.4 <= s.distance < 0.5


def test_lying_still_or_shifting_a_little_is_not_slide():
    """넘어져 누워 있거나 꿈틀대며 몸길이의 0.1배쯤 움직인 건 슬라이딩이 아니다."""
    def fallen(k):
        return person(300, lying=True)

    def shift(k):
        return person(100 + k, lying=True)

    assert slides(frames(fallen, shift, n=20)) == []


def test_scene_cut_restarts_tracking():
    def a(k):
        return person(100)

    before = frames(a, n=5)
    after = frames(a, n=5, t0=2.0)  # 1.6초 비었다 — 다른 장면
    ids = track(before + after)
    assert ids[0] == [0] and ids[-1] == [1]


def test_bent_over_fielder_with_wide_box_is_not_lying():
    """허리 숙여 땅볼을 잡는 수비 — 팔을 옆으로 뻗어 관절 범위는 넓어도 상체는 서 있다."""
    p = list(person(300))
    p[9], p[10] = (300 - 160, 420, 0.9), (300 + 160, 420, 0.9)  # 두 손목을 옆으로 멀리
    p[7], p[8] = (300 - 120, 380, 0.9), (300 + 120, 380, 0.9)
    b = body(tuple(p))
    assert b.x1 - b.x0 >= 1.2 * (b.y1 - b.y0) and not b.lying


def test_double_detection_of_one_person_is_dropped():
    a = person(300)
    twin = tuple((x + 3, y + 2, s) for x, y, s in a)
    assert duplicate(body(a), body(twin))
    assert not duplicate(body(a), body(person(300 + 0.4 * H + 10)))
    kept = bodies(PeopleFrame(0.0, (a, twin, person(600))))
    assert kept[0] is not None and kept[1] is None and kept[2] is not None


def test_people_smaller_than_min_scale_are_ignored():
    def a(k):
        return person(100)

    def b(k):
        return person(400 - 30 * min(k, 10))

    assert contacts(frames(a, b, n=20), min_scale=H * 2) == []


def test_camera_pan_alone_is_not_slide():
    """넘어져 가만히 누운 사람을 카메라가 빠르게 훑는다 — 화면에선 움직여도 세상에선 멈춰 있다."""
    def fallen(k):
        return person(300, lying=True)

    assert slides(frames(fallen, n=20, pan=40)) == []


def test_slide_followed_by_camera_is_still_slide():
    """카메라가 미끄러지는 주자를 따라가 화면 속 주자는 제자리 — cam이 움직임을 돌려준다."""
    def runner(k):
        return person(300 + 30 * k, lying=True)

    [s] = slides(frames(runner, n=8, pan=-30))
    assert s.distance >= 1.0


def test_camera_pan_does_not_break_tracking():
    def a(k):
        return person(100)

    ids = track(frames(a, n=10, pan=80))  # 0.1초에 키의 0.4배 넘게 화면이 밀린다
    assert {row[0] for row in ids} == {0}


def test_cut_flag_restarts_tracking():
    shots = [PeopleFrame(0.0, (person(100),)), PeopleFrame(0.1, (person(100),), cut=True)]
    assert track(shots) == [[0], [1]]


def test_analyze_shares_tracking_between_contacts_and_slides():
    sc, found, slid = analyze(_slide_into_tag())
    assert [c.kind for c in found] == ["slide_tag"] and len(slid) == 1
    assert found[0].a == slid[0].who or found[0].b == slid[0].who
    assert len(sc.ids) == len(sc.frames)


def _zoomed(n, lying=False):
    """카메라가 0.1초마다 10%씩 당긴다 — 화면 속 사람은 커지고 화면 가운데에서 멀어진다."""
    out = []
    for k in range(n):
        z = 1.1 ** k
        p = tuple((x * z, y * z, s) for x, y, s in person(300, ground=300, h=100, lying=lying))
        out.append(PeopleFrame(round(k / FPS, 2), (p,), cam=(z, 0.0, 0.0, 0.0)))
    return out


def test_camera_zoom_keeps_track_and_world_size():
    sc = analyze(_zoomed(10))[0]
    assert {row[0] for row in sc.ids} == {0}
    sizes = [b.world_scale for row in sc.bodies for b in row]
    assert max(sizes) - min(sizes) < 1


def test_camera_zoom_on_lying_player_is_not_slide():
    assert analyze(_zoomed(10, lying=True))[2] == []


def test_one_slide_split_across_two_track_ids_is_merged():
    """관절을 놓쳐 추적 번호가 바뀌어도 이어진 같은 자리의 슬라이딩은 하나."""
    def runner(k):
        return person(100 + 30 * k, lying=True)

    run = frames(runner, n=10)
    gap = [PeopleFrame(f.t, f.people, cut=(k == 5)) for k, f in enumerate(run)]  # 0.5초에 끊김
    assert len(track(gap)[0]) == 1 and track(gap)[5] == [1]
    [s] = slides(gap)
    assert s.t0 == 0.0 and s.t1 == 0.9


def test_close_up_without_legs_is_not_collision():
    """덕아웃 클로즈업 — 상체만 보이는 두 사람이 빠르게 겹쳐도 넘어짐을 믿을 수 없다."""
    def upper(p):
        return tuple((x, y, 0.0 if k in (13, 14, 15, 16) else s) for k, (x, y, s) in enumerate(p))

    found = contacts([PeopleFrame(f.t, tuple(upper(p) for p in f.people)) for f in _collision()])
    assert [c.kind for c in found] != ["collision"]


def test_slide_into_catcher_and_umpire_is_one_tag():
    """주자가 포수에 닿고 곧 심판에도 닿는다 — 태그는 한 번."""
    def runner(k):
        return person(100 + 30 * min(k, 12), lying=True)

    def catcher(k):
        return person(500)

    def umpire(k):
        return person(600) if k >= 11 else None

    shots = frames(runner, catcher, umpire, n=24)
    last = bodies(shots[-1])
    assert touching(last[0], last[1]) and touching(last[0], last[2])  # 주자가 둘 다에 닿았다
    tags = [c for c in contacts(shots) if c.kind == "slide_tag"]
    assert [(c.a, c.b) for c in tags] == [(0, 1)]  # 먼저 닿은 포수와의 태그 하나


def test_tag_held_for_seconds_is_still_slide_tag():
    """포수가 미끄러져 들어온 주자에게 태그를 3초 넘게 대고 있다 — 겹쳐 선 것과 다르다."""
    [c] = [c for c in contacts(_slide_into_tag() + [
        PeopleFrame(round(2.4 + k / FPS, 2), _slide_into_tag()[-1].people) for k in range(20)
    ]) if c.kind == "slide_tag"]
    assert c.t1 - c.t0 > 2.5


def test_merge_looks_past_another_players_dive():
    """주자가 두 번에 나눠 잡힌 사이 멀리서 다른 선수가 다이빙해도 주자 슬라이딩은 하나."""
    def runner(k):
        if 5 <= k < 10:
            return person(100 + 30 * k)  # 0.5초 서서 달림(관절이 흔들려 일어난 것처럼)
        return person(100 + 30 * k, lying=True)

    def diver(k):
        return person(900 + 30 * k, ground=300, lying=True) if 3 <= k <= 8 else None

    found = slides(frames(runner, diver, n=16))
    assert len(found) == 2 and found[0].t0 == 0.0 and found[0].t1 == 1.5
