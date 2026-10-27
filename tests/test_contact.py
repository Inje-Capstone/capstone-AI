"""선수 접촉·충돌·슬라이딩 — 합성 관절로 충돌·태그·깊이 다름·정지 겹침·슬라이딩·패닝 갈래."""

from app.domain.contact import (
    Contact,
    PeopleFrame,
    Slide,
    _merge,
    _one_tag_per_slide,
    analyze,
    bodies,
    body,
    contacts,
    duplicate,
    gap,
    inside,
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


def test_person_cut_by_bottom_edge_is_not_lying():
    """앞줄 관중처럼 화면 아래 끝에 걸린 사람은 누웠는지 모른다."""
    p = person(300, ground=480, lying=True)
    assert bodies(PeopleFrame(0.0, (p,)))[0].lying
    assert not bodies(PeopleFrame(0.0, (p,), height=480))[0].lying
    assert bodies(PeopleFrame(0.0, (person(300, ground=400, lying=True),), height=480))[0].lying


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


def test_slides_in_two_shots_stay_two():
    """화면이 바뀌어 다른 각도로 다시 보여준 슬라이딩은 따로 — 정답 표도 장면마다 센다."""
    def runner(k):
        return person(100 + 30 * k, lying=True)

    run = frames(runner, n=10)
    shots = [PeopleFrame(f.t, f.people, cut=(k == 5)) for k, f in enumerate(run)]  # 0.5초에 전환
    assert track(shots)[0] == [0] and track(shots)[5] == [1]
    assert [(s.t0, s.t1) for s in slides(shots)] == [(0.0, 0.4), (0.5, 0.9)]


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


def test_dive_after_running_is_slide_even_if_it_stops_when_lying():
    """달려오다(0.1초에 40px) 몸을 던지고 누운 뒤엔 거의 안 움직이는 다이빙 — 던지기 전부터 잰다."""
    def diver(k):
        if k < 8:
            return person(100 + 40 * k)
        return person(100 + 40 * 8, lying=True)

    [s] = slides(frames(diver, n=16))
    assert s.t0 == 0.8 and s.distance >= 1.2


def test_falling_in_place_is_not_slide():
    """제자리에 서 있다가 쓰러져 누움(몸에 맞는 공) — 던지기 전 자리와 거의 같다."""
    def batter(k):
        return person(300) if k < 8 else person(300, lying=True)

    assert slides(frames(batter, n=20)) == []


def test_long_slow_motion_slide_is_slide():
    """느린 화면 — 2.5초 동안 0.15키/초. 1.5초 넘게 누운 채(마지막 0.5초 빼고) 0.25배 넘게."""
    def runner(k):
        return person(100 + 3 * k, lying=True)

    [s] = slides(frames(runner, n=26))
    assert 0.25 <= s.distance < 0.4 and not s.thrown


def test_lying_long_then_pushing_up_is_not_slow_slide():
    """제자리에 쓰러져 1.6초 누워 있다가 손을 짚고 몸을 일으킨다 — 위로 뜰 뿐 미끄러지지 않는다."""
    def fallen(k):
        if k < 5:
            return person(300)
        if k < 21:
            return person(300, lying=True)
        return person(300, ground=500 - 0.15 * H, lying=True)

    assert slides(frames(fallen, n=23)) == []


def test_short_slow_shift_is_not_slide():
    def runner(k):
        return person(100 + 3 * k, lying=True)

    assert slides(frames(runner, n=12)) == []  # 1.1초 — 느린 화면 기준 시간이 안 된다


def test_tag_needs_the_standing_players_hand():
    """주자가 미끄러져 와 발에 닿았지만 서 있는 사람(심판) 손은 위로 — 태그가 아니다."""
    def runner(k):
        return person(100 + 30 * min(k, 9), lying=True)

    def umpire(k):
        p = list(person(500))
        for w in (9, 10):  # 손목을 머리 위로
            p[w] = (p[w][0], 500 - 1.1 * H, 0.9)
        p[7], p[8] = (p[7][0], 500 - 0.95 * H, 0.9), (p[8][0], 500 - 0.95 * H, 0.9)
        return tuple(p)

    found = contacts(frames(runner, umpire, n=24))
    assert found and all(c.kind != "slide_tag" for c in found)
    assert found[0].slid and found[0].hand > 0.3


def test_touch_while_still_diving_becomes_tag_when_runner_lies():
    """몸을 던지는 중에 먼저 닿고 0.1초 뒤에 눕는다 — 닿아 있는 동안 누웠으면 태그."""
    def runner(k):
        if k < 9:
            return person(100 + 40 * k)
        return person(100 + 40 * 8 + 30, lying=True)

    def fielder(k):
        return person(100 + 40 * 8 + 160)

    tags = [c for c in contacts(frames(runner, fielder, n=20)) if c.kind == "slide_tag"]
    assert len(tags) == 1


def _stretched(cx, reach=1.6):
    """머리부터 슬라이딩 — 누워 팔을 머리 위로 뻗어 관절 범위가 키의 reach배."""
    p = list(person(cx, lying=True))
    ext = (reach - 0.95) * H
    for k, f in ((9, 1.0), (10, 1.0), (7, 0.5), (8, 0.5)):
        p[k] = (cx + 0.475 * H + ext * f, p[k][1], 0.9)
    return tuple(p)


def test_stretched_lying_runner_is_same_depth_as_fielder():
    """팔을 뻗고 누우면 관절 범위가 키보다 길다(실측 1.4~1.65배) — 머리~발 길이로 견준다."""
    lying, standing = body(_stretched(300)), body(person(500))
    assert lying.scale > 1.5 * standing.scale and same_depth(lying, standing)


def test_feet_first_runner_is_not_same_depth_as_small_far_person():
    """팔을 안 뻗은 발 먼저 슬라이딩과, 뒤쪽 멀리 작게(0.6배) 선 사람 — 다른 깊이."""
    assert not same_depth(body(person(300, lying=True)), body(person(360, ground=470, h=120)))


def test_fielder_reaching_a_little_behind_the_lying_runner_is_same_depth():
    """포수가 주자보다 조금 뒤(발이 키의 0.45배 위)에서 손을 뻗는다 — 둘 다 서 있으면 다른 깊이."""
    lying = body(person(300, lying=True))
    reaching = body(person(500, ground=500 - 0.45 * 0.95 * H))
    assert same_depth(lying, reaching)
    assert not same_depth(body(person(300)), reaching)


def test_part_of_one_person_detected_again_is_not_a_contact():
    """클로즈업에서 머리·상체만 따로 한 사람으로 잡혔다 — 관절 범위가 큰 사람 안에 다 든다."""
    whole = body(person(300))
    upper = [(x + 6, y + 4, s if k <= 10 else 0.0) for k, (x, y, s) in enumerate(person(300))]
    part = body(tuple(upper))  # 얼굴·어깨·팔만 — 관절이 사람 몸 위쪽에 모인다
    assert inside(part, whole) > 0.8 and gap(part, whole) < 0.15 and not touching(part, whole)
    assert inside(whole, body(person(300 + 0.4 * H + 10))) < 0.8


def test_slide_resumed_where_the_last_one_ended_is_merged():
    """추적이 끊겨 다른 번호로 이어진 긴 슬라이딩 — 앞 슬라이딩이 끝난 자리에서 다시 시작."""
    def run(t0, x0, x1, h=H):
        start, end = body(person(x0, h=h, lying=True)), body(person(x1, h=h, lying=True))
        return [(t0, start), (t0 + 0.6, end)]

    first = (Slide(0.0, 0.6, 0, 1.9), run(0.0, 100, 460))
    second = (Slide(0.7, 1.3, 1, 1.9), run(0.7, 520, 880))  # 시작 자리가 앞 슬라이딩 시작과는 멀다
    [s] = _merge([first, second])
    assert (s.t0, s.t1, s.who) == (0.0, 1.3, 0)
    assert len(_merge([first, second], cuts=[0.7])) == 2  # 화면이 바뀌었으면 다른 다시보기
    zoomed = (Slide(0.7, 1.3, 1, 1.9), run(0.7, 520, 880, h=H * 1.4))
    assert len(_merge([first, zoomed])) == 2  # 디졸브 — 전환은 안 잡혔지만 몸 크기가 튄다


def test_partial_detection_moving_together_is_one_slide():
    """한 주자의 상체만 따로 잡힌 검출이 같은 때 같은 자리에서 함께 미끄러진다 — 하나."""
    def runner(k):
        return person(100 + 30 * k, lying=True)

    def upper(k):
        full = runner(k)
        return tuple((x + 25, y - 10, s if j <= 12 else 0.0) for j, (x, y, s) in enumerate(full))

    assert len(slides(frames(runner, upper, n=10))) == 1


def test_two_players_sliding_at_the_same_base_stay_two():
    """주자가 베이스에 멈춘 동안 다른 선수가 옆에서 미끄러져 온다 — 둘."""
    def runner(k):
        return person(40 * min(k, 10), lying=True)

    def other(k):
        return person(700 - 40 * (k - 8), lying=True) if 8 <= k <= 13 else None

    assert len(slides(frames(runner, other, n=18))) == 2


def test_fielder_diving_into_a_sliding_runner_is_tag_not_collision():
    """주자가 미끄러져 들어오고 수비가 몸을 날려 태그하다 함께 넘어진다 — 충돌이 아니라 태그."""
    def runner(k):
        return person(100 + 30 * min(k, 9), lying=True)

    def fielder(k):
        if k < 9:
            return person(500)
        return person(470, lying=True)  # 태그하며 넘어짐

    kinds = [c.kind for c in contacts(frames(runner, fielder, n=24))]
    assert "slide_tag" in kinds and "collision" not in kinds


def test_walking_then_falling_forward_is_not_slide():
    """1초 걷다가(0.75키/초) 앞으로 넘어진다 — 자세만 바뀐 이동이라 몸을 던진 게 아니다."""
    def walker(k):
        if k < 10:
            return person(300 + 15 * k)
        return person(300 + 15 * 9 + 0.475 * H, lying=True)

    assert slides(frames(walker, n=20)) == []


def test_sprint_dive_that_breaks_tracking_is_still_slide():
    """전력질주(0.35키/프레임) 뒤 다이빙 — 눕는 순간 중심이 튀어 추적 번호가 바뀌어도 이어 본다."""
    def sprinter(k):
        if k < 8:
            return person(100 + 70 * k)
        return person(100 + 70 * 7 + 70 + 0.475 * H, lying=True)

    run = frames(sprinter, n=16)
    assert track(run)[7] != track(run)[8]  # 추적이 끊겼다
    [s] = slides(run)
    assert s.thrown and s.distance >= 1.2


def test_tag_needs_the_wrists_seen():
    """수비 손목을 못 봤으면 태그로 치지 않는다 — 손목 없는 부분 검출이 태그로 잡히던 오탐(실측).
    대신 글러브가 주자 몸에 가린 진짜 태그는 놓친다(한계)."""
    def runner(k):
        return person(100 + 30 * min(k, 9), lying=True)

    def fielder(k):
        p = list(person(500))
        p[9], p[10] = (p[9][0], p[9][1], 0.1), (p[10][0], p[10][1], 0.1)
        return tuple(p)

    found = contacts(frames(runner, fielder, n=24))
    assert found and all(c.kind != "slide_tag" for c in found) and found[0].hand is None


def test_one_tag_per_slide_even_when_touching_again_later():
    """미끄러져 들어와 태그된 뒤 2초 넘게 지나 다시 닿아도 같은 슬라이딩의 태그는 하나."""
    def runner(k):
        return person(100 + 30 * min(k, 9), lying=True)

    def fielder(k):
        return person(500) if k < 14 or k >= 34 else person(800)  # 잠깐 물러났다 다시 온다

    tags = [c for c in contacts(frames(runner, fielder, n=40)) if c.kind == "slide_tag"]
    assert len(tags) == 1


def test_umpire_lowering_arms_long_after_contact_is_not_tag():
    """심판이 손을 들고 있다가 1.4초 뒤에 내린다 — 손은 닿기 시작한 뒤 1초 안에서만 본다."""
    def runner(k):
        return person(100 + 30 * min(k, 9), lying=True)

    def umpire(k):
        p = list(person(500))
        if k < 23:
            for w in (9, 10):
                p[w] = (p[w][0], 500 - 1.1 * H, 0.9)
            p[7], p[8] = (p[7][0], 500 - 0.95 * H, 0.9), (p[8][0], 500 - 0.95 * H, 0.9)
        return tuple(p)

    assert all(c.kind != "slide_tag" for c in contacts(frames(runner, umpire, n=30)))


def test_one_tag_when_one_runner_is_detected_twice():
    """한 주자가 두 추적 번호(0·1)로 잡혀 합친 슬라이딩 하나 — 두 번호 각각의 태그도 하나로 센다."""
    merged = [Slide(0.0, 2.0, 0, 1.5, members=(0, 1))]
    tags = [Contact(0.9, 1.0, 0, 2, None, False, True, "slide_tag", 0.1),
            Contact(1.6, 1.7, 1, 3, None, False, True, "slide_tag", 0.1)]
    assert [(c.a, c.b) for c in _one_tag_per_slide(tags, merged)] == [(0, 2)]
    alone = [Slide(0.0, 2.0, 0, 1.5), Slide(0.0, 2.0, 1, 1.5)]  # 합치지 않았으면 둘
    assert len(_one_tag_per_slide(tags, alone)) == 2



def test_thrown_needs_the_run_up_seen_right_before_lying():
    """달리던 모습이 0.5초 끊겼다 다른 자리에 누워 나타난다(무더기 속 추적) — 던진 게 아니다."""
    def player(k):
        if k < 4:
            return person(100 + 30 * k)
        if k < 9:
            return None
        return person(600, lying=True)

    assert slides(frames(player, n=14)) == []
