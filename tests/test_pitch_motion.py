"""투수 동작 분석(실험) — 합성 관절 좌표로 세트·정지·던지기·중단 갈래를 확인한다."""

from app.adapters.video.pose import pick_pitcher
from app.domain.pitch_motion import PoseFrame, analyze_motions

FPS = 10


def _pose(t, wrist_dx=0.0, wrist_dy=0.0, knee_up=0.0, together=True):
    """몸통 길이 100px 투수. 손목은 가슴 앞(together면 붙어 있음)."""
    p = [(0.0, 0.0, 0.0)] * 17
    p = list(p)
    p[0] = (200, 90, 0.9)  # 코
    p[5], p[6] = (180, 100, 0.9), (220, 100, 0.9)  # 어깨
    p[11], p[12] = (185, 200, 0.9), (215, 200, 0.9)  # 엉덩이
    p[13], p[14] = (185, 260 - knee_up, 0.9), (215, 260, 0.9)  # 무릎
    p[15], p[16] = (185, 320, 0.9), (215, 320, 0.9)  # 발목
    gap = 5 if together else 60
    p[9] = (200 - gap + wrist_dx, 150 + wrist_dy, 0.9)
    p[10] = (200 + gap + wrist_dx, 150 + wrist_dy, 0.9)
    return PoseFrame(t, tuple(p))


def _seq(parts):
    frames, t = [], 0.0
    for count, kw in parts:
        for k in range(count):
            frames.append(
                _pose(round(t, 2), **{key: (v(k) if callable(v) else v) for key, v in kw.items()})
            )
            t += 1 / FPS
    return frames


def test_set_stop_then_leg_kick_is_a_legal_pitch():
    frames = _seq([(6, {}), (6, {"knee_up": lambda k: 15 * k, "wrist_dy": lambda k: -20 * k})])
    (m,) = analyze_motions(frames)
    assert m.outcome == "throw" and m.suspect is None and m.stop_sec >= 0.3


def test_leg_kick_without_stop_is_flagged():
    # 손이 천천히 내려오며 모이다가(멈춤 없이) 바로 다리를 든다
    frames = _seq(
        [
            (5, {"wrist_dy": lambda k: 5 * k}),
            (6, {"knee_up": lambda k: 15 * k, "wrist_dy": lambda k: 25 - 20 * k}),
        ]
    )
    motions = analyze_motions(frames)
    assert motions and motions[0].suspect == "no_stop"


def test_motion_that_stops_without_a_throw_is_flagged_interrupted():
    frames = _seq([(6, {}), (3, {"wrist_dy": lambda k: 25 * k}), (20, {"wrist_dy": 75})])
    motions = analyze_motions(frames)
    assert motions and (motions[0].outcome, motions[0].suspect) == ("aborted", "interrupted")


def test_pick_pitcher_prefers_big_lower_center_person_in_center_field_view():
    def person(cx, cy, h):
        return [(cx + (i % 3 - 1) * h * 0.1, cy + (i / 17 - 0.5) * h, 0.9) for i in range(17)]

    plate = [person(600, 150, 100), person(700, 160, 90), person(650, 170, 80)]
    pitcher = person(640, 450, 300)
    assert pick_pitcher(plate + [pitcher], 1280, 720) == 3
    assert pick_pitcher([pitcher], 1280, 720) is None  # 위쪽에 타자·포수가 없으면 다른 화면
