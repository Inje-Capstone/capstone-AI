"""투구 뒤 화면 — 넓은 경기장(공이 맞아 나감)과 선수 클로즈업(안 맞음)을 사람 키로 가른다."""

from app.domain.contact import PeopleFrame
from app.domain.play_view import IN_PLAY, NO_PLAY, play_view, tallest

H = 1080


def person(top: float, bottom: float, score: float = 0.9) -> tuple:
    """머리~발 사이에 관절 17개를 고르게 둔 사람."""
    step = (bottom - top) / 16
    return tuple((500.0, top + k * step, score) for k in range(17))


def frames(t0: float, n: int, tall: float, fps: float = 10.0) -> list[PeopleFrame]:
    return [PeopleFrame(t0 + k / fps, (person(100, 100 + tall * H),), height=H) for k in range(n)]


def test_tallest_uses_scored_joints_and_screen_height():
    f = PeopleFrame(0.0, (person(100, 316), person(0, 54, score=0.1)), height=H)
    assert abs(tallest(f) - 0.2) < 1e-6
    assert tallest(PeopleFrame(0.0, (), height=H)) is None
    assert tallest(PeopleFrame(0.0, (person(0, 500),), height=0)) is None


def test_wide_field_after_the_pitch_means_the_ball_was_put_in_play():
    assert play_view(frames(10.5, 25, tall=0.08), 10.0) == IN_PLAY


def test_close_up_after_the_pitch_means_it_was_not():
    assert play_view(frames(10.5, 8, tall=0.9), 10.0) == NO_PLAY


def test_unclear_or_missing_views_say_nothing():
    assert play_view([], 10.0) is None  # 경기장이 아닌 화면 — 관절이 건너뜀
    assert play_view(frames(10.5, 5, tall=0.08), 10.0) is None  # 넓은 화면 0.5초뿐
    assert play_view(frames(10.5, 20, tall=0.4), 10.0) is None  # 중간 크기
    assert play_view(frames(20.0, 20, tall=0.08), 10.0) is None  # 투구 뒤 3.5초 밖
