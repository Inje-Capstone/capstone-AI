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
    # 투구 뒤 3.5초 밖의 넓은 화면은 세지 않는다 — 그 사이가 경기장 밖 화면이면 안 맞은 것
    assert play_view(frames(20.0, 20, tall=0.08), 10.0) == NO_PLAY


def test_pitch_followed_only_by_off_field_frames_is_no_play():
    """투구 뒤 경기장 화면이 한 장도 없다(실내 클로즈업 건너뜀) — 안 맞았다. 영상 끝은 모른다."""
    later = [PeopleFrame(5.0, (), height=1080)]  # 분석이 구간 뒤까지 이어졌다
    assert play_view(later, 0.0) == NO_PLAY
    assert play_view([], 0.0) is None  # 영상·조각이 끝났다
