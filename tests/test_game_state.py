import pytest

from app.adapters.relay.fixture import FixtureRelaySource
from app.domain.game_state import GameSim, GameStateError, bases_from_list, replay
from app.domain.models import RelayEvent

GAME_ID = "20260823LGOB"


@pytest.fixture(scope="module")
def feed():
    return FixtureRelaySource().load(GAME_ID)


def _ev(**kw):
    base = dict(id="x", t=0, inning=1, half="top", kind="result", text="", detail={})
    base.update(kw)
    return RelayEvent(**base)


# ── 경계값 ──────────────────────────────────────────────────────────────
def test_bases_from_list():
    assert bases_from_list([]) == (False, False, False)
    assert bases_from_list([1, 3]) == (True, False, True)
    with pytest.raises(GameStateError):
        bases_from_list([4])


def test_third_out_advances_half_and_clears_bases():
    sim = GameSim("두산", "LG")
    sim.apply(_ev(id="a", detail={"outs_made": 2, "bases_after": [1, 2], "runs": 0}))
    assert sim.state.outs == 2
    assert sim.state.bases == (True, True, False)

    sim.apply(_ev(id="b", t=1, detail={"outs_made": 1, "bases_after": [1, 2], "runs": 0}))
    # 3아웃 → 1회말로 넘어가고 주자·카운트가 지워진다.
    assert (sim.state.inning, sim.state.half) == (1, "bot")
    assert sim.state.outs == 0
    assert sim.state.bases == (False, False, False)


def test_fourth_out_is_impossible():
    sim = GameSim("두산", "LG")
    sim.apply(_ev(id="a", detail={"outs_made": 2, "bases_after": [], "runs": 0}))
    with pytest.raises(GameStateError, match="아웃 카운트"):
        sim.apply(_ev(id="b", t=1, detail={"outs_made": 2, "bases_after": [], "runs": 0}))


def test_count_accumulates_and_resets_on_result():
    sim = GameSim("두산", "LG")
    sim.apply(_ev(id="ab", kind="atbat", batter="임하람"))
    sim.apply(_ev(id="p1", kind="pitch", detail={"result": "ball"}))
    sim.apply(_ev(id="p2", kind="pitch", detail={"result": "strike"}))
    sim.apply(_ev(id="p3", kind="pitch", detail={"result": "swing_strike"}))
    assert (sim.state.balls, sim.state.strikes) == (1, 2)

    # 투스트라이크 이후 파울은 카운트를 늘리지 않는다.
    sim.apply(_ev(id="p4", kind="pitch", detail={"result": "foul"}))
    assert sim.state.strikes == 2

    sim.apply(_ev(id="r", kind="result", detail={"outs_made": 1, "bases_after": [], "runs": 0}))
    assert (sim.state.balls, sim.state.strikes) == (0, 0)


def test_runs_go_to_the_batting_team():
    sim = GameSim("두산", "LG")
    sim.apply(_ev(id="a", detail={"outs_made": 0, "bases_after": [], "runs": 2}))
    assert (sim.state.away_score, sim.state.home_score) == (2, 0)

    sim.apply(_ev(id="b", t=1, half="bot", detail={"outs_made": 0, "bases_after": [], "runs": 1}))
    assert (sim.state.away_score, sim.state.home_score) == (2, 1)


def test_walk_off_ends_the_game_immediately():
    sim = GameSim("두산", "LG")
    sim.state.inning, sim.state.half = 9, "bot"
    sim.state.away_score, sim.state.home_score = 3, 3
    sim.apply(
        _ev(id="w", inning=9, half="bot", detail={"outs_made": 0, "bases_after": [], "runs": 1})
    )
    assert sim.state.game_over is True


def test_game_ends_after_top_of_ninth_when_home_leads():
    sim = GameSim("두산", "LG")
    sim.state.inning, sim.state.half = 9, "top"
    sim.state.away_score, sim.state.home_score = 3, 5
    sim.state.outs = 2
    sim.apply(
        _ev(id="z", inning=9, half="top", detail={"outs_made": 1, "bases_after": [], "runs": 0})
    )
    assert sim.state.game_over is True
    # 9회말은 열리지 않는다.
    assert (sim.state.inning, sim.state.half) == (9, "top")


# ── fixture 정합성: 이 테스트가 곧 fixture의 검증기다 ────────────────────
def test_fixture_replays_to_the_declared_final_score(feed):
    final = replay(feed.events, feed.meta.away_team, feed.meta.home_team)
    assert final.away_score == feed.meta.final_away
    assert final.home_score == feed.meta.final_home
    assert final.game_over is True


def test_fixture_every_half_inning_records_exactly_three_outs(feed):
    """모든 반이닝이 정확히 3아웃으로 끝나야 한다.

    하나라도 어긋나면 `_sync_half`가 다음 이벤트의 이닝 표기를 보고 조용히 덮어버려
    상태가 그럴싸하게 틀린 채로 흘러간다. fixture의 정합성 검사기이자 회귀 방지선.
    """
    totals = {}
    order = []
    for event in feed.events:
        key = (event.inning, event.half)
        if key not in totals:
            totals[key] = 0
            order.append(key)
        totals[key] += int(event.detail.get("outs_made", 0))

    for key in order:
        inning, half = key
        assert totals[key] == 3, (
            f"{inning}회{'초' if half == 'top' else '말'} 아웃 합계가 {totals[key]}이다"
        )

    # 9회초까지 진행하고 9회말은 열리지 않는다(홈팀 리드).
    assert order[0] == (1, "top")
    assert order[-1] == (9, "top")
    assert (9, "bot") not in totals


def test_scoreboard_text_matches_wireframe_format(feed):
    state = replay(feed.events, feed.meta.away_team, feed.meta.home_team, until_t=7810)
    # 와이어프레임 S4 좌하단 표기 형식
    assert state.scoreboard_text() == "두산 3 : 4 LG · 7회말 · B2 S1 O2"


def test_state_is_a_pure_function_of_timecode(feed):
    """시킹 대응의 근거 — 같은 t면 몇 번을 계산해도 같은 상태."""
    a = replay(feed.events, feed.meta.away_team, feed.meta.home_team, until_t=7550)
    b = replay(feed.events, feed.meta.away_team, feed.meta.home_team, until_t=7550)
    assert a.model_dump() == b.model_dump()


def test_balk_moves_the_runner_to_second(feed):
    before = replay(feed.events, feed.meta.away_team, feed.meta.home_team, until_t=7540)
    after = replay(feed.events, feed.meta.away_team, feed.meta.home_team, until_t=7550)
    assert before.bases == (True, False, False)  # 보크 직전: 1루
    assert after.bases == (False, True, False)  # 보크 후: 2루
    assert after.outs == 0
