"""영상 판정 — 점수판 전이 × 장면 단서 → 플레이 판정, 그리고 문자중계 채점."""

import importlib.util
from pathlib import Path

import pytest

from app.domain.detectors import detect
from app.domain.game_state import replay
from app.domain.grading import grade, summary
from app.domain.models import GameFeed, GameMeta
from app.domain.scorebug import (
    Reading,
    advanced_all_one_base,
    runner_removed,
    single_runner_advance,
    stabilize,
    transitions,
)
from app.domain.video_judge import Cue, judge, judge_transition, to_relay_events

ROOT = Path(__file__).resolve().parents[1]
E = (False, False, False)


def R(t, outs=0, bases=E, balls=0, strikes=0, away=0, home=0, inning=3, half="top"):
    return Reading(float(t), inning, half, balls, strikes, outs, bases, away, home)


def twice(*readings):
    """점수판은 같은 화면이 여러 번 읽혀야 인정된다 — 판독마다 0.5초 뒤 한 번 더."""
    out = []
    for r in readings:
        out += [r, Reading(r.t + 0.5, *r.key())]
    return out


def one(before, after, cues=()):
    (tr,) = transitions(stabilize(twice(before, after)))
    return judge_transition(tr, cues)


# ── 점수판 ─────────────────────────────────────────────────────────────
def test_stabilize_drops_single_frame_misreads_and_impossible_values():
    good = twice(R(0), R(10, outs=1))
    noise = [R(5, outs=2, bases=(True, True, True)),  # 한 번만 보인 오독
             Reading(6.0, 3, "top", 4, 0, 0, E, 0, 0)]  # 4볼 — 불가능
    stable = stabilize(good + noise)
    assert [r.outs for r in stable] == [0, 1]


def test_stabilize_rejects_score_going_down_and_inning_going_back():
    readings = twice(R(0, away=2), R(10, away=1), R(20, inning=2), R(30, away=3))
    assert [(r.away, r.inning) for r in stabilize(readings)] == [(2, 3), (3, 3)]


def test_half_change_counts_remaining_outs_and_top_to_bot_is_forward():
    tr, = transitions(stabilize(twice(R(0, outs=1), R(10, half="bot"))))
    assert tr.half_change and tr.outs_made == 2


def test_base_shapes():
    assert advanced_all_one_base((True, False, True), (False, True, False))
    assert not advanced_all_one_base((True, False, False), (False, False, True))
    assert single_runner_advance((True, False, False), (False, True, False)) == 2
    assert single_runner_advance((True, True, False), (True, False, True)) == 3
    assert single_runner_advance((True, True, False), (False, True, True)) is None
    assert runner_removed((True, False, False), E)


# ── 판정 규칙 ───────────────────────────────────────────────────────────
def test_balk_needs_all_runners_one_base_and_a_cue():
    before, after = R(0, bases=(True, False, False), balls=1), R(10, bases=(False, True, False),
                                                                  balls=1)
    j = one(before, after, [Cue(8, "speech", "보크")])
    assert (j.kind, j.code) == ("call", "balk") and j.confidence >= 0.5
    assert any("보크" in e for e in j.evidence)
    assert one(before, after).confidence < 0.5  # 모양만 보크 — 카드 안 뜬다


def test_steal_vs_wild_pitch_by_cue():
    before, after = R(0, bases=(True, False, False), strikes=1), R(10, bases=(False, True, False),
                                                                     strikes=1)
    assert one(before, after, [Cue(7, "slide")]).code == "steal"
    assert one(before, after, [Cue(7, "slide"), Cue(9, "speech", "폭투")]).code == "wild_pitch"


def test_steal_at_zero_zero_count_is_not_mistaken_for_a_new_batter():
    j = one(R(0, bases=(True, False, False)), R(10, bases=(False, True, False)),
            [Cue(6, "speech", "2루 도루 성공")])
    assert j.code == "steal"


def test_caught_stealing():
    j = one(R(0, bases=(True, False, False), balls=1), R(10, outs=1, balls=1),
            [Cue(7, "slide")])
    assert (j.kind, j.code, j.detail["caught"]) == ("steal", "caught_stealing", True)


def test_walk_from_three_balls_without_any_cue():
    j = one(R(0, balls=3, strikes=1), R(10, bases=(True, False, False)))
    assert j.code == "walk" and j.confidence >= 0.5


def test_hbp_and_dropped_third_strike():
    reach = (R(0, strikes=2), R(10, bases=(True, False, False)))
    assert one(*reach, [Cue(5, "hit_by_pitch")]).code == "hbp"
    j = one(*reach, [Cue(5, "catcher_miss"), Cue(6, "speech", "낫아웃")])
    assert (j.kind, j.code) == ("call", "dropped_third_strike")
    out = one(R(0, strikes=2), R(10, outs=1), [Cue(5, "catcher_miss")])
    assert out.code == "dropped_third_strike" and out.detail["out"] is True


def test_homerun_sac_fly_infield_fly_double_play():
    hr = one(R(0, bases=(True, False, False)), R(10, away=2), [Cue(5, "ball_over_fence")])
    assert hr.code == "homerun"
    sf = one(R(0, bases=(False, False, True)), R(10, outs=1, away=1), [Cue(5, "deep_fly")])
    assert sf.code == "sac_fly"
    iff = one(R(0, bases=(True, True, False)), R(10, outs=1, bases=(True, True, False)),
              [Cue(8, "speech", "인필드플라이")])
    assert iff.code == "infield_fly"
    dp = one(R(0, bases=(True, False, False)), R(10, outs=2))
    assert dp.code == "double_play"
    k_cs = one(R(0, bases=(True, False, False), strikes=2), R(10, outs=2),
               [Cue(5, "speech", "삼진")])
    assert k_cs.code == "strikeout"  # 삼진+도루실패는 병살타 카드가 아니다


def test_pitch_only_transition_is_not_a_play():
    tr, = transitions(stabilize(twice(R(0), R(10, balls=1))))
    assert judge_transition(tr, []) is None


def test_pitching_change_takes_half_from_scoreboard():
    readings = twice(R(0, outs=2), R(100, inning=3, half="bot"))
    js = judge(readings, [Cue(105, "pitching_change"), Cue(130, "speech", "투수 교체")])
    subs = [j for j in js if j.kind == "sub"]
    assert len(subs) == 1 and subs[0].detail == {"inning": 3, "half": "bot"}


# ── 기존 엔진으로 연결 ──────────────────────────────────────────────────
def test_events_replay_in_game_sim_and_fire_detectors():
    readings = twice(
        R(0, inning=1), R(30, inning=1, balls=3), R(60, inning=1, bases=(True, False, False)),
        R(90, inning=1, bases=(False, True, False)),
        R(120, inning=1, outs=1, bases=(False, True, False)),
    )
    cues = [Cue(85, "slide"), Cue(88, "speech", "도루")]
    events = to_relay_events(judge(readings, cues))
    assert [e.kind for e in events] == ["result", "steal", "result"]
    assert events[0].detail["source"] == "video" and "evidence" in events[0].detail
    meta = GameMeta(id="V", date="", stadium="", away_team="A", home_team="H")
    feed = GameFeed(meta=meta, events=events)
    state = replay(events, "A", "H")
    assert state.outs == 1 and state.bases == (False, True, False)
    rules = {s.rule_id for s in detect(feed)}
    assert {"walk", "steal"} <= rules


def test_low_confidence_keeps_state_but_drops_the_label():
    readings = twice(R(0, bases=(True, False, False), balls=1),
                     R(10, bases=(False, True, False), balls=1))
    (ev,) = to_relay_events(judge(readings, []))  # 단서 없음 → '주자 진루'(0.4)
    assert ev.kind == "result" and ev.detail["result"] == "reach"
    assert ev.detail["bases_after"] == [2]


# ── 채점 ───────────────────────────────────────────────────────────────
def _sit(rule, t):
    from app.domain.models import GameState, Situation

    st = GameState(away_team="A", home_team="H")
    return Situation(id=f"{rule}{t}", t=t, rule_id=rule, term_id=rule, category="basic_rules",
                     label=rule, trigger_text="", event_ids=[], state=st)


def test_grade_matches_within_tolerance_once():
    truth = [_sit("balk", 100), _sit("balk", 500), _sit("steal", 300)]
    video = [_sit("balk", 410), _sit("balk", 790), _sit("steal", 900)]
    scores = {s.rule_id: s for s in grade(truth, video, offset=300, tolerance=15)}
    assert (scores["balk"].matched, scores["balk"].expected, scores["balk"].found) == (2, 2, 2)
    assert scores["steal"].matched == 0
    total = summary(list(scores.values()))
    assert total["matched"] == 2
    windowed = {s.rule_id: s for s in grade(truth, video, offset=300, tolerance=15,
                                             window=(0, 450))}
    assert windowed["balk"].expected == 1  # 영상이 덮지 않은 구간의 정답은 세지 않는다


# ── 실경기 상한 측정 (완벽한 눈) ─────────────────────────────────────────
@pytest.fixture(scope="module")
def sim():
    spec = importlib.util.spec_from_file_location(
        "simulate_video_judge", ROOT / "scripts" / "simulate_video_judge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_perfect_eye_on_real_games_matches_the_relay(sim):
    totals = sim.run(["20260920HHLG02026", "20260920HTNC02026", "20260920OBKT02026"],
                     cue_drop=0.0, misread=0.0, seed=0)
    expected = sum(v[0] for v in totals.values())
    matched = sum(v[2] for v in totals.values())
    found = sum(v[1] for v in totals.values())
    assert expected >= 30
    assert matched == expected == found  # 판정 규칙 자체는 실경기에서 빠짐없이 맞는다


def test_noise_degrades_gracefully(sim):
    totals = sim.run(["20260920HHLG02026", "20260920HTNC02026", "20260920OBKT02026"],
                     cue_drop=0.3, misread=0.1, seed=3)
    expected = sum(v[0] for v in totals.values())
    matched = sum(v[2] for v in totals.values())
    found = sum(v[1] for v in totals.values())
    assert matched / expected >= 0.8
    assert matched / found >= 0.85


# ── VLM 출력 v2 → 판독·단서 → 영상 판정 데이터 ───────────────────────────
def test_parse_analysis_scoreboard_speech_and_events(tmp_path):
    import json

    from app.adapters.video.base import VideoClip, parse_analysis

    clip = VideoClip(tmp_path / "c.mp4", 600.0, 60.0)
    text = "<think>..</think><answer>" + json.dumps({
        "scoreboard": [
            {"t": 2, "inning": 3, "half": "말", "balls": 1, "strikes": 2, "outs": 1,
             "bases": "1,3", "away": 2, "home": 4},
            {"t": 5, "inning": 3, "half": "bot", "balls": 1, "strikes": 2, "outs": 1,
             "bases": [True, False, True], "away": 2, "home": 4},
            {"t": 9, "inning": "?", "half": "top", "balls": 0, "strikes": 0, "outs": 0,
             "bases": [], "away": 0, "home": 0},  # 못 읽은 값 — 버린다
        ],
        "events": [{"start": 8, "end": 9, "type": "slide", "description": "2루 슬라이딩"}],
        "speech": [{"t": 8.5, "text": "도루 성공입니다"}, {"t": 99, "text": "청크 밖"}],
    }, ensure_ascii=False) + "</answer>"
    a = parse_analysis(text, clip, "p", "m")
    assert [(r["t"], r["half"], r["bases"]) for r in a.scoreboard] == [
        (602.0, "bot", [1, 3]), (605.0, "bot", [1, 3])]
    assert [(e.t_start, e.event_type) for e in a.events] == [(608.0, "slide")]
    assert a.speech == [{"t": 608.5, "text": "도루 성공입니다"}]


def test_build_video_feed_from_snapshot_serves_through_engine(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "build_video_feed", ROOT / "scripts" / "build_video_feed.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def board(t, **kw):
        base = {"t": t, "inning": 1, "half": "top", "balls": 0, "strikes": 0, "outs": 0,
                "bases": [], "away": 0, "home": 0}
        return [{**base, **kw}, {**base, **kw, "t": t + 1}]

    snap = {
        "provider": "fake", "model": "fake", "prompt_version": "video-scoreboard-v2",
        "scoreboard": board(10) + board(40, balls=3) + board(70, bases=[1])
        + board(100, bases=[2]),
        "events": [{"t_start": 95, "t_end": 96, "event_type": "slide", "description": "슬라이딩"}],
        "speech": [{"t": 97, "text": "도루!"}],
    }
    feed = mod.build("G1", snap, {"away_team": "A", "home_team": "H"})
    assert feed["source"] == "video" and feed["game"]["has_video"] is True
    assert feed["analysis"]["judgments"] == {"walk": 1, "steal": 1}
    import json

    from app.adapters.relay.fixture import _parse_fixture

    path = tmp_path / "video_G1.json"
    path.write_text(json.dumps(feed, ensure_ascii=False), encoding="utf-8")
    parsed = _parse_fixture(path)
    situations = detect(parsed)
    steal = next(s for s in situations if s.rule_id == "steal")
    assert any(r.startswith("영상 근거:") and "slide" in r for r in steal.reasons)
    assert "영상 판정" in steal.reasons[0]


# ── 영상 판정 + 데이터 보강 (선수·구종·기록은 데이터) ─────────────────────
def test_enrich_adds_names_pitch_type_and_atbats_without_using_data_results():
    from app.domain.enrich import enrich
    from app.domain.models import RelayEvent

    def ev(id_, t, kind, **kw):
        return RelayEvent(id=id_, t=t, inning=1, half="top", kind=kind, text="", **kw)

    data = [
        ev("a1", 100, "atbat", batter="가나다", pitcher="투수A",
           detail={"stats": {"season_avg": "0.300", "today": "오늘 첫 타석"}}),
        ev("p1", 110, "pitch", detail={"result": "ball", "pitch_type": "직구", "speed": 145}),
        ev("p2", 125, "pitch", detail={"result": "swing_strike", "pitch_type": "포크볼",
                                       "speed": 132}),
        ev("r1", 125, "result", detail={"result": "strikeout"}),  # 데이터의 결과 — 쓰면 안 된다
        ev("s1", 200, "sub", detail={"sub_type": "pitcher", "pitcher": "투수B"}),
    ]
    video = [  # 영상 시각 = 데이터 + 50
        ev("v1", 176, "result", detail={"result": "strikeout", "outs_made": 1}),
        ev("v2", 255, "sub", detail={"sub_type": "pitcher"}),
    ]
    stable = [R(140), R(300, outs=1)]
    out = enrich(video, data, offset=50, stable=stable)
    kinds = [(e.t, e.kind) for e in out]
    assert kinds == [(150, "atbat"), (176, "pitch"), (176, "result"), (255, "sub")]
    atbat = out[0]
    # 이닝은 영상 점수판을 따른다
    assert (atbat.batter, atbat.pitcher, atbat.inning) == ("가나다", "투수A", 3)
    assert atbat.detail["stats"]["season_avg"] == "0.300"
    pitch = out[1]
    assert pitch.detail["pitch_type"] == "포크볼" and pitch.detail["decisive"] is True
    assert pitch.detail["result"] == "unknown"  # 데이터의 투구 결과는 가져오지 않는다
    assert out[2].batter == "가나다" and out[2].id == "v1"
    assert out[3].detail["pitcher"] == "투수B"
    assert not any(e.id == "dr1" for e in out)  # 데이터 결과 이벤트는 들어오지 않는다


def test_enrich_without_scoreboard_returns_video_only():
    from app.domain.enrich import enrich
    from app.domain.models import RelayEvent

    v = [RelayEvent(id="v1", t=1, inning=1, half="top", kind="result", text="",
                    detail={"result": "walk"})]
    assert enrich(v, [], offset=0, stable=[]) == v
