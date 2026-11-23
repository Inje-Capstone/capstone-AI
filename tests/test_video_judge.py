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
    fold_cleared,
    forced_to_first,
    possible_step,
    runner_removed,
    single_runner_advance,
    stabilize,
    third_out_cleared,
    transitions,
)
from app.domain.video_judge import (
    Cue,
    from_snapshot,
    judge,
    judge_calls,
    judge_checked,
    judge_from_cues,
    judge_transition,
    observe,
    pose_cues,
    split_cues,
    to_relay_events,
)

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
    noise = [
        R(5, outs=2, bases=(True, True, True)),  # 한 번만 보인 오독
        Reading(6.0, 3, "top", 4, 0, 0, E, 0, 0),
    ]  # 4볼 — 불가능
    stable = stabilize(good + noise)
    assert [r.outs for r in stable] == [0, 1]


def test_stabilize_rejects_score_going_down_and_inning_going_back():
    readings = twice(R(0, away=2), R(10, away=1), R(20, inning=2), R(30, away=3))
    assert [(r.away, r.inning) for r in stabilize(readings)] == [(2, 3), (3, 3)]


def test_a_misread_later_inning_does_not_lock_out_the_real_ones():
    """타순 4를 회차로 읽은 판독 몇 장이 진짜 2회 판독 수십 장을 버리게 하면 안 된다."""
    reads = [R(0, inning=2), R(5, inning=4), R(6, inning=4), R(10, inning=2, outs=1),
             R(15, inning=2, outs=2), R(18, inning=2, outs=2, balls=1),
             R(20, inning=3, half="bot"), R(25, inning=3, half="bot", outs=1)]
    stable = stabilize(reads, min_repeat=1)
    assert [(r.inning, r.half, r.outs) for r in stable] == [
        (2, "top", 0), (2, "top", 1), (2, "top", 2), (2, "top", 2), (3, "bot", 0), (3, "bot", 1)]
    # 반복을 요구하는 경기 영상 방식도 같다 — 오독 덩어리가 짧으면 진 사슬이다
    assert [r.inning for r in stabilize(twice(*reads), min_repeat=2)] == [2, 2, 2, 2, 3, 3]


def test_stabilize_never_lets_outs_go_down_within_a_half():
    """같은 반 이닝에서 아웃이 줄면 오독 — 받아들이면 다음 판독에서 아웃을 또 센다."""
    readings = twice(R(0, outs=1), R(10, outs=0, balls=1), R(20, outs=2))
    assert [r.outs for r in stabilize(readings)] == [1, 2]


def test_possible_step_checks_runner_out_run_bookkeeping():
    on1 = (True, False, False)
    assert possible_step(R(0), R(1, strikes=1))  # 투구 하나
    assert possible_step(R(0, strikes=2), R(1, outs=1))  # 삼진 — 새 타자
    assert possible_step(R(0, balls=3), R(1, bases=on1))  # 볼넷 — 타자가 주자로
    assert possible_step(R(0, bases=on1), R(1, bases=(False, True, False)))  # 도루
    assert not possible_step(R(0), R(1, outs=2, balls=1))  # 아웃 둘이 갑자기
    assert not possible_step(R(0, strikes=2), R(1))  # 새 타자인데 아무 일도 없음
    assert not possible_step(R(0, bases=on1), R(1))  # 주자가 그냥 사라짐
    assert possible_step(R(0, outs=2), R(1, half="bot"))  # 반 이닝이 바뀌면 순서만


def test_single_read_spike_is_dropped_and_replay_never_overflows_outs():
    """점수판 전용 판독은 상태마다 한 장 — 튄 한 장이 아웃을 두 번 세게 하면 경기 상태가 깨졌다.

    시뮬레이션(scripts/simulate_video_judge.py --reads 1 --min-repeat 1 --misread 0.1) 실측:
    GameStateError '아웃 카운트가 3을 넘는다'. 앞뒤와 수지가 안 맞는 한 장을 버린다.
    """
    readings = [
        R(0),
        R(10, outs=1),
        R(20, outs=2),
        R(30, balls=1),  # 튄 판독 — 2아웃에서 0아웃으로
        R(40, outs=2, strikes=1),  # 예전엔 여기서 아웃 둘을 또 셌다 (2 + 2)
        R(50, half="bot"),
    ]
    assert [r.outs for r in stabilize(readings, min_repeat=1)] == [0, 1, 2, 2, 0]
    events = to_relay_events(judge(readings, [], min_repeat=1))
    replay(events, "A", "H")  # 예전엔 GameStateError


def test_skipped_plays_before_a_half_change_are_not_a_spike():
    """하이라이트가 홈런 뒤 다음 타석으로 건너뛰면 '1사 1루 → 2사 주자 없음·2점'이 한 번에 보인다.

    한 플레이로는 불가능한 모양이라, 다음 판독이 반 이닝이 바뀐 것(늘 '가능')이면 튄 판독으로
    버려져 홈런 + 아웃 둘이 이닝 끝 전이 하나('아웃')로 합쳐졌다(2026-11-08 한화-삼성 정은원 홈런).
    두 번 이상 읽혔고 앞뒤 모두 타석 여러 개로 설명되면 남긴다.
    """
    on1 = (True, False, False)
    seen = [R(0, outs=1, bases=on1, balls=1), R(30, outs=2, away=2), R(31, outs=2, away=2),
            R(40, half="bot", away=2)]
    assert [r.outs for r in stabilize(seen, min_repeat=1)] == [1, 2, 0]
    cues = [_view(2, "in_play")]
    assert [j.code for j in judge(seen, cues, min_repeat=1)][0] == "homerun"
    # 한 장뿐이면 예전처럼 튄 판독, 아무 일 없이 새 타자(카운트만 0-0)도 튄 판독
    assert [r.outs for r in stabilize(seen[:2] + seen[3:], min_repeat=1)] == [1, 0]
    reset = [R(0, bases=on1, balls=3, strikes=1), R(5, bases=on1), R(6, bases=on1),
             R(40, half="bot")]
    assert [r.balls for r in stabilize(reset, min_repeat=1)] == [3, 0]
    # 3아웃 판독은 남기지 않는다 — 다음 반 이닝 전이가 아웃 없는 '출루'가 됐다(독립 리뷰)
    three = [R(0, outs=1), R(10, outs=3), R(11, outs=3), R(20, half="bot")]
    assert [j.code for j in judge(three, [], min_repeat=1)] == ["out"]


def test_skipped_half_inning_before_a_strikeout_pair_is_not_a_spike():
    """반 이닝을 통째로 건너뛴 뒤 두 번 읽힌 새 이닝 첫 판독 — 뒤로도 삼진 둘을 건너뛰어도 남긴다.

    PO 1차전(2026-11-08): '5회말 2사 1루 3-2' → '6회말 무사 1-2'(두 장) → '6회말 2사'. 가운데가
    버려져 5회말 세 번째 아웃(삼진)이 6회말 아웃 둘과 한 전이로 합쳐져 놓쳤다. 점수가 줄면 오독.
    """
    on1 = (True, False, False)
    seen = [R(0, outs=2, bases=on1, balls=3, strikes=2, half="bot"),
            R(20, inning=4, half="bot", balls=1, strikes=2), R(20.5, inning=4, half="bot", balls=1,
                                                                 strikes=2),
            R(40, inning=4, half="bot", outs=2)]
    assert [(r.inning, r.outs) for r in stabilize(seen, min_repeat=1)] == [(3, 2), (4, 0), (4, 2)]
    # 점수를 높게 잘못 읽은 두 장(1:2) 뒤 다음 반 이닝이 0:0이면 예전처럼 튄 판독
    misread = [R(0, half="bot"), R(10, half="bot", away=1, home=2),
               R(11, half="bot", away=1, home=2), R(20, inning=4)]
    assert [(r.inning, r.away) for r in stabilize(misread, min_repeat=1)] == [(3, 0), (4, 0)]


def test_half_change_counts_remaining_outs_and_top_to_bot_is_forward():
    (tr,) = transitions(stabilize(twice(R(0, outs=1), R(10, half="bot"))))
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
    before, after = (
        R(0, bases=(True, False, False), balls=1),
        R(10, bases=(False, True, False), balls=1),
    )
    j = one(before, after, [Cue(8, "speech", "보크")])
    assert (j.kind, j.code) == ("call", "balk") and j.confidence >= 0.5
    assert any("보크" in e for e in j.evidence)
    assert one(before, after).confidence < 0.5  # 모양만 보크 — 카드 안 뜬다


def test_steal_vs_wild_pitch_by_cue():
    before, after = (
        R(0, bases=(True, False, False), strikes=1),
        R(10, bases=(False, True, False), strikes=1),
    )
    assert one(before, after, [Cue(7, "slide")]).code == "steal"
    assert one(before, after, [Cue(7, "slide"), Cue(9, "speech", "폭투")]).code == "wild_pitch"


def test_steal_at_zero_zero_count_is_not_mistaken_for_a_new_batter():
    j = one(
        R(0, bases=(True, False, False)),
        R(10, bases=(False, True, False)),
        [Cue(6, "speech", "2루 도루 성공")],
    )
    assert j.code == "steal"


def test_caught_stealing():
    j = one(R(0, bases=(True, False, False), balls=1), R(10, outs=1, balls=1), [Cue(7, "slide")])
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
    iff = one(
        R(0, bases=(True, True, False)),
        R(10, outs=1, bases=(True, True, False)),
        [Cue(8, "speech", "인필드플라이")],
    )
    assert iff.code == "infield_fly"
    dp = one(R(0, bases=(True, False, False)), R(10, outs=2))
    assert dp.code == "double_play"
    k_cs = one(
        R(0, bases=(True, False, False), strikes=2), R(10, outs=2), [Cue(5, "speech", "삼진")]
    )
    assert k_cs.code == "strikeout"  # 삼진+도루실패는 병살타 카드가 아니다


def test_pitch_only_transition_is_not_a_play():
    (tr,) = transitions(stabilize(twice(R(0), R(10, balls=1))))
    assert judge_transition(tr, []) is None


def test_pitching_change_takes_half_from_scoreboard():
    readings = twice(R(0, outs=2), R(100, inning=3, half="bot"))
    js = judge(readings, [Cue(105, "pitching_change"), Cue(130, "speech", "투수 교체")])
    subs = [j for j in js if j.kind == "sub"]
    assert len(subs) == 1 and subs[0].detail == {"inning": 3, "half": "bot"}


# ── 기존 엔진으로 연결 ──────────────────────────────────────────────────
def test_events_replay_in_game_sim_and_fire_detectors():
    readings = twice(
        R(0, inning=1),
        R(30, inning=1, balls=3),
        R(60, inning=1, bases=(True, False, False)),
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
    readings = twice(
        R(0, bases=(True, False, False), balls=1), R(10, bases=(False, True, False), balls=1)
    )
    (ev,) = to_relay_events(judge(readings, []))  # 단서 없음 → '주자 진루'(0.4)
    assert ev.kind == "result" and ev.detail["result"] == "reach"
    assert ev.detail["bases_after"] == [2]


# ── 채점 ───────────────────────────────────────────────────────────────
def _sit(rule, t):
    from app.domain.models import GameState, Situation

    st = GameState(away_team="A", home_team="H")
    return Situation(
        id=f"{rule}{t}",
        t=t,
        rule_id=rule,
        term_id=rule,
        category="basic_rules",
        label=rule,
        trigger_text="",
        event_ids=[],
        state=st,
    )


def test_grade_matches_within_tolerance_once():
    truth = [_sit("balk", 100), _sit("balk", 500), _sit("steal", 300)]
    video = [_sit("balk", 410), _sit("balk", 790), _sit("steal", 900)]
    scores = {s.rule_id: s for s in grade(truth, video, offset=300, tolerance=15)}
    assert (scores["balk"].matched, scores["balk"].expected, scores["balk"].found) == (2, 2, 2)
    assert scores["steal"].matched == 0
    total = summary(list(scores.values()))
    assert total["matched"] == 2
    windowed = {
        s.rule_id: s for s in grade(truth, video, offset=300, tolerance=15, window=(0, 450))
    }
    assert windowed["balk"].expected == 1  # 영상이 덮지 않은 구간의 정답은 세지 않는다


# ── 실경기 상한 측정 (완벽한 눈) ─────────────────────────────────────────
@pytest.fixture(scope="module")
def sim():
    spec = importlib.util.spec_from_file_location(
        "simulate_video_judge", ROOT / "scripts" / "simulate_video_judge.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_perfect_eye_on_real_games_matches_the_relay(sim):
    totals = sim.run(
        ["20260920HHLG02026", "20260920HTNC02026", "20260920OBKT02026"],
        cue_drop=0.0,
        misread=0.0,
        seed=0,
    )
    expected = sum(v[0] for v in totals.values())
    matched = sum(v[2] for v in totals.values())
    found = sum(v[1] for v in totals.values())
    assert expected >= 30
    assert matched == expected == found  # 판정 규칙 자체는 실경기에서 빠짐없이 맞는다


def test_noise_degrades_gracefully(sim):
    totals = sim.run(
        ["20260920HHLG02026", "20260920HTNC02026", "20260920OBKT02026"],
        cue_drop=0.3,
        misread=0.1,
        seed=3,
    )
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
    text = (
        "<think>..</think><answer>"
        + json.dumps(
            {
                "scoreboard": [
                    {
                        "t": 2,
                        "inning": 3,
                        "half": "말",
                        "balls": 1,
                        "strikes": 2,
                        "outs": 1,
                        "bases": "1,3",
                        "away": 2,
                        "home": 4,
                    },
                    {
                        "t": 5,
                        "inning": 3,
                        "half": "bot",
                        "balls": 1,
                        "strikes": 2,
                        "outs": 1,
                        "bases": [True, False, True],
                        "away": 2,
                        "home": 4,
                    },
                    {
                        "t": 9,
                        "inning": "?",
                        "half": "top",
                        "balls": 0,
                        "strikes": 0,
                        "outs": 0,
                        "bases": [],
                        "away": 0,
                        "home": 0,
                    },  # 못 읽은 값 — 버린다
                ],
                "events": [{"start": 8, "end": 9, "type": "slide", "description": "2루 슬라이딩"}],
                "speech": [{"t": 8.5, "text": "도루 성공입니다"}, {"t": 99, "text": "청크 밖"}],
            },
            ensure_ascii=False,
        )
        + "</answer>"
    )
    a = parse_analysis(text, clip, "p", "m")
    assert [(r["t"], r["half"], r["bases"]) for r in a.scoreboard] == [
        (602.0, "bot", [1, 3]),
        (605.0, "bot", [1, 3]),
    ]
    assert [(e.t_start, e.event_type) for e in a.events] == [(608.0, "slide")]
    assert a.speech == [{"t": 608.5, "text": "도루 성공입니다"}]


def test_build_video_feed_from_snapshot_serves_through_engine(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "build_video_feed", ROOT / "scripts" / "build_video_feed.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def board(t, **kw):
        base = {
            "t": t,
            "inning": 1,
            "half": "top",
            "balls": 0,
            "strikes": 0,
            "outs": 0,
            "bases": [],
            "away": 0,
            "home": 0,
        }
        return [{**base, **kw}, {**base, **kw, "t": t + 1}]

    snap = {
        "provider": "fake",
        "model": "fake",
        "prompt_version": "video-scoreboard-v2",
        "scoreboard": board(10) + board(40, balls=3) + board(70, bases=[1]) + board(100, bases=[2]),
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


def test_full_game_feed_prefers_crop_scoreboard_and_whisper_like_clip_eval():
    """경기 분석도 클립 평가(63%)와 같은 입력 — 점수판 전용 판독(상태마다 한 장) · Whisper 해설.

    VLM은 해설을 '보크'로 지어냈고(실측 유형), 주자 한 명 한 베이스는 보크 모양과도 같다.
    Whisper가 들은 '도루'가 이겨야 한다. 화면 자막은 Whisper가 있어도 남는다.
    """
    spec = importlib.util.spec_from_file_location(
        "build_video_feed", ROOT / "scripts" / "build_video_feed.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def read(t, **kw):
        base = {"t": t, "inning": 1, "half": "top", "balls": 0, "strikes": 0, "outs": 0,
                "bases": [], "away": 0, "home": 0}
        return {**base, **kw}

    snap = {
        "provider": "fake", "model": "fake", "prompt_version": "video-scoreboard-v2",
        "scoreboard": [read(5, inning=7, outs=2)],  # 영상 통째 판독 — 틀렸고 한 번뿐
        "scorebug": {"scoreboard": [read(10), read(40, balls=3), read(70, bases=[1]),
                                    read(100, bases=[2])]},
        "events": [{"t_start": 95, "t_end": 96, "event_type": "slide", "description": "슬라이딩"}],
        "speech": [{"t": 97, "text": "보크입니다"}, {"t": 98, "text": "[자막] 도루 성공"}],
        "asr": {"segments": [{"t": 90, "text": "뛰었습니다 도루!"}, {"t": 120, "text": ""}]},
    }
    readings, cues, min_repeat = observe(snap)
    assert min_repeat == 1 and [r.t for r in readings] == [10, 40, 70, 100]
    # 경기 영상 방식(N초마다)은 같은 상태를 여러 번 읽었다 → 반복으로 거른다
    periodic = {**snap, "scorebug": {**snap["scorebug"], "every": 8}}
    assert observe(periodic)[2] == 2
    # 하이라이트를 N초마다 읽으면 한 상태가 한 번만 읽힌다 — 분석 때 정한 기준을 따른다
    highlight = {**snap, "scorebug": {**snap["scorebug"], "every": 8, "min_repeat": 1}}
    assert observe(highlight)[2] == 1
    speech = [c.text for c in cues if c.kind == "speech"]
    assert speech == ["뛰었습니다 도루!", "[자막] 도루 성공"]

    feed = mod.build("G1", snap, {"away_team": "A", "home_team": "H"})
    a = feed["analysis"]
    assert (a["scoreboard_source"], a["speech_source"]) == ("crop", "whisper")
    assert a["judgments"] == {"walk": 1, "steal": 1}
    # 프론트에 넘길 대표 시점 후보 — 카드가 뜰 장면만, 영상 타임코드로
    assert [m.split(" (")[0] for m in mod.demo_moments(feed)] == ["01:10 walk", "01:40 steal"]


def test_feed_scoreboard_matches_broadcast_when_recording_starts_mid_game():
    """녹화가 5회말 2아웃 1·2루 3:2에서 시작해도 화면 스코어보드가 그대로 — 예전엔 0:0 0아웃."""
    from app.domain.models import RelayEvent

    spec = importlib.util.spec_from_file_location(
        "build_video_feed", ROOT / "scripts" / "build_video_feed.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def read(t, **kw):
        base = {"t": t, "inning": 5, "half": "bot", "balls": 1, "strikes": 1, "outs": 2,
                "bases": [1, 2], "away": 3, "home": 2}
        return {**base, **kw}

    snap = {  # 경기 영상 방식(8초마다) — 상태마다 두 번 이상 읽힘
        "provider": "fake", "model": "fake", "prompt_version": "v",
        "scorebug": {"every": 8, "scoreboard": [
            read(10), read(18), read(26, strikes=2), read(34, strikes=2),
            read(42, balls=0, strikes=0, bases=[2, 3]), read(50, balls=0, strikes=0, bases=[2, 3]),
        ]},
    }
    feed = mod.build("G1", snap, {"away_team": "A", "home_team": "H"})
    events = [RelayEvent(**e) for e in feed["events"]]
    at = {t: replay(events, "A", "H", until_t=t) for t in (20, 40)}
    assert (at[20].away_score, at[20].home_score, at[20].outs) == (3, 2, 2)
    assert (at[20].inning, at[20].half, at[20].bases) == (5, "bot", (True, True, False))
    assert (at[40].balls, at[40].strikes) == (1, 2)
    final = replay(events, "A", "H")
    assert final.bases == (False, True, True) and final.outs == 2


def test_pose_slide_reaches_the_card_as_evidence(tmp_path):
    """VLM이 슬라이딩을 못 봐도 관절 슬라이딩이 도루를 받치고, 카드 근거에 무엇을 봤는지 남는다."""
    spec = importlib.util.spec_from_file_location(
        "build_video_feed", ROOT / "scripts" / "build_video_feed.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    base = {"inning": 1, "half": "top", "balls": 1, "strikes": 0, "outs": 0, "away": 0, "home": 0}
    snap = {
        "provider": "fake", "model": "fake", "prompt_version": "video-scoreboard-v2",
        "scoreboard": [{**base, "t": t, "bases": bases}
                       for t, bases in ((10, [1]), (11, [1]), (40, [2]), (41, [2]))],
        "events": [], "speech": [],
        "pose": {"slides": [{"t0": 30.2, "t1": 31.0, "distance": 1.6, "thrown": False}],
                 "contacts": [{"t0": 30.9, "t1": 31.1, "kind": "slide_tag", "why": ""}]},
    }
    feed = mod.build("G2", snap, {"away_team": "A", "home_team": "H"})
    assert feed["analysis"]["judgments"] == {"steal": 1}
    import json

    from app.adapters.relay.fixture import _parse_fixture

    path = tmp_path / "video_G2.json"
    path.write_text(json.dumps(feed, ensure_ascii=False), encoding="utf-8")
    steal = next(s for s in detect(_parse_fixture(path)) if s.rule_id == "steal")
    said = "영상 근거: [관절] 슬라이딩·다이빙 — 누운 채 몸길이 1.6배 미끄러짐, 30.2~31.0초"
    assert any(r.startswith(said) for r in steal.reasons)


def test_balk_card_says_which_action_was_the_balk(tmp_path):
    """입문자는 '보크'만 들어선 무슨 동작이 반칙인지 모른다 — 다시보기 해설의 설명이 카드 근거로."""
    import json

    from app.adapters.relay.fixture import _parse_fixture

    spec = importlib.util.spec_from_file_location(
        "build_video_feed", ROOT / "scripts" / "build_video_feed.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    base = {"inning": 3, "half": "bot", "balls": 1, "strikes": 1, "outs": 1, "away": 0, "home": 0}
    snap = {
        "provider": "fake", "model": "fake", "prompt_version": "video-scoreboard-v2",
        "scoreboard": [{**base, "t": t, "bases": bases}
                       for t, bases in ((10, [1]), (11, [1]), (40, [2]), (41, [2]))],
        "events": [],
        "speech": [{"t": 30, "text": "아 보크 선언이 나오네요"},
                   {"t": 55, "text": "다시 보면 세트 들어가다 한 번 멈췄다가 움직였어요 보크죠"}],
    }
    feed = mod.build("G3", snap, {"away_team": "A", "home_team": "H"})
    assert feed["analysis"]["judgments"] == {"balk": 1}
    path = tmp_path / "video_G3.json"
    path.write_text(json.dumps(feed, ensure_ascii=False), encoding="utf-8")
    balk = next(s for s in detect(_parse_fixture(path)) if s.rule_id == "balk")
    assert any("반칙 동작: 세트 자세로 들어가다 한 번 멈췄다가" in r for r in balk.reasons)
    assert any("해설이 말한 동작: 이중 동작" in r for r in balk.reasons)


def test_balk_without_described_action_keeps_the_plain_reason():
    readings = twice(R(10, bases=(True, False, False)), R(40, bases=(False, True, False)))
    j = judge(readings, [Cue(30, "speech", "보크입니다")])[0]
    assert j.code == "balk" and "balk_kind" not in j.detail
    assert "balk_kind" not in to_relay_events([j])[0].detail


# ── 영상 판정 + 데이터 보강 (선수·구종·기록은 데이터) ─────────────────────
def test_enrich_adds_names_pitch_type_and_atbats_without_using_data_results():
    from app.domain.enrich import enrich
    from app.domain.models import RelayEvent

    def ev(id_, t, kind, **kw):
        return RelayEvent(id=id_, t=t, inning=1, half="top", kind=kind, text="", **kw)

    data = [
        ev(
            "a1",
            100,
            "atbat",
            batter="가나다",
            pitcher="투수A",
            detail={"stats": {"season_avg": "0.300", "today": "오늘 첫 타석"}},
        ),
        ev("p1", 110, "pitch", detail={"result": "ball", "pitch_type": "직구", "speed": 145}),
        ev(
            "p2",
            125,
            "pitch",
            detail={"result": "swing_strike", "pitch_type": "포크볼", "speed": 132},
        ),
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

    v = [
        RelayEvent(
            id="v1", t=1, inning=1, half="top", kind="result", text="", detail={"result": "walk"}
        )
    ]
    assert enrich(v, [], offset=0, stable=[]) == v


# ── 하이라이트 클립에서 드러난 경우 (2026-10-25) ───────────────────────
def test_walk_off_balk_even_when_scoreboard_clears_bases():
    """끝내기 후 점수판이 주자를 지워도 결승점 + 보크 단서면 보크."""
    before = R(0, bases=(True, True, True), away=6, home=6, inning=9, half="bot")
    after = R(18, away=6, home=7, inning=9, half="bot")
    j = one(before, after, [Cue(12, "pitcher_stops")])
    assert (j.code, j.detail.get("walk_off")) == ("balk", True)


def test_walk_off_without_cue_falls_back_to_normal_rules():
    before = R(0, bases=(True, True, True), away=6, home=6, inning=9, half="bot", balls=3)
    after = R(18, bases=(True, True, True), away=6, home=7, inning=9, half="bot")
    assert one(before, after).code == "walk"  # 만루 3볼 → 밀어내기는 모양만으로


def test_english_commentary_counts_as_speech_cue():
    before = R(0, bases=(False, False, True), outs=1)
    after = R(10, outs=1, home=1)
    j = one(before, after, [Cue(8, "speech", "And that's a BALK, the run scores!")])
    assert j.code == "balk"


def test_walk_off_word_is_not_a_walk():
    before = R(0, bases=(False, True, False), away=2, home=2, inning=9, half="bot")
    after = R(15, away=2, home=3, inning=9, half="bot")
    j = one(before, after, [Cue(10, "speech", "walk-off single!")])
    assert j is None or j.code != "walk"


def test_single_reading_clip_needs_min_repeat_one():
    readings = [R(0, bases=(False, False, True), outs=1), R(10, outs=1, home=1)]
    cues = [Cue(8, "speech", "보크")]
    assert judge(readings, cues) == []  # 경기 전체 기준(2번 연속)으론 못 믿는다
    assert [j.code for j in judge(readings, cues, min_repeat=1)] == ["balk"]


def test_cue_only_prefers_rare_play_and_marks_lower_confidence():
    from app.domain.video_judge import judge_from_cues

    cues = [Cue(3, "speech", "삼진 아웃인데요, 낫아웃입니다!"), Cue(5, "swing_miss", "헛스윙")]
    j = judge_from_cues(cues)
    assert (j.code, j.confidence) == ("dropped_third_strike", 0.5)
    assert judge_from_cues([Cue(1, "pitch", "투구")]) is None


def test_cue_only_trusts_captions_then_event_order():
    from app.domain.video_judge import judge_from_cues

    caption = judge_from_cues([Cue(1, "speech", "[자막] 끝내기 보크!"), Cue(2, "speech", "홈런!")])
    assert (caption.code, caption.confidence) == ("balk", 0.6)
    outs = [Cue(t, "out_at_base", "베이스 아웃") for t in (10, 13, 16)]
    assert judge_from_cues(outs).code == "triple_play"
    sf = [Cue(5, "deep_fly", "외야수에게 잡힌다"), Cue(12, "runner_scores", "득점")]
    assert judge_from_cues(sf).code == "sac_fly"
    steal = [Cue(3, "slide", "홈으로 슬라이딩"), Cue(4, "runner_scores", "득점")]
    assert judge_from_cues(steal).code == "steal"


# ── 규칙 성립 조건 (2026-10-26 저장 규칙 대조) ──────────────────────────
def test_dropped_third_strike_needs_first_base_open_or_two_outs():
    """5.09(a)(3): 1루 주자가 있고 2아웃 전이면 공을 놓쳐도 타자는 아웃이다."""
    blocked = one(
        R(0, outs=1, bases=(True, False, False), strikes=2),
        R(10, outs=1, bases=(True, True, False)),
        [Cue(8, "catcher_miss")],
    )
    assert blocked is None or blocked.code != "dropped_third_strike"
    open_first = one(
        R(0, outs=1, strikes=2), R(10, outs=1, bases=(True, False, False)), [Cue(8, "catcher_miss")]
    )
    assert open_first.code == "dropped_third_strike"


def test_infield_fly_needs_first_and_second_with_less_than_two_outs():
    cue = [Cue(8, "speech", "인필드 플라이 선언")]
    only_first = one(
        R(0, outs=0, bases=(True, False, False)), R(10, outs=1, bases=(True, False, False)), cue
    )
    assert only_first.code != "infield_fly"
    first_second = one(
        R(0, outs=0, bases=(True, True, False)), R(10, outs=1, bases=(True, True, False)), cue
    )
    assert first_second.code == "infield_fly"


def test_balk_keeps_the_count_but_wild_pitch_and_double_steal_add_a_pitch():
    bases = (True, True, False)
    balk = one(
        R(0, bases=bases, balls=1),
        R(10, bases=(False, True, True), balls=1),
        [Cue(8, "pitcher_stops")],
    )
    assert balk.code == "balk"
    double = one(
        R(0, bases=bases, balls=1), R(10, bases=(False, True, True), balls=2), [Cue(8, "slide")]
    )
    assert (double.code, double.detail.get("double")) == ("steal", True)
    shape_only = one(R(0, bases=bases, balls=1), R(10, bases=(False, True, True), balls=2))
    assert shape_only.code != "balk"


def test_sac_fly_needs_less_than_two_outs_before():
    two_outs = one(
        R(0, outs=2, bases=(False, False, True)),
        R(10, outs=0, home=1, half="bot"),
        [Cue(5, "deep_fly")],
    )
    assert two_outs is None or two_outs.code != "sac_fly"


def test_sac_fly_from_shape_when_the_forced_runner_on_first_held():
    """만루·1사에서 아웃 하나·3루 주자만 득점·1·2루 그대로 — 땅볼이면 1루 주자는 포스로 뛴다.

    해설에 뜬공 말이 없어도(삼성-LG 희생플라이 0-2, 2026-11-08) 공이 맞아 나간 화면이면 희생플라이.
    """
    loaded, after = (True, True, True), (True, True, False)
    sf = one(R(0, outs=1, bases=loaded, strikes=1), R(10, outs=2, bases=after, away=1),
             [_view(4, "in_play")])
    assert (sf.code, sf.detail.get("shape")) == ("sac_fly", True)
    # 공이 안 맞았으면(폭투로 득점 + 삼진 같은 것) 아니다
    no_view = one(R(0, outs=1, bases=loaded, strikes=1), R(10, outs=2, bases=after, away=1))
    assert no_view.code != "sac_fly"
    # 만루가 아니면 가르지 않는다 — 2·3루는 땅볼 아웃 사이 득점, 1·3루는 2루 포스아웃 + 타자 1루
    # + 득점(야수선택)과 모양이 같다
    second_third = one(R(0, outs=1, bases=(False, True, True)),
                       R(10, outs=2, bases=(False, True, False), away=1), [_view(4, "in_play")])
    first_third = one(R(0, outs=1, bases=(True, False, True)),
                      R(10, outs=2, bases=(True, False, False), away=1), [_view(4, "in_play")])
    assert second_third.code != "sac_fly" and first_third.code != "sac_fly"


def test_pose_slide_supports_steal_and_shows_what_was_seen():
    """관절로 잰 슬라이딩은 VLM 'slide'와 같은 단서 — 카드 근거에 무엇을 봤는지 남긴다."""
    snap = {
        "scoreboard": [
            {"t": 0, "inning": 3, "half": "top", "balls": 1, "strikes": 0, "outs": 0,
             "bases": [1], "away": 0, "home": 0},
            {"t": 10, "inning": 3, "half": "top", "balls": 1, "strikes": 1, "outs": 0,
             "bases": [2], "away": 0, "home": 0},
        ],
        "pose": {"slides": [{"t0": 6.0, "t1": 6.6, "distance": 1.7}],
                 "contacts": [{"t0": 6.6, "t1": 6.7, "kind": "slide_tag", "why": ""},
                              {"t0": 7.0, "t1": 7.1, "kind": "contact", "why": ""}]},
    }
    readings, cues = from_snapshot(snap)
    assert [c.kind for c in cues] == ["slide", "slide_tag"]
    [j] = judge(readings, cues, min_repeat=1)
    assert j.code == "steal"
    said = "[관절] 슬라이딩·다이빙 — 누운 채 몸길이 1.7배 미끄러짐, 6.0~6.6초"
    assert any(said in e for e in j.evidence)


def test_scoring_slide_after_a_hit_is_not_home_steal():
    """2루타에 1루 주자가 홈으로 슬라이딩 — 타구가 12초 전이어도 홈스틸이 아니다."""
    cues = [Cue(0.0, "contact"), Cue(11.0, "slide", "[관절] 슬라이딩·다이빙 11.0~11.5초"),
            Cue(12.0, "runner_scores")]
    assert judge_from_cues(cues) is None
    assert judge_from_cues(cues[1:]).code == "steal"  # 타구가 없었으면 홈스틸


def test_pose_collision_confirms_home_collision_call():
    """해설 '홈 충돌' + 관절로 잰 충돌 → 실제로 부딪힌 것이 확인된 심판 선언."""
    said = Cue(10.0, "speech", "홈 충돌 상황입니다")
    assert judge_calls([said]) == []
    [j] = judge_calls([said, Cue(9.5, "collision", "[관절] 빠르게 부딪혀 넘어짐")])
    assert j.code == "home_collision" and "[관절] 빠르게 부딪혀 넘어짐 (10s)" in j.evidence


def test_thrown_dive_evidence_says_it_was_thrown():
    """몸을 던진 다이빙은 누운 뒤엔 거의 안 움직인다 — 누운 채 미끄러졌다고 쓰지 않는다."""
    [cue] = from_snapshot({"pose": {"slides": [
        {"t0": 0.8, "t1": 1.5, "distance": 1.7, "thrown": True}]}})[1]
    assert cue.text == "[관절] 슬라이딩·다이빙 — 달려와 몸을 던져 몸길이 1.7배 이동, 0.8~1.5초"


# ── 영상 먼저, 해설은 검사 ─────────────────────────────────────────────────
FIRST = (True, False, False)
SECOND = (False, True, False)


def checked(cues, same_count=False):
    after = R(10, bases=SECOND, balls=0 if same_count else 1)
    (j,) = judge_checked(twice(R(0, bases=FIRST), after), cues)
    return j


def test_split_cues_keeps_captions_with_what_was_seen():
    slide, caption = Cue(1, "slide"), Cue(2, "speech", "[자막] 보크")
    talk = Cue(3, "speech", "보크")
    assert split_cues([slide, caption, talk]) == ([slide, caption], [talk])


def test_video_judgment_stands_when_commentary_is_silent():
    j = checked([Cue(5, "slide")])
    assert (j.code, j.detail["check"]) == ("steal", "video_only")


def test_caption_alone_does_not_count_as_commentary():
    j = checked([Cue(5, "slide"), Cue(6, "speech", "[자막] 도루 성공")])
    assert (j.code, j.detail["check"]) == ("steal", "video_only")


def test_commentary_confirms_video_judgment():
    j = checked([Cue(5, "slide"), Cue(6, "speech", "도루 성공입니다")])
    assert (j.code, j.detail["check"]) == ("steal", "confirmed")


def test_commentary_names_a_kind_the_video_left_generic():
    j = checked([Cue(6, "speech", "도루 성공입니다")])
    assert (j.code, j.detail["check"], j.detail["video_code"]) == ("steal", "refined", "runner")


def test_disagreement_holds_the_card_but_keeps_the_score_state():
    j = checked([Cue(5, "slide"), Cue(6, "speech", "폭투로 주자가 갑니다")])
    assert (j.code, j.detail["check"], j.detail["video_code"]) == (
        "wild_pitch", "conflict", "steal")
    assert j.confidence < 0.5
    (event,) = to_relay_events([j])
    assert event.detail["result"] == "reach"
    assert event.detail["bases_after"] == [2]
    assert event.detail["check"] == "conflict"


def test_commentary_naming_another_play_holds_even_if_the_rule_ignored_it():
    j = checked([Cue(5, "pitcher_stops"), Cue(6, "speech", "2루 도루 성공입니다")], same_count=True)
    assert (j.detail["check"], j.detail["video_code"]) == ("conflict", "balk")
    assert j.confidence < 0.5


def test_commentary_the_rule_never_read_still_confirms():
    (j,) = judge_checked(twice(R(0, balls=1), R(10, bases=FIRST)),
                         [Cue(5, "hit_by_pitch"), Cue(6, "speech", "몸에 맞는 공입니다")])
    assert (j.code, j.detail["check"]) == ("hbp", "confirmed")


def test_old_talk_outside_the_check_window_does_not_change_the_video_judgment():
    j = checked([Cue(-8, "speech", "아까 폭투로 한 점 줬었죠"), Cue(5, "slide")])
    assert (j.code, j.detail["check"]) == ("steal", "video_only")


def test_more_specific_name_of_the_same_play_is_not_a_disagreement():
    (j,) = judge_checked(twice(R(0, strikes=2), R(10, outs=1)),
                         [Cue(5, "swing_miss"), Cue(6, "speech", "낫아웃입니다")])
    assert j.detail["check"] == "confirmed"
    assert j.confidence >= 0.5


def test_hold_survives_a_lower_card_threshold():
    j = checked([Cue(5, "slide"), Cue(6, "speech", "폭투로 주자가 갑니다")])
    (event,) = to_relay_events([j], min_confidence=0.3)
    assert event.detail["result"] == "reach"


def test_call_event_of_a_two_part_judgment_carries_the_check():
    (j,) = judge_checked(twice(R(0, strikes=2), R(10, bases=FIRST)),
                         [Cue(5, "catcher_miss"), Cue(6, "speech", "낫아웃입니다")])
    call, result = to_relay_events([j])
    assert call.detail["check"] == result.detail["check"] == "confirmed"


def test_batter_took_strike_three_needs_no_commentary():
    cues = [Cue(5, "no_swing", "[관절] 타자가 공을 그냥 보냄")]
    j = one(R(0, strikes=2), R(10, outs=1), cues)
    assert (j.code, j.detail["looking"]) == ("strikeout", True)
    assert j.confidence >= 0.5
    assert "[관절] 타자가 공을 그냥 보냄" in j.evidence[1]


def test_slow_replay_of_a_swing_is_not_a_take():
    cues = [Cue(5, "swing"), Cue(9, "no_swing")]
    j = one(R(0, strikes=2), R(12, outs=1), cues)
    assert j.code == "out"


def test_swing_on_an_earlier_pitch_does_not_hide_the_take():
    cues = [Cue(0.5, "swing"), Cue(15, "no_swing")]
    j = one(R(0, strikes=2), R(18, outs=1), cues)
    assert j.code == "strikeout"


def test_take_on_an_earlier_pitch_says_nothing_about_the_out():
    assert one(R(0, strikes=2), R(19, outs=1), [Cue(1, "no_swing")]).code == "out"


def test_slow_replay_after_the_scoreboard_change_is_ignored():
    cues = [Cue(100, "swing"), Cue(109.5, "no_swing")]
    assert one(R(90, strikes=2), R(104, outs=1), cues).code == "out"


def test_out_on_two_strikes_without_swing_cues_stays_generic():
    assert one(R(0, strikes=2), R(10, outs=1)).code == "out"


def test_runner_out_on_two_strikes_is_not_a_called_third_strike():
    cues = [Cue(5, "no_swing")]
    j = one(R(0, bases=FIRST, strikes=2), R(10, outs=1, strikes=2), cues)
    assert j.code != "strikeout"


def test_take_during_a_steal_of_home_is_not_a_called_third_strike():
    cues = [Cue(5, "no_swing")]
    third = (False, False, True)
    j = one(R(0, bases=third, strikes=2), R(10, outs=1, away=1), cues)
    assert j.code != "strikeout"


def test_official_walk_stays_when_the_batter_seemed_to_swing():
    """심판이 체크스윙을 안 돌았다고 봤으면 볼넷이다 — 영상 관측은 기록만."""
    cues = [Cue(5, "swing"), Cue(7, "speech", "볼넷으로 걸어 나갑니다")]
    j = one(R(0, balls=3), R(10, bases=FIRST), cues)
    assert (j.code, j.detail["observed"]) == ("walk", "swing_on_ball_four")
    (event,) = to_relay_events([j])
    assert event.detail["result"] == "walk"
    assert event.detail["observed"] == "swing_on_ball_four"


def test_asr_spelling_of_walk():
    """2025 한국시리즈 1차전 1회말 — 3-2 파울과 볼넷이 한 투구 화면으로 이어져 관절이 휘두름을
    봤고, 받아쓰기가 '볼넷 출루'를 '본래 출루'로 적어 '출루'였다. '본래'는 보통 말이라
    '출루' 앞·문장 끝만 받는다."""
    walk_said = [Cue(5, "swing"), Cue(7, "speech", "다시 풀카운트. 벗어납니다 본래 출루 홍창기.")]
    assert one(R(0, balls=3, strikes=2), R(10, bases=FIRST), walk_said).code == "walk"
    other = [Cue(5, "swing"), Cue(7, "speech", "본래 컨택이 좋은 타자입니다")]
    assert one(R(0, balls=3, strikes=2), R(10, bases=FIRST), other).code == "reach"


def test_tag_seen_on_a_safe_steal_keeps_the_safe_call():
    cues = [Cue(5, "slide"), Cue(5.5, "slide_tag")]
    j = one(R(0, bases=FIRST), R(10, bases=SECOND, balls=1), cues)
    assert (j.code, j.detail["observed"]) == ("steal", "tag_on_safe_runner")
    assert to_relay_events([j])[0].detail["observed"] == "tag_on_safe_runner"


def test_no_observation_note_when_video_agrees():
    j = one(R(0, bases=FIRST), R(10, bases=SECOND, balls=1), [Cue(5, "slide")])
    assert "observed" not in j.detail
    assert "observed" not in to_relay_events([j])[0].detail


def test_dive_on_grass_then_an_out_is_a_diving_play():
    second = (False, True, False)
    cues = [Cue(5, "dive", "[관절] 수비가 잔디 위로 몸을 던짐")]
    j = one(R(0, bases=second, balls=1, strikes=2), R(10, outs=1, bases=second), cues)
    assert (j.kind, j.code) == ("call", "dive_out")
    call, result = to_relay_events([j])
    assert (call.kind, call.detail["call"]) == ("call", "dive_out")
    assert (result.detail["result"], result.detail["outs_made"]) == ("out", 1)


def test_diving_card_leaves_the_game_state_as_a_plain_out_would():
    """카드만 더한다 — 아웃·카운트·이닝은 일반 아웃 판정과 같아야 한다."""
    from app.domain.game_state import replay

    def state(events):
        s = replay(events, "A", "B")
        return s.inning, s.half, s.outs, s.balls, s.strikes, s.bases, s.game_over

    for before, after in ((R(0, outs=1, balls=1, strikes=2), R(10, outs=2)),
                          (R(0, outs=2, balls=1, strikes=2), R(10, inning=3, half="bot")),
                          (R(0, balls=2, strikes=2), R(10, bases=FIRST))):
        (tr,) = transitions(stabilize(twice(before, after)))
        dive = judge_transition(tr, [Cue(5, "dive")])
        plain = judge_transition(tr, [])
        assert dive.code in ("dive_out", "dive_hit")
        assert state(to_relay_events([dive])) == state(to_relay_events([plain]))


def test_dive_then_the_batter_on_base_is_a_missed_dive():
    j = one(R(0, balls=2, strikes=2), R(10, bases=FIRST), [Cue(5, "dive")])
    assert j.code == "dive_hit"
    call, result = to_relay_events([j])
    assert result.detail["result"] == "reach"


def test_a_dive_before_the_previous_board_is_a_replay_of_an_earlier_play():
    """외야 다이빙 캐치 다시보기(직전 상태 전)가 다음 타석 땅볼을 '몸을 던진 수비'로 만들었다."""
    j = one(R(20, outs=1, balls=1, strikes=2), R(30, outs=2), [Cue(15, "dive")])
    assert j.code == "out"
    later = one(R(20, outs=1, balls=1, strikes=2), R(30, outs=2), [Cue(25, "dive")])
    assert later.code == "dive_out"


def test_half_change_with_the_new_half_under_way_is_two_plays():
    """'3회말 2사' → '4회초 무사 1루' — 3아웃 땅볼과 4회초 안타가 '아웃' 하나로 합쳐졌다.

    앞 반 이닝이 2사였고 사이에 투구 화면이 둘 이상이면 첫 투구에서 앞 반 이닝이 끝났다고 본다.
    """
    seen = twice(R(0, outs=2, half="bot", away=0, home=4), R(20, inning=4, bases=FIRST, home=4))
    views = [_view(4, "in_play"), _view(16, "in_play")]
    js = judge(seen, views)
    assert [(j.code, j.t) for j in js] == [("out", 4.0), ("reach", 20.0)]
    (tr,) = [j.transition for j in js if j.code == "out"]
    assert tr.half_change and tr.after.half == "top" and not any(tr.after.bases)
    # 투구 화면이 하나뿐이거나, 앞 반 이닝이 2사가 아니면(남은 아웃이 여럿) 가르지 않는다
    assert len(judge(seen, views[:1])) == 1
    one_out = twice(R(0, outs=1, half="bot", home=4), R(20, inning=4, bases=FIRST, home=4))
    assert len(judge(one_out, views)) == 1
    # 반 이닝을 건너뛰었으면 가르지 않는다
    skipped = twice(R(0, outs=2, half="bot", home=4), R(20, inning=5, bases=FIRST, home=4))
    assert len(judge(skipped, views)) == 1


def test_split_new_half_does_not_reuse_the_previous_halfs_cues():
    """독립 리뷰 재현 두 가지 — 앞 반 이닝 3아웃의 다이빙이 새 반 이닝 안타에 붙고, 같은 시각
    투구 화면 단서 둘이 두 전이를 같은 시각으로 만들어 대조에서 판정이 섞였다."""
    seen = twice(R(0, outs=2, half="bot", home=4), R(20, inning=4, bases=FIRST, home=4))
    dived = judge(seen, [_view(4, "in_play"), Cue(8, "dive"), _view(16, "in_play")])
    assert [j.code for j in dived] == ["dive_out", "reach"]
    same_time = [_view(20, "no_play"), _view(20, "view_unknown")]
    checked = judge_checked(seen, same_time)
    assert len(checked) == 1
    events = to_relay_events(judge_checked(seen, [_view(4, "in_play"), _view(16, "in_play")]))
    state = replay(events, "A", "H")
    assert (state.inning, state.half, state.outs, state.bases) == (4, "top", 0, FIRST)


def test_runner_on_first_makes_a_dive_out_ambiguous():
    """1루 주자 포스아웃 + 타자 1루와 타자 아웃 + 주자 그대로는 점수판 모양이 같다."""
    j = one(R(0, bases=FIRST), R(10, outs=1, bases=FIRST), [Cue(5, "dive")])
    assert j.code not in ("dive_out", "dive_hit")


def test_runners_moving_without_a_new_runner_is_not_a_missed_dive():
    both = (True, True, False)
    corners = (True, False, True)
    j = one(R(0, bases=both), R(10, bases=corners), [Cue(5, "dive")])
    assert j.code not in ("dive_out", "dive_hit")


def test_third_out_with_runners_on_is_not_called_a_diving_catch():
    second = (False, True, False)
    (tr,) = transitions(stabilize(twice(R(0, outs=2, bases=second), R(10, inning=3, half="bot"))))
    assert judge_transition(tr, [Cue(5, "dive")]).code != "dive_out"


def test_dive_with_a_run_scoring_says_nothing_specific():
    third = (False, False, True)
    j = one(R(0, bases=third), R(10, outs=1, away=1), [Cue(5, "dive")])
    assert j.code not in ("dive_out", "dive_hit")


def test_runner_slide_on_dirt_stays_a_slide_and_grass_becomes_a_dive():
    pose = {"slides": [{"t0": 1.0, "t1": 2.0, "distance": 2.0, "grass": 0.02},
                       {"t0": 5.0, "t1": 6.0, "distance": 2.4, "grass": 0.96},
                       {"t0": 7.0, "t1": 8.0, "distance": 1.0, "grass": 0.52},  # 주자 최대
                       {"t0": 9.0, "t1": 9.5, "distance": 1.0}]}
    assert [c.kind for c in pose_cues(pose)] == ["slide", "dive", "slide", "slide"]


def test_swing_on_ball_three_means_a_hit_not_a_walk():
    cues = [Cue(5, "swing")]
    assert one(R(0, balls=3), R(10, bases=FIRST), cues).code == "reach"


def test_a_swing_before_the_three_ball_board_belongs_to_the_previous_batter():
    """3-1이 보인 뒤에 결정구를 던졌다 — 그 전 스윙은 앞 타석(3아웃) 공이다(삼성-LG 11-08)."""
    j = one(R(20, balls=3, strikes=1), R(23, bases=FIRST), [Cue(11, "swing")])
    assert j.code == "walk"
    # 3-1이 보인 뒤의 스윙은 이 타석 공이다
    assert one(R(20, balls=3, strikes=1), R(30, bases=FIRST), [Cue(25, "swing")]).code == "reach"


def test_take_on_ball_three_backs_the_walk():
    j = one(R(0, balls=3), R(10, bases=FIRST), [Cue(5, "no_swing", "[관절] 타자가 공을 그냥 보냄")])
    assert j.code == "walk"
    assert j.confidence > 0.6


def test_pose_cues_turn_swings_into_cues_and_skip_unclear_ones():
    pose = {"swings": [{"t": 3.0, "swing": True}, {"t": 9.0, "swing": False},
                       {"t": 20.0, "swing": False, "clear": False}]}
    assert [(c.t, c.kind) for c in pose_cues(pose)] == [(3.0, "swing"), (9.0, "no_swing")]


def test_call_only_heard_in_commentary_is_marked_for_moving_to_video():
    cues = [Cue(5, "umpire_signal"), Cue(6, "speech", "비디오 판독 들어갑니다")]
    (j,) = judge_checked(twice(R(0), R(10)), cues)
    assert (j.code, j.detail["check"]) == ("video_review", "speech_only")


def test_whisper_writing_strikeout_with_a_digit_still_counts():
    """받아쓰기가 '삼진'을 '3진'으로 적어도 삼진이다(하이라이트 실측 '4타자 연속 3진')."""
    before = R(0, outs=2, balls=3, strikes=2, bases=(True, True, False))
    after = R(10, inning=4)
    (tr,) = transitions(stabilize(twice(before, after)))
    j = judge_transition(tr, [Cue(8, "speech", "3진. 4타자 연속 삼진 잡아내고")])
    assert j.code == "strikeout"
    j = judge_transition(tr, [Cue(8, "speech", "- 3진.- 4타자 연속")])
    assert j.code == "strikeout"


def test_one_spoken_strikeout_explains_one_strikeout():
    """30초 받아쓰기 조각 하나에 앞 타자 삼진과 다음 타자 땅볼이 같이 든다."""
    reads = twice(R(0, balls=1, strikes=2), R(10, outs=1, balls=1, strikes=2), R(20, outs=2))
    heard = [Cue(5, "speech", "투 앤 투에서 헛스윙 삼진! 다음 타자 땅볼 1루에서 아웃")]
    assert [j.code for j in judge(reads, heard)] == ["strikeout", "out"]
    twice_said = [Cue(5, "speech", "삼진! 또 삼진입니다")]
    assert [j.code for j in judge(reads, twice_said)] == ["strikeout", "strikeout"]


def test_count_dropping_alone_is_a_misread_not_a_new_batter():
    """카운트만 줄고 아웃·주자·득점이 그대로면 판정하지 않는다(독립 리뷰 재현)."""
    reads = twice(R(0, balls=1, strikes=2, outs=1), R(10, balls=0, strikes=1, outs=1))
    assert judge(reads, []) == []
    hit = twice(R(0, balls=2, strikes=2, outs=2, bases=(False, True, False)),
                R(10, balls=0, strikes=1, outs=2, bases=(True, False, True)))
    assert [j.code for j in judge(hit, [])] == ["reach"]


def test_a_compound_word_does_not_count_as_a_second_mention():
    reads = twice(R(0), R(10, away=1), R(20, away=2))
    # 관절 화면 단서가 있는 분석
    heard = [Cue(5, "speech", "홈런스타 이벤트, 홈런!"), _view(15, "no_play")]
    assert [j.code for j in judge(reads, heard)] == ["homerun", "reach"]


def _view(t, kind):
    return Cue(t, kind, "[관절] 투구 뒤 화면")


def test_ball_not_put_in_play_with_two_strikes_then_out_is_a_strikeout():
    """해설 없이 — 투구 뒤 선수 클로즈업(공이 안 맞음) + 2스트라이크 + 아웃 = 삼진."""
    (tr,) = transitions(stabilize(twice(R(0, balls=1, strikes=2), R(10, outs=1, balls=0,
                                                                     strikes=1))))
    assert judge_transition(tr, [_view(4, "no_play")]).code == "strikeout"
    # 3아웃이면 점수판이 주자를 지워도 삼진이다
    (tr,) = transitions(stabilize(twice(R(0, outs=2, strikes=2, bases=(True, False, False)),
                                        R(10, inning=4))))
    assert judge_transition(tr, [_view(4, "no_play")]).code == "strikeout"


def test_ball_put_in_play_overrides_spoken_strikeout_and_walk():
    """공이 맞아 나갔으면 해설의 '삼진'·'볼넷'은 다른 타석 말이다."""
    (tr,) = transitions(stabilize(twice(R(0, balls=1, strikes=2), R(10, outs=1))))
    said = Cue(6, "speech", "앞 타자 헛스윙 삼진")
    assert judge_transition(tr, [said, _view(4, "in_play")]).code == "out"
    (tr,) = transitions(stabilize(twice(R(0, balls=3, strikes=1), R(10, bases=(True, False,
                                                                              False)))))
    walk_said = Cue(6, "speech", "볼넷으로 걸어 나갑니다")
    assert judge_transition(tr, [walk_said, _view(4, "in_play")]).code == "reach"
    assert judge_transition(tr, [_view(4, "no_play")]).code == "walk"


def test_not_put_in_play_early_in_the_count_then_first_base_is_a_hit_by_pitch():
    (tr,) = transitions(stabilize(twice(R(0, balls=1), R(10, bases=(True, False, False)))))
    assert judge_transition(tr, [_view(4, "no_play")]).code == "hbp"
    # 2스트라이크면 낫아웃일 수도 있어 말하지 않는다
    (tr,) = transitions(stabilize(twice(R(0, balls=1, strikes=2),
                                        R(10, bases=(True, False, False)))))
    assert judge_transition(tr, [_view(4, "no_play")]).code != "hbp"


def test_ball_in_play_and_everyone_scores_is_a_home_run_without_the_word():
    (tr,) = transitions(stabilize(twice(R(0, balls=1, bases=(True, False, False)),
                                        R(10, away=2))))
    assert judge_transition(tr, [_view(4, "in_play")]).code == "homerun"
    # 화면 단서가 있는데 공이 안 맞았으면(클로즈업) 홈런이 아니다
    # (단서가 아예 없으면 모양만으로 — 실시간)
    assert judge_transition(tr, [_view(4, "no_play")]).code != "homerun"


def test_the_view_must_belong_to_this_at_bat():
    """직전 상태가 보이기 전의 투구 화면은 이 타석 것이 아니다."""
    (tr,) = transitions(stabilize(twice(R(20, balls=1, strikes=2), R(30, outs=1))))
    assert judge_transition(tr, [_view(5, "no_play")]).code == "out"


def test_an_unclear_last_pitch_does_not_borrow_the_previous_pitch_view():
    """마지막 공 화면을 못 가르면 같은 타석 앞 공의 '안 맞음'을 쓰지 않는다(독립 리뷰 재현)."""
    (tr,) = transitions(stabilize(twice(R(100, strikes=2), R(130, outs=1))))
    earlier, last = _view(101, "no_play"), _view(120, "view_unknown")
    assert judge_transition(tr, [earlier, last]).code == "out"
    (tr,) = transitions(stabilize(twice(R(100, balls=1, strikes=1),
                                        R(130, bases=(True, False, False)))))
    assert judge_transition(tr, [earlier, last]).code == "reach"
    # 직전 상태가 보이기 전(앞 타석 끝)의 화면도 쓰지 않는다
    assert judge_transition(tr, [_view(99.5, "no_play")]).code == "reach"


def test_swings_without_a_view_become_unknown_view_cues():
    pose = {"swings": [{"t": 5.0, "t0": 3.0, "t1": 6.0, "swing": True, "clear": True},
                       {"t": 25.0, "t0": 23.0, "t1": 26.0, "swing": False, "clear": True,
                        "view": "no_play"}]}
    kinds = [(c.t, c.kind) for c in pose_cues(pose) if "view" in c.kind or "play" in c.kind]
    assert kinds == [(6.0, "view_unknown"), (26.0, "no_play")]


def test_a_new_runner_means_the_at_bat_ended_even_if_the_count_did_not_drop():
    """1·2루에서 2점이 나고 2루에 주자가 남았다 — 타자가 나갔다(다음 타자 첫 판독이 이미 2-1)."""
    reads = twice(R(0, balls=2, strikes=1, outs=2, bases=(True, True, False)),
                  R(10, balls=2, strikes=1, outs=2, bases=(False, True, False), home=2))
    (tr,) = transitions(stabilize(reads))
    assert tr.count_reset
    assert judge_transition(tr, []).code == "reach"
    # 도루는 주자 수가 그대로다 — 새 타자로 보지 않는다
    steal = twice(R(0, balls=1, bases=(True, False, False)),
                  R(10, balls=1, strikes=1, bases=(False, True, False)))
    (tr,) = transitions(stabilize(steal))
    assert not tr.count_reset


def test_a_dive_does_not_explain_a_half_change_merged_with_the_next_half():
    """3아웃 뒤 새 이닝 첫 타자 안타까지 한 전이로 합쳐지면 다이빙 아웃이라 말하지 않는다."""
    (tr,) = transitions(stabilize(twice(R(0, outs=2, half="bot"),
                                        R(10, inning=4, bases=(True, False, False)))))
    assert judge_transition(tr, [Cue(5, "dive")]).code != "dive_out"


def test_batter_scoring_is_a_home_run_even_when_later_outs_are_merged():
    """3점 홈런 뒤 2아웃까지 한 전이로 합쳐져도 홈런 — 타자가 홈을 밟는 병살은 없다."""
    reads = twice(R(0, outs=0, bases=(True, True, False)), R(10, outs=2, away=3))
    (tr,) = transitions(stabilize(reads))
    assert judge_transition(tr, [_view(4, "in_play")]).code == "homerun"
    assert judge_transition(tr, [Cue(6, "speech", "강민호의 스리런!")]).code == "homerun"
    assert judge_transition(tr, []).code == "out"  # 단서가 없으면 예전대로(합쳐진 전이)
    # 이닝이 바뀌면 새 이닝은 늘 빈 베이스 — 해설 말이 있어야 한다
    half = twice(R(0, outs=1, bases=(True, False, False)), R(10, inning=4, away=2))
    (tr,) = transitions(stabilize(half))
    assert judge_transition(tr, [_view(4, "in_play")]).code != "homerun"
    assert judge_transition(tr, [Cue(6, "speech", "투런 홈런")]).code == "homerun"


def test_home_run_words_said_long_before_the_new_scoreboard_still_count():
    """다시보기가 길어 새 점수판이 30초 넘게 뒤에 보여도, 그 사이 '솔로포'는 이 전이의 말이다."""
    (tr,) = transitions(stabilize(twice(R(0, outs=1), R(40, inning=4, away=1))))
    assert judge_transition(tr, [Cue(12, "speech", "김용훈의 솔로포")]).code == "homerun"


def test_two_outs_across_a_half_change_need_the_word_double_play():
    """1아웃에서 이닝이 바뀐 전이(아웃 둘) — 하이라이트에선 대개 여러 플레이가 합쳐진 것이다."""
    (tr,) = transitions(stabilize(twice(R(0, outs=1, bases=(True, False, False)),
                                        R(10, inning=4))))
    assert judge_transition(tr, []).code == "out"
    assert judge_transition(tr, [Cue(6, "speech", "6-4-3 병살!")]).code == "double_play"
    # 같은 이닝 안의 아웃 둘은 예전대로 병살
    (tr,) = transitions(stabilize(twice(R(0, outs=0, bases=(True, False, False)), R(10, outs=2))))
    assert judge_transition(tr, []).code == "double_play"


def test_home_run_with_outs_in_the_same_transition_is_marked_merged():
    (tr,) = transitions(stabilize(twice(R(0, outs=1), R(10, inning=4, away=1))))
    j = judge_transition(tr, [Cue(6, "speech", "솔로포!")])
    assert (j.code, j.detail.get("merged")) == ("homerun", True)


def test_the_last_out_of_the_game_comes_from_the_last_pitch_after_the_last_scoreboard():
    """9회말 2아웃, 원정이 앞섬 — 마지막 점수판 뒤 투구가 있으면 그 공이 마지막 아웃이다."""
    reads = twice(R(0, inning=9, half="bot", outs=2, balls=1, strikes=2, away=5, home=1))
    js = judge(reads, [_view(20, "no_play")], finished=True)
    assert [(j.code, j.t) for j in js] == [("strikeout", 21.0)]
    assert judge(reads, [_view(20, "no_play")]) == []  # 실시간(다 안 봄)에는 말하지 않는다
    (event,) = [e for e in to_relay_events(js) if e.kind == "result"]
    assert event.detail["outs_made"] == 1  # 상태도 세 번째 아웃으로 넘어간다
    # 동점이면 연장 — 마지막 아웃으로 끝난다고 말하지 않는다
    tied = twice(R(0, inning=9, half="bot", outs=2, away=1, home=1))
    assert judge(tied, [_view(20, "in_play")], finished=True) == []
    # 투구 화면이 없으면(점수판 뒤 바로 영상 끝) 말하지 않는다
    assert judge(reads, [], finished=True) == []


def test_a_walk_off_word_after_the_last_scoreboard_stops_the_game_end_out():
    reads = twice(R(0, inning=9, half="bot", outs=2, bases=(True, False, False), away=5, home=4))
    said = Cue(25, "speech", "끝내기 투런 홈런!")
    assert judge(reads, [_view(20, "in_play"), said], finished=True) == []


def test_an_out_and_a_late_run_from_third_are_one_sacrifice_fly():
    """점수판이 아웃을 먼저, 11초 뒤 득점을 따로 반영했다 — 한 플레이다."""
    loaded = (True, True, True)
    reads = twice(R(0, balls=2, strikes=2, bases=loaded), R(10, outs=1, bases=loaded),
                  R(21, outs=1, bases=(True, True, False), away=1))
    said = Cue(5, "speech", "가볍게 걷어 올렸습니다 센터쪽 중견수")
    hit = _view(5, "in_play")
    assert [j.code for j in judge(reads, [said, hit])] == ["sac_fly"]
    # 말이 없어도 한 플레이 — 만루에서 1루 주자가 그대로라 모양만으로 희생플라이(확신 낮춤)
    [shape] = judge(reads, [hit])
    assert (shape.code, shape.confidence) == ("sac_fly", 0.65)
    # 공이 맞았다는 영상이 없으면 합치지 않는다 — 삼진 뒤 폭투 득점일 수 있다
    assert len(judge(reads, [said])) == 2
    assert len(judge(reads, [hit, Cue(18, "speech", "폭투! 3루 주자 홈인")])) == 2
    # 다음 타자 초구 카운트와 함께 보여도 같다
    pitched = twice(R(0, balls=2, strikes=2, bases=loaded), R(10, outs=1, bases=loaded),
                    R(21, outs=1, balls=1, bases=(True, True, False), away=1))
    assert [j.code for j in judge(pitched, [said, hit])] == ["sac_fly"]
    # 다른 주자도 움직였으면(폭투) 따로 일어난 일이다
    wild = twice(R(0, balls=2, strikes=2, bases=loaded), R(10, outs=1, bases=loaded),
                 R(21, outs=1, balls=1, bases=(False, True, True), away=1))
    assert len(judge(wild, [])) == 2


def test_a_runner_tagging_up_to_third_keeps_the_sacrifice_fly_one_play():
    """PO 2차전: 1사 2·3루 희생플라이 — 점수판이 아웃을 먼저, 11초 뒤 득점과 2루 주자 3루를 함께.
    카운트가 그대로이고 사이에 투구가 없으면 태그업이다. 투구가 있었으면 폭투일 수 있다."""
    corners = (False, True, True)
    reads = twice(R(0, outs=1, balls=3, strikes=2, bases=corners),
                  R(13, outs=2, bases=corners), R(24, outs=2, bases=(False, False, True), away=1))
    said = Cue(20, "speech", "희생플라이 타점으로 또 한 점을 더했습니다")
    hit = _view(5, "in_play")
    assert [j.code for j in judge(reads, [said, hit])] == ["sac_fly"]
    assert len(judge(reads, [said, hit, _view(18, "no_play")])) == 2


def test_tag_up_words_after_the_late_run_count_for_the_sacrifice_fly():
    """PO 2차전: 1사 1·3루 희생플라이 — '노바운드 캐치… 태그업'이 득점이 늦게 반영된 뒤(합친
    전이의 아웃 시각 + 6초 밖)에 나왔다. 새 상태가 보인 때까지 듣고 '태그업'도 희생플라이 말."""
    corners = (True, False, True)
    reads = twice(R(0, outs=1, balls=1, bases=corners), R(8, outs=2, bases=corners),
                  R(16, outs=2, bases=FIRST, away=1))
    said = Cue(15, "speech", "노바운드 캐치를 잘 해줬고 태그업 주자")
    assert [j.code for j in judge(reads, [said, _view(3, "in_play")])] == ["sac_fly"]


def test_home_run_words_during_the_replay_after_the_new_board():
    """PO 2차전 최형우 솔로 — '홈런입니다'가 새 점수판 9~19초 뒤 다시보기 동안 나왔다."""
    reads = twice(R(0, outs=1, half="bot", home=7), R(12, outs=1, half="bot", home=8))
    # 이 투구 화면 단서가 '못 가름'이면 말이 있어야 홈런이다(단서가 아예 없으면 모양만으로 — 아래)
    late = [_view(5, "view_unknown"), Cue(21, "speech", "역대 플레이오프 최고령 홈런입니다")]
    assert [j.code for j in judge(reads, late)] == ["homerun"]
    too_late = [_view(5, "view_unknown"), Cue(40, "speech", "역대 플레이오프 최고령 홈런입니다")]
    assert [j.code for j in judge(reads, too_late)] != ["homerun"]


def test_review_cases_for_tag_up_and_late_words():
    """독립 리뷰 재현 넷 — 태그업 합치기가 보크·도루를 삼키고, 늘린 창에 다음 타석 '뜬공'이 섞이고,
    앞 플레이 '태그업'이 땅볼 타점을, 앞 홈런이 다음 홈런의 다시보기 말을 가져갔다."""
    corners = (False, True, True)
    reads = twice(R(0, outs=1, bases=corners), R(13, outs=2, bases=corners),
                  R(24, outs=2, bases=(False, False, True), away=1))
    hit = _view(5, "in_play")
    balk = judge(reads, [hit, Cue(22, "pitcher_stops")])
    assert [j.code for j in balk] == ["out", "balk"]
    steal = judge(reads, [hit, Cue(22, "slide")])
    assert len(steal) == 2
    next_fly = judge(reads, [hit, Cue(22, "pitcher_stops"),
                             Cue(28, "speech", "다음 타자 뜬공 잡았습니다")])
    assert "sac_fly" not in [j.code for j in next_fly]
    grounder = twice(R(0, outs=1, balls=1, strikes=1, bases=(False, False, True)),
                     R(10, outs=2, away=1))
    said = [_view(4, "in_play"), Cue(-5, "speech", "앞서 태그업으로 3루까지 갔죠")]
    assert judge(grounder, said)[0].code != "sac_fly"
    two = twice(R(0, outs=1), R(12, outs=1, away=1), R(34, outs=1, away=2))
    back = [_view(5, "in_play"), _view(-30, "no_play"), Cue(31, "speech", "백투백 홈런")]
    assert [j.code for j in judge(two, back)] == ["homerun", "homerun"]


def test_a_strikeout_mark_inside_the_out_transition_is_not_a_second_strikeout():
    """PO 2차전 3K — 표시 뒤 아웃 판독이 34초 늦게 와 표시만으로 낸 삼진과 둘이 됐다. 그 아웃 전이가
    표시를 품고 사이에 다른 투구가 없으면 같은 삼진이다. 다른 투구가 있었으면 따로다(정우주 3K)."""
    reads = twice(R(0, outs=1, strikes=2), R(40, outs=2))
    mark = Cue(5, "k_mark", "[점수판] 삼진 표시(K)")
    one_k = judge(reads, [_view(4.5, "no_play"), mark], finished=True)
    assert [j.code for j in one_k] == ["strikeout"]
    two = judge(reads, [_view(4.5, "no_play"), mark, _view(20, "no_play")], finished=True)
    assert [j.code for j in two].count("strikeout") >= 1 and len(two) == 2


def test_outs_raised_one_by_one_within_seconds_are_one_double_play():
    """점수판이 병살 아웃을 4초 간격으로 하나씩 올렸다 — 투구 없이 이어지면 한 플레이다."""
    reads = twice(R(0, outs=1, balls=0, strikes=1, bases=(True, False, False)),
                  R(10, outs=2, bases=(True, False, False)), R(14, inning=4, half="bot"))
    said = Cue(12, "speech", "더블 플레이로 이닝을 끝내버리는")
    assert [j.code for j in judge(reads, [said])] == ["double_play"]
    # 사이에 투구가 있었으면 따로 일어난 아웃이다
    assert len(judge(reads, [said, _view(12, "no_play")])) == 2


def test_talk_about_hoping_for_a_double_play_before_the_play_is_not_a_double_play():
    (tr,) = transitions(stabilize(twice(R(0, outs=1, bases=(True, False, False)),
                                        R(40, inning=4, half="bot"))))
    assert judge_transition(tr, [Cue(5, "speech", "여기서 병살 노려야죠")]).code == "out"


def _board(t, balls, strikes, outs, bases=(), inning=1, half="top", mark=None):
    r = {"t": t, "inning": inning, "half": half, "balls": balls, "strikes": strikes,
         "outs": outs, "bases": list(bases), "away": 0, "home": 0}
    if mark:
        r["mark"] = mark
    return r


def test_strikeout_mark_reading_is_dropped_and_names_the_next_out():
    """점수판이 주자·아웃 칸을 'nK'로 덮은 판독 — 상태에서 빼고, 뒤이은 아웃을 삼진으로."""
    board = [_board(0, 2, 2, 1), _board(3, 0, 0, 0, [3], mark="strikeout"),
             _board(4, 0, 0, 0, [3], mark="strikeout"),  # 같은 표시를 한 번 더 읽음
             _board(5, 0, 0, 2), _board(8, 1, 1, 2),
             _board(12, 0, 0, 0, inning=1, half="bot")]  # 다음 아웃(이닝 끝)은 다른 타자다
    readings, cues, _ = observe({"scorebug": {"scoreboard": board}})
    assert len(readings) == 4 and [c.t for c in cues if c.kind == "k_mark"] == [3.0, 4.0]
    js = judge(readings, cues, min_repeat=1)
    assert [j.code for j in js] == ["strikeout", "out"]
    assert any("삼진 표시(K)" in e for e in js[0].evidence)
    # 표시가 없으면(같은 판독에서 덮인 두 장을 뺀 것) 공이 맞았는지 모르니 그냥 아웃이다
    plain = [r for r in board if "mark" not in r]
    readings, cues, _ = observe({"scorebug": {"scoreboard": plain}})
    assert [j.code for j in judge(readings, cues, min_repeat=1)] == ["out", "out"]


def test_strikeout_mark_needs_two_strikes_before_it_and_a_single_out():
    """표시 직전 판독이 2스트라이크가 아니면 뒤 아웃을 삼진으로 바꾸지 않는다(리뷰 시나리오)."""
    board = [_board(0, 1, 1, 1), _board(3, 0, 0, 0, [3], mark="strikeout"),
             _board(5, 0, 0, 2)]
    readings, cues, _ = observe({"scorebug": {"scoreboard": board}})
    assert [j.code for j in judge(readings, cues, min_repeat=1)] == ["out"]


def test_strikeout_mark_names_a_merged_highlight_jump_but_keeps_it_merged():
    """하이라이트가 건너뛰어 아웃 둘이 한 전이(이닝 끝) — 보이는 플레이는 삼진, merged는 남긴다."""
    board = [_board(0, 1, 2, 1), _board(3, 0, 0, 0, mark="strikeout"),
             _board(6, 0, 0, 0, inning=1, half="bot")]
    readings, cues, _ = observe({"scorebug": {"scoreboard": board}})
    [j] = judge(readings, cues, min_repeat=1)
    assert j.code == "strikeout" and j.detail.get("merged")


def test_strikeout_mark_reaches_a_late_jump_only_without_a_pitch_between():
    """표시 뒤 20초에 건너뛴 점수판 — 사이에 투구 화면이 없으면 그 아웃이 삼진, 있으면 아니다."""
    board = [_board(0, 1, 2, 1), _board(3, 0, 0, 0, mark="strikeout"),
             _board(23, 0, 0, 0, inning=1, half="bot")]
    readings, cues, _ = observe({"scorebug": {"scoreboard": board}})
    cues = sorted(cues + [_view(1, "view_unknown")], key=lambda c: c.t)  # 삼진 투구 자체(표시 전)
    assert [j.code for j in judge(readings, cues, min_repeat=1)] == ["strikeout"]
    pitched = sorted(cues + [_view(15, "in_play")], key=lambda c: c.t)
    assert [j.code for j in judge(readings, pitched, min_repeat=1)] == ["out"]


def test_strikeout_mark_does_not_name_an_out_that_scored_or_a_jump_without_views():
    """같은 이닝에 점수가 난 아웃(희생플라이)은 삼진이 아니다. 투구 화면 단서가 없으면 8초까지만."""
    board = [_board(0, 1, 2, 1, [3]), _board(3, 0, 0, 0, mark="strikeout"), _board(6, 0, 0, 2)]
    board[-1]["away"] = 1
    readings, cues, _ = observe({"scorebug": {"scoreboard": board}})
    assert [j.code for j in judge(readings, cues, min_repeat=1)] != ["strikeout"]
    jump = [_board(0, 1, 2, 1), _board(3, 0, 0, 0, mark="strikeout"),
            _board(23, 0, 0, 0, inning=1, half="bot")]
    readings, cues, _ = observe({"scorebug": {"scoreboard": jump}})
    no_views = [c for c in cues if not c.text.startswith("[관절]")]
    assert [j.code for j in judge(readings, no_views, min_repeat=1)] == ["out"]


def test_late_board_after_review_still_finds_the_steal_slide():
    """비디오 판독 뒤 2분 늦게 고친 점수판(2루→3루, 카운트 그대로) — 투구가 없었으면 앞 판독 무렵의
    슬라이딩이 도루 단서다. 사이에 투구가 있었으면 다른 플레이일 수 있어 쓰지 않는다."""
    reads = twice(R(0, outs=1, balls=1, strikes=1, bases=(False, True, False)),
                  R(120, outs=1, balls=1, strikes=1, bases=(False, False, True)))
    slide = Cue(1.0, "slide", "[관절] 슬라이딩·다이빙 1.0~1.5초")
    earlier = _view(-3, "in_play")  # 앞 투구 화면(관절 화면 단서가 있는 영상)
    assert [j.code for j in judge(reads, [earlier, slide])] == ["steal"]
    pitched = [earlier, slide, _view(60, "no_play")]
    assert [j.code for j in judge(reads, pitched)] != ["steal"]
    # 앞 판독 전의 슬라이딩(앞 플레이 몫), 관절 화면 단서가 없는 영상, '폭투' 해설은 도루가 아니다
    before = Cue(-1.0, "slide", "[관절] 슬라이딩·다이빙 -1.0~-0.5초")
    assert [j.code for j in judge(reads, [earlier, before])] != ["steal"]
    assert [j.code for j in judge(reads, [slide])] != ["steal"]
    wp = Cue(2.0, "speech", "아 폭투가 나오면서")
    assert [j.code for j in judge(reads, [earlier, slide, wp])] != ["steal"]


def test_strikeout_pitch_view_ending_just_after_the_mark_does_not_block_the_jump():
    """표시가 그 삼진 투구 화면보다 0.5초 먼저 떴다 — 그 화면은 '사이의 투구'가 아니다."""
    board = [_board(0, 0, 2, 2), _board(3, 0, 0, 0, mark="strikeout"),
             _board(15, 2, 2, 1, [1], inning=2, half="top")]
    readings, cues, _ = observe({"scorebug": {"scoreboard": board}})
    cues = sorted(cues + [_view(3.5, "no_play")], key=lambda c: c.t)
    assert "strikeout" in [j.code for j in judge(readings, cues, min_repeat=1)]


def test_strikeout_mark_alone_at_the_end_of_a_finished_video():
    """다 본 영상 끝 2초 전 삼진 — 뒤에 아웃 판독이 없다. 표시만으로 삼진(선언),
    그 투구 화면 끝 시각. 실시간(finished=False)은 아웃 판독이 뒤에 올 수 있어 내지 않는다."""
    board = [_board(0, 1, 2, 2), _board(3, 0, 0, 0, mark="strikeout")]
    readings, cues, _ = observe({"scorebug": {"scoreboard": board}})
    cues = sorted(cues + [_view(-20, "in_play"), _view(3.5, "no_play")], key=lambda c: c.t)
    [j] = judge(readings, cues, min_repeat=1, finished=True)
    assert (j.code, j.kind, j.t, j.transition) == ("strikeout", "call", 3.5, None)
    assert judge(readings, cues, min_repeat=1) == []


def test_scanned_marks_become_strikeout_cues():
    """화소로 훑어 찾은 표시(scorebug.marks)도 판독 표시와 같은 단서다."""
    board = [_board(0, 1, 2, 1), _board(5, 0, 0, 2)]
    readings, cues, _ = observe({"scorebug": {"scoreboard": board, "marks": [3.0]}})
    assert [c.t for c in cues if c.kind == "k_mark"] == [3.0]
    assert [j.code for j in judge(readings, cues, min_repeat=1)] == ["strikeout"]


def test_mark_alone_needs_a_recent_reading_and_scanned_marks_need_no_crop_reads():
    """판독이 오래전에 끊긴 뒤의 표시는 그 타석으로 볼 수 없다.
    점수판 판독이 없어도 훑은 표시는 단서다."""
    board = [_board(0, 1, 2, 2), _board(1, 1, 2, 2)]
    readings, cues, _ = observe({"scorebug": {"scoreboard": board, "marks": [300.0]}})
    cues = sorted(cues + [_view(-5, "in_play")], key=lambda c: c.t)
    assert judge(readings, cues, min_repeat=1, finished=True) == []
    _, cues, _ = observe({"scorebug": {"scoreboard": [], "marks": [5.0]}})
    assert [c.t for c in cues if c.kind == "k_mark"] == [5.0]


def test_homerun_from_the_board_shape_alone_when_there_are_no_view_cues():
    """실시간(관절 없음) — 아웃 없이 주자 1명 + 타자까지 들어와 루가 비었다(2점). 말이 없어도 홈런.
    관절 화면 단서가 있는 분석에서는 예전처럼 넓은 화면이나 말이 있어야 한다."""
    reads = twice(R(0, outs=1, balls=1, strikes=1, bases=(True, False, False)),
                  R(20, outs=1, away=2))
    assert [j.code for j in judge(reads, [])] == ["homerun"]
    assert [j.code for j in judge(reads, [_view(5, "no_play")])] != ["homerun"]


# ── 2026-11-09 한국시리즈 1차전(TVING, 처음 본 경기) ─────────────────────────────


def test_a_state_read_many_times_survives_a_single_misread_right_after_it():
    """KS1 2타점 2루타 — 5번 읽힌 '1사 2루 2점'이 바로 뒤 한 번 읽힌 '2사 주자 없음'(태그업 주자가
    3루에 그려지기 전)과 안 맞아 튄 판독으로 버려졌다. 튄 것은 한 번 읽힌 쪽이다."""
    corners, second, third = (True, False, True), (False, True, False), (False, False, True)
    seen = [R(0, outs=1, bases=corners, strikes=2)] * 6 + [
        R(30, outs=1, bases=second, balls=1, strikes=2, away=2)] * 5 + [
        R(45, outs=2, away=2), R(46, outs=2, bases=third, away=2),
        R(47, outs=2, bases=third, away=2)]
    stable = stabilize(seen, min_repeat=1)
    assert [(r.outs, r.bases, r.away) for r in stable] == [
        (1, corners, 0), (1, second, 2), (2, third, 2)]
    assert [j.code for j in judge(seen, [], min_repeat=1)] == ["reach", "out"]
    # 다음 판독도 여러 번 읽혔으면 어느 쪽이 튄 것인지 모른다 — 예전처럼 앞쪽을 버린다
    twice_next = seen[:11] + [R(45, outs=2, away=2), R(45.5, outs=2, away=2)] + seen[12:]
    assert [r.bases for r in stabilize(twice_next, min_repeat=1)] == [corners, E, third]


def test_walk_moves_only_forced_runners():
    """볼넷·사구는 1루부터 이어진 주자만 한 베이스씩 — 2·3루 주자가 1루 빈 채로 들어오면 안타다."""
    assert forced_to_first((False, True, True), (True, True, True), 0)
    assert forced_to_first((True, True, True), (True, True, True), 1)
    assert forced_to_first((False, False, True), (True, False, True), 0)
    assert not forced_to_first((False, True, True), (True, False, False), 2)
    reads = twice(R(0, outs=2, bases=(False, True, True), balls=3), R(10, outs=2, away=2,
                                                                     bases=(True, False, False)))
    assert [j.code for j in judge(reads, [_view(4, "no_play")])] == ["reach"]


def test_batter_home_shape_is_a_home_run_when_this_pitch_was_not_seen():
    """KS1 2점 홈런 — 관절이 그 투구 화면을 못 봐(단서 없음) '출루'였다. 다른 투구의 단서만
    있으면 이 전이는 못 본 것과 같다 — 실시간처럼 모양만으로 홈런."""
    reads = twice(R(0, outs=2, bases=(True, False, False), balls=1, half="bot", home=6),
                  R(12, outs=2, half="bot", home=8))
    assert [j.code for j in judge(reads, [_view(-30, "no_play")])] == ["homerun"]
    # 그 투구 화면이 '안 맞음'이면 모양만으로 홈런이라 하지 않는다 — 점수판이 그 화면 도중에
    # 바뀌어 화면이 뒤에 끝나도 같다(독립 리뷰)
    assert [j.code for j in judge(reads, [_view(5, "no_play")])] != ["homerun"]
    late_view = [Cue(2, "pitch_seen", ""), _view(14, "no_play")]
    assert [j.code for j in judge(reads, late_view)] != ["homerun"]


def _pitch(t_pitch, t_end, kind="no_play"):
    return [Cue(t_pitch, "pitch_seen", "[관절] 투구 순간"), _view(t_end, kind)]


def test_board_changed_during_the_pitch_view_is_that_pitch_result():
    """TVING은 투구 화면이 이어지는 동안 점수판을 바꾼다 — 그 판정은 이 투구의 결과다
    (`pitch_t` = 투구 화면 끝, 판정 시각은 점수판이 바뀐 때 그대로)이고 그 투구 화면 단서
    (클로즈업)를 쓴다. 투구 순간 전에 바뀐 점수판은 앞 플레이라 그대로."""
    walk = twice(R(0, outs=2, balls=3, strikes=2, bases=(True, True, False)),
                 R(10, outs=2, bases=(True, True, True)))
    (j,) = judge(walk, _pitch(8, 12))
    assert j.code == "walk" and j.t == pytest.approx(10.0) and j.detail["pitch_t"] == 12
    (j,) = judge(walk, _pitch(11, 14))  # 투구 순간이 점수판 뒤 — 다음 투구다
    assert "pitch_t" not in j.detail
    # 2스트라이크에서 아웃 — 바뀐 뒤 끝난 투구 화면이 클로즈업이면 삼진
    k = twice(R(0, outs=1, balls=1, strikes=2), R(10, outs=2))
    (j,) = judge(k, _pitch(8, 12))
    assert j.code == "strikeout" and j.detail["pitch_t"] == 12
    # 관절이 없으면(실시간) 없다
    (j,) = judge(walk, [])
    assert "pitch_t" not in j.detail


def test_a_count_read_long_after_the_restamped_time_does_not_block_the_pitch_view():
    """2025 와일드카드 1차전 4회초 — 판독기가 시각을 주자가 바뀐 초(10)로 앞당긴 판독의 카운트
    0-1은 실제로 읽은 화면(15초, 다음 타자 첫 공 뒤) 값이다. '새 타자 카운트가 벌써 진행'으로 보지
    않고 그 투구 화면의 볼넷으로 둔다. 읽은 시각이 없거나 가까우면 예전대로."""
    from dataclasses import replace

    from app.domain.scorebug import Reading

    def walk(read_t):
        after = Reading(10, 3, "top", 0, 1, 1, (True, False, False), 0, 0, read_t)
        return [R(0, outs=1, balls=3, strikes=1)] * 2 + [after, replace(after, t=11)]

    (j,) = judge(walk(15.0), _pitch(8, 12))
    assert j.code == "walk" and j.detail["pitch_t"] == 12
    for read_t in (None, 11.0):
        (j,) = judge(walk(read_t), _pitch(8, 12))
        assert "pitch_t" not in j.detail


def test_speech_over_a_new_half_pitch_does_not_name_the_last_out():
    """2025 와일드카드 1차전 7회초 2사 만루 2-2 2루 포스아웃(3아웃) — 새 반 이닝이 보인 뒤(45초)
    시작해 7회말 투구(53초)를 덮는 받아쓰기 조각 '스윙 삼진'이 3아웃을 '삼진'으로 만들었다.
    새 반 이닝 투구가 없으면(관절 없음) 예전대로."""
    loaded = (True, True, True)
    seen = twice(R(0, outs=2, balls=2, strikes=2, bases=loaded), R(45, half="bot"))
    said = [Cue(50, "speech", "결국 참아내지 못합니다 스윙 삼진")]
    assert [j.code for j in judge(seen, said + [Cue(53, "pitch_seen", "")])] == ["out"]
    assert [j.code for j in judge(seen, said)] == ["strikeout"]


def test_in_progress_view_is_not_borrowed_after_a_batted_ball_or_across_halves():
    """독립 리뷰 — 직전 상태 뒤 공이 맞아 나간 화면이 이미 끝났으면 늦게 보인 그 타구의 결과다
    (다음 투구·다시보기 화면을 결정구로 잡아 출루를 사구로 냈다). 직전 상태가 보이기 전의
    투구, 반 이닝이 바뀐 전이도 아니다."""
    on1, on12 = (True, False, False), (True, True, False)
    reach = twice(R(0, outs=1, balls=1, bases=on1), R(10, outs=1, bases=on12))
    cues = [Cue(2, "pitch_seen", ""), _view(5, "in_play")] + _pitch(8, 12)
    (j,) = judge(reach, cues)
    assert "pitch_t" not in j.detail and j.code != "hbp"
    (j,) = judge(reach, _pitch(-3, 12))  # 투구 순간이 직전 상태보다 앞
    assert "pitch_t" not in j.detail
    half = twice(R(0, outs=2, strikes=2), R(10, half="bot"))
    (j,) = judge(half, _pitch(8, 12))
    assert "pitch_t" not in j.detail
    # 그 투구의 화면 단서가 빠졌으면 다음 투구 화면을 빌리지 않는다
    missing = [Cue(8, "pitch_seen", ""), Cue(20, "pitch_seen", ""), _view(24, "no_play")]
    (j,) = judge(reach, missing)
    assert "pitch_t" not in j.detail and j.code != "hbp"


def test_strikeout_mark_shown_after_the_board_changed_in_the_same_view():
    """독립 리뷰 — 점수판이 투구 화면 도중 먼저 바뀌고(10초) 표시가 화면이 끝나기 전(11.5초)에
    떴다. 표시 뒤 아웃 판정만 찾으면 못 쓰고 '아웃'으로 남았다."""
    k = twice(R(0, outs=1, balls=1, strikes=2), R(10, outs=2))
    cues = _pitch(8, 12, "view_unknown") + [Cue(11.5, "k_mark", "[점수판] 삼진 표시(K)")]
    (j,) = [j for j in judge(k, cues) if j.transition is not None]
    assert j.code == "strikeout"
    # 표시와 새 점수판이 같은 프레임에서 읽혀도(같은 시각) 그 전이의 표시다
    same = _pitch(8, 12, "view_unknown") + [Cue(10.0, "k_mark", "[점수판] 삼진 표시(K)")]
    (j,) = [j for j in judge(k, same) if j.transition is not None]
    assert j.code == "strikeout"


def test_swing_on_ball_four_with_a_close_up_is_a_walk():
    """3볼에서 휘둘렀는데 투구 뒤 클로즈업(공이 안 맞음)이고 1루 — 체크스윙을 '안 돌았다'로 본
    볼넷이다(관측만 남긴다). 넓은 경기장이면 예전처럼 출루."""
    reads = twice(R(0, outs=2, balls=3, bases=(False, True, True)),
                  R(10, outs=2, bases=(True, True, True)))
    swung = [Cue(3, "swing", "[관절] 타자가 배트를 휘두름")]
    (j,) = judge(reads, swung + [_view(4, "no_play")])
    assert j.code == "walk" and j.detail.get("observed") == "swing_on_ball_four"
    (j,) = judge(reads, swung + [_view(4, "in_play")])
    assert j.code == "reach"


def test_runs_with_two_outs_before_the_inning_ends_came_from_a_non_out_play():
    """2아웃에서 점수가 나고 반 이닝이 끝났다 — 세 번째 아웃 플레이로는 점수가 안 난다(5.08(a)).
    하이라이트가 적시타 뒤 3아웃을 건너뛰면 '아웃'으로 나갔다(조정용·PO 1차전)."""
    third = (False, False, True)
    reads = twice(R(0, outs=2, bases=third, balls=1, half="top"), R(20, half="bot", away=1))
    (j,) = judge(reads, [])
    assert j.code == "reach" and j.detail.get("merged")
    # 반 이닝을 여럿 건너뛰면(투수 모음) 점수가 어느 반 이닝 것인지 모른다 — 예전처럼 아웃
    skip = twice(R(0, outs=2, bases=third, balls=1, half="top"),
                 R(20, inning=4, half="top", away=1))
    assert judge(skip, [])[0].code == "out"
    # 상대 팀(다음 반 이닝 공격 팀)의 점수만 늘었으면 아니다
    other = twice(R(0, outs=2, bases=third, balls=1, half="top"), R(20, half="bot", home=1))
    assert judge(other, [])[0].code == "out"


# ── 2026-11-09 한국시리즈 3차전(KBS N 상자, 처음 본 경기) ──────────────────────


def test_inning_read_backwards_is_repaired_between_stable_states():
    """KS3 — VLM이 7회초~8회말 판독 35장을 '1회'로 읽었다(점수판은 7▲·8▲·8▼). 회차는 거꾸로 가지
    않고 점수도 줄지 않으니 회차만 틀렸다 — 초·말과 아웃 순서로 다시 매긴다."""
    on2 = (False, True, False)
    # 실제 경기처럼 앞뒤 안정 상태가 더 많이 읽혔다(앞 경기 판독이 사슬을 받친다)
    seen = [R(0, inning=7, outs=1, bases=on2, balls=2, strikes=1, away=1, home=4)] * 8 + [
        R(10, inning=1, outs=2, bases=on2, strikes=2, away=1, home=4)] * 3 + [
        R(30, inning=1, strikes=1, away=1, home=4)] * 3 + [
        R(40, inning=1, outs=1, away=1, home=4)] * 3 + [
        R(60, inning=8, half="bot", outs=1, balls=3, strikes=1, away=1, home=4)] * 8
    stable = stabilize(seen, min_repeat=1)
    assert [(r.inning, r.half, r.outs) for r in stable] == [
        (7, "top", 1), (7, "top", 2), (8, "top", 0), (8, "top", 1), (8, "bot", 1)]
    # 한 장 회차 오독이 진짜 판독을 끌어올리지 않는다 — 안정 상태 사이만 고친다
    early = [R(0, inning=2, outs=1)] * 2 + [R(5, inning=4, outs=1)] + [R(10, inning=2, outs=2)] * 2
    assert [r.inning for r in stabilize(early, min_repeat=1)] == [2, 2]
    # 아웃 오독 한 장('1회 무사' 한 장)이 회차를 올리지 않는다 — 다음 회차는 두 번 이상 읽혔을 때만
    noisy = [R(0, inning=7, outs=1, away=1, home=4)] * 8 + [
        R(10, inning=1, outs=1, away=1, home=4)] * 3 + [
        R(20, inning=1, away=1, home=4)] + [R(30, inning=1, outs=2, away=1, home=4)] * 3 + [
        R(60, inning=8, half="bot", outs=1, away=1, home=4)] * 8
    assert all(r.inning != 8 or r.half == "bot" for r in stabilize(noisy, min_repeat=1))


def test_single_read_that_starts_the_next_half_is_not_a_spike():
    """KS3 — 새 반 이닝 첫 판독('5회초 무사', 한 장)이 다음 판독('2사', 타석을 건너뜀)과 한 번에 안
    이어진다고 버려졌다. 앞 → 뒤가 반 이닝을 넘으면 늘 '가능'이라 생긴 일 — 새 반 이닝에서 이어지면
    남긴다(삼진 6K를 놓침)."""
    seen = [R(0, half="bot", outs=2, balls=1, strikes=1)] * 2 + [
        R(30, inning=5, balls=1, strikes=1)] + [R(40, inning=5, outs=2, strikes=2)] * 2
    assert [(r.inning, r.outs) for r in stabilize(seen, min_repeat=1)] == [(3, 2), (5, 0), (5, 2)]


def test_board_read_after_the_next_batter_saw_pitches_is_not_that_pitch():
    """KS3 — 삼진 뒤 점수판이 다음 타자 3-1에서야 읽혀 그 투구 화면 도중이 됐다. 새 타자 카운트가
    벌써 진행됐으면 진행 중인 투구는 새 타자 것 — 앞 타석 결과를 그 투구에 붙이지 않는다."""
    k = twice(R(0, outs=1, strikes=2), R(10, outs=2, balls=3, strikes=1))
    (j,) = [j for j in judge(k, _pitch(8, 12)) if j.transition is not None]
    assert "pitch_t" not in j.detail


def test_game_end_out_in_the_top_half_ignores_home_run_recap_words():
    """KS3 — 9회초 2사(홈 팀 리드) 뒤 마지막 투구가 경기 끝 아웃인데, 경기를 정리하는 해설의 '홈런'
    때문에 끝내기 홈런으로 보고 막았다. 초 공격엔 끝내기가 없다."""
    last = twice(R(0, inning=9, outs=2, bases=(True, False, False), home=5, away=1))
    cues = [_view(20, "in_play"), Cue(30, "speech", "구자욱의 홈런으로 앞서 나간 삼성")]
    codes = [j.code for j in judge(last, cues, finished=True)]
    assert codes == ["out"]


def test_board_changed_in_the_view_where_the_before_state_first_showed():
    """KS4 — 직전 상태(3-0 만루 전)가 그 투구 화면 안에서 처음 보였고 화면이 끝나기 전에 볼넷으로
    바뀌었다. 관절이 잰 투구 순간(스윙 시각)이 판독 시각보다 늦게 잡혀 앞 투구 칸으로 갔다 —
    투구 화면 시작으로도 그 투구의 결과를 안다."""
    on23 = (False, True, True)
    walk = twice(R(10, outs=1, balls=3, bases=on23), R(14, outs=1, bases=(True, True, True)))
    cues = [Cue(9.9, "shot_start", ""), Cue(15, "pitch_seen", ""), _view(17, "no_play")]
    (j,) = judge(walk, cues)
    assert j.code == "walk" and j.detail["pitch_t"] == 17
    # 직전 상태가 화면이 시작되기 전에 보였어도 그 사이 끝난 다른 투구 화면이 없으면 이 화면의
    # 공이다(판독기 v5 — 준플레이오프 2차전·한국시리즈 1차전 볼넷). 다른 투구 화면이 끝났으면
    # (그 공의 결과가 늦게 보였을 수 있다) 투구 순간이 있어야 붙인다
    early = [Cue(12, "shot_start", ""), Cue(15, "pitch_seen", ""), _view(17, "no_play")]
    (j,) = judge(walk, early)
    assert j.detail.get("pitch_t") == 17
    between = [_view(11, "no_play")] + early
    (j,) = judge(walk, between)
    assert "pitch_t" not in j.detail
    # 독립 리뷰 — 투구 하나짜리가 아닌 긴 화면(다른 투구가 섞임)이나, 화면에서 잰 투구가 직전
    # 상태보다 앞이면(이 전이의 투구가 아님) 붙이지 않는다
    long_shot = [Cue(9.9, "shot_start", ""), Cue(25, "pitch_seen", ""), _view(40, "in_play")]
    (j,) = judge(walk, long_shot)
    assert "pitch_t" not in j.detail and j.code == "walk"
    before_pitch = [Cue(8, "shot_start", ""), Cue(9, "pitch_seen", ""), _view(17, "no_play")]
    (j,) = judge(walk, before_pitch)
    assert "pitch_t" not in j.detail


def test_pose_cues_carry_the_pitch_view_start():
    swing = {"t": 5.0, "t0": 3.0, "t1": 7.0, "swing": True, "view": "no_play"}
    cues = pose_cues({"swings": [swing]})
    assert sorted((c.t, c.kind) for c in cues) == [
        (3.0, "shot_start"), (5.0, "pitch_seen"), (5.0, "swing"), (7.0, "no_play")]


def test_single_read_that_ends_a_half_with_more_outs_is_not_a_spike():
    """KS5 1회초 — '무사 2-2' 뒤 '2사 2-2'(삼진 + 못 본 아웃, 다음 타자도 2-2) 한 장이
    카운트가 그대로라
    한 번에 못 바뀐다고 버려졌다. 같은 반 이닝에서 아웃만 늘고 다음이 새 반 이닝이면 남긴다."""
    seen = [R(0, strikes=2, balls=2)] * 2 + [R(9, outs=2, strikes=2, balls=2)] + [
        R(30, half="bot", inning=3)] * 2
    assert [(r.half, r.outs) for r in stabilize(seen, min_repeat=1)] == [
        ("top", 0), ("top", 2), ("bot", 0)]
    # 주자까지 바뀐 한 장은 못 본 플레이를 판정으로 만든다(투수 모음) — 예전처럼 버린다
    runner = [R(0, outs=1)] * 2 + [R(9, outs=2, strikes=2, bases=(False, True, False))] + [
        R(30, inning=4)] * 2
    assert [(r.inning, r.outs) for r in stabilize(runner, min_repeat=1)] == [(3, 1), (4, 0)]


def test_count_read_as_score_takes_the_score_of_the_states_around_it():
    """KS5 MBC 상자 — 가운데 '3-0' 볼카운트를 점수 3:0으로도 읽었다. 앞뒤 안정 상태의 점수(1:0)가
    같으면 그 점수로 고쳐, 볼넷 직전 상태를 살린다. 0-0·0:0은 우연히 늘 같아 고치지 않는다."""
    seen = [R(0, outs=2, away=1, bases=(False, True, False))] * 2 + [
        R(20, half="bot", outs=1, balls=3, away=3)] + [
        R(28, half="bot", outs=1, away=1, bases=(True, False, False))] * 2
    states = stabilize(seen, min_repeat=1)
    assert [(r.half, r.balls, r.away) for r in states] == [("top", 0, 1), ("bot", 3, 1),
                                                           ("bot", 0, 1)]
    walk = judge(seen, [_view(25, "no_play")], min_repeat=1)
    assert "walk" in [j.code for j in walk]
    # 0-0 카운트와 0:0 점수는 고치지 않는다(PO 2차전 — 득점 전 그림이 잠깐 다시 보임)
    flick = [R(0, outs=2, away=1, bases=(True, False, False))] + [
        R(1, outs=2, bases=(True, False, True))] + [
        R(9, outs=2, away=1, bases=(True, False, False))]
    fixed = [r for r in stabilize(flick, min_repeat=1) if r.bases == (True, False, True)]
    assert all(r.away == 0 for r in fixed)


def test_runs_beyond_the_runner_on_third_are_not_a_balk():
    """KS5 — 1·2루 → 2·3루 + 1점(적시 2루타)을 '모양은 보크'로 봤다. 3루 주자 없이 들어온 점수는
    투구 하나의 진루로 안 난다 — 공이 맞아 나간 출루다."""
    j = one(R(0, outs=1, bases=(True, True, False)), R(10, outs=1, bases=(False, True, True),
                                                       home=1))
    assert j.code != "balk" and j.code == "reach"


def test_outfielder_catch_then_run_is_a_sac_fly():
    """KS5 — '중견수 뒷걸음질 잡았고 3루 주자 홈으로'. '희생플라이' 말이 없어도 외야수가
    잡고 득점이면
    희생플라이다. 내야수가 잡은 땅볼 득점은 아니다."""
    before = R(0, outs=1, bases=(False, True, True))
    after = R(10, outs=2, bases=(False, True, False), away=1)
    told = [Cue(8, "speech", "센터 쪽 중견수 뒷걸음질 잡았고 3루 주자 홈으로 들어갑니다")]
    assert one(before, after, told).code == "sac_fly"
    ground = [Cue(8, "speech", "유격수 잡아서 1루 아웃 그사이 3루 주자 홈으로")]
    assert one(before, after, ground).code != "sac_fly"


def test_strikeout_merged_with_an_unseen_out_keeps_the_strikeout():
    """KS5 1회초 — 2스트라이크 삼진 + 못 본 아웃이 한 전이(무사 → 2사)로 합쳐졌다.
    보인 투구가 공이 안
    맞았고 해설이 삼진이면 그 타석은 삼진(합쳐짐 표시)."""
    j = one(R(0, balls=2, strikes=2), R(10, outs=2, balls=2, strikes=2),
            [_view(4, "no_play"), Cue(5, "speech", "삼진으로 잡아내면서 출발합니다")])
    assert (j.code, j.detail.get("merged")) == ("strikeout", True)
    plain = one(R(0, balls=2, strikes=2), R(10, outs=2, balls=2, strikes=2), [_view(4, "no_play")])
    assert plain.code == "out"


def test_strikeout_when_the_two_strike_count_was_never_read():
    """KS5 2회말 — 직전 판독이 타석 시작(0-0)이라 2스트라이크를 못 봤다. 휘둘렀는데 공이 안 맞았고
    해설이 '탈삼진'이면 삼진. 새 판독은 다음 타자 카운트(0-1)다. 해설이 없으면 그냥 아웃."""
    before, after = R(0, outs=1, bases=(True, False, False)), R(10, outs=2, strikes=1,
                                                                bases=(True, False, False))
    cues = [Cue(4, "pitch_seen", "[관절] 투구 순간"), Cue(4, "swing", "[관절] 스윙"),
            _view(6, "no_play")]
    assert one(before, after, cues + [Cue(7, "speech", "두 번째 탈삼진")]).code == "strikeout"
    assert one(before, after, cues).code == "out"


def test_review_guards_on_the_new_ks5_rules():
    """독립 리뷰 — 새 규칙이 엉뚱한 곳에서 서지 않게."""
    # 외야수가 '안타'를 잡아 송구한 아웃은 희생플라이가 아니다. 3루 주자가 없으면 보지 않는다
    sf_before = R(0, outs=1, bases=(False, True, True))
    sf_after = R(10, outs=2, bases=(False, True, False), away=1)
    hit = [Cue(8, "speech", "우익수 앞 안타 잡아서 2루 송구 타자 주자 아웃 그사이 득점")]
    assert one(sf_before, sf_after, hit).code != "sac_fly"
    no_third = one(R(0, outs=1, bases=(True, True, False)),
                   R(10, outs=2, bases=(False, True, False), away=1),
                   [Cue(8, "speech", "중견수 잡았고 홈으로")])
    assert no_third.code != "sac_fly"
    # 폭투 해설이 있으면 2·3루에서 둘 다 들어와도 같은 타석의 진루다
    wp = one(R(0, outs=1, bases=(False, True, True), strikes=1),
             R(10, outs=1, strikes=2, home=2), [Cue(8, "speech", "폭투 2루 주자까지 홈으로")])
    assert wp.code == "wild_pitch"
    # 못 본 2스트라이크 삼진 — 직전 상태 전의 '삼진'·공이 맞은 단서는 근거가 아니다
    before, after = R(0, outs=1, bases=(True, False, False)), R(10, outs=2, strikes=1,
                                                                bases=(True, False, False))
    cues = [Cue(4, "pitch_seen", "[관절] 투구 순간"), Cue(4, "swing", "[관절] 스윙"),
            _view(6, "no_play")]
    assert one(before, after, cues + [Cue(-1, "speech", "앞 타자 삼진")]).code == "out"
    assert one(before, after, cues + [Cue(7, "speech", "삼진"), Cue(5, "contact")]).code == "out"
    # 합쳐진 전이에 주자가 늘었으면(출루 + 삼진) 삼진으로 보지 않는다
    merged = one(R(0, balls=2, strikes=2, bases=(True, False, False)),
                 R(10, outs=2, balls=2, strikes=2, bases=(False, False, True)),
                 [_view(4, "no_play"), Cue(5, "speech", "삼진")])
    assert merged.code != "strikeout"


def test_double_play_split_across_a_long_gap_when_commentary_says_so():
    """KS2 — 1사 1루 직선타 병살: 점수판이 2사(주자 그대로)를 먼저 보이고 18초 뒤 다음 반 이닝.
    사이에 투구가 없고 해설이 '더블 플레이'면 두 아웃을 한 병살로 합친다. 말이 없으면 그대로 둘."""
    seen = twice(R(0, outs=1, bases=(True, False, False)), R(9, outs=2, bases=(True, False, False)),
                 R(27, half="bot"))
    cues = [_view(5, "in_play"), Cue(10, "speech", "이렇게 더블 플레이")]
    codes = [j.code for j in judge(seen, cues) if j.transition is not None]
    assert codes == ["double_play"]
    assert [j.code for j in judge(seen, [_view(5, "in_play")]) if j.transition is not None] == [
        "out", "out"]


def test_two_out_hit_before_skipped_half_innings_is_a_hit():
    """KS2 — 4회말 2사 3루 '좌익수 앞에 떨어지는 안타'로 1점, 다음 판독은 6회초(5회를 건너뜀).
    상대 팀 점수가 그대로이고 공이 맞아 나갔고 해설이 안타면 3아웃 전 안타(출루)다."""
    before = R(0, inning=4, half="bot", outs=2, bases=(False, False, True))
    after = R(10, inning=6, away=0, home=1)
    hit = [_view(4, "in_play"), Cue(8, "speech", "좌익수 앞에 떨어지는 안타")]
    assert one(before, after, hit).code == "reach"
    assert one(before, after, [_view(4, "in_play")]).code != "reach"  # 말이 없으면 못 가른다
    other = R(10, inning=6, away=1, home=1)  # 건너뛴 반 이닝에 상대 팀도 점수 — 섞였다
    assert one(before, other, hit).code != "reach"


def test_ks2_rules_do_not_fire_on_lookalike_words_or_other_plays():
    """독립 리뷰 — '안타깝다'는 안타가 아니고, 새 반 이닝에 주자가 있으면(그 반 이닝 안타일 수 있음)
    안 쓴다. '병살 노려야'·30초 넘게 떨어진 두 아웃은 합치지 않는다."""
    before = R(0, inning=4, half="bot", outs=2, bases=(False, False, True))
    after = R(10, inning=6, home=1)
    sorry = [_view(4, "in_play"), Cue(8, "speech", "안타깝게 놓쳤네요")]
    assert one(before, after, sorry).code != "reach"
    runner = R(10, inning=6, home=1, bases=(True, False, False))
    hit = [_view(4, "in_play"), Cue(8, "speech", "좌익수 앞 안타")]
    assert one(before, runner, hit).code != "reach"
    seen = twice(R(0, outs=1, bases=(True, False, False)), R(9, outs=2, bases=(True, False, False)),
                 R(27, half="bot"))
    aim = [_view(5, "in_play"), Cue(8, "speech", "여기서 병살 노려야죠")]
    assert [j.code for j in judge(seen, aim) if j.transition is not None] == ["out", "out"]
    far = twice(R(0, outs=1, bases=(True, False, False)), R(9, outs=2, bases=(True, False, False)),
                R(60, half="bot"))
    said = [_view(5, "in_play"), Cue(10, "speech", "더블 플레이")]
    assert [j.code for j in judge(far, said) if j.transition is not None] == ["out", "out"]


def test_runner_vanishing_without_an_out_means_the_double_play_is_not_finished():
    """PO 3차전 — 1사 1루 직선타 병살: 점수판이 '2사 주자 없음'(주자가 아웃 없이 사라짐)을
    먼저 보이고
    41초 뒤 다음 반 이닝. 사람 수가 안 맞으면 다음 아웃까지 한 플레이 — 병살이다(해설에 '병살'이
    없어도). 주자가 그대로 남은 2사(따로 난 아웃일 수 있음)는 그렇게 보지 않는다."""
    seen = twice(R(0, outs=1, bases=(True, False, False)), R(9, outs=2), R(50, half="bot"))
    codes = [j.code for j in judge(seen, [_view(5, "in_play")]) if j.transition is not None]
    assert codes == ["double_play"]
    kept = twice(R(0, outs=1, bases=(True, False, False)), R(9, outs=2, bases=(True, False, False)),
                 R(50, half="bot"))
    assert [j.code for j in judge(kept, [_view(5, "in_play")]) if j.transition is not None] == [
        "out", "out"]


def test_caught_stealing_is_not_a_vanished_runner():
    """독립 리뷰 — 도루 실패도 주자가 사라지지만 타자는 그대로(카운트가 이어짐)라 병살로
    합치지 않는다."""
    cs = twice(R(0, outs=1, bases=(True, False, False), balls=1, strikes=1),
               R(9, outs=2, balls=1, strikes=1), R(50, half="bot"))
    codes = [j.code for j in judge(cs, [Cue(6, "slide"), Cue(7, "speech", "도루 실패")])
             if j.transition is not None]
    assert "double_play" not in codes


def _shot(t_start, t_pitch, t_end, kind):
    return [Cue(t_start, "shot_start", "[관절] 투구 화면 시작"),
            Cue(t_pitch, "pitch_seen", "[관절] 투구 순간"), _view(t_end, kind)]


def test_the_pitch_view_that_made_the_state_is_not_the_third_out_pitch():
    """준플레이오프 1차전 — '2사 2-2'를 만든 볼 화면이 끝난 직후(0.03초 뒤)에 그 상태가 읽혀,
    반 이닝이
    바뀐 전이를 가를 때 그 화면이 3아웃 투구로 쓰였다(삼진이 앞 칸에 붙음). 그 화면의 투구 순간이
    상태보다 앞이면 그 상태를 만든 투구다 — 3아웃은 그 뒤 첫 투구 화면."""
    seen = twice(R(10, outs=2, balls=2, strikes=2), R(40, half="bot", outs=1))
    cues = (_shot(5, 7, 10.03, "no_play") + _shot(17, 20, 22, "view_unknown")
            + _shot(26, 28, 30, "view_unknown")
            + [Cue(21, "speech", "낮은 코스 들어갔습니다 삼진")])
    js = [(j.code, round(j.t)) for j in judge(seen, cues) if j.transition is not None]
    assert js == [("strikeout", 22), ("out", 40)]
    # 관절이 그 화면의 투구 순간을 못 쟀으면 알 수 없다 — 예전대로 첫 화면
    bare = [_view(10.03, "no_play"), _view(22, "view_unknown"), _view(30, "view_unknown")]
    assert round([j for j in judge(seen, bare) if j.transition is not None][0].t) == 10


def test_sac_fly_whose_run_shows_only_in_the_next_half():
    """준플레이오프 1차전 — 점수판이 아웃만 먼저 올리고(3루 주자·점수 그대로) 득점은 다음 반 이닝
    판독에서야 보였다(남은 아웃은 하이라이트가 건너뜀). 외야수가 잡았다는 말이 있고
    공이 맞아 나갔으면
    그 득점은 희생플라이다. 받아쓰기는 '우익수'를 '오익수'로도 적는다."""
    first_third = (True, False, True)
    seen = twice(R(0, half="bot", strikes=1, bases=first_third, away=1, home=2),
                 R(10, half="bot", outs=1, bases=first_third, away=1, home=2),
                 R(30, inning=4, balls=1, strikes=2, away=1, home=3))
    said = Cue(8, "speech", "오른쪽 오익수 잡았습니다 3루주자 여유있게 홈플레이트를 밟았습니다")
    codes = [j.code for j in judge(seen, [_view(6, "in_play"), said]) if j.transition is not None]
    assert codes == ["sac_fly", "out"]
    # 말이 없으면 뒤 타자의 적시타일 수도 있다 — 그대로 아웃 둘
    plain = [j.code for j in judge(seen, [_view(6, "in_play")]) if j.transition is not None]
    assert plain == ["out", "out"]
    # 공이 안 맞은 화면이면(삼진 뒤 다른 플레이의 득점) 붙이지 않는다
    no_play = [j.code for j in judge(seen, [_view(6, "no_play"), said]) if j.transition is not None]
    assert "sac_fly" not in no_play


def test_review_guards_on_the_spo1_rules():
    """독립 리뷰 — 투구 순간을 못 잰 화면이 앞 화면의 투구 순간을 가져가지 않고, 늦은 득점은 정확히
    1점·들어왔다는 말이 있을 때만, 외야수 낱말은 이름('김익수')·'못 잡았'·'2루타'에 안 맞는다."""
    seen = twice(R(10, outs=2, balls=2, strikes=2), R(40, half="bot", outs=1))
    cues = (_shot(5, 7, 10.03, "no_play") + [_view(22, "view_unknown"), _view(30, "view_unknown")]
            + [Cue(21, "speech", "낮은 코스 들어갔습니다 삼진")])
    js = [(j.code, round(j.t)) for j in judge(seen, cues) if j.transition is not None]
    assert js == [("strikeout", 22), ("out", 40)]

    first_third = (True, False, True)
    before = R(0, half="bot", strikes=1, bases=first_third, away=1, home=2)
    out1 = R(10, half="bot", outs=1, bases=first_third, away=1, home=2)

    def codes(after, text):
        cues = [_view(6, "in_play"), Cue(8, "speech", text)]
        return [j.code for j in judge(twice(before, out1, after), cues) if j.transition is not None]

    one_run = R(30, inning=4, balls=1, strikes=2, away=1, home=3)
    two_runs = R(30, inning=4, balls=1, strikes=2, away=1, home=4)
    assert codes(one_run, "우익수 잡았습니다 3루 주자 홈으로 들어옵니다") == ["sac_fly", "out"]
    assert "sac_fly" not in codes(two_runs, "우익수 잡았습니다 3루 주자 홈으로 들어옵니다")
    assert "sac_fly" not in codes(one_run, "우익수 잡았습니다 주자는 움직이지 못합니다")
    assert "sac_fly" not in codes(one_run, "김익수가 잡았고 득점")
    assert "sac_fly" not in codes(one_run, "우익수 못 잡았습니다 득점")
    assert "sac_fly" not in codes(one_run, "우익수 키를 넘는 2루타 잡아서 송구 득점")


def test_count_cleared_after_the_third_out_is_not_a_play():
    """준플레이오프 2차전(SBS 상자) — 2사에서 타석이 아웃으로 끝나면 아웃 셋을 안 그리고 카운트만
    지운 '2사 0-0'을 2~3초 보인다. 그 판독은 플레이가 아니다: 빼고 전이를 만들어 직전 2스트라이크가
    3아웃 전이의 직전 상태로 남는다(삼진). 그대로 두면 '아웃' + 볼카운트 0-0이었다."""
    first = (True, False, False)
    two = R(0, outs=2, balls=2, strikes=2, bases=first)
    cleared = R(8, outs=2, bases=first)
    assert third_out_cleared(two, cleared)
    assert not third_out_cleared(R(0, outs=1, balls=2, strikes=2), R(8, outs=1))  # 2사에서만
    assert not third_out_cleared(two, R(8, outs=2))  # 주자가 바뀌었으면 다른 플레이다
    assert not third_out_cleared(R(0, outs=2, bases=first), cleared)  # 원래 0-0이면 지운 게 아니다
    assert [r.t for r in fold_cleared([two, cleared, R(30, half="bot")])] == [0.0, 30.0]
    seen = twice(two, cleared, R(30, half="bot", balls=3, strikes=2))
    codes = [j.code for j in judge(seen, _pitch(3, 6)) if j.transition is not None]
    assert codes == ["strikeout"]
    # 공이 맞아 나갔으면 삼진이 아니다 — 종류 모를 아웃 하나
    batted = [j.code for j in judge(seen, _pitch(3, 6, "in_play")) if j.transition is not None]
    assert batted == ["out"]


def test_game_ending_strikeout_when_the_last_read_only_cleared_the_count():
    """준플레이오프 2차전 9회초 2사 2-2 — 삼진 뒤 마지막 판독이 '2사 0-0'이었다. 그 판독 뒤엔 투구가
    없어 경기 끝 아웃을 못 냈고, 카운트만 지운 전이가 '출루'가 됐다."""
    seen = twice(R(0, inning=9, outs=2, balls=2, strikes=2, away=3, home=6),
                 R(5, inning=9, outs=2, away=3, home=6))
    cues = _pitch(2, 3) + [Cue(3, "speech", "스윙 삼진 경기 끝")]
    js = [j.code for j in judge(seen, cues, finished=True) if j.transition is not None]
    assert js == ["strikeout"]


def test_runner_read_once_before_the_count_cleared_third_out_is_kept():
    """준플레이오프 2차전 4회말 — 2사 2-2에 사구로 나간 주자 한 장('2사 2-2 1루') 뒤 다음 타자의
    3아웃으로 '2사 0-0 1루'가 보였다. 타석이 끝났는데 아무도 안 늘어 그 앞 한 장이 튄 판독으로
    버려졌다. 지운 판독이 그 반 이닝의 마지막일 때만 남긴다 — 같은 반 이닝이 이어지면 카운트
    오독."""
    first = (True, False, False)
    seen = [R(0, outs=2, balls=2, strikes=2, half="bot"),
            R(29, outs=2, balls=2, strikes=2, half="bot", bases=first),
            R(38, outs=2, half="bot", bases=first), R(45, inning=4, outs=1, balls=2, strikes=2)]
    assert [(r.t, r.bases[0]) for r in stabilize(seen, min_repeat=1)][:2] == [(0.0, False),
                                                                             (29.0, True)]
    # 2-2에서 휘두르지 않은 공이 안 맞아 나갔는데 1루, 해설 '팔에 맞았습니다' — 몸에 맞는 공.
    # 다음 타자의 3아웃은 따로
    cues = _pitch(-1, 0.7) + _pitch(33, 34, "in_play") + [Cue(-4, "speech", "지금 팔에 맞았습니다")]
    codes = [j.code for j in judge(seen, cues, min_repeat=1) if j.transition is not None]
    assert codes == ["hbp", "out"]
    # 같은 반 이닝이 이어지면(전환 중 카운트 오독 '2-1' 한 장) 예전처럼 그 한 장을 버린다
    going = [R(0, outs=2, balls=3, strikes=2), R(3, outs=2, balls=2, strikes=1, bases=first),
             R(8, outs=2, bases=first), R(16, outs=2, balls=2, strikes=2, bases=first)]
    assert [r.t for r in stabilize(going, min_repeat=1)] == [0.0, 8.0, 16.0]


def test_hit_by_pitch_with_two_strikes_needs_a_walk_to_be_impossible():
    """2스트라이크에서도 휘두르지 않은 공이 안 맞아 나갔는데 1루면 몸에 맞는 공 — 단 직전 상태 뒤
    던진 공으로 볼 넷을 채울 수 없을 때만. PO 3차전 8회말: 2-2 뒤 투구 화면이 둘(못 읽은 3-2를
    거친 볼넷)이라 사구가 아니다. 휘둘렀으면 낫아웃일 수 있다."""
    before, after = R(0, outs=1, balls=2, strikes=2), R(20, outs=1, bases=(True, False, False))
    said = [Cue(-4, "speech", "자 지금 팔에 맞았습니다")]  # 조각 시작은 말보다 앞선다
    assert one(before, after, _pitch(3, 5) + said).code == "hbp"
    assert one(before, after, _pitch(3, 5)).code == "reach"  # 말이 없으면 낫아웃일 수도 있다
    two_pitches = _pitch(3, 5) + _pitch(10, 14)
    assert one(before, after, two_pitches + said).code == "reach"
    swung = _pitch(15, 17) + [Cue(15, "swing", "[관절] 스윙")]
    assert one(before, after, swung + said).code == "reach"
    # 타구가 투수 몸에 맞은 말은 공이 맞아 나간 화면이라 여기 안 든다
    comebacker = [_view(5, "in_play"), Cue(4, "speech", "타구가 투수 다리에 맞고 굴절됩니다")]
    assert one(before, after, comebacker).code == "reach"
    assert one(before, after, _pitch(3, 5) + [Cue(4, "speech", "타이밍 등에 맞춰 나옵니다")]
               ).code == "reach"


def test_swing_with_two_strikes_merged_with_an_unseen_out_is_a_strikeout():
    """준플레이오프 2차전 2회초 '무사 0-2 → 2사' — 해설은 말이 없었지만 관절이 그 공의 스윙을
    봤고 공이 안 맞아 나갔다. 휘두르지 않았으면(볼일 수 있다) 예전대로 아웃."""
    before, after = R(0, strikes=2), R(9, outs=2)
    swung = _pitch(0.4, 2.2) + [Cue(0.4, "swing", "[관절] 타자가 배트를 휘두름")]
    j = one(before, after, swung)
    assert (j.code, j.detail.get("merged")) == ("strikeout", True)
    assert one(before, after, _pitch(0.4, 2.2)).code == "out"
    took = _pitch(0.4, 2.2) + [Cue(0.4, "no_swing", "[관절] 타자가 공을 그냥 보냄")]
    assert one(before, after, took).code == "out"


def test_runner_picked_off_for_the_third_out():
    """준플레이오프 2차전 3회초 2사 1·2루 — '1루 견제 아웃 … 마지막 아웃 카운트'. 반 이닝이 바뀐
    전이엔 주자 모양이 없어 '아웃'이 됐다. 직전 상태 무렵 해설이 견제사·도루 실패를 말했고 그 뒤
    공이 맞아 나간 화면도 삼진이라는 말도 없을 때만."""
    on = (True, True, False)
    before, after = R(10, outs=2, balls=2, strikes=2, bases=on), R(43, inning=4, half="bot")
    said = Cue(9, "speech", "바깥쪽 스위퍼 그리고 1루 견제 아웃 베이스 위에서 마지막 아웃 카운트")
    tag = Cue(9.9, "slide_tag", "[관절] 미끄러져 들어온 주자에게 수비 손이 닿음(태그)")
    j = one(before, after, [said, tag])
    assert (j.code, j.detail.get("caught")) == ("caught_stealing", True)
    # 관절이 없어도(실시간) 같은 말이 반 이닝이 끝났다고 하면 — '마지막 아웃 카운트'
    assert one(before, after, [said]).code == "caught_stealing"
    assert one(before, after, [tag]).code == "out"  # 태그만으로는 모른다
    assert one(before, after, [Cue(9, "speech", "견제 한 번 하고 다시 승부")]).code == "out"
    assert one(before, after, [said, tag, _view(14, "in_play")]).code == "out"  # 타구가 있었다
    k_said = Cue(12, "speech", "헛스윙 삼진")
    assert one(before, after, [said, tag, k_said]).code != "caught_stealing"
    assert one(before, after, [Cue(60, "speech", "견제 아웃 마지막 아웃")]).code == "out"  # 한참 뒤
    empty = R(10, outs=2, balls=2, strikes=2)  # 주자가 없으면 잡힐 주자도 없다
    assert one(empty, after, [said, tag]).code == "out"
    one_out = R(10, outs=1, balls=2, strikes=2, bases=on)  # 2사가 아니면 아웃이 둘 이상이다
    assert one(one_out, after, [said, tag]).code != "caught_stealing"


def test_review_guards_on_the_spo2_rules():
    """독립 리뷰 — 3아웃 견제사는 통계 이야기('도루 저지율')·앞의 둘째 아웃 이야기·삼진 단서에 서지
    않고, 2스트라이크 사구는 '몸에 맞고 파울'에 서지 않는다. 1스트라이크 표시가 안 쓰인 판정은 다음
    표시가 쓸 수 있다."""
    on = (True, False, False)
    two_out = R(10, outs=2, balls=1, strikes=1, bases=on)
    nxt = R(43, inning=4, half="bot")
    rate = Cue(12, "speech", "이 포수가 도루 저지율이 리그 최고죠")
    assert one(two_out, nxt, [rate]).code == "out"
    assert one(two_out, nxt, [rate, Cue(30, "speech", "내야 땅볼 아웃 이닝 끝")]).code == "out"
    # 앞의 둘째 아웃이 견제사였고 셋째 아웃은 못 본 플레이 — 태그도 '이닝 끝' 말도 없으면 모른다
    earlier = Cue(5, "speech", "1루 견제 아웃 됐습니다")
    assert one(two_out, nxt, [earlier]).code == "out"
    # 2사 1-2에서 헛스윙 삼진으로 끝났으면 앞 견제사 말이 있어도 삼진
    k_before = R(10, outs=2, balls=1, strikes=2, bases=on)
    k_cues = [earlier, _view(20, "no_play"), Cue(20, "swing_miss", "[장면] 헛스윙"),
              Cue(7.5, "slide_tag", "[관절] 태그")]
    assert one(k_before, nxt, k_cues).code == "strikeout"
    # 한참 전(직전 상태 3초보다 앞)의 태그는 앞 플레이의 것이다
    old_tag = Cue(2, "slide_tag", "[관절] 태그")
    assert one(two_out, nxt, [earlier, old_tag]).code == "out"

    b2, a2 = R(0, outs=1, balls=2, strikes=2), R(20, outs=1, bases=on)
    assert one(b2, a2, _pitch(3, 5) + [Cue(4, "speech", "공이 발에 맞고 파울이 됩니다")]
               ).code == "reach"
    assert one(b2, a2, _pitch(3, 5) + [Cue(4, "speech", "손에 맞았습니다 파울")]).code == "reach"


def test_sac_fly_split_without_a_pitch_view_needs_the_commentary():
    """준플레이오프 2차전 7회말 1사 1·3루 — 아웃이 먼저, 5초 뒤 득점이 따로 올라왔고 그 투구 화면은
    다른 각도라 못 봤다(공이 맞아 나갔는지 모름). 해설이 희생플라이·태그업을 말했을 때만 합친다.
    받아쓰기는 '태그업'을 '탭업'으로도 적는다."""
    corners = (True, False, True)
    seen = twice(R(0, half="bot", outs=1, balls=1, strikes=1, bases=corners, away=2, home=5),
                 R(19, half="bot", outs=2, bases=corners, away=2, home=5),
                 R(24, half="bot", outs=2, bases=(True, False, False), away=2, home=6))

    def codes(cues):
        return [j.code for j in judge(seen, cues) if j.transition is not None]

    assert codes([Cue(26, "speech", "나성범의 희생 플라이")]) == ["sac_fly"]
    assert codes([Cue(16, "speech", "3루 주자 탭업")]) == ["sac_fly"]
    assert codes([]) == ["out", "reach"]  # 말이 없으면 그대로 둘
    assert "sac_fly" not in codes([Cue(26, "speech", "희생 플라이"), _view(10, "no_play")])
    assert "sac_fly" not in codes([Cue(26, "speech", "폭투로 3루 주자 득점 희생 플라이는 아니고")])


def test_strikeout_mark_when_only_one_strike_was_read():
    """준플레이오프 2차전 5회초 — '1-1' 판독 15초 뒤 삼진 표시(K), 1-2는 못 읽었다. 진행 중인
    타석이 방금 보였으면 그 타석의 삼진이다. 오래된 판독·0-0(아웃이 이미 올라감)은 아니다."""
    first = (True, False, False)
    seen = twice(R(0, outs=1, balls=1, strikes=1, bases=first), R(15, outs=2, bases=first))
    mark = Cue(14.9, "k_mark", "[점수판] 삼진 표시(K)")
    (j,) = [j for j in judge(seen, _pitch(11, 13) + [mark]) if j.transition is not None]
    assert j.code == "strikeout"
    # 그 아웃의 공이 맞아 나갔으면(표시를 잘못 본 한 장) 삼진으로 바꾸지 않는다
    (j,) = [j for j in judge(seen, _pitch(11, 13, "in_play") + [mark]) if j.transition is not None]
    assert j.code == "out"
    # 1-1 판독이 오래됐고(45초 전) 아웃 판독도 표시와 떨어져(5초 뒤) 있으면 그 타석인지 모른다
    stale = twice(R(0, outs=1, balls=1, strikes=1, bases=first), R(50, outs=2, bases=first))
    late = Cue(44.9, "k_mark", "[점수판] 삼진 표시(K)")
    (j,) = [j for j in judge(stale, _pitch(41, 43) + [late]) if j.transition is not None]
    assert j.code == "out"
    # 표시뿐이고 아웃 판독이 없으면(영상 끝) 2스트라이크를 못 본 표시로는 삼진을 내지 않는다
    alone = judge(twice(R(0, outs=1, balls=1, strikes=1, bases=first)), [mark], finished=True)
    assert [j.code for j in alone] == []


def test_whisper_spellings_of_strikeout_and_intentional_walk():
    """준플레이오프 2차전 받아쓰기 — '스윙 섬진', '고의사고를 해 주는 거죠'. '섬진강'은 아니다."""
    k = one(R(0, outs=1, balls=1, strikes=2), R(10, outs=2),
            [_view(5, "view_unknown"), Cue(6, "speech", "한준수 스윙 섬진")])
    assert k.code == "strikeout"
    river = one(R(0, outs=1, balls=1, strikes=2), R(10, outs=2),
                [_view(5, "view_unknown"), Cue(6, "speech", "섬진강 재첩 광고")])
    assert river.code == "out"
    ibb = one(R(0, outs=1, bases=(False, False, True)), R(10, outs=1, bases=(True, False, True)),
              [Cue(6, "speech", "그냥 고의사고를 해 주는 거죠")])
    assert ibb.code == "walk"


def test_strikeout_word_in_the_piece_that_holds_a_mid_at_bat_state():
    """준플레이오프 3차전 — 받아쓰기는 10초 조각 시작 시각에 찍힌다. 직전 상태 '2사 2-1'이
    471초에 보였고 '여섯 번째 탈삼진'은 470초 조각에 들었다. 직전 상태가 타석 중간이면 그 조각의
    말을 듣는다 — 타석이 막 시작했으면(0-0) 앞 타자 말일 수 있어 듣지 않는다(독립 리뷰 가드는
    그대로)."""
    before, after = R(10, outs=2, balls=2, strikes=1), R(25, inning=4, half="bot")
    cues = [Cue(13, "pitch_seen", "[관절] 투구 순간"), Cue(13, "swing", "[관절] 스윙"),
            _view(15, "no_play"), Cue(9, "speech", "이렇게 무너집니다. 여섯 번째 탈삼진")]
    assert one(before, after, cues).code == "strikeout"
    fresh = R(10, outs=2)
    assert one(fresh, after, cues).code == "out"
    far = cues[:3] + [Cue(-5, "speech", "여섯 번째 탈삼진")]  # 조각이 직전 상태를 안 품음
    assert one(before, after, far).code == "out"


def test_strikeout_mark_shown_after_the_out_inside_the_same_view():
    """MBC 포스트시즌 상자는 아웃을 먼저 올리고 1초 뒤 'nK' 띠를 띄운다 — 표시가 그 판정의 투구 화면
    안이다. 2스트라이크째를 못 읽어 직전 판독이 1스트라이크여도 그 타석의 삼진이다(준플레이오프
    3차전 1회말). 0스트라이크면(카운트를 못 읽음) 쓰지 않는다."""
    seen = twice(R(0, balls=0, strikes=1), R(10, outs=1))
    cues = _pitch(6, 12) + [Cue(11, "k_mark", "[점수판] 삼진 표시(K)")]
    (j,) = [j for j in judge(seen, cues) if j.transition is not None]
    assert j.code == "strikeout"
    # 0스트라이크(카운트를 못 읽음)면 판정 화면 안이라는 것만으로는 안 쓴다 — 아웃 직후(K_MARK_LAG)
    # 표시가 아니면 아웃 그대로
    zero = twice(R(0), R(10, outs=1))
    late = _pitch(6, 14) + [Cue(13.5, "k_mark", "[점수판] 삼진 표시(K)")]
    (j,) = [j for j in judge(zero, late) if j.transition is not None]
    assert j.code == "out"


def test_strikeout_mark_right_after_the_out_reading():
    """MBC 포스트시즌 상자는 아웃을 먼저 올리고 1초쯤 뒤 'nK' 띠를 띄운다. 직전 판독이 2스트라이크가
    아니어도(못 읽음) 표시 직전 3초 안에 아웃 하나가 올라갔고 공이 맞아 나간 화면이 없으면 그 아웃의
    삼진이다. 타구가 있었거나 주자가 움직였거나 표시가 한참 뒤면 아니다."""
    mark = Cue(11.1, "k_mark", "[점수판] 삼진 표시(K)")
    seen = twice(R(0, strikes=1), R(10, outs=1))
    (j,) = [j for j in judge(seen, [mark]) if j.transition is not None]
    assert j.code == "strikeout"
    (j,) = [j for j in judge(seen, [mark, _view(6, "in_play")]) if j.transition is not None]
    assert j.code == "out"
    moved = twice(R(0, strikes=1, bases=(True, False, False)),
                  R(10, outs=1, bases=(False, True, False)))
    assert [j.code for j in judge(moved, [mark]) if j.transition is not None] != ["strikeout"]
    late = Cue(14.5, "k_mark", "[점수판] 삼진 표시(K)")
    (j,) = [j for j in judge(seen, [late]) if j.transition is not None]
    assert j.code == "out"


def test_board_changed_in_a_pitch_view_that_started_after_the_before_state():
    """판독 시각을 화소가 바뀐 초로 앞당기면(판독기 v5) 점수판이 관절이 잰 투구 순간보다 1초쯤
    먼저 바뀐 것으로 찍힌다(준플레이오프 2차전 3-0 볼넷). 직전 상태가 보인 뒤 시작한 투구 화면이고
    그 사이 끝난 다른 투구 화면이 없으면 그 화면의 결과다. 새 상태 카운트가 벌써 진행됐거나(다음
    타자 공), 화면이 시작하고 공을 던지기 전에 바뀌었으면 아니다."""
    walk = twice(R(0, balls=3, bases=(False, True, False)),
                 R(4, bases=(True, True, False)))
    cues = _shot(1.0, 5.2, 10.4, "no_play")
    (j,) = [j for j in judge(walk, cues) if j.transition is not None]
    assert (j.code, j.detail.get("pitch_t")) == ("walk", 10.4)
    counted = twice(R(0, balls=3, bases=(False, True, False)),
                    R(4, balls=0, strikes=2, bases=(True, True, False)))
    (j,) = [j for j in judge(counted, cues) if j.transition is not None]
    assert "pitch_t" not in j.detail
    early = _shot(3.8, 6.5, 9.0, "in_play")  # 화면 시작 0.2초 뒤 바뀜 — 앞 플레이
    (j,) = [j for j in judge(walk, early) if j.transition is not None]
    assert "pitch_t" not in j.detail


def test_review_guards_on_the_spo3_rules():
    """독립 리뷰 — 앞 타석 화면 단서가 빠졌을 때 다음 타자 화면이 앞 플레이를 가져가지 않고(홈런이
    '출루', 땅볼이 '삼진'), 뒤쪽 다른 화면의 투구 순간을 빌리지 않는다. 표시가 판독 4초 뒤에
    판정 화면 안에서 떠도 1스트라이크 타석의 삼진이다. 'nK' 띠 뒤에 찍힌 아웃, 3아웃 잔루 삼진도."""
    hr = twice(R(0, balls=1, strikes=1), R(14, away=1))
    cues = [Cue(10, "shot_start", ""), Cue(16, "pitch_seen", ""), _view(20, "no_play")]
    assert [j.code for j in judge(hr, cues) if j.transition is not None] == ["homerun"]
    grounder = twice(R(0, balls=1, strikes=2), R(14, outs=1))
    b_view = [Cue(2, "shot_start", ""), Cue(12, "shot_start", ""), Cue(15, "pitch_seen", ""),
              _view(18, "no_play")]
    assert [j.code for j in judge(grounder, b_view) if j.transition is not None] == ["out"]
    reach = twice(R(0, balls=1), R(14, bases=(True, False, False)))
    own_no_pitch = [Cue(11, "shot_start", ""), _view(17, "no_play"),
                    Cue(30, "shot_start", ""), Cue(33, "pitch_seen", ""), _view(36, "no_play")]
    (j,) = [j for j in judge(reach, own_no_pitch) if j.transition is not None]
    assert "pitch_t" not in j.detail
    # 표시가 판독 4초 뒤, 판정 화면 안(in_pa own) — 1스트라이크 타석
    one_k = twice(R(0, strikes=1), R(10, outs=1))
    (j,) = [j for j in judge(one_k, _pitch(6, 16) + [Cue(14, "k_mark", "")])
            if j.transition is not None]
    assert j.code == "strikeout"
    # 띠가 뜬 초는 화소가 없어 앞당긴 아웃 판독이 띠 뒤(13초)로 찍힘 — 그 사이 투구가 없으면 삼진
    after_bar = twice(R(0), R(13, outs=1))
    (j,) = [j for j in judge(after_bar, [Cue(11.2, "k_mark", "")]) if j.transition is not None]
    assert j.code == "strikeout"
    pitched = [Cue(11.2, "k_mark", ""), Cue(12.0, "pitch_seen", "")]
    (j,) = [j for j in judge(after_bar, pitched) if j.transition is not None]
    assert j.code == "out"
    # 2사 1루 삼진으로 반 이닝 끝 — 점수판이 주자를 지운다
    third = twice(R(0, outs=2, bases=(True, False, False)), R(10, half="bot"))
    (j,) = [j for j in judge(third, [Cue(11, "k_mark", "")]) if j.transition is not None]
    assert j.code == "strikeout"
    # 직전 상태가 0스트라이크면 조각 확장으로 '삼진'을 듣지 않는다(회상일 수 있다)
    early = R(10, outs=1, balls=1, strikes=0)
    k_cues = [Cue(13, "pitch_seen", ""), Cue(13, "swing", ""), _view(15, "no_play"),
              Cue(3, "speech", "오늘 삼진 여섯 개째")]
    assert one(early, R(18, outs=2), k_cues).code == "out"


def test_view_gap_counts_from_the_last_read_of_the_before_state():
    """PO 4차전 — 직전 상태 '1사 2-2'가 처음 보인 건 4초 넘게 전이지만 투구 화면 시작 0.1초 전에
    다시 읽혔다. '곧바로 시작한 화면'은 직전 상태가 마지막으로 확인된 때부터 잰다. 다시 읽히지
    않았으면(4.1초 전 한 번) 그 사이 공이 있었을 수 있어 붙이지 않는다."""
    first = [R(0, outs=1, balls=2, strikes=2), R(0.5, outs=1, balls=2, strikes=2)]
    reread = first + [R(4, outs=1, balls=2, strikes=2)] + twice(R(8, outs=2))
    cues = _shot(4.1, 8.8, 9.6, "no_play")
    (j,) = [j for j in judge(reread, cues) if j.transition is not None]
    assert j.detail.get("pitch_t") == 9.6
    once = first + twice(R(8, outs=2))
    (j,) = [j for j in judge(once, cues) if j.transition is not None]
    assert "pitch_t" not in j.detail


def test_a_runner_state_survives_the_next_reads_count_misread():
    """와일드카드 1차전 6회초 — 뜬공 뒤 '2사 1루'(2-0) 한 장이 다음 판독(도루 뒤 '2사 2루')의 카운트
    오독('2-1'을 '0-0') 때문에 튄 판독으로 버려져 뜬공 아웃과 도루가 '아웃' 하나로 합쳐졌다. 다음
    판독이 주자만 옮겨 갔고 그 뒤 판독이 이 카운트에서 이어지면 받는다. 거꾸로 이 판독의 카운트가 튄
    한 장이면(독립 리뷰: 1-0 → 3-2 → 1-1 → 2-1) 예전처럼 버린다."""
    first, second = (True, False, False), (False, True, False)
    seen = [R(0, outs=1, bases=first), R(10, outs=2, balls=2, bases=first),
            R(18, outs=2, bases=second), R(24, outs=2, balls=2, strikes=2, bases=second)]
    # 카운트를 잘못 읽은 그 한 장(18초)은 제 차례에 버려진다
    assert [(r.t, r.outs, r.bases) for r in stabilize(seen, min_repeat=1)] == [
        (0.0, 1, first), (10.0, 2, first), (24.0, 2, second)]
    # 뜬공 아웃과 주자 진루(단서가 없어 도루인지는 모른다)가 따로 난다
    assert [j.code for j in judge(seen, [], min_repeat=1)] == ["out", "runner"]
    odd = [R(0, outs=1, balls=1, bases=first), R(5, outs=1, balls=3, strikes=2, bases=first),
           R(9, outs=1, balls=1, strikes=1, bases=second), R(14, outs=1, balls=2, strikes=1,
                                                            bases=second)]
    assert [r.t for r in stabilize(odd, min_repeat=1)] == [0.0, 9.0, 14.0]


def test_new_batter_read_survives_the_board_clearing_the_count_first():
    """2025 플레이오프 1차전 8회말 — SBS 상자는 타석이 끝나면 카운트부터 0-0으로 지우고 주자는
    몇 초 뒤에 바꾼다. 볼넷 뒤 다음 타자 '2사 1루 1-2' 판독이, 그 타자의 안타로 카운트만 지운
    '0-0' 판독 때문에 튄 판독으로 버려져 볼넷과 다음 타자의 스윙이 한 전이('출루')로 합쳐졌다.
    같은 타석의 카운트 진행(0-2)은 예전처럼 버린다 — 살리면 뒤의 사구가 '2스트라이크 출루'가 된다
    (플레이오프 4차전)."""
    first, first_third = (True, False, False), (True, False, True)
    seen = [R(0, outs=2, balls=3), R(8, outs=2, balls=1, strikes=2, bases=first),
            R(16, outs=2, bases=first), R(17, outs=2, bases=first_third)]
    assert [r.t for r in stabilize(seen, min_repeat=1)] == [0.0, 8.0, 17.0]
    cues = [Cue(5, "speech", "볼넷입니다"), Cue(9, "swing"), Cue(9.4, "in_play")]
    assert [j.code for j in judge(seen, cues, min_repeat=1) if j.transition] == ["walk", "reach"]
    second, first_second = (False, True, False), (True, True, False)
    same_pa = [R(0, outs=1, bases=second), R(4, outs=1, strikes=2, bases=second),
               R(8, outs=1, bases=second), R(12, outs=1, bases=first_second)]
    assert [r.t for r in stabilize(same_pa, min_repeat=1)] == [0.0, 12.0]


def test_double_play_said_before_the_stale_board_across_the_half_change():
    """2025 플레이오프 1차전 4회말 — 1사 1·3루 병살타 해설 조각(660초)이 병살 뒤에도 그대로인 직전
    상태(667초)보다 앞서고, 새 상태는 이닝 사이 화면 뒤(681초)라 판정 시각 근처 창 밖이었다.
    반 이닝이 바뀐 전이는 직전 상태 때의 조각부터 본다. 반 이닝이 안 바뀐 전이는 그대로."""
    corners = (True, False, True)
    seen = [R(7, outs=1, bases=corners, half="bot"), R(21, inning=4)]
    said = [Cue(0, "speech", "당긴 타구 3루수 막아놓고 앞선 주자 그리고 1루 이어집니다 병살타")]
    assert [j.code for j in judge(seen, said, min_repeat=1)] == ["double_play"]
    assert [j.code for j in judge(seen, [], min_repeat=1)] == ["out"]
    # 직전 상태가 보이기 10초 넘게 전의 조각은 앞 플레이 말이다(독립 리뷰)
    late = [R(25, outs=1, bases=corners, half="bot"), R(30, outs=1, bases=corners, half="bot"),
            R(50, inning=4)]
    early = [Cue(12, "speech", "여기서 병살타 하나면 끝나요")]
    assert [j.code for j in judge(late, early, min_repeat=1)] == ["out"]


def test_a_hit_said_with_no_pitch_view_before_a_skipped_half():
    """2025 플레이오프 1차전 3회초 2사 2루 — 적시타를 다시보기로만 보여 주고(투구 화면 없음) 3회말을
    건너 4회초로 갔다. 2사에서 점수가 났으니 3아웃 전 플레이이고, 직전 상태 뒤 투구 화면이 없으니
    안타 해설은 앞 반 이닝 것이다. 공이 안 맞은 투구 화면이 있으면 그대로 '아웃'."""
    second = (False, True, False)
    seen = [R(0, outs=2, strikes=2, bases=second, away=4, home=5), R(27, inning=4, away=5, home=5)]
    said = [Cue(15, "speech", "김영웅의 적시타가 터졌습니다. 동점타입니다")]
    assert [j.code for j in judge(seen, said, min_repeat=1)] == ["reach"]
    no_hit = said + [Cue(3, "pitch_seen"), _view(4, "no_play")]
    assert [j.code for j in judge(seen, no_hit, min_repeat=1)] == ["out"]
    # 화면 단서가 없을 땐 새 상태가 보인 뒤의 말·폭투 득점 말은 안 쓴다(독립 리뷰)
    after = [Cue(30, "speech", "선두 타자 안타로 출루합니다")]
    assert [j.code for j in judge(seen, after, min_repeat=1)] == ["out"]
    wild = said + [Cue(10, "speech", "폭투로 3루 주자가 들어왔습니다")]
    assert [j.code for j in judge(seen, wild, min_repeat=1)] == ["out"]


def test_a_one_out_hit_before_two_skipped_outs():
    """2025 한국시리즈 3차전 2회말·준플레이오프 1차전 4회초 — 1사에서 적시타 뒤 아웃 둘(병살 등)을
    건너뛰고 다음 반 이닝으로 갔다. 첫 투구 화면이 맞아 나갔고 해설이 안타면 출루. 해설이 없거나
    희생플라이면, 첫 화면이 공이 안 맞은 것이면 예전대로 '아웃'."""
    on12 = (True, True, False)
    seen = twice(R(0, outs=1, strikes=1, bases=on12), R(30, half="bot", away=1))
    hit = [Cue(3, "pitch_seen", ""), _view(4, "in_play"),
           Cue(10, "speech", "좌익수 앞에 안타 타점")]
    assert [j.code for j in judge(seen, hit)] == ["reach"]
    assert [j.code for j in judge(seen, hit[:2])] == ["out"]
    sf = hit[:2] + [Cue(10, "speech", "희생플라이 안타 하나 없이 한 점")]
    assert [j.code for j in judge(seen, sf)] == ["out"]
    looked = [Cue(3, "pitch_seen", ""), _view(4, "no_play"), hit[2]]
    assert [j.code for j in judge(seen, looked)] != ["reach"]
    # 폭투·실책 득점 말이 있으면 안타 말이 있어도 아니다(독립 리뷰)
    wild = hit + [Cue(12, "speech", "공이 빠졌어요 폭투로 들어옵니다")]
    assert [j.code for j in judge(seen, wild)] == ["out"]


def test_asr_spelling_of_hit_by_pitch():
    """와일드카드 1차전 7회초 — 받아쓰기가 '몸에 맞는 공'을 '못 맞는 공'으로 적어 2-2 사구가
    '출루'였다."""
    two_three, loaded = (False, True, True), (True, True, True)
    seen = [R(0, outs=1, balls=2, strikes=2, bases=two_three), R(12, outs=1, bases=loaded)]
    said = [Cue(10, "speech", "후반 두 이닝에서 못 맞는 공이 나와요. 비어있던 1루 마저도")]
    assert [j.code for j in judge(seen, said, min_repeat=1) if j.transition is not None] == ["hbp"]
    assert [j.code for j in judge(seen, [], min_repeat=1) if j.transition is not None] == ["reach"]


def test_a_blank_read_during_the_half_change_does_not_drop_the_reads_before_it():
    """와일드카드 1차전 2회초 — 직선타 병살로 반 이닝이 끝나는 순간 전환 효과를 '0아웃 주자
    없음'으로 읽은 한 장 때문에 그 앞 '1사 1루' 판독 셋이 버려졌다(같은 반 이닝에서 아웃은 줄 수
    없다). 그 한 장은 건너 다음 반 이닝 판독과 견준다."""
    first = (True, False, False)
    one = [R(t, outs=1, balls=2, strikes=2, bases=first) for t in (10, 11, 12)]
    seen = [R(0)] + one + [R(20, balls=2, strikes=2), R(30, half="bot")]
    stable = stabilize(seen, min_repeat=1)
    assert [(r.t, r.outs, r.bases) for r in stable] == [(0.0, 0, E), (10.0, 1, first), (30.0, 0, E)]


def test_check_swing_called_a_swing_at_two_strikes_is_a_strikeout():
    """2025 준PO 4차전 2회초 — 3-2에서 체크스윙 비디오 판독 끝에 '스윙이군요. 스윙입니다'로 3아웃.
    해설에 '삼진'이 없어 '아웃'이었다. '노스윙입니다'(받아쓰기 '너스윙')는 반대 판정이라 아니다."""
    first = (True, False, False)
    seen = [R(0, outs=2, balls=3, strikes=2, bases=first), R(14, half="bot")]
    pitch = [Cue(3, "pitch_seen", ""), _view(6, "view_unknown")]
    called = pitch + [Cue(6, "speech", "스윙이군요. 스윙입니다.")]

    def codes(cues):
        return [j.code for j in judge(seen, cues, min_repeat=1) if j.transition is not None]

    assert codes(called) == ["strikeout"]
    assert codes(pitch + [Cue(6, "speech", "판단 결과 노스윙입니다.")]) == ["out"]
    # 말 첫머리의 판정만 — 스윙을 꾸미는 말·같은 조각의 반대 판정은 아니다(독립 리뷰)
    for said in ("풀스윙입니다", "좋은 스윙입니다 2루수 땅볼 아웃", "큰 스윙입니다. 파울입니다",
                 "스윙입니다. 판독 결과 노스윙입니다", "스윙 판정이 아닙니다",
                 "좋은 스윙입니다. 스윙 판정 후 파울 처리",
                 "공 쳤습니다. 스윙 판정은 없고 땅볼입니다"):
        assert codes(pitch + [Cue(6, "speech", said)]) == ["out"], said
    for said in ("체크스윙 비디오 판독 결과 스윙입니다", "- 배트가.- 스윙 판정입니다."):
        assert codes(pitch + [Cue(6, "speech", said)]) == ["strikeout"], said


def test_asr_spelling_of_steal():
    """2025 와일드카드 2차전 8회말 — 받아쓰기가 '기습 도루'를 '기습 토루'로 적어, 카운트가 그대로인
    2루 주자의 3루 진루가 '보크'였다."""
    second, third = (False, True, False), (False, False, True)
    seen = [R(0, outs=1, balls=1, bases=second), R(6, outs=1, balls=1, bases=third)]
    said = [Cue(3, "speech", "3루에 세이프! 김헌곤 선수의 기습 토루가 완벽하게 성공했고요.")]

    def codes(cues):
        return [j.code for j in judge(seen, cues, min_repeat=1) if j.transition is not None]

    assert codes(said) == ["steal"]
    assert codes([]) == ["balk"]


def test_strikeout_call_in_commentary():
    """2025 한국시리즈 1차전 5회초·6회말 — 해설이 '삼진' 없이 '스윙 스트라이크 아웃'·
    '루킹 스트라이크 아웃'(받아쓰기 '로킹 스트라이크아웃')으로만 말해 '아웃'이었다. 앞말 없는
    '스트라이크 아웃'은 땅볼 이야기일 수 있다('땅볼을 유도했지만 스트라이크 아웃을 잡은 거나
    진배없어요')."""
    seen = [R(0, outs=2, balls=1, strikes=2, bases=(False, False, True)), R(14, half="bot")]
    pitch = [Cue(3, "pitch_seen", ""), _view(6, "view_unknown")]

    def codes(cues):
        return [j.code for j in judge(seen, cues, min_repeat=1) if j.transition is not None]

    assert codes(pitch) == ["out"]
    # 받아쓰기의 '스윙 삼진' 다른 표기(2025 와일드카드 1차전·플레이오프 1차전·준PO 1차전)
    for said in ("높은 공에 스윙 스트라이크 아웃. 위기 상황.", "- 들어갔어요. 로킹 스트라이크아웃.",
                 "바깥쪽 변화골로 킹 스트라이크 아웃!", "바깥쪽 볼입니다. 스윙 선진! 경기 끝!",
                 "스프링 섬지! 스프링 섬지!", "가장 많이 승리를 거두는 투수입니다.- 6번째 탈삼."):
        assert codes(pitch + [Cue(6, "speech", said)]) == ["strikeout"], said
    for said in ("땅볼을 유도했지만 스트라이크 아웃을 잡은 거나 진배 없어요.",
                 "선진 야구를 보여 줍니다"):
        assert codes(pitch + [Cue(6, "speech", said)]) == ["out"], said


def test_asr_spellings_from_po225():
    """2025 플레이오프 2차전 — 받아쓰기가 '신삼진'을 '신성진!'·'신상진.'으로, '몸에 맞는 공'을
    '몸을 맞는 공'으로, '볼넷'을 '골렛'으로 적어 삼진·사구·볼넷이 '아웃'·'출루'였다."""
    seen_k = [R(0, outs=2, balls=1, strikes=2, bases=(False, False, True)), R(14, half="bot")]
    pitch = [Cue(3, "pitch_seen", ""), _view(6, "view_unknown")]

    def codes(seen, cues):
        return [j.code for j in judge(seen, cues, min_repeat=1) if j.transition is not None]

    for said in ("신성진! 자, 강민호 선수가 노련하네요.", "구구째. 신상진. 끈질기게 승부했던"):
        assert codes(seen_k, pitch + [Cue(6, "speech", said)]) == ["strikeout"], said
    two_three, loaded = (False, True, True), (True, True, True)
    seen_hbp = [R(0, outs=1, balls=1, strikes=2, bases=two_three), R(12, outs=1, bases=loaded)]
    assert codes(seen_hbp, [Cue(10, "speech", "-몸을 맞는 공.")]) == ["hbp"]
    second, on12 = (False, True, False), (True, True, False)
    seen_bb = [R(0, outs=1, balls=2, strikes=2, bases=second), R(12, outs=1, strikes=1, bases=on12)]
    assert codes(seen_bb, []) == ["reach"]
    for said in ("풀카운트 다시 빠지네요 또 빠져요 골렛", "선두타자에게 볼렛 내줍니다",
                 "1루까지 골넷으로 출루"):
        assert codes(seen_bb, [Cue(8, "speech", said)]) == ["walk"], said


def test_speech_that_ended_before_the_state_is_an_earlier_play():
    """준플레이오프 1차전 3회초 — 앞 타자 볼넷 말('골렛입니다', 480초 조각)이 491초에 보인 상태
    뒤의 사구 전이(493초)에서 '볼넷'이 됐다. 직전 상태가 보이기 전에 끝난 조각은 앞 플레이
    말이다."""
    on12, loaded = (True, True, False), (True, True, True)
    before, after = R(20, outs=2, balls=2, bases=on12), R(30, outs=2, bases=loaded)
    pitch = [Cue(26, "pitch_seen", ""), _view(27, "no_play")]
    # 판정 시각 근처 창(20초) 안이지만 직전 상태(20초)가 보이기 전에 끝난 조각(10~20초)
    j = one(before, after, [Cue(10, "speech", "3번 스트리트. 골렛입니다.")] + pitch)
    assert j is not None and j.code != "walk"
    # 직전 상태가 보인 뒤에 걸친 조각은 그대로 쓴다
    j = one(before, after, [Cue(15, "speech", "또 빠져요 골렛입니다.")] + pitch)
    assert j is not None and j.code == "walk"


def test_strikeout_speech_starting_when_the_state_is_seen():
    """2025 플레이오프 2차전 5회초 — 상태 500.0초 1사 만루 0-0(하이라이트가 카운트를 건너뜀),
    '신상진.' 조각도 500.0초에 시작, 휘두른 공이 안 맞고 아웃. 그 조각은 직전 상태 뒤의 말이다."""
    loaded = (True, True, True)
    seen = [R(500, outs=1, bases=loaded), R(511, outs=2, balls=2, strikes=2, bases=loaded)]
    cues = [Cue(500, "speech", "주자가 꽉 찹니다. 투투. 구구째. 신상진."),
            Cue(503.8, "swing", ""), Cue(503.8, "pitch_seen", ""), _view(505.7, "no_play")]
    codes = [j.code for j in judge(seen, cues, min_repeat=1) if j.transition is not None]
    assert codes == ["strikeout"]


def test_one_out_hit_when_the_board_updates_after_the_runners_score():
    """2025 플레이오프 2차전 3회초 1사 2·3루 — 295.4초 김영웅 2타점 적시타('내야를 관통합니다')
    뒤에도
    점수판은 298.5초까지 직전 상태였고, 아웃 둘을 건너뛰고 3회말. 직전 상태가 보인 동안의 마지막
    투구가 맞아 나갔고 그 뒤 다른 투구가 없으면 그 투구다. 그 투구가 5초보다 앞이면 아니다."""
    on23 = (False, True, True)
    seen = twice(R(290, outs=1, strikes=1, bases=on23, away=2, home=1),
                 R(298, outs=1, strikes=1, bases=on23, away=2, home=1),
                 R(327, half="bot", away=4, home=1))
    hit = [Cue(295.4, "pitch_seen", ""), _view(295.8, "in_play"),
           Cue(290, "speech", "이 타구는 빠르게 내야를 관통합니다. 3루 주자 들어옵니다")]
    assert [j.code for j in judge(seen, hit)] == ["reach"]
    early = [Cue(291.4, "pitch_seen", ""), _view(291.8, "in_play"), hit[2]]
    assert [j.code for j in judge(seen, early)] == ["out"]
    assert [j.code for j in judge(seen, hit[:2])] == ["out"]


def test_two_out_hit_said_long_before_a_skipped_half():
    """2025 플레이오프 2차전 4회초 2사 1·3루 — 디아즈 적시 2루타('안타요' 400초, '2루타' 430초),
    1루 주자 홈에서 아웃으로 3아웃, 4회말을 건너뛰고 5회초 상태가 457초에. 투구 화면 단서가 없으면
    직전 상태가 마지막으로 보인 뒤 30초 안의 말도 본다."""
    on13 = (True, False, True)
    seen = twice(R(395, outs=2, balls=1, strikes=1, bases=on13, away=4, home=1, inning=4),
                 R(457, balls=3, away=5, home=1, inning=5))
    said = [Cue(400, "speech", "초구. 옳죠. 안타요. 라인 안쪽에 들어왔어요."),
            Cue(430, "speech", "디아즈의 2루타. 한 점을 더 추가했는데 구자욱 선수가 잡혔어요")]
    assert [j.code for j in judge(seen, said)] == ["reach"]
    assert [j.code for j in judge(seen, [])] == ["out"]
    # 직전 상태 뒤 30초 밖이고 판정 시각 근처 창(20초)도 밖인 말 — 그대로
    late = [Cue(430, "speech", "자 디아즈의 2루타")]
    assert [j.code for j in judge(seen, late)] == ["out"]
