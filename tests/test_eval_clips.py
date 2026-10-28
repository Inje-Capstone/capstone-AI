"""하이라이트 클립 평가 — 정답지 형식과 채점 함수 (영상·API 없이)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_clips  # noqa: E402

from app.domain.video_judge import CALL_TABLE  # noqa: E402

CATALOG = {
    r["id"]
    for r in json.loads((ROOT / "data" / "rules_catalog.json").read_text(encoding="utf-8"))["rules"]
}


def _sb(t, outs, bases, away=0, home=0, balls=0, strikes=0):
    return {
        "t": t,
        "inning": 9,
        "half": "bot",
        "balls": balls,
        "strikes": strikes,
        "outs": outs,
        "bases": bases,
        "away": away,
        "home": home,
    }


def test_labels_point_to_catalog_and_known_codes():
    labels = json.loads(eval_clips.LABELS.read_text(encoding="utf-8"))["clips"]
    ids = [c["id"] for c in labels]
    assert len(ids) == len(set(ids))
    codes = {
        "balk",
        "steal",
        "caught_stealing",
        "wild_pitch",
        "passed_ball",
        "strikeout",
        "double_play",
        "triple_play",
        "homerun",
        "sac_fly",
        "infield_fly",
        "hbp",
        "dropped_third_strike",
        "walk",
    } | {code for code, _words, _kinds in CALL_TABLE}  # 심판 선언
    for c in labels:
        assert c["situation"] in CATALOG, c["id"]
        assert c["expect"] is None or c["expect"] in codes, c["id"]
        assert c["keywords"] and c["league"] in {"KBO", "MLB"}


def test_balk_clip_scores_judge_and_cue():
    snap = {
        "scoreboard": [
            _sb(1, 1, [3]),
            _sb(2, 1, [3]),
            _sb(9, 1, [], home=1),
            _sb(10, 1, [], home=1),
        ],
        "events": [{"t_start": 7, "event_type": "pitcher_stops", "description": "투수가 멈칫한다"}],
        "speech": [{"t": 8, "text": "보크입니다!"}],
    }
    label = {
        "id": "x",
        "league": "KBO",
        "situation": "balk_walkoff",
        "expect": "balk",
        "keywords": ["보크"],
    }
    row = eval_clips.score_clip(label, snap)
    assert row["judged"] is True and row["cue_hit"] and "balk" in row["codes"]


def test_clip_without_scoreboard_counts_cue_only():
    snap = {"scoreboard": [], "events": [], "speech": [{"t": 3, "text": "Ground-rule double"}]}
    label = {
        "id": "y",
        "league": "MLB",
        "situation": "ground_rule_2b",
        "expect": None,
        "keywords": ["ground-rule"],
    }
    row = eval_clips.score_clip(label, snap)
    assert row["judged"] is None and row["cue_hit"] and row["readings"] == 0
    report = eval_clips.render([row], {"model": "m", "thinking": "끔"})
    assert "단서 포착" in report and "0/0" in report


def test_whisper_transcript_replaces_vlm_speech_but_keeps_captions():
    snap = {
        "scoreboard": [],
        "events": [],
        "speech": [{"t": 1, "text": "You're gonna be a star."}, {"t": 2, "text": "[자막] 모먼트"}],
        "asr": {"segments": [{"t": 0, "text": "믿을 수 없는 트리플 플레이!"}]},
    }
    label = {
        "id": "z",
        "league": "KBO",
        "situation": "triple_play_r",
        "expect": "triple_play",
        "keywords": ["트리플"],
    }
    row = eval_clips.score_clip(label, snap)
    assert row["judged"] is True and row["codes"] == ["triple_play"]


def test_whisper_pieces_cover_clip():
    from app.adapters.video.speech_asr import plan_pieces

    assert plan_pieces(65.5, 30) == [(0.0, 30.0), (30.0, 30.0), (60.0, 5.5)]
    assert plan_pieces(0.5, 30) == []
    # 경기 영상은 분석한 구간만 — 시각은 영상 기준
    assert plan_pieces(40, 30, start=600) == [(600.0, 30.0), (630.0, 10.0)]


def test_pose_slide_is_a_scene_cue_but_not_commentary():
    """관절로 잰 슬라이딩은 장면 단서 — 타구 없이 슬라이딩 득점이면 홈스틸."""
    snap = {
        "scoreboard": [],
        "events": [{"t_start": 9.0, "event_type": "runner_scores", "description": ""}],
        "speech": [],
    }
    label = {"id": "h", "league": "KBO", "situation": "steal_home", "expect": "steal",
             "keywords": []}
    assert eval_clips.score_clip(label, snap)["judged"] is False
    snap["pose"] = {"slides": [{"t0": 7.5, "t1": 8.2, "distance": 1.8}],
                    "contacts": [{"t0": 8.2, "t1": 8.3, "kind": "slide_tag", "why": "x"}]}
    row = eval_clips.score_clip(label, snap)
    assert row["judged"] is True and row["codes"] == ["steal"]


def test_ablation_splits_sound_from_screen():
    """해설만 = 소리(받아쓰기), 화면만 = 점수판·장면·화면 자막 — 같은 판정 규칙에 입력만 바꾼다."""
    snap = {
        "scoreboard": [_sb(1, 0, [1])],
        "events": [{"t_start": 2.0, "event_type": "slide", "description": ""}],
        "speech": [{"t": 3, "text": "[자막] 도루 성공"}, {"t": 4, "text": "VLM 받아쓰기"}],
        "asr": {"segments": [{"t": 0, "text": "도루!"}]},
        "scorebug": {"scoreboard": [_sb(1, 0, [1])]},
        "pose": {"slides": []},
    }
    talk = eval_clips.commentary_only(snap)
    assert talk["scoreboard"] == [] and talk["events"] == [] and "scorebug" not in talk
    assert [s["text"] for s in talk["speech"]] == ["VLM 받아쓰기"]
    assert talk["asr"] and "pose" not in talk
    screen = eval_clips.screen_only(snap)
    assert "asr" not in screen and [s["text"] for s in screen["speech"]] == ["[자막] 도루 성공"]
    assert screen["events"] and screen["scorebug"]


def test_ablation_report_counts_what_the_screen_added():
    label = {"id": "a", "league": "KBO", "situation": "s", "expect": "triple_play", "keywords": []}
    snap = {"scoreboard": [], "events": [], "speech": [],
            "asr": {"segments": [{"t": 0, "text": "트리플 플레이!"}]}}
    report = eval_clips.render_ablation([(label, snap)])
    assert "| 화면 + 해설 | 1/1 (100%) | 0/1 |" in report
    assert "| 해설만 | 1/1 (100%) | 0/1 |" in report and "| 화면만 | 0/1 (0%) | 0/1 |" in report
    assert "화면이 더한 클립 0개" in report
