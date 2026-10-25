"""하이라이트 클립 평가 — 정답지 형식과 채점 함수 (영상·API 없이)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_clips  # noqa: E402

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
    }
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
