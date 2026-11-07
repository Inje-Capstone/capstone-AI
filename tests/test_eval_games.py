"""경기 하이라이트 채점 — 판정을 정답 칸에 나눠 담고 맞음·틀림·놓침을 가르는 순수 부분."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_games", ROOT / "scripts" / "eval_games.py")
eval_games = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_games)

PLAYS = [
    {"t": 10.0, "pitch_end": 13.0, "end": 30.0, "code": "reach", "sure": True},
    {"t": 30.0, "pitch_end": 33.0, "end": 50.0, "code": "none", "sure": True},
    {"t": 50.0, "pitch_end": 52.0, "end": 60.0, "code": "out_any", "sure": False},
]


def test_a_change_seen_at_the_next_pitch_belongs_to_the_play_before_it():
    """하이라이트는 다시보기 뒤 다음 투구 화면에서야 새 점수판이 보인다."""
    judged = [{"t": 5.0, "code": "out"},  # 첫 투구 전 — 버림
              {"t": 30.5, "code": "reach"},  # 다음 투구 화면 시작 = 앞 칸
              {"t": 34.0, "code": "walk"},  # 두 번째 투구가 끝난 뒤 = 두 번째 칸
              {"t": 61.0, "code": "strikeout"}]
    cells = eval_games.assign(PLAYS, judged)
    assert [[j["code"] for j in c] for c in cells] == [["reach"], ["walk"], ["strikeout"]]


def test_verdicts():
    assert eval_games.verdict("reach", [{"code": "reach"}]) == "맞음"
    assert eval_games.verdict("reach", [{"code": "out"}]) == "틀림"
    assert eval_games.verdict("reach", []) == "놓침"
    assert eval_games.verdict("out_any", [{"code": "strikeout"}]) == "맞음"
    assert eval_games.verdict("out_any", [{"code": "out"}, {"code": "walk"}]) == "맞음"
    assert eval_games.verdict("none", [{"code": "walk"}]) == "헛판정"
    assert eval_games.verdict("none", []) == "조용함"
    assert eval_games.verdict("?", [{"code": "walk"}]) is None


def test_shipped_labels_use_known_codes_and_increasing_times():
    games = json.loads(eval_games.LABELS.read_text(encoding="utf-8"))["games"]
    known = set(eval_games.ACCEPT) | eval_games.SKIP | {eval_games.NONE}
    for game in games.values():
        plays = game["plays"]
        assert all(p["code"] in known for p in plays)
        assert all(a["t"] < b["t"] for a, b in zip(plays, plays[1:]))
        assert all(p["t"] <= p["pitch_end"] <= p["end"] for p in plays)


def test_a_judgment_in_a_replay_cell_belongs_to_the_play_before_it():
    plays = [{"t": 10.0, "pitch_end": 13.0, "end": 30.0, "code": "homerun", "sure": True},
             {"t": 30.0, "pitch_end": 33.0, "end": 60.0, "code": "none", "sure": False},
             {"t": 60.0, "pitch_end": 62.0, "end": 70.0, "code": "out", "sure": True}]
    cells = eval_games.assign(plays, [{"t": 50.0, "code": "homerun"}])
    assert [[j["code"] for j in c] for c in cells] == [["homerun"], [], []]


def test_only_play_cells_take_judgments_from_idle_cells():
    plays = [{"t": 0.0, "pitch_end": 2.0, "end": 10.0, "code": "none", "sure": True},
             {"t": 10.0, "pitch_end": 12.0, "end": 20.0, "code": "?", "sure": False}]
    cells = eval_games.assign(plays, [{"t": 15.0, "code": "out"}])
    assert [[j["code"] for j in c] for c in cells] == [[], ["out"]]
