"""관절 접촉 점검 스크립트 — 정답 표 비교(영상 없이)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import contact_check  # noqa: E402


def test_score_matches_each_truth_once_within_tolerance():
    hit, false, miss = contact_check.score([5.0, 20.0, 40.0], [(4.5, 5.2), (5.0, 5.5), (30, 31)])
    assert (hit, false, miss) == (1, 2, 2)


def test_detected_groups_by_kind():
    summary = {
        "slides": [{"t0": 1.0, "t1": 1.5, "distance": 2.0}],
        "contacts": [{"t0": 1.5, "t1": 1.6, "kind": "slide_tag", "why": ""}],
    }
    found = contact_check.detected(summary)
    assert found == {"slide": [(1.0, 1.5)], "slide_tag": [(1.5, 1.6)], "collision": []}


def test_score_matching_is_optimal_for_long_spans():
    """긴 구간이 짧은 구간의 정답을 먼저 가져가면 안 된다."""
    assert contact_check.score([4.5, 7.5], [(3.0, 7.0), (4.0, 4.0)]) == (2, 0, 0)


def test_unsure_windows_are_not_counted_as_false():
    assert contact_check.score([], [(10.0, 10.5)], unsure=[[9.8, 10.2]]) == (0, 0, 0)
    assert contact_check.score([], [(10.0, 10.5)], unsure=[[20, 21]]) == (0, 1, 0)


def test_cache_names_split_by_span():
    video = Path("g.mp4")
    assert contact_check.caches(video) == (Path("g.people.json"), Path("g.cam.json"))
    assert contact_check.caches(video, 600.0, 1200.0)[0] == Path("g.600-1200.people.json")


def test_long_video_is_summarized_window_by_window(monkeypatch):
    seen = []

    def fake(video, args, start, end):
        seen.append((start, end))
        return {"slides": [{"t0": start + 1, "t1": start + 2, "distance": 1.0}], "contacts": []}

    monkeypatch.setattr(contact_check, "summarize", fake)
    out = contact_check.summarize_long(Path("g.mp4"), None, 0.0, 1300.0, window=600.0)
    assert seen == [(0.0, 600.0), (600.0, 1200.0), (1200.0, 1300.0)]
    assert [s["t0"] for s in out["slides"]] == [1.0, 601.0, 1201.0]


def test_truth_can_be_a_scene_span():
    """다이빙(5.4초) 뒤 누워 있다 일어나는(9.0초) 한 장면 — 그 안 어디서 잡아도 같은 정답."""
    assert contact_check.score([[5.4, 9.0]], [(7.6, 9.0)]) == (1, 0, 0)
    assert contact_check.score([[5.4, 9.0]], [(7.6, 8.0), (8.5, 9.0)]) == (1, 1, 0)
