"""분석 데이터 표 — 엔진 결과를 정답과 잇고 요약하는 순수 부분."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("cv2")
spec = importlib.util.spec_from_file_location(
    "analysis_data", ROOT / "scripts" / "analysis_data.py"
)
analysis_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis_data)


def test_what_the_engine_said():
    assert analysis_data.said(None) == "못 찾음"
    assert analysis_data.said({"swing": True}) == "스윙"
    assert analysis_data.said({"swing": False, "clear": True}) == "안 침"
    assert analysis_data.said({"swing": False, "clear": False}) == "모름"


def test_labeled_shot_is_matched_by_overlap_or_by_start(tmp_path, monkeypatch):
    truth = tmp_path / "truth.json"
    truth.write_text(json.dumps({"shots": [
        {"video": "v", "folder": "f", "t0": 10.0, "t1": 14.0, "label": "S", "set": "tune"},
        {"video": "v", "folder": "f", "t0": 30.0, "t1": None, "label": "T", "set": "test"},
        {"video": "v", "folder": "f", "t0": 50.0, "t1": 52.0, "label": "T", "set": "tune"},
    ]}), encoding="utf-8")
    engine = [{"t0": 10.5, "t1": 14.2, "t": 12.0, "moved": 0.5, "swing": True, "clear": True},
              {"t0": 30.1, "t1": 33.0, "t": 31.0, "moved": 0.1, "swing": False, "clear": True}]
    monkeypatch.setattr(analysis_data, "SWING_TRUTH", truth)
    monkeypatch.setattr(analysis_data, "load_swings", lambda _video: engine)
    rows = analysis_data.swing_rows(tmp_path)
    assert [r["said"] for r in rows] == ["스윙", "안 침", "못 찾음"]
    assert rows[0]["moved"] == 0.5


def test_summary_counts_test_and_tune_apart():
    rows = [{"set": "test", "label": "S", "said": "스윙"},
            {"set": "test", "label": "T", "said": "모름"},
            {"set": "tune", "label": "S", "said": "안 침"},
            {"set": "tune", "label": "?", "said": "스윙"}]
    text = analysis_data.summary(rows, [], [], {})
    assert "**test** (2장면)" in text and "**tune** (1장면)" in text
    assert "| 스윙 | 1 | 0 | 0 | 0 |" in text
