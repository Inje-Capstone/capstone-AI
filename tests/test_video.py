"""영상 분석(VSS 방식) — 파서, 백엔드 요청 모양, 영상-중계 정합, 분석 스크립트.

실제 NVIDIA API·경기 영상 없이 돈다. 외부 응답 모양은 문서 기준 가정이라
(TODO 스키마 미검증) 파서가 흔들림(<think>, 코드펜스, "00:12" 시각)을 견디는지를 본다.
"""

import importlib.util
import io
import json
import random
import urllib.error
from pathlib import Path

import pytest

from app.adapters.video import http as video_http
from app.adapters.video.base import VideoAnalyzerError, VideoClip, VideoEvent, parse_events
from app.adapters.video.cosmos import CosmosAnalyzer
from app.adapters.video.vss import VssAnalyzer
from app.domain.models import RelayEvent
from app.domain.video_sync import Anchor, estimate_offset, relay_anchors

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "analyze_video", ROOT / "scripts" / "analyze_video.py"
)
analyze_video = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze_video)


def _clip(tmp_path, start=600.0, duration=60.0):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00\x00fake-mp4")
    return VideoClip(path, start, duration)


# ── 파서 ────────────────────────────────────────────────────────────────
def test_parse_strips_think_and_answer_and_shifts_to_video_time(tmp_path):
    text = (
        "<think>타자가 친다...</think>\n<answer>\n"
        '{"events": [{"start": 12, "end": 15, "type": "hit", "description": "우전 안타"},'
        ' {"start": "00:40", "end": "00:41", "type": "HOME_RUN"}]}\n</answer>'
    )
    events = parse_events(text, _clip(tmp_path), "p", "m")
    assert [(e.t_start, e.t_end, e.event_type) for e in events] == [
        (612.0, 615.0, "hit"), (640.0, 641.0, "home_run")]
    assert events[0].description == "우전 안타" and events[0].provider == "p"


def test_parse_code_fence_list_and_drops_bad_rows(tmp_path):
    text = """```json
[{"start_time": 5, "type": "bat_flip"},
 {"start": 999, "type": "hit"},
 {"type": "hit"},
 "noise",
 {"timestamp": "0:01:00", "event_type": "celebration"}]
```"""
    events = parse_events(text, _clip(tmp_path), "p", "m")
    assert [(e.t_start, e.event_type) for e in events] == [(605.0, "other"), (660.0, "celebration")]


def test_parse_without_json_raises(tmp_path):
    with pytest.raises(VideoAnalyzerError):
        parse_events("I cannot see the video.", _clip(tmp_path), "p", "m")


# ── 백엔드 요청 ─────────────────────────────────────────────────────────
class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _completion(content):
    return _FakeResponse(json.dumps({"choices": [{"message": {"content": content}}]}).encode())


def test_cosmos_sends_base64_video_and_parses(tmp_path, monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["auth"] = req.get_header("Authorization")
        seen["body"] = json.loads(req.data)
        return _completion('{"events": [{"start": 3, "type": "strikeout"}]}')

    monkeypatch.setattr(video_http.urllib.request, "urlopen", fake_urlopen)
    analyzer = CosmosAnalyzer("KEY", base_url="http://nim:8000/v1/", fps=1.0)
    events = analyzer.analyze(_clip(tmp_path))

    assert seen["url"] == "http://nim:8000/v1/chat/completions"
    assert seen["auth"] == "Bearer KEY"
    body = seen["body"]
    assert body["model"] == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    video, prompt = body["messages"][0]["content"]
    assert video["video_url"]["url"].startswith("data:video/mp4;base64,")
    assert "JSON" in prompt["text"]
    assert body["media_io_kwargs"] == {"video": {"fps": 1.0}}
    assert [(e.t_start, e.event_type) for e in events] == [(603.0, "strikeout")]


def test_vss_summarize_request_and_structured_output(tmp_path, monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data)
        content = json.dumps({"video_summary": "…", "events": [
            {"start_time": 20.5, "end_time": 24, "type": "pitching_change",
             "description": "교체"}]})
        return _completion(content)

    monkeypatch.setattr(video_http.urllib.request, "urlopen", fake_urlopen)
    analyzer = VssAnalyzer("http://vss:38111", url_for=lambda c: f"http://files/{c.path.name}")
    events = analyzer.analyze(_clip(tmp_path))

    assert seen["url"] == "http://vss:38111/v1/summarize"
    body = seen["body"]
    assert body["url"] == "http://files/clip.mp4"
    assert body["enable_vlm_structured_output"] is True
    assert "pitching_change" in body["events"] and "other" not in body["events"]
    assert "model" not in body  # 비우면 VSS 기본 VLM
    assert [(e.t_start, e.event_type) for e in events] == [(620.5, "pitching_change")]


def _http_error(code):
    return urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO(b"nope"))


def test_post_json_retries_5xx_but_not_429(monkeypatch):
    calls = []
    waits = []
    monkeypatch.setattr(video_http, "_sleep", waits.append)

    def flaky(req, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise _http_error(503)
        return _completion("ok")

    monkeypatch.setattr(video_http.urllib.request, "urlopen", flaky)
    assert video_http.post_json("http://x", {})["choices"]
    assert len(calls) == 2
    assert waits == [5.0]  # 수용량 초과(503)엔 기다렸다가 다시

    calls.clear()

    def limited(req, timeout):
        calls.append(1)
        raise _http_error(429)

    monkeypatch.setattr(video_http.urllib.request, "urlopen", limited)
    with pytest.raises(VideoAnalyzerError, match="429"):
        video_http.post_json("http://x", {})
    assert len(calls) == 1  # 쿼터를 두 배로 태우지 않는다


# ── 영상-중계 정합 ──────────────────────────────────────────────────────
def _relay_game():
    """중계 이벤트 모양만 흉내 낸 가상 경기 (첫 투구 기준 초)."""
    rng = random.Random(7)
    events, t = [], 0
    for i in range(60):
        t += rng.randint(60, 200)
        kind = rng.choice(["in_play", "strikeout", "walk"])
        if kind == "in_play":
            events.append(RelayEvent(id=f"p{i}", t=t, inning=1, half="top", kind="pitch",
                                     text="타격", detail={"result": "in_play"}))
        else:
            events.append(RelayEvent(id=f"r{i}", t=t, inning=1, half="top", kind="result",
                                     text=kind, detail={"result": kind}))
    events.append(RelayEvent(id="s1", t=t + 30, inning=1, half="top", kind="sub",
                             text="투수 교체", detail={"sub_type": "pitcher", "pitcher": "가"}))
    return events


def test_relay_anchors_map_kinds():
    kinds = {a.kind for a in relay_anchors(_relay_game())}
    assert kinds == {"hit", "strikeout", "walk", "pitching_change"}


def test_offset_recovered_despite_jitter_misses_and_false_positives():
    relay = _relay_game()
    rng = random.Random(1)
    true_offset = 1234.0
    video = []
    for a in relay_anchors(relay):
        if rng.random() < 0.35:  # VLM이 놓친 장면
            continue
        video.append(Anchor(a.t + true_offset + rng.uniform(-3, 3), a.kind))
    for _ in range(15):  # VLM 오탐 (광고·리플레이 화면)
        video.append(Anchor(rng.uniform(0, 12000), rng.choice(["hit", "strikeout"])))

    est = estimate_offset(relay, video)
    assert est is not None and est.confident
    assert abs(est.offset - true_offset) <= 2
    assert est.residual <= 3


def test_offset_not_confident_with_few_anchors():
    relay = _relay_game()
    first = relay_anchors(relay)[:2]
    est = estimate_offset(relay, [Anchor(a.t + 50, a.kind) for a in first])
    assert est is not None and not est.confident


def test_offset_none_without_matching_kinds():
    assert estimate_offset(_relay_game(), [Anchor(10, "crowd_cheer")]) is None


# ── 분석 스크립트 ───────────────────────────────────────────────────────
def test_plan_clips_covers_range_and_drops_tiny_tail():
    assert analyze_video.plan_clips(130.5, 60) == [(0.0, 60.0), (60.0, 60.0), (120.0, 10.5)]
    assert analyze_video.plan_clips(120.4, 60) == [(0.0, 60.0), (60.0, 60.0)]
    assert analyze_video.plan_clips(1000, 60, start=100, end=200) == [(100.0, 60.0), (160.0, 40.0)]


def test_ffmpeg_cut_cmd_is_low_res_and_silent():
    cmd = analyze_video.ffmpeg_cut_cmd("ffmpeg", Path("g.mp4"), 60, 30, Path("o.mp4"), 360)
    assert cmd[cmd.index("-ss") + 1] == "60.000" and cmd[cmd.index("-t") + 1] == "30.000"
    assert "scale=-2:360" in cmd and "-ac" in cmd and cmd[-1] == "o.mp4"  # 해설 음성 유지
    silent = analyze_video.ffmpeg_cut_cmd("ffmpeg", Path("g.mp4"), 0, 5, Path("o.mp4"), audio=False)
    assert "-an" in silent and "-ac" not in silent


class _FakeAnalyzer:
    provider = "fake"
    model = "fake-1"

    def __init__(self, fail_at=()):
        self.calls = []
        self.fail_at = set(fail_at)

    def analyze(self, clip):
        self.calls.append(clip.start)
        assert clip.path.read_bytes() == b"clip"
        if clip.start in self.fail_at:
            raise VideoAnalyzerError("boom")
        return [VideoEvent(t_start=clip.start + 5, t_end=clip.start + 6, event_type="hit",
                           provider=self.provider, model=self.model)]


def _cut(start, length, dest):
    dest.write_bytes(b"clip")


def test_run_accumulates_resumes_and_retries_failed_clips(tmp_path):
    out = tmp_path / "video" / "G.json"
    clips = [(0.0, 60.0), (60.0, 60.0), (120.0, 60.0)]

    first = _FakeAnalyzer(fail_at={60.0})
    snap = analyze_video.run("G", Path("g.mp4"), first, clips, _cut, out)
    assert first.calls == [0.0, 60.0, 120.0]
    assert snap["done"] == [0.0, 120.0] and snap["failed"] == [60.0]
    assert [e["t_start"] for e in snap["events"]] == [5.0, 125.0]

    second = _FakeAnalyzer()
    snap = analyze_video.run("G", Path("g.mp4"), second, clips, _cut, out)
    assert second.calls == [60.0]  # 끝난 청크는 다시 부르지 않는다
    assert snap["failed"] == []
    saved = json.loads(out.read_text("utf-8"))["events"]
    assert [e["t_start"] for e in saved] == [5.0, 65.0, 125.0]


def test_run_restarts_when_model_changes(tmp_path):
    out = tmp_path / "G.json"
    analyze_video.run("G", Path("g.mp4"), _FakeAnalyzer(), [(0.0, 60.0)], _cut, out)
    other = _FakeAnalyzer()
    other.model = "fake-2"
    snap = analyze_video.run("G", Path("g.mp4"), other, [(0.0, 60.0)], _cut, out)
    assert other.calls == [0.0] and snap["model"] == "fake-2"


def test_apply_offset_updates_only_that_game(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"game": {"id": "A", "relay_video_offset_sec": 0}}))
    (tmp_path / "b.json").write_text(json.dumps({"game": {"id": "B", "relay_video_offset_sec": 0}}))
    path = analyze_video.apply_offset(tmp_path, "B", 95.6)
    assert path.name == "b.json"
    game_b = json.loads((tmp_path / "b.json").read_text())["game"]
    assert game_b["relay_video_offset_sec"] == 96 and game_b["has_video"] is True
    assert json.loads((tmp_path / "a.json").read_text())["game"]["relay_video_offset_sec"] == 0
