"""방금 장면 한 줄 요약 — 타임코드의 결정적 함수이고, 모델이 죽어도 한 줄은 뜬다."""

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.adapters.llm.base import LLMError, LLMResult
from app.adapters.relay.fixture import FixtureRelaySource
from app.api.deps import reset_services
from app.config import get_settings
from app.main import app
from app.services.moment_service import MomentService, clean_line

GAME_ID = "20260823LGOB"
DEMO = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "game_20260823_LG_OB.json"
BALK_T = 7550


@pytest.fixture
def relay(tmp_path):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    shutil.copy(DEMO, fixtures / DEMO.name)
    return FixtureRelaySource(fixtures)


class _FakeLLM:
    source = "llm"
    model = "fake-haiku"

    def __init__(self, reply="보크로 주자가 2루까지 공짜로 갔어요.", fail=False):
        self.reply, self.fail, self.calls = reply, fail, 0

    def complete(self, **kw):
        self.calls += 1
        if self.fail:
            raise LLMError("down")
        return LLMResult(text=self.reply, model=self.model, source="llm")


class _Mock:
    source = "mock"
    model = "mock"

    def complete(self, **kw):  # pragma: no cover - 호출되면 안 된다
        raise AssertionError("mock에선 LLM을 부르지 않고 조립 문장을 쓴다")


# ── 조립 문장 ───────────────────────────────────────────────────────────
def test_before_first_play(relay):
    m = MomentService(relay, _Mock()).moment(GAME_ID, 30)
    assert m.event_id is None and m.source == "template"


def test_template_describes_change(relay):
    svc = MomentService(relay, _Mock())
    dp = svc.moment(GAME_ID, 300)
    assert dp.event_id == "e006" and "병살타" in dp.text and "공수 교대" in dp.text
    balk = svc.moment(GAME_ID, BALK_T)
    assert balk.event_id == "e117" and balk.text.startswith("7회말")


def test_scoring_play_mentions_score(relay):
    svc = MomentService(relay, _Mock())
    scoring = [m for m in svc.all(GAME_ID) if "점 (" in m.text]
    assert scoring
    assert all("두산" in m.text and "LG" in m.text for m in scoring)


def test_moment_is_deterministic_in_t(relay):
    svc = MomentService(relay, _Mock())
    ts = [m.t for m in svc.all(GAME_ID)]
    assert ts == sorted(ts)
    for m in svc.all(GAME_ID)[:10]:
        assert svc.moment(GAME_ID, m.t).event_id == m.event_id  # 그 순간에 딱 그 장면


# ── 생성·폴백·스냅샷 ────────────────────────────────────────────────────
def test_llm_line_is_cleaned_cached_and_snapshotted(relay, tmp_path):
    llm = _FakeLLM(reply='  "보크로 주자가\n2루까지 갔어요."  ')
    svc = MomentService(relay, llm, snapshot_dir=tmp_path / "snap")
    first = svc.moment(GAME_ID, BALK_T, level=0)
    again = svc.moment(GAME_ID, BALK_T, level=0)
    assert first.source == "llm" and first.text == "보크로 주자가 2루까지 갔어요."
    assert again == first and llm.calls == 1
    assert (tmp_path / "snap" / f"{GAME_ID}.moments.json").exists()

    reborn = MomentService(relay, _FakeLLM(fail=True), snapshot_dir=tmp_path / "snap")
    cached = reborn.moment(GAME_ID, BALK_T, level=0)
    assert cached.source == "snapshot" and cached.text == first.text


def test_llm_failure_falls_back_to_template_and_is_not_cached(relay):
    llm = _FakeLLM(fail=True)
    svc = MomentService(relay, llm)
    assert svc.moment(GAME_ID, BALK_T).source == "template"
    svc.moment(GAME_ID, BALK_T)
    assert llm.calls == 2  # 조립 문장은 캐시하지 않아 모델이 살아나면 다시 시도한다


def test_levels_are_cached_separately(relay):
    llm = _FakeLLM()
    svc = MomentService(relay, llm)
    svc.moment(GAME_ID, BALK_T, level=0)
    svc.moment(GAME_ID, BALK_T, level=2)
    assert llm.calls == 2


def test_snapshot_ignored_without_real_model(relay, tmp_path):
    MomentService(relay, _FakeLLM(), snapshot_dir=tmp_path).moment(GAME_ID, BALK_T)
    m = MomentService(relay, _Mock(), snapshot_dir=tmp_path).moment(GAME_ID, BALK_T)
    assert m.source == "template"


def test_clean_line_caps_length():
    assert len(clean_line("가" * 200)) == 60
    assert clean_line("“인용부호”") == "인용부호"


# ── API ─────────────────────────────────────────────────────────────────
def test_moment_endpoint_uses_summary_model_setting(monkeypatch, tmp_path):
    monkeypatch.setenv("ROOKIE_LLM_BACKEND", "mock")
    monkeypatch.setenv("ROOKIE_SNAPSHOT_DIR", str(tmp_path / "snapshots"))
    monkeypatch.setenv("ROOKIE_SUMMARY_MODEL", "claude-haiku-4-5")
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    shutil.copy(DEMO, fixtures / DEMO.name)
    monkeypatch.setenv("ROOKIE_FIXTURE_DIR", str(fixtures))
    get_settings.cache_clear()
    reset_services()
    try:
        client = TestClient(app)
        body = client.get(f"/api/games/{GAME_ID}/moment", params={"t": BALK_T}).json()
        assert body["event_id"] == "e117" and body["source"] == "template"
        assert body["scoreboard"].startswith("두산 3 : 4 LG")
        assert get_settings().summary_model == "claude-haiku-4-5"
        assert client.get("/api/games/nope/moment").status_code == 404
    finally:
        get_settings.cache_clear()
        reset_services()
