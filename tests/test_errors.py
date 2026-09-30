"""에러 계약 — 프론트가 다뤄야 할 응답 모양을 고정한다 (백엔드 팀 합의 2026-09-30).

`detail`이 문자열인 것(404·500)과 배열인 것(422 파라미터 검증)만 남기고,
평문 500이 나가지 않게 막는다.
"""

import pytest
from fastapi.testclient import TestClient

from app.adapters.relay.fixture import FixtureRelaySource
from app.api.deps import reset_services
from app.config import get_settings
from app.main import app

GAME_ID = "20260823LGOB"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ROOKIE_LLM_BACKEND", "mock")
    get_settings.cache_clear()
    reset_services()
    # 500 응답을 테스트에서 받아 보려면 예외를 다시 던지지 않게 해야 한다.
    yield TestClient(app, raise_server_exceptions=False)
    get_settings.cache_clear()
    reset_services()


def test_missing_resource_is_404_with_string_detail(client):
    for path in ("/api/games/nope/state", "/api/glossary/nope"):
        response = client.get(path)
        assert response.status_code == 404
        assert isinstance(response.json()["detail"], str)


def test_bad_parameter_is_422_with_list_detail(client):
    response = client.get(f"/api/games/{GAME_ID}/state", params={"t": -1})
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


def test_unexpected_error_is_json_500_not_plain_text(client, monkeypatch):
    """FastAPI 기본 500은 평문이라 프론트의 JSON 파서가 깨진다."""

    def boom(self):
        raise RuntimeError("의도한 폭발")

    monkeypatch.setattr(FixtureRelaySource, "list_games", boom)
    response = client.get("/api/games")
    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    assert isinstance(response.json()["detail"], str)
    assert "의도한 폭발" not in response.text  # 내부 정보를 응답에 싣지 않는다


def test_health_survives_a_broken_relay_source(client, monkeypatch):
    """헬스체크는 어떤 이유로도 500이 되면 안 된다 — EC2·ALB 헬스체크가 이걸 본다."""

    def boom(self):
        raise RuntimeError("의도한 폭발")

    monkeypatch.setattr(FixtureRelaySource, "list_games", boom)
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["relay_ok"] is False
