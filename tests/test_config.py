"""환경 설정 — 배포 환경변수가 제대로 읽히는지."""

from app.config import Settings


def test_cors_origins_default_is_dev_servers(monkeypatch):
    monkeypatch.delenv("ROOKIE_CORS_ORIGINS", raising=False)
    assert Settings().cors_origins == ["http://localhost:5173", "http://localhost:3000"]


def test_cors_origins_from_env_trims_and_drops_blanks(monkeypatch):
    monkeypatch.setenv("ROOKIE_CORS_ORIGINS", " https://rookie.example.com , ,http://a.test ")
    assert Settings().cors_origins == ["https://rookie.example.com", "http://a.test"]
