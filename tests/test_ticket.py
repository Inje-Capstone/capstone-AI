"""티켓 인증 — 읽은 값 검사(날짜·구장·좌석·시험용 표시)와 백엔드 전용 API.

실제 Claude API는 부르지 않는다(가짜 판독기).
"""

import base64
from datetime import date

import pytest
from fastapi.testclient import TestClient

import app.api.routes_ticket as routes_ticket
from app.adapters.vision.ticket import SCHEMA, TicketReaderError, UnavailableTicketReader
from app.api.deps import reset_services
from app.config import get_settings
from app.domain.ticket import check_ticket, parse_date, seat_part, stadium_of, team_of
from app.main import app

TODAY = date(2026, 9, 1)  # GOOD 경기(8/23) 9일 뒤
GOOD = {
    "is_ticket": True, "game_date": "2026-08-23", "stadium": "잠실야구장",
    "home_team": "LG 트윈스", "away_team": "KIA 타이거즈", "seat_section": "3루 블루석",
    "seat_block": "314블록", "seat_row": "11열", "seat_number": "6번", "kind": "paper",
    "marked_sample": False,
}


def test_good_ticket_passes_with_normalized_seat_key():
    c = check_ticket(GOOD, TODAY)
    assert c.ok, c.checks
    assert (c.game_date, c.stadium, c.home_team, c.away_team) == ("2026-08-23", "잠실", "LG", "KIA")
    assert (c.block, c.row, c.number) == ("314", "11", "6")
    assert c.seat_key == "2026-08-23|잠실|314|11|6"
    # 같은 좌석을 다르게 적어도 같은 키(모바일 화면·종이 티켓 표기 차이로 돌려쓰기 방지)
    same = {**GOOD, "game_date": "2026.08.23(토)", "seat_block": "314블럭", "seat_row": "11",
            "seat_number": "006번석", "seat_section": "블루석 3루", "stadium": "서울 잠실 야구장",
            "home_team": "LG Twins"}
    assert check_ticket(same, TODAY).seat_key == c.seat_key


@pytest.mark.parametrize("change,failed", [
    ({"is_ticket": False}, "is_ticket"),
    ({"marked_sample": True}, "not_sample"),
    ({"game_date": "2026-12-01"}, "date"),  # 미래
    ({"game_date": ""}, "date"),
    ({"game_date": "2026-02-30"}, "date"),
    ({"stadium": "도쿄돔"}, "stadium"),
    ({"stadium": "고척스카이돔"}, "stadium"),  # 고척은 LG 홈이 아님
    ({"stadium": "고척스카이돔", "home_team": "LG트윈스"}, "stadium"),  # 띄어쓰기 없어도
    ({"away_team": "LG"}, "teams"),
    ({"home_team": "인천 유나이티드", "stadium": "인천문학경기장"}, "teams"),  # 모르는 팀
    ({"away_team": "대한민국"}, "teams"),
    ({"game_date": "2015-05-05"}, "date"),  # 오래된 티켓
    ({"game_date": "2026-08-01"}, "date"),  # 31일 전
    ({"seat_number": ""}, "seat"),
    ({"seat_row": ""}, "seat"),  # 열을 가린 사진 — 다른 키로 같은 좌석을 또 쓰지 못하게
    ({"seat_block": ""}, "seat"),  # 구역 이름만 있고 블록이 없음
])
def test_each_problem_fails_its_check_and_has_no_seat_key(change, failed):
    c = check_ticket({**GOOD, **change}, TODAY)
    assert not c.ok and c.seat_key is None
    assert [x.name for x in c.checks if not x.ok] == [failed]


def test_home_team_is_filled_from_single_home_stadium_and_letters_kept():
    c = check_ticket({**GOOD, "stadium": "대전 한화생명볼파크", "home_team": "",
                      "away_team": "삼성 라이온즈", "seat_block": "b-12"}, TODAY)
    assert c.ok and c.home_team == "한화" and c.block == "B12"
    shared = check_ticket({**GOOD, "home_team": ""}, TODAY)  # 잠실은 두 팀 홈 → 비워 둔다
    assert shared.ok and shared.home_team is None


def test_date_window_edges():
    assert check_ticket({**GOOD, "game_date": "2026-08-02"}, TODAY).ok  # 딱 30일 전
    assert check_ticket({**GOOD, "game_date": "2026-09-01"}, TODAY).ok  # 오늘


def test_seat_and_team_normalizers():
    assert seat_part("B-12") == seat_part("b12") == seat_part("B 12블록") == "B12"
    assert seat_part("006번석") == "6" and seat_part("0") == "0" and seat_part("열") is None
    assert team_of("KIA타이거즈") == "KIA" and team_of("기아") == "KIA"
    assert team_of("두산베어스") == "두산" and team_of("kt wiz") == "KT"
    assert team_of("LG 대 두산") is None and team_of("인천 유나이티드") is None


def test_garbage_from_reader_does_not_crash():
    c = check_ticket({"is_ticket": "yes", "game_date": 20260823, "stadium": None,
                      "seat_number": 6, "kind": "phone"}, TODAY)
    assert not c.ok and c.kind == "unknown" and c.number is None


def test_parse_date_and_stadium_helpers():
    assert parse_date("2026년 8월 23일 토요일") == date(2026, 8, 23)
    assert parse_date("08/23") is None and parse_date(None) is None
    assert stadium_of("인천SSG랜더스필드") == ("인천", ("SSG",))
    assert stadium_of("광주-기아 챔피언스필드")[0] == "광주"
    assert stadium_of("") == (None, ())


def test_schema_is_strict_and_has_no_union_types():
    """구조화 출력: 모든 필드 필수·추가 필드 금지·유니언 타입 금지(400 방지)."""
    assert set(SCHEMA["required"]) == set(SCHEMA["properties"])
    assert SCHEMA["additionalProperties"] is False
    assert all(isinstance(p["type"], str) for p in SCHEMA["properties"].values())


# ── API ──────────────────────────────────────────────────

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


class FakeReader:
    model = "fake"

    def __init__(self, out=None, error=None):
        self.out, self.error, self.calls = out, error, []

    def read(self, image, media_type):
        self.calls.append(media_type)
        if self.error:
            raise self.error
        return self.out


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ROOKIE_LLM_BACKEND", "mock")
    get_settings.cache_clear()
    reset_services()
    monkeypatch.setattr(routes_ticket, "today_kst", lambda: TODAY)
    yield TestClient(app)
    get_settings.cache_clear()
    reset_services()


def _post(client, image=PNG, **extra):
    body = {"imageBase64": base64.b64encode(image).decode(), **extra}
    return client.post("/api/backend/tickets/read", json=body)


def test_api_reads_and_checks(client, monkeypatch):
    fake = FakeReader(GOOD)
    monkeypatch.setattr(routes_ticket, "get_ticket_reader", lambda: fake)
    r = _post(client, mediaType="image/jpeg")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] and body["reasons"] == [] and body["model"] == "fake"
    assert body["seatKey"] == "2026-08-23|잠실|314|11|6"
    assert body["seat"] == {"section": "3루 블루석", "block": "314", "row": "11", "number": "6"}
    assert body["scheduleChecked"] is False
    assert fake.calls == ["image/png"]  # 선언이 틀려도 바이트로 판단

    fake.out = {**GOOD, "marked_sample": True}
    body = _post(client).json()
    assert not body["ok"] and body["seatKey"] is None and body["reasons"]


def test_api_rejects_bad_input_before_calling_the_model(client, monkeypatch):
    fake = FakeReader(GOOD)
    monkeypatch.setattr(routes_ticket, "get_ticket_reader", lambda: fake)
    assert client.post("/api/backend/tickets/read",
                       json={"imageBase64": "@@not base64@@@@"}).status_code == 422
    assert _post(client, image=b"GIF00a" + b"\x00" * 40).status_code == 415
    big = b"\xff\xd8\xff" + b"\x00" * (4 * 1024 * 1024)
    r = _post(client, image=big)
    assert r.status_code == 413 and len(r.content) < 1000  # 사진을 되돌려 보내지 않는다
    edge = b"\xff\xd8\xff" + b"\x00" * (routes_ticket.MAX_IMAGE_BYTES - 2)  # 디코드 뒤 한도 초과
    assert _post(client, image=edge).status_code == 413
    assert fake.calls == []


def test_api_reader_unavailable_or_failing_is_503(client, monkeypatch):
    monkeypatch.setattr(routes_ticket, "get_ticket_reader", lambda: UnavailableTicketReader())
    r = _post(client)
    assert r.status_code == 503 and "키" in r.json()["detail"]
    monkeypatch.setattr(routes_ticket, "get_ticket_reader",
                        lambda: FakeReader(error=TicketReaderError("잠시 뒤 다시")))
    assert _post(client).status_code == 503
    monkeypatch.setattr(routes_ticket, "get_ticket_reader",
                        lambda: FakeReader(error=TicketReaderError("사진을 읽을 수 없어요", 422)))
    assert _post(client).status_code == 422  # 깨진 사진은 다시 보내도 같다 → 4xx


def test_mock_backend_never_builds_a_real_reader(client):
    from app.api.deps import get_ticket_reader

    assert isinstance(get_ticket_reader(), UnavailableTicketReader)
