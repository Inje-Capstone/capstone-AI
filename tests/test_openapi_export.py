"""커밋된 OpenAPI 명세가 코드와 일치하는지.

백엔드 팀(Spring Boot)이 `docs/openapi.json`으로 DTO를 생성한다. 계약을 바꾸고 이 파일을
다시 내보내지 않으면 남의 코드가 조용히 낡은 계약을 따라간다 — 그래서 여기서 실패시킨다.
"""

import json
from pathlib import Path

from app.main import app

SPEC = Path(__file__).resolve().parents[1] / "docs" / "openapi.json"
REGENERATE = ".venv/bin/python scripts/export_openapi.py"


def test_committed_spec_matches_the_code():
    assert SPEC.exists(), f"docs/openapi.json이 없다 — `{REGENERATE}`를 돌려라"
    committed = json.loads(SPEC.read_text(encoding="utf-8"))
    current = json.loads(json.dumps(app.openapi()))
    assert committed == current, (
        f"커밋된 명세가 코드와 다르다 — `{REGENERATE}`를 다시 돌리고 커밋해라"
    )
