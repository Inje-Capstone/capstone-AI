"""OpenAPI 명세를 파일로 내보낸다 — 백엔드 팀이 DTO를 생성하는 원본.

    .venv/bin/python scripts/export_openapi.py

`docs/openapi.json`을 덮어쓴다. 계약이 바뀌면 이 명령을 다시 돌려야 하고, 안 돌리면
`tests/test_openapi_export.py`가 실패한다 — 커밋된 명세가 조용히 낡는 걸 막는다.
"""

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402

OUT = ROOT / "docs" / "openapi.json"


def dumps(spec: dict[str, Any]) -> str:
    """정렬해서 쓴다 — 필드 순서가 흔들려 diff가 지저분해지지 않게."""
    return json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main() -> None:
    spec = app.openapi()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(dumps(spec), encoding="utf-8")
    operations = sum(len(ops) for ops in spec["paths"].values())
    print(f"{OUT.relative_to(ROOT)} 갱신 — 경로 {len(spec['paths'])}개 · 오퍼레이션 {operations}개")


if __name__ == "__main__":
    main()
