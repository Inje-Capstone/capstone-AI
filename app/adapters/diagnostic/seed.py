"""온보딩 수준 진단 문제은행(data/diagnostic_seed.json) 읽기.

고정 문제은행이다 — LLM을 부르지 않는다(user-flow B①). 문항이 DB나 CMS로 옮겨가면
이 모듈만 교체한다. 용어 사전 시드 로더와 같은 모양으로 맞췄다.
"""

import json
from pathlib import Path
from typing import Any

DIAGNOSTIC_PATH = Path(__file__).resolve().parents[3] / "data" / "diagnostic_seed.json"


def load_diagnostic(path: Path = DIAGNOSTIC_PATH) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("questions", [])
