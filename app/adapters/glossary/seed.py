"""용어 사전 시드(data/glossary_seed.json) 읽기.

용어 사전 화면(S5)과 mock LLM이 같은 원본을 쓴다. 본문 콘텐츠가 DB나 CMS로 옮겨가면
이 모듈만 교체한다.
"""

import json
from pathlib import Path
from typing import Any

GLOSSARY_PATH = Path(__file__).resolve().parents[3] / "data" / "glossary_seed.json"


def load_glossary(path: Path = GLOSSARY_PATH) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("terms", {})
