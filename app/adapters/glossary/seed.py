"""용어 사전 시드(data/glossary_seed.json)와 규칙 전체 목록(data/rules_catalog.json) 읽기.

용어 사전 화면(S5)과 mock LLM이 같은 시드를 쓴다. 규칙 목록은 S5의 규칙 탭이 따로 쓴다.
본문 콘텐츠가 DB나 CMS로 옮겨가면 이 모듈만 교체한다.
"""

import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
GLOSSARY_PATH = DATA_DIR / "glossary_seed.json"
CATALOG_PATH = DATA_DIR / "rules_catalog.json"


def load_glossary(path: Path = GLOSSARY_PATH) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("terms", {})


def load_rules(path: Path = CATALOG_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")).get("rules", [])
