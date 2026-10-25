"""야구 규칙 전체 목록(data/rules_catalog.json)과 커버리지 보고서가 코드와 어긋나지 않는지."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_rules_catalog as catalog  # noqa: E402


def test_catalog_references_exist_in_code():
    assert catalog.validate(catalog.load_catalog()) == []


def test_committed_files_are_up_to_date():
    rows = catalog.load_catalog()
    committed = json.loads(catalog.CATALOG_PATH.read_text(encoding="utf-8"))["rules"]
    assert committed == rows, "scripts/build_rules_catalog.py를 다시 돌려라"
    assert catalog.REPORT_PATH.read_text(encoding="utf-8") == catalog.report(rows)


def test_every_rule_can_be_explained():
    """영상으로 못 보는 규칙도 질문이 오면 근거를 대고 설명할 수 있어야 한다."""
    rows = catalog.load_catalog()
    assert [r["id"] for r in rows if not r["glossary"]] == []
