"""야구 규칙 전체 목록 → data/rules_catalog.json + RULES_COVERAGE.md (엔진 커버리지 보고서).

커버리지는 손으로 적지 않고 코드에서 센다:
- 설명: 용어 사전(data/glossary_seed.json)에 키가 있는가
- 카드: 감지 규칙(app/domain/detectors.RULES)에 있는가
- 영상: 영상 판정(app/domain/video_judge.py)이 내는 코드인가

사용: python scripts/build_rules_catalog.py   (원본: scripts/rules_catalog_src.py)
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rules_catalog_src import R  # noqa: E402

from app.adapters.glossary.seed import load_glossary  # noqa: E402
from app.domain.detectors import RULES  # noqa: E402

CATALOG_PATH = ROOT / "data" / "rules_catalog.json"
REPORT_PATH = ROOT / "RULES_COVERAGE.md"
FIELDS = ("id", "chapter", "rule", "name", "action", "ruling", "observe",
          "glossary", "detector", "video", "tier")
OBSERVE = {"S": "점수판", "V": "화면 동작", "A": "해설", "U": "심판·판독 화면", "X": "영상 불가"}
TIERS = {
    1: "지금 영상으로 결과를 알아본다",
    2: "심판 판정을 알아보면 된다 (단서·규칙 추가)",
    3: "AI가 동작을 직접 판단해야 한다 (영상 학습 대상)",
    4: "영상 판정 대상 아님 — 설명만",
}


def video_codes() -> set[str]:
    """video_judge가 실제로 내는 판정 코드 (make("kind", "code", …) + 교체)."""
    src = (ROOT / "app" / "domain" / "video_judge.py").read_text(encoding="utf-8")
    return set(re.findall(r'make\("\w+", "(\w+)"', src)) | {"sub"}


def load_catalog() -> list[dict]:
    rows = [dict(zip(FIELDS, r)) for r in R]
    ids = Counter(r["id"] for r in rows)
    dup = [k for k, n in ids.items() if n > 1]
    if dup:
        raise SystemExit(f"중복 id: {dup}")
    return rows


def validate(rows: list[dict]) -> list[str]:
    glossary = load_glossary()
    detectors = {r.id for r in RULES}
    codes = video_codes()
    errors = []
    for r in rows:
        if r["glossary"] and r["glossary"] not in glossary:
            errors.append(f"{r['id']}: 용어 사전에 없는 키 {r['glossary']}")
        if r["detector"] and r["detector"] not in detectors:
            errors.append(f"{r['id']}: 없는 감지 규칙 {r['detector']}")
        if r["video"] and r["video"] not in codes:
            errors.append(f"{r['id']}: 영상 판정에 없는 코드 {r['video']}")
        if r["tier"] not in TIERS or set(r["observe"]) - set(OBSERVE):
            errors.append(f"{r['id']}: 단계·보이는 방법 값 오류")
    return errors


def pct(n: int, total: int) -> str:
    return f"{n}/{total} ({n / total:.0%})" if total else "0/0"


def report(rows: list[dict]) -> str:
    total = len(rows)
    has_g = sum(1 for r in rows if r["glossary"])
    has_d = sum(1 for r in rows if r["detector"])
    has_v = sum(1 for r in rows if r["video"])
    tiers = Counter(r["tier"] for r in rows)
    filmable = [r for r in rows if r["tier"] != 4]

    out = [
        "# 엔진 커버리지 — 야구 규칙 전체",
        "",
        "> `python scripts/build_rules_catalog.py`가 만든다. "
        "직접 고치지 말고 `scripts/rules_catalog_src.py`를 고친다.",
        "> 숫자는 코드에서 센다(용어 사전·감지 규칙·영상 판정 코드). "
        "정확도가 아니라 '다룰 수 있느냐'다.",
        "",
        "## 요약",
        "",
        "| 항목 | 커버 |",
        "|---|---|",
        f"| 규칙·상황 전체 | {total}개 |",
        f"| 설명할 수 있다 (용어 사전 근거) | {pct(has_g, total)} |",
        f"| 문자 데이터로 카드를 띄운다 (감지 규칙) | {pct(has_d, total)} |",
        f"| 영상에서 결과를 알아본다 (영상 판정 코드) | {pct(has_v, total)} |",
        "",
        "| 단계 | 뜻 | 개수 |",
        "|---|---|---|",
    ]
    for t, label in TIERS.items():
        out.append(f"| {t} | {label} | {tiers.get(t, 0)} |")
    out += [
        "",
        f"영상으로 다룰 수 있는 것(1~3단계) {len(filmable)}개 중 지금 결과를 알아보는 것 "
        f"{sum(1 for r in filmable if r['video'])}개.",
        "",
        "보이는 방법: " + " · ".join(f"**{k}** {v}" for k, v in OBSERVE.items()),
        "",
    ]
    chapters: dict[str, list[dict]] = {}
    for r in rows:
        chapters.setdefault(r["chapter"], []).append(r)
    for ch, items in chapters.items():
        g = sum(1 for r in items if r["glossary"])
        v = sum(1 for r in items if r["video"])
        out += [f"## {ch} — 설명 {g}/{len(items)} · 영상 {v}/{len(items)}", "",
                "| 조항 | 상황 | 행동 → 판정 | 보임 | 설명 | 카드 | 영상 | 단계 |",
                "|---|---|---|---|---|---|---|---|"]
        for r in items:
            out.append(
                f"| {r['rule']} | {r['name']} | {r['action']} → {r['ruling']} | {r['observe']} | "
                f"{'✓' if r['glossary'] else '·'} | {'✓' if r['detector'] else '·'} | "
                f"{r['video'] or '·'} | {r['tier']} |"
            )
        out.append("")
    gaps = [r for r in rows if not r["glossary"]]
    out += ["## 설명 근거(용어 사전)가 없는 항목", ""]
    out += [f"- {r['name']} `{r['rule']}`" for r in gaps] or ["- 없음"]
    out.append("")
    return "\n".join(out)


def main() -> None:
    rows = load_catalog()
    errors = validate(rows)
    if errors:
        raise SystemExit("\n".join(errors))
    CATALOG_PATH.write_text(
        json.dumps({"_note": "build_rules_catalog.py가 생성. 원본 rules_catalog_src.py",
                    "rules": rows}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(report(rows), encoding="utf-8")
    print(f"{len(rows)}개 → {CATALOG_PATH.name}, {REPORT_PATH.name}")


if __name__ == "__main__":
    main()
