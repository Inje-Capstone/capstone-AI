"""야구 규칙 전체 목록 → data/rules_catalog.json + RULES_COVERAGE.md (엔진 커버리지 보고서).

커버리지는 손으로 적지 않고 코드에서 센다:
- 설명: 용어 사전(data/glossary_seed.json)에 키가 있는가
- 카드: 감지 규칙(app/domain/detectors.RULES)에 있는가
- 영상: 영상 판정(app/domain/video_judge.py)이 내는 코드인가
- 행동 로직: 영상으로 다룰 규칙(단계 1~3)마다 성립 조건·관측 신호·판정(scripts/action_logic_src.py),
  신호가 실제 구현됐는지(신호 어휘의 구현 위치·해설 키워드)
- 조항: 공식 야구규칙(data/obr_provisions.json, 번호만)의 모든 조항이 어느 행엔가 걸리는가

사용: python scripts/build_rules_catalog.py   (원본: scripts/rules_catalog_src.py)
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from action_logic_src import ACTION, SIGNALS  # noqa: E402
from rules_catalog_src import COVERS, JUDGE, MOTION, R  # noqa: E402

from app.adapters.glossary.seed import load_glossary  # noqa: E402
from app.domain.balk_judge import BALK_WORDS  # noqa: E402
from app.domain.detectors import RULES  # noqa: E402
from app.domain.pitch_motion import PITCH_ONLY_RULES  # noqa: E402
from app.domain.video_judge import SPEECH_KEYWORDS  # noqa: E402

CATALOG_PATH = ROOT / "data" / "rules_catalog.json"
REPORT_PATH = ROOT / "RULES_COVERAGE.md"
PROVISIONS_PATH = ROOT / "data" / "obr_provisions.json"
REF = re.compile(r"\d\.\d\d(?:\([a-zA-Z0-9]+\))*")
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
    found = set(re.findall(r'make\("\w+", "(\w+)"', src))
    found |= set(re.findall(r'\("(\w+)", "(?:call|result|steal)", \(', src))  # 단서만 판정표
    found |= set(re.findall(r'make\("(\w+)", "(?:call|result|steal)"', src))  # 장면 순서 판정
    return (found - {"call", "result"}) | {"sub", "pitcher", "out", "reach"}


def provisions() -> list[str]:
    """공식 야구규칙 조항 번호 (부모가 자식보다 먼저 온다)."""
    return json.loads(PROVISIONS_PATH.read_text(encoding="utf-8"))["provisions"]


def refs(row: dict) -> list[str]:
    """행이 덮는 조항 — 조항 칸의 번호 + COVERS(묶음). '*'로 끝나면 그 번호로 시작하는 전부."""
    return REF.findall(row["rule"]) + list(COVERS.get(row["id"], ()))


def parent(pid: str) -> str | None:
    return pid.rsplit("(", 1)[0] if "(" in pid else None


def coverage(rows: list[dict]) -> list[str]:
    """어느 행에도 안 걸린 조항. 하위 조항이 모두 걸리면 제목 조항도 걸린 것으로 본다."""
    ids = provisions()
    exact = {x for r in rows for x in refs(r) if not x.endswith("*")}
    prefixes = tuple(x[:-1] for r in rows for x in refs(r) if x.endswith("*"))
    covered = {p for p in ids if p in exact or p.startswith(prefixes)}
    kids: dict[str, list[str]] = {}
    for p in ids:
        kids.setdefault(parent(p) or "", []).append(p)
    for p in reversed(ids):
        if kids.get(p) and all(k in covered for k in kids[p]):
            covered.add(p)
    return [p for p in ids if p not in covered]


def signal_known(tok: str) -> bool:
    return all(a in SIGNALS or a.startswith("kw:") for a in tok.split("|"))


def signal_built(tok: str) -> bool:
    """신호가 지금 코드로 관측되나 — 대체 신호(a|b)는 하나만 돼도 된다."""
    words = {k.replace(" ", "") for k in SPEECH_KEYWORDS}
    for a in tok.split("|"):
        if a.startswith("kw:"):
            w = a[3:].replace(" ", "")
            if w in words or BALK_WORDS.fullmatch(w):
                return True
        elif SIGNALS.get(a, ("", "", ""))[2]:
            return True
    return False


# 동작을 직접 보는 신호 — 나머지는 판정을 알아보는 신호
JUDGE_PREFIX = ("pose.", "ball.", "obj.", "time.")


def split_signals(rid: str) -> tuple[list[str], list[str]]:
    """(인식 신호, 판단 신호). 인식 = 점수판·장면·해설·심판 동작으로 '심판 판정을 알아본다',
    판단 = 자세·공·물체·시간으로 '동작을 직접 판단한다'."""
    toks = ACTION[rid][1].split()
    judge = [s for s in toks if all(a.startswith(JUDGE_PREFIX) for a in s.split("|"))]
    return [s for s in toks if s not in judge], judge


def action_status(rid: str) -> tuple[str, str]:
    """(인식, 판단) 상태. 인식 가능 = 계기 신호(장면·해설·심판·누적 상태)가 하나라도 구현
    (계기 없이 점수판만 쓰는 규칙은 점수판이 다 구현되면). 판단 가능 = 판단 신호가 모두 구현."""
    recog, judge = split_signals(rid)
    triggers = [s for s in recog if not s.startswith("sb.")]
    if triggers:
        r = "가능" if any(signal_built(s) for s in triggers) else "불가"
    else:
        r = "가능" if recog and all(signal_built(s) for s in recog) else "—"
    j = "—" if not judge else ("가능" if all(signal_built(s) for s in judge) else "불가")
    return r, j


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
    known = set(provisions())
    ids = {r["id"] for r in rows}
    for r in rows:
        for x in refs(r):
            if not x.endswith("*") and x not in known:
                errors.append(f"{r['id']}: 공식 규칙에 없는 조항 번호 {x}")
    for r in rows:
        if r["tier"] != 4 and r["id"] not in ACTION:
            errors.append(f"{r['id']}: 행동 판정 로직(action_logic_src.ACTION)이 없다")
    for rid, (_pre, sigs, _how) in ACTION.items():
        if rid not in ids:
            errors.append(f"행동 로직: 카탈로그에 없는 id {rid}")
        for tok in sigs.split():
            if not signal_known(tok):
                errors.append(f"행동 로직 {rid}: 신호 어휘에 없는 {tok}")
    for rid in COVERS:
        if rid not in ids:
            errors.append(f"묶음 커버: 카탈로그에 없는 id {rid}")
    for code in sorted(codes - set(JUDGE) - {"sub"}):
        errors.append(f"판정 로직표에 없는 영상 판정 코드: {code}")
    for r in rows:
        if r["video"] and r["video"] != "sub" and r["video"] not in JUDGE:
            errors.append(f"{r['id']}: 판정 로직표에 없는 코드 {r['video']}")
    for rid, (how, _why) in MOTION.items():
        if rid not in ids:
            errors.append(f"동작 판정표: 카탈로그에 없는 id {rid}")
        if how not in ("자세", "해설", "점수판", "불가"):
            errors.append(f"동작 판정표: {rid} 방법 값 오류 {how}")
    code_rules = set(PITCH_ONLY_RULES.values())
    table_pose = {rid for rid, (how, _w) in MOTION.items() if how == "자세"}
    for rid in code_rules - table_pose:
        errors.append(f"코드(pitch_motion)가 재는데 표엔 '자세'가 아님: {rid}")
    for rid in table_pose - code_rules - {"set_position", "hop_step"}:
        errors.append(f"표엔 '자세'인데 코드(pitch_motion)가 안 잼: {rid}")
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
        f"| 공식 야구규칙 조항 (OBR {len(provisions())}개) | "
        f"{pct(len(provisions()) - len(coverage(rows)), len(provisions()))} |",
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
    out += ["## 영상 판정 로직 — 저장 규칙 대조", "",
            "`app/domain/video_judge.py`가 내는 코드마다 근거 조항·점수판 모양·단서·성립 조건. "
            "성립 조건이 안 맞으면 그 판정을 내지 않는다.", "",
            "| 코드 | 조항 | 점수판 모양 | 단서 | 성립 조건 |", "|---|---|---|---|---|"]
    for code, (rule, shape, cues, cond) in JUDGE.items():
        out.append(f"| {code} | {rule} | {shape} | {cues} | {cond} |")
    out.append("")
    names = {r["id"]: (r["rule"], r["name"]) for r in rows}
    out += ["## 보크·투구 반칙 — 동작 판정 방법", "",
            "성립 조건: 축발이 투수판에 닿아 있고(먼저 빼면 보크 아님) 주자가 있을 때. "
            "판정은 해설 + 동작, 점수판은 보이면 '주자 한 베이스씩 진루'로 검증 "
            "(`app/domain/balk_judge.py`).",
            "", "| 조항 | 상황 | 방법 | 어떻게 |", "|---|---|---|---|"]
    for rid, (how, why) in MOTION.items():
        rule, name = names.get(rid, ("?", rid))
        out.append(f"| {rule} | {name} | {how} | {why} |")
    out.append("")
    out += action_report(rows)
    gaps = [r for r in rows if not r["glossary"]]
    out += ["## 설명 근거(용어 사전)가 없는 항목", ""]
    out += [f"- {r['name']} `{r['rule']}`" for r in gaps] or ["- 없음"]
    out.append("")
    return "\n".join(out)


def action_report(rows: list[dict]) -> list[str]:
    acted = [r for r in rows if r["id"] in ACTION]
    st = {r["id"]: action_status(r["id"]) for r in acted}
    recog = Counter(v[0] for v in st.values())
    judge = Counter(v[1] for v in st.values())
    blocked: Counter = Counter()
    for r in acted:
        for tok in dict.fromkeys(ACTION[r["id"]][1].split()):
            if not signal_built(tok):
                blocked[tok] += 1
    out = ["## 행동 판정 로직 — 규칙마다 성립 조건 · 관측 신호 · 판정", "",
           "원본 `scripts/action_logic_src.py`. 판정은 두 길이다 — "
           "**인식**: 점수판·장면·해설·심판 동작으로 심판이 내린 판정을 알아본다 · "
           "**판단**: 자세·공·물체·시간으로 동작을 직접 본다(심판 판정과 어긋나면 심판을 따른다). "
           "상태는 코드에서 센다. ~~취소선~~ 신호는 아직 못 본다.", "",
           f"영상 대상 {len(acted)}개 — 인식 가능 {recog.get('가능', 0)}"
           f" · 불가 {recog.get('불가', 0)}"
           f" / 판단 가능 {judge.get('가능', 0)} · 불가 {judge.get('불가', 0)}"
           f" · 판단 신호 없음 {judge.get('—', 0)}", "",
           "### 미구현 신호 — 막힌 규칙 수 (구현 우선순위)", "",
           "| 신호 | 무엇 | 걸린 규칙 |", "|---|---|---|"]
    for tok, n in blocked.most_common():
        what = "해설·자막 키워드" if tok.startswith("kw:") else " / ".join(
            SIGNALS[a][1] for a in tok.split("|") if a in SIGNALS)
        out.append(f"| `{tok}` | {what} | {n} |")
    out += ["", "| 규칙 | 성립 조건 | 관측 신호 | 판정 | 인식 | 판단 |",
            "|---|---|---|---|---|---|"]
    for r in acted:
        pre, sigs, how = ACTION[r["id"]]
        shown = " ".join(f"`{s}`" if signal_built(s) else f"~~{s}~~" for s in sigs.split())
        rs, js = st[r["id"]]
        out.append(f"| {r['name']} | {pre} | {shown} | {how} | {rs} | {js} |")
    out.append("")
    return out


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
