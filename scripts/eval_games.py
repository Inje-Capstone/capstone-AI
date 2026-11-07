#!/usr/bin/env python
"""1080p 경기 하이라이트 채점 — 영상 판정을 사람이 매긴 정답(data/game_eval/labels.json)과 맞춘다.

정답은 투구 화면마다 하나다: 그 투구부터 다음 투구까지 사이에 끝난 타석의 결과.
엔진 판정의 시각(새 점수판이 처음 보인 때)이 그 투구 화면이 끝난 뒤부터 다음 투구 화면이 끝나기
전까지면 그 정답의 판정으로 본다. 결과는 투구가 끝나야 보이고, 하이라이트는 다시보기가 길어
새 점수판이 다음 투구 화면에서야 보이는 일이 많다.

- 맞음: 그 칸의 판정 중 하나가 정답과 같은 종류
- 틀림: 판정은 냈는데 종류가 다름
- 놓침: 판정이 없음
- 헛판정: 정답이 '변화 없음(none)'인 칸에 낸 판정
정답 '?'(결과를 못 봄) 칸은 채점하지 않는다. 투수 교체는 채점하지 않는다.

사용(analyze_video.py --no-vlm --scorebug --layout … --asr --audio … --pose 결과가 있어야 한다):
    python scripts/eval_games.py --snapshots ../clips_private/game_eval
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.domain.video_judge import judge_checked, observe  # noqa: E402

LABELS = ROOT / "data" / "game_eval" / "labels.json"
OUT = ROOT / "data" / "game_eval"
# 정답 종류 → 맞다고 볼 엔진 판정 코드
ACCEPT = {
    "reach": {"reach", "dive_hit", "ground_rule_double"},
    "out": {"out", "dive_out"},
    "strikeout": {"strikeout", "dropped_third_strike"},
    "out_any": {"out", "strikeout", "dive_out", "dropped_third_strike"},
    "walk": {"walk"},
    "hbp": {"hbp"},
    "walk_or_hbp": {"walk", "hbp"},
    "steal": {"steal"},
    "homerun": {"homerun"},
    "sac_fly": {"sac_fly"},
    "double_play": {"double_play"},
}
SKIP = {"?"}
LATE = 8.0  # 앞 칸 판정이 다음 칸으로 늦게 넘어올 수 있는 시간(초)
NONE = "none"


def windows(plays: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """정답 칸마다 판정을 받을 시간 [시작, 끝) — 투구 화면 끝부터 다음 투구 화면 끝까지."""
    out = []
    for i, p in enumerate(plays):
        start = p.get("pitch_end", p["t"])
        nxt = plays[i + 1] if i + 1 < len(plays) else None
        end = nxt.get("pitch_end", nxt["t"]) if nxt else p["end"] + 15.0
        out.append((start, end))
    return out


def assign(plays: list[dict[str, Any]], judged: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """판정을 정답 칸에 나눠 담는다. 어느 칸에도 안 드는 판정(첫 투구 전)은 버린다.

    빈 칸 바로 다음 칸에 판정이 둘 이상이고 그 첫 판정이 칸 시작 LATE초 안이면 앞 칸 몫으로
    옮긴다 — 다시보기 뒤 새 점수판이 다음 투구 화면이 끝난 몇 초 뒤에야 보이는 일이 있다.
    """
    spans = windows(plays)
    cells: list[list[dict[str, Any]]] = [[] for _ in plays]
    for j in judged:
        for k, (a, b) in enumerate(spans):
            if a <= j["t"] < b:
                cells[k].append(j)
                break
    for k in range(len(cells) - 1):
        nxt = cells[k + 1]
        if not cells[k] and len(nxt) >= 2 and nxt[0]["t"] < spans[k + 1][0] + LATE:
            cells[k].append(nxt.pop(0))
    return cells


def verdict(code: str, got: list[dict[str, Any]]) -> Optional[str]:
    """정답 한 칸의 결과. 채점하지 않는 칸은 None."""
    if code in SKIP:
        return None
    if code == NONE:
        return "헛판정" if got else "조용함"
    if not got:
        return "놓침"
    return "맞음" if any(j["code"] in ACCEPT.get(code, {code}) for j in got) else "틀림"


def engine_judgments(snap: dict[str, Any]) -> list[dict[str, Any]]:
    readings, cues, repeat = observe(snap)
    out = []
    for j in judge_checked(readings, cues, min_repeat=repeat):
        if j.kind == "sub":
            continue
        out.append({"t": round(j.t, 1), "code": j.code, "kind": j.kind,
                    "confidence": round(j.confidence, 2), "check": j.detail.get("check", ""),
                    "merged": bool(j.detail.get("merged")), "evidence": j.evidence})
    return out


def grade(stem: str, game: dict[str, Any], snap: dict[str, Any]) -> list[dict[str, Any]]:
    plays = game["plays"]
    judged = engine_judgments(snap)
    rows = []
    for p, got in zip(plays, assign(plays, judged)):
        rows.append({"video": stem, "t": p["t"], "truth": p["code"], "sure": p["sure"],
                     "engine": " ".join(j["code"] for j in got),
                     "check": " ".join(j["check"] for j in got),
                     "evidence": " | ".join("; ".join(j["evidence"]) for j in got),
                     "verdict": verdict(p["code"], got) or "제외", "note": p.get("note", "")})
    return rows


def report(rows: list[dict[str, Any]], titles: dict[str, str]) -> str:
    lines = ["# 1080p 경기 하이라이트 채점", "",
             "> `python scripts/eval_games.py --snapshots ../clips_private/game_eval`로 "
             "다시 만든다. "
             "정답은 `labels.json`(사람이 투구 화면마다 매김), 칸별 결과는 `plays.csv`.", ""]
    head = ["영상", "채점 칸", "맞음", "틀림", "놓침", "헛판정/조용함"]
    table = []
    for stem in list(dict.fromkeys(r["video"] for r in rows)) + ["합계"]:
        part = [r for r in rows if stem in ("합계", r["video"])]
        c = Counter(r["verdict"] for r in part)
        plays = c["맞음"] + c["틀림"] + c["놓침"]
        name = "합계" if stem == "합계" else f"{stem} ({titles.get(stem, '')})"
        quiet = f"{c['헛판정']}/{c['조용함']}"
        table.append([name, plays, c["맞음"], c["틀림"], c["놓침"], quiet])
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head),
              *["| " + " | ".join(str(x) for x in t) + " |" for t in table], ""]
    sure = [r for r in rows if r["sure"] and r["verdict"] in ("맞음", "틀림", "놓침")]
    if sure:
        c = Counter(r["verdict"] for r in sure)
        lines += [f"확실한 정답만: {len(sure)}칸 중 맞음 {c['맞음']}, 틀림 {c['틀림']}, "
                  f"놓침 {c['놓침']}.", ""]
    pairs = Counter((r["truth"], r["engine"] or "(없음)") for r in rows
                    if r["verdict"] in ("맞음", "틀림", "놓침"))
    lines += ["## 정답 × 엔진", "", "| 정답 | 엔진 | 칸 |", "|---|---|---|",
              *[f"| {t} | {e} | {n} |" for (t, e), n in sorted(pairs.items())], ""]
    return "\n".join(lines)


ABLATIONS = (
    ("점수판+관절+해설", lambda s: s),
    ("점수판+관절(해설 없이)", lambda s: {**s, "asr": {}, "speech": []}),
    ("점수판+해설(관절 없이)", lambda s: {**s, "pose": {}}),
    ("점수판만", lambda s: {**s, "asr": {}, "speech": [], "pose": {}}),
)


def ablation(labels: dict[str, Any], snaps: dict[str, dict[str, Any]]) -> str:
    lines = ["## 입력별", "", "| 입력 | 맞음 | 틀림 | 놓침 | 헛판정 |", "|---|---|---|---|---|"]
    for name, cut in ABLATIONS:
        c: Counter = Counter()
        for stem, snap in snaps.items():
            c.update(r["verdict"] for r in grade(stem, labels[stem], cut(snap)))
        lines.append(f"| {name} | {c['맞음']} | {c['틀림']} | {c['놓침']} | {c['헛판정']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshots", type=Path, required=True, help="analyze_video 결과 폴더")
    parser.add_argument("--only", nargs="*", default=None, help="이 영상만")
    parser.add_argument("--ablation", action="store_true",
                        help="입력을 하나씩 빼고도 채점한다(해설 없이·관절 없이·점수판만)")
    parser.add_argument("--scorebug-from", type=Path, default=None,
                        help="점수판 판독만 이 폴더의 스냅샷에서 가져온다(읽기 방식 비교)")
    args = parser.parse_args()
    labels = json.loads(LABELS.read_text(encoding="utf-8"))["games"]
    rows, titles = [], {}
    snaps: dict[str, dict[str, Any]] = {}
    for stem, game in labels.items():
        snap_path = args.snapshots / f"{stem}.json"
        if (args.only and stem not in args.only) or not snap_path.exists():
            continue
        snap = json.loads(snap_path.read_text(encoding="utf-8"))
        if args.scorebug_from is not None:
            other = json.loads((args.scorebug_from / f"{stem}.json").read_text(encoding="utf-8"))
            snap["scorebug"] = other["scorebug"]
        rows += grade(stem, game, snap)
        titles[stem] = game["title"]
        snaps[stem] = snap
    if not rows:
        raise SystemExit("채점할 영상이 없다 — 스냅샷 폴더를 확인한다")
    with (OUT / "plays.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = report(rows, titles)
    if args.ablation:
        text += "\n" + ablation(labels, snaps)
    (OUT / "REPORT.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
