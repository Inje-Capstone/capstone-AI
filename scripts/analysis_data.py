#!/usr/bin/env python
"""분석 데이터 기록 — 엔진이 영상에서 잰 값과 사람이 매긴 정답을 표로 모은다(data/analysis/).

발표·보고서의 숫자는 이 표에서 나온다. 영상·관절 캐시는 비공개 폴더(clips_private)에 두고,
여기에는 영상 id·시각·잰 값·정답·판정만 남긴다. 중계 화면은 남기지 않는다.

- swings.csv: 투구 화면마다 엔진이 잰 투수 발 들림·타자 손 이동량과 정답(swing_truth.json)
- slides.csv: KBO 정답 클립의 슬라이딩·다이빙마다 거리·몸 아래 잔디·흙 비율과 정답
- judgments.csv: KBO 평가 클립마다 입력(화면+해설·해설만·화면만)별 판정 코드
- SUMMARY.md: 위 표의 요약. 손으로 쓰는 측정 기록은 LOG.md

사용(캐시가 있어야 한다 — swing_check.py · contact_check.py · eval_clips.py가 만든다):
    python scripts/analysis_data.py --clips ../clips_private
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
sys.path.insert(0, str(ROOT / "scripts"))

from contact_check import MIN_SCALE, _boxes_at, _span, load_frames  # noqa: E402
from contact_check import summarize as contact_summary  # noqa: E402
from eval_clips import (  # noqa: E402
    ABLATIONS,
    CHECK_LABELS,
    LABELS,
    check_rows,
    score_clip,
)
from swing_check import load_cached as load_swings  # noqa: E402

from app.domain.contact import analyze  # noqa: E402

OUT = ROOT / "data" / "analysis"
SWING_TRUTH = OUT / "swing_truth.json"
CONTACT_TRUTH = ROOT / "data" / "clip_eval" / "contact_truth.json"
MATCH = 0.5  # 정답 화면과 이 비율 넘게 겹치는 엔진 화면을 같은 장면으로 본다
SAME_START = 0.2  # 시작만 적힌 정답은 엔진 화면 시작이 이만큼 안이면 같은 장면
SLIDE_NEAR = 1.0  # 정답 시각과 이만큼(초) 안이면 같은 슬라이딩


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:  # 엑셀이 한글을 바로 읽게
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def said(row: Optional[dict[str, Any]]) -> str:
    if row is None:
        return "못 찾음"
    if row["swing"]:
        return "스윙"
    return "안 침" if row.get("clear", True) else "모름"


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def swing_rows(clips: Path) -> list[dict[str, Any]]:
    truth = json.loads(SWING_TRUTH.read_text(encoding="utf-8"))["shots"]
    out = []
    for shot in truth:
        engine = load_swings(clips / shot["folder"] / f"{shot['video']}.mp4")
        if shot["t1"] is None:
            found = [r for r in engine if abs(r["t0"] - shot["t0"]) <= SAME_START]
        else:
            span = shot["t1"] - shot["t0"]
            found = [r for r in engine
                     if _overlap(r["t0"], r["t1"], shot["t0"], shot["t1"]) > MATCH * span]
        r = found[0] if found else None
        out.append({
            "video": shot["video"], "set": shot["set"], "t0": shot["t0"], "t1": shot["t1"],
            "label": shot["label"], "first_label": shot.get("first_label", ""),
            "engine_t": r["t"] if r else None, "lift": r.get("lift") if r else None,
            "moved": r["moved"] if r else None, "said": said(r),
        })
    return out


def _ground(img: Any, body: Any) -> tuple[float, float]:
    import cv2

    x0, x1, y1 = int(max(body.x0, 0)), int(body.x1), int(body.y1)
    tall = int(body.y1 - body.y0)
    strip = img[min(y1, img.shape[0] - 2):min(y1 + tall // 2 + 6, img.shape[0]),
                x0:max(x1, x0 + 1)]
    if strip.size == 0:
        return 0.0, 0.0
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    grass = (h >= 30) & (h <= 90) & (s >= 50) & (v >= 40)  # pose.field_ratio와 같은 색 기준
    dirt = (h >= 5) & (h <= 25) & (s >= 60) & (v >= 60)
    return round(float(grass.mean()), 3), round(float(dirt.mean()), 3)


def slide_rows(clips: Path) -> list[dict[str, Any]]:
    from app.adapters.video.frames import open_frames

    truth = json.loads(CONTACT_TRUTH.read_text(encoding="utf-8"))["clips"]
    leagues = {c["yt"]: c["league"] for c in _labels()}
    args = argparse.Namespace(fps=10.0, mode="balanced", min_scale=MIN_SCALE)
    out = []
    for vid, gold in truth.items():
        video = clips / "videos" / f"{vid}.mp4"
        if leagues.get(vid) != "KBO" or not video.with_suffix(".people.json").exists():
            continue
        frames, height = load_frames(video, args)
        scene, _found, slid = analyze(frames, min_scale=args.min_scale * height)
        for s in slid:
            t = (s.t0 + s.t1) / 2
            boxes = _boxes_at(scene, t, s.members or (s.who,))
            first = next(open_frames(video, t, t + 0.3).frames, None)
            grass, dirt = (0.0, 0.0)
            if boxes and first:
                grass, dirt = _ground(first[1], next(iter(boxes.values())))
            hit = any(_span(x)[0] - SLIDE_NEAR <= s.t1 and s.t0 <= _span(x)[1] + SLIDE_NEAR
                      for x in gold.get("slide", []))
            out.append({"video": vid, "split": gold.get("split", ""), "t0": s.t0, "t1": s.t1,
                        "distance": s.distance, "thrown": s.thrown, "grass": grass,
                        "dirt": dirt, "truth_slide": hit})
    return out


def _labels() -> list[dict[str, Any]]:
    raw = json.loads(LABELS.read_text(encoding="utf-8"))
    return raw if isinstance(raw, list) else raw.get("clips") or raw.get("labels") or []


def judgment_rows(clips: Path) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    args = argparse.Namespace(fps=10.0, mode="balanced", min_scale=MIN_SCALE)
    pairs = []
    for label in _labels():
        cache = clips / "results" / f"{label['id']}.json"
        if label["league"] != "KBO" or not label.get("expect") or not cache.exists():
            continue
        snap = json.loads(cache.read_text(encoding="utf-8"))
        video = clips / "videos" / f"{label['yt']}.mp4"
        if all(video.with_suffix(s).exists() for s in (".people.json", ".cam.json")):
            snap["pose"] = contact_summary(video, args)
        swings = load_swings(video)
        if swings:
            snap["pose"] = {**(snap.get("pose") or {}), "swings": swings}
        pairs.append((label, snap))
    out = []
    for label, snap in pairs:
        for name, fn in ABLATIONS:
            row = score_clip(label, fn(snap) if fn else snap)
            out.append({"clip": label["id"], "expect": label["expect"], "input": name,
                        "right": bool(row["judged"]), "wrong": " ".join(row["wrong"]),
                        "codes": " ".join(row["codes"])})
    return out, check_rows(pairs)


def _table(head: list[str], rows: list[list[Any]]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head),
            *["| " + " | ".join(str(c) for c in r) + " |" for r in rows]]


def _check_name(key: str) -> str:
    names = dict(CHECK_LABELS, missed="정답 코드 없음")
    cue = key.startswith("cue:")
    # 3.9엔 str.removeprefix가 없다
    return ("단서 대체 " if cue else "") + names[key[4:] if cue else key]


def summary(swings: list[dict[str, Any]], slides: list[dict[str, Any]],
            judgments: list[dict[str, Any]], checks: dict[str, list[str]]) -> str:
    lines = ["# 분석 데이터 요약", "",
             "> `python scripts/analysis_data.py --clips ../clips_private`로 다시 만든다. "
             "원본 표는 같은 폴더의 CSV, 손으로 쓰는 측정 기록은 LOG.md.", ""]
    lines += ["## 타자 스윙 (swings.csv)", "",
              "정답(행) × 엔진이 말한 것(열). tune = 기준값을 정할 때 본 장면, "
              "test = 기준을 고정한 뒤 처음 본 장면.", ""]
    for part in ("test", "tune"):
        rows = [r for r in swings if r["set"] == part and r["label"] in ("S", "T")]
        count = Counter((r["label"], r["said"]) for r in rows)
        cols = ["스윙", "안 침", "모름", "못 찾음"]
        lines += [f"**{part}** ({len(rows)}장면)", "",
                  *_table(["정답"] + cols, [[{"S": "스윙", "T": "안 침"}[y]] +
                                            [count[(y, c)] for c in cols] for y in ("S", "T")]),
                  ""]
    if slides:
        by_truth = {True: [s for s in slides if s["truth_slide"]],
                    False: [s for s in slides if not s["truth_slide"]]}
        lines += ["## 슬라이딩·다이빙 (slides.csv)", "",
                  f"관절로 찾은 {len(slides)}건 중 정답과 맞는 것 {len(by_truth[True])}건. "
                  f"몸 아래 잔디 비율 0.8 이상 {sum(s['grass'] >= 0.8 for s in slides)}건.", ""]
    if judgments:
        lines += ["## 판정 (judgments.csv)", "",
                  *_table(["입력", "정답", "오판"],
                          [[name, f"{sum(j['right'] for j in judgments if j['input'] == name)}"
                            f"/{sum(1 for j in judgments if j['input'] == name)}",
                            sum(1 for j in judgments if j["input"] == name and j["wrong"])]
                           for name, _ in ABLATIONS]), "",
                  "영상 먼저 판정 → 해설 대조에서 정답이 나온 경로"
                  "(단서 대체 = 점수판 없이 단서만): " +
                  ", ".join(f"{_check_name(k)} {len(v)}" for k, v in checks.items() if v), ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clips", type=Path, required=True, help="clips_private 폴더")
    parser.add_argument("--skip-slides", action="store_true", help="슬라이딩 표는 건너뛴다")
    args = parser.parse_args()
    swings = swing_rows(args.clips)
    slides = [] if args.skip_slides else slide_rows(args.clips)
    judgments, checks = judgment_rows(args.clips)
    write_csv(OUT / "swings.csv", swings)
    write_csv(OUT / "slides.csv", slides)
    write_csv(OUT / "judgments.csv", judgments)
    (OUT / "SUMMARY.md").write_text(summary(swings, slides, judgments, checks), encoding="utf-8")
    print(f"→ {OUT} (스윙 {len(swings)} · 슬라이딩 {len(slides)} · 판정 {len(judgments)})")


if __name__ == "__main__":
    main()
