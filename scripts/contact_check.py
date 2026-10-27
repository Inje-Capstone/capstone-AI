#!/usr/bin/env python
"""영상 → 화면 속 모든 사람 관절 → 선수 접촉·충돌·슬라이딩 (app/domain/contact.py).

선택 설치: pip install -e ".[pose]". CPU로 돈다(사람 수만큼 느려진다).
관절 좌표는 영상 옆 `<영상>.people.json`, 카메라 이동은 `<영상>.cam.json`에 캐시해 기준만 바꿔
다시 볼 때는 빠르다.
`--shots 폴더`면 후보마다 그 순간 프레임에 사람 관절 범위를 그려 저장한다 — 정답 표를
눈으로 만들 때 쓴다(영상과 같은 비공개 폴더에 둘 것). `--truth`면 눈으로 확인한 정답 표
(data/clip_eval/contact_truth.json)와 비교해 종류별 정밀도·재현율을 낸다.

    python scripts/contact_check.py clip.mp4
    python scripts/contact_check.py clips/*.mp4 --fps 10 --shots ../clips_private/contact_shots
    python scripts/contact_check.py clips/*.mp4 --truth data/clip_eval/contact_truth.json
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.adapters.video.pose import (  # noqa: E402
    PeoplePoseExtractor,
    attach_camera,
    camera_motion,
    people_from_json,
    people_to_json,
)
from app.domain.contact import Body, Scene, analyze  # noqa: E402

MIN_SCALE = 0.1  # 화면 높이 대비 — 이보다 작은 사람(먼 외야·관중석)은 접촉을 가릴 수 없다
WINDOW = 600.0  # 경기 전체는 이만큼씩 잘라 관절을 뽑고 잰다
FPS, MODE = 10.0, "balanced"  # 관절 추출 기본 — 다르면 캐시를 따로 둔다
KINDS = ("slide", "slide_tag", "collision")
TOLERANCE = 1.0  # 정답 시각과 이만큼(초) 안이면 같은 동작
REVIEW = 1.0  # 그냥 접촉도 이 속도(키/초) 이상 다가왔으면 프레임을 남긴다 — 놓친 충돌 찾기


def detected(summary: dict[str, Any]) -> dict[str, list[tuple[float, float]]]:
    """summarize 결과 → 종류별 [(시작, 끝)]."""
    out: dict[str, list[tuple[float, float]]] = {k: [] for k in KINDS}
    out["slide"] = [(s["t0"], s["t1"]) for s in summary["slides"]]
    for c in summary["contacts"]:
        out[c["kind"]].append((c["t0"], c["t1"]))
    return out


def _span(x: Any) -> tuple[float, float]:
    """정답 한 개 — 시각 하나 또는 [시작, 끝](다이빙 뒤 누워 있다 일어나기까지 이어진 한 장면)."""
    return (float(x[0]), float(x[1])) if isinstance(x, (list, tuple)) else (float(x), float(x))


def score(truth: Sequence[Any], found: list[tuple[float, float]], tol: float = TOLERANCE,
          unsure: Sequence[Sequence[float]] = ()) -> tuple[int, int, int]:
    """(맞힘, 헛것, 놓침). 잡은 구간 앞뒤 tol초가 정답 장면과 겹치면 맞힘(정답 하나에 하나).

    unsure: 눈으로 봐도 갈리지 않은 구간 [시작, 끝] — 여기 걸친 검출은 셈에서 뺀다.
    """
    left = sorted((_span(x) for x in truth), key=lambda span: span[1])
    hit = false = 0
    for t0, t1 in sorted(found, key=lambda span: span[1]):  # 끝이 이른 구간부터 — 최적 매칭
        match = next((x for x in left if t0 - tol <= x[1] and x[0] <= t1 + tol), None)
        if match is not None:
            left.remove(match)
            hit += 1
        elif not any(a <= t1 and t0 <= b for a, b in unsure):
            false += 1
    return hit, false, len(left)


def _boxes_at(sc: Scene, t: float, who: tuple[int, ...]) -> dict[int, Body]:
    k = min(range(len(sc.frames)), key=lambda i: abs(sc.frames[i].t - t))
    return {tid: b for b, tid in zip(sc.bodies[k], sc.ids[k]) if tid in who and b is not None}


def caches(video: Path, start: float = 0.0, end: Optional[float] = None,
           fps: float = FPS, mode: str = MODE) -> tuple[Path, Path]:
    """(관절, 카메라) 캐시 경로. 구간을 잘라 보면 구간마다, 관절 모델·fps가 기본과 다르면 따로."""
    span = "" if start == 0 and end is None else f".{start:g}-{end:g}"
    model = "" if (fps, mode) == (FPS, MODE) else f".{mode}{fps:g}"
    return (video.with_suffix(f"{span}{model}.people.json"),
            video.with_suffix(f"{span}.cam.json"))  # 카메라는 원본 프레임마다라 모델과 무관


def load_frames(video: Path, extractor_args: argparse.Namespace, start: float = 0.0,
                end: Optional[float] = None) -> tuple[list, int]:
    """(카메라 이동을 붙인 사람 관절 프레임, 화면 높이) — 없으면 뽑아서 캐시한다."""
    cache, cam_cache = caches(video, start, end, extractor_args.fps, extractor_args.mode)
    if not cache.exists():
        extractor = PeoplePoseExtractor(fps=extractor_args.fps, mode=extractor_args.mode)
        doc = people_to_json(extractor.extract(video, start, end), extractor.size)
        cache.write_text(json.dumps(doc), encoding="utf-8")
    doc = json.loads(cache.read_text(encoding="utf-8"))  # 캐시를 거쳐 읽어 늘 같은 결과
    if not cam_cache.exists():
        cam_cache.write_text(json.dumps(camera_motion(video, start, end)), encoding="utf-8")
    motion = [tuple(r) for r in json.loads(cam_cache.read_text(encoding="utf-8"))]
    return attach_camera(people_from_json(doc), motion), int(doc["size"][1])


def summarize(video: Path, args: argparse.Namespace, start: float = 0.0,
              end: Optional[float] = None) -> dict[str, Any]:
    """판정 단서로 쓸 것만 — 슬라이딩·다이빙, 미끄러져 들어와 닿음, 충돌. (eval_clips --pose)"""
    frames, height = load_frames(video, args, start, end)
    _sc, found, slid = analyze(frames, min_scale=args.min_scale * height)
    return {
        "slides": [{"t0": s.t0, "t1": s.t1, "distance": s.distance, "thrown": s.thrown}
                   for s in slid],
        "contacts": [
            {"t0": c.t0, "t1": c.t1, "kind": c.kind, "why": c.why}
            for c in found if c.kind != "contact"
        ],
    }


def summarize_long(video: Path, args: argparse.Namespace, start: float, end: float,
                   window: float = WINDOW) -> dict[str, Any]:
    """경기 전체 — window초씩 잘라 본다(3시간 = 프레임 10만 개, 한 번에 들면 메모리 부족).

    잘린 자리에 걸친 슬라이딩은 놓칠 수 있다(창 600초에 경계 하나).
    """
    out: dict[str, Any] = {"slides": [], "contacts": []}
    t = start
    while t < end:
        part = summarize(video, args, t, min(t + window, end))
        out["slides"] += part["slides"]
        out["contacts"] += part["contacts"]
        t += window
    return out


def save_shot(video: Path, t: float, boxes: dict[int, Body], dest: Path, label: str) -> None:
    import cv2

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return
    for tid, b in boxes.items():
        cv2.rectangle(img, (int(b.x0), int(b.y0)), (int(b.x1), int(b.y1)), (0, 255, 255), 2)
        cv2.putText(img, str(tid), (int(b.x0), int(b.y0) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)
    cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), img)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--mode", default=MODE, help="lightweight | balanced | performance")
    parser.add_argument("--shots", type=Path, default=None, help="후보 순간 프레임을 저장할 폴더")
    parser.add_argument(
        "--min-scale", type=float, default=MIN_SCALE, help="화면 높이 대비 최소 사람 키"
    )
    parser.add_argument("--json", action="store_true", help="결과를 JSON으로")
    parser.add_argument("--truth", type=Path, default=None, help="정답 표와 비교")
    args = parser.parse_args()
    report: dict[str, Any] = {}
    truth = json.loads(args.truth.read_text(encoding="utf-8"))["clips"] if args.truth else None
    totals: dict[tuple[str, str], list[int]] = {}  # (묶음, 종류) → [맞힘, 헛것, 놓침]
    for video in args.videos:
        if truth is not None:
            if video.stem not in truth:
                continue
            found_by_kind = detected(summarize(video, args))
            gold = truth[video.stem]
            line = []
            for k in KINDS:
                if k != "slide" and gold.get("tags_checked") is False:  # 태그·충돌은 아직 안 봄
                    continue
                h, f, m = score(gold.get(k, []), found_by_kind[k],
                                unsure=gold.get("unsure", {}).get(k, []))
                key = (gold.get("split", "-"), k)
                totals[key] = [a + b for a, b in zip(totals.get(key, [0, 0, 0]), (h, f, m))]
                line.append(f"{k} 맞힘{h} 헛것{f} 놓침{m}")
            print(f"== {video.stem} [{gold.get('split', '-')}]: " + " · ".join(line))
            continue
        frames, height = load_frames(video, args)
        sc, found, slid = analyze(frames, min_scale=args.min_scale * height)
        report[video.name] = {
            "contacts": [{**c.__dict__, "why": c.why} for c in found],
            "slides": [s.__dict__ for s in slid],
        }
        if args.shots:
            for c in found:
                if c.kind == "contact" and (c.closing is None or c.closing < REVIEW):
                    continue
                name = f"{video.stem}_{c.t0:06.1f}_{c.kind}_{c.a}-{c.b}.jpg"
                save_shot(video, c.t0, _boxes_at(sc, c.t0, (c.a, c.b)), args.shots / name,
                          f"{c.kind} {c.t0:.1f}s")
            for s in slid:
                name = f"{video.stem}_{s.t1:06.1f}_slide_{s.who}.jpg"
                save_shot(video, s.t1, _boxes_at(sc, s.t1, (s.who,)), args.shots / name,
                          f"slide {s.t0:.1f}-{s.t1:.1f}s")
        if not args.json:
            people = sum(len(f.people) for f in frames) / max(len(frames), 1)
            print(f"== {video.name}: 프레임 {len(frames)} · 평균 {people:.1f}명")
            for c in found:
                print(f"  {c.kind:9s} {c.why}")
            for s in slid:
                print(f"  slide     {s.t0:.1f}~{s.t1:.1f}s 사람 {s.who} 거리 {s.distance}몸길이")
    if truth is not None:
        for (split, k), (h, f, m) in sorted(totals.items()):
            precision = f"{h / (h + f):.0%}" if h + f else "—"
            recall = f"{h / (h + m):.0%}" if h + m else "—"
            print(f"[{split}] {k}: 정밀도 {precision} ({h}/{h + f})"
                  f" · 재현율 {recall} ({h}/{h + m})")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
