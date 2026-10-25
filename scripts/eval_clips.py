#!/usr/bin/env python
"""하이라이트 클립 평가 — 클립마다 VLM 판독 → 영상 판정 → 정답(data/clip_eval/labels.json)과 비교.

경기 전체가 아니라 짧은 클립 하나씩이라 점수판 흐름이 끊긴다. 그래서 두 가지를 따로 잰다.
- 판정: 영상 판정(video_judge)이 정답 코드를 냈나 (expect가 있는 클립만)
- 단서: VLM 관찰(장면 설명·해설·이벤트 종류)에 정답 키워드가 나왔나 — 판정 규칙을 늘릴 근거

영상 파일은 레포 밖 비공개 폴더에 둔다(저작권). VLM 결과는 클립별로 캐시해 다시 돌려도
API를 다시 부르지 않는다. 요약만 data/clip_eval/REPORT.md로 남긴다.

사용:
    python scripts/eval_clips.py --clips "../clips_private" --limit 3      # 앞 3개만 시험
    python scripts/eval_clips.py --clips "../clips_private"                # 전체
    python scripts/eval_clips.py --clips "../clips_private" --report-only  # 캐시로 보고서만
"""

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_video import (  # noqa: E402
    build_analyzer,
    ffmpeg_cut_cmd,
    find_tool,
    plan_clips,
    probe_duration,
)

from app.adapters.video.base import PROMPT_VERSION, VideoAnalyzerError, VideoClip  # noqa: E402
from app.domain.video_judge import MIN_CONFIDENCE, from_snapshot, judge  # noqa: E402

LABELS = ROOT / "data" / "clip_eval" / "labels.json"
REPORT = ROOT / "data" / "clip_eval" / "REPORT.md"


# ── 순수 함수 (테스트 대상) ───────────────────────────────────────────────
def observed_text(snap: dict[str, Any]) -> str:
    """VLM이 본 것·들은 것을 한 문자열로 (키워드 검사용, 소문자)."""
    parts = [
        f"{e.get('event_type', '')} {e.get('description', '')}" for e in snap.get("events") or []
    ]
    parts += [s.get("text", "") for s in snap.get("speech") or []]
    return " ".join(parts).lower()


def score_clip(label: dict[str, Any], snap: dict[str, Any]) -> dict[str, Any]:
    readings, cues = from_snapshot(snap)
    judgments = judge(readings, cues, min_repeat=1)  # 클립은 점수판이 몇 번 안 나온다
    shown = [j.code for j in judgments if j.confidence >= MIN_CONFIDENCE]
    every = [j.code for j in judgments]
    text = observed_text(snap)
    hits = [k for k in label.get("keywords", []) if k.lower() in text]
    expect = label.get("expect")
    return {
        "id": label["id"],
        "league": label["league"],
        "situation": label["situation"],
        "expect": expect,
        "judged": expect in shown if expect else None,
        "judged_low": (expect in every and expect not in shown) if expect else None,
        "codes": every,
        "cue_hit": bool(hits),
        "hits": hits,
        "readings": len(readings),
        "transitions": len(judgments),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    judged = [r for r in rows if r["expect"]]
    by_code: dict[str, list[dict]] = defaultdict(list)
    for r in judged:
        by_code[r["expect"]].append(r)
    by_league: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_league[r["league"]].append(r)
    return {
        "clips": len(rows),
        "judge_hit": sum(1 for r in judged if r["judged"]),
        "judge_total": len(judged),
        "cue_hit": sum(1 for r in rows if r["cue_hit"]),
        "no_scoreboard": sum(1 for r in rows if r["readings"] == 0),
        "by_code": {
            k: (sum(1 for r in v if r["judged"]), len(v)) for k, v in sorted(by_code.items())
        },
        "by_league": {
            k: (
                sum(1 for r in v if r["judged"]),
                sum(1 for r in v if r["expect"]),
                sum(1 for r in v if r["cue_hit"]),
                len(v),
            )
            for k, v in sorted(by_league.items())
        },
    }


def render(rows: list[dict[str, Any]], meta: dict[str, str]) -> str:
    s = summarize(rows)

    def pct(a: int, b: int) -> str:
        return f"{a}/{b} ({a / b:.0%})" if b else "0/0"

    out = [
        "# 하이라이트 클립 평가",
        "",
        f"> 모델 {meta.get('model', '?')} · 프롬프트 {PROMPT_VERSION} · "
        f"추론 {meta.get('thinking', '?')} · "
        "`python scripts/eval_clips.py`로 재생성. 정답은 영상 제목 기준(사람 확인 전).",
        "",
        "| 지표 | 값 |",
        "|---|---|",
        f"| 클립 | {s['clips']}개 |",
        f"| 판정 정답 (영상 판정이 정답 코드를 냄) | {pct(s['judge_hit'], s['judge_total'])} |",
        f"| 단서 포착 (VLM 관찰에 정답 키워드) | {pct(s['cue_hit'], s['clips'])} |",
        f"| 점수판을 하나도 못 읽은 클립 | {s['no_scoreboard']}개 |",
        "",
        "| 리그 | 판정 | 단서 |",
        "|---|---|---|",
    ]
    for league, (jh, jt, ch, n) in s["by_league"].items():
        out.append(f"| {league} | {pct(jh, jt)} | {pct(ch, n)} |")
    out += ["", "| 판정 코드 | 정답 |", "|---|---|"]
    for code, (h, n) in s["by_code"].items():
        out.append(f"| {code} | {pct(h, n)} |")
    out += [
        "",
        "## 클립별",
        "",
        "| 클립 | 정답 | 판정 | 낸 코드 | 단서 | 점수판 판독 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        verdict = (
            "·"
            if r["judged"] is None
            else ("✓" if r["judged"] else ("낮은 확신" if r["judged_low"] else "✗"))
        )
        out.append(
            f"| {r['id']} | {r['expect'] or '(단서만)'} | {verdict} | "
            f"{', '.join(r['codes']) or '-'} | {', '.join(r['hits']) or '✗'} | {r['readings']} |"
        )
    out.append("")
    return "\n".join(out)


# ── 실행 ────────────────────────────────────────────────────────────────
def analyze_clip(
    video: Path, analyzer: Any, ffmpeg: str, ffprobe: str, clip_sec: float, height: int
) -> dict[str, Any]:
    """클립 하나를 60초 청크로 나눠 분석해 analyze_video와 같은 스냅샷 모양으로 돌려준다."""
    snap: dict[str, Any] = {
        "provider": analyzer.provider,
        "model": analyzer.model,
        "prompt_version": PROMPT_VERSION,
        "events": [],
        "scoreboard": [],
        "speech": [],
        "failed": [],
    }
    duration = probe_duration(ffprobe, video)
    with tempfile.TemporaryDirectory() as tmp:
        for start, length in plan_clips(duration, clip_sec):
            dest = Path(tmp) / f"c_{int(start):05d}.mp4"
            subprocess.run(
                ffmpeg_cut_cmd(ffmpeg, video, start, length, dest, height=height), check=True
            )
            try:
                result = analyzer.analyze(VideoClip(dest, start, length))
            except VideoAnalyzerError as exc:
                print(f"  {start:.0f}s 실패: {exc}", file=sys.stderr)
                snap["failed"].append(start)
                continue
            snap["events"] += [e.model_dump() for e in result.events]
            snap["scoreboard"] += result.scoreboard
            snap["speech"] += result.speech
    for key, t in (("events", "t_start"), ("scoreboard", "t"), ("speech", "t")):
        snap[key].sort(key=lambda x: x[t])
    return snap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clips", type=Path, required=True, help="videos/·results/가 있는 비공개 폴더"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", default=None, help="이 문자열이 id에 든 클립만")
    parser.add_argument(
        "--report-only", action="store_true", help="API를 부르지 않고 캐시로 보고서만"
    )
    parser.add_argument("--clip-sec", type=float, default=60.0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--think", dest="no_think", action="store_false")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--ffmpeg", default=None)
    parser.add_argument("--ffprobe", default=None)
    args = parser.parse_args()
    args.backend, args.vss_url, args.clip_base_url = "cosmos", None, None

    labels = json.loads(LABELS.read_text(encoding="utf-8"))["clips"]
    if args.only:
        labels = [c for c in labels if args.only in c["id"]]
    if args.limit:
        labels = labels[: args.limit]
    results_dir = args.clips / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    analyzer: Optional[Any] = None
    rows, meta = [], {}
    for i, label in enumerate(labels, 1):
        cache = results_dir / f"{label['id']}.json"
        snap = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else None
        if snap is None or snap.get("prompt_version") != PROMPT_VERSION or snap.get("failed"):
            if args.report_only:
                print(f"[{i}] {label['id']} 캐시 없음 — 건너뜀")
                continue
            video = args.clips / "videos" / f"{label['yt']}.mp4"
            if not video.exists():
                print(f"[{i}] {label['id']} 영상 없음 ({video.name}) — 건너뜀")
                continue
            if analyzer is None:
                analyzer = build_analyzer(args)
                ffmpeg, ffprobe = (
                    find_tool("ffmpeg", args.ffmpeg),
                    find_tool("ffprobe", args.ffprobe),
                )
            snap = analyze_clip(video, analyzer, ffmpeg, ffprobe, args.clip_sec, args.height)
            snap["thinking"] = not args.no_think
            cache.write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
        meta = {"model": snap.get("model", "?"), "thinking": "켬" if snap.get("thinking") else "끔"}
        row = score_clip(label, snap)
        rows.append(row)
        print(
            f"[{i}/{len(labels)}] {label['id']}: 판정 {row['judged']} 단서 {row['hits'] or '없음'} "
            f"코드 {row['codes']}"
        )
    if rows:
        REPORT.write_text(render(rows, meta), encoding="utf-8")
        s = summarize(rows)
        print(
            f"판정 {s['judge_hit']}/{s['judge_total']} · 단서 {s['cue_hit']}/{s['clips']} "
            f"→ {REPORT.name}"
        )


if __name__ == "__main__":
    main()
