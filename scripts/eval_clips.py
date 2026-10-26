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
from app.domain.video_judge import (  # noqa: E402
    GENERIC_CODES,
    MIN_CONFIDENCE,
    Cue,
    from_snapshot,
    judge,
    judge_from_cues,
)

LABELS = ROOT / "data" / "clip_eval" / "labels.json"
REPORT = ROOT / "data" / "clip_eval" / "REPORT.md"
CLIP_WINDOW = 600.0
SCOREBUG_VERSION = "scorebug-crop-v1"
ASR_VERSION = "whisper-large-v3-ko-30s"


# ── 순수 함수 (테스트 대상) ───────────────────────────────────────────────
def observed_text(snap: dict[str, Any]) -> str:
    """VLM이 본 것·들은 것 + 해설 받아쓰기를 한 문자열로 (키워드 검사용, 소문자)."""
    parts = [
        f"{e.get('event_type', '')} {e.get('description', '')}" for e in snap.get("events") or []
    ]
    parts += [s.get("text", "") for s in snap.get("speech") or []]
    parts += [s.get("text", "") for s in (snap.get("asr") or {}).get("segments") or []]
    return " ".join(parts).lower()


def consistent(readings: list[Any]) -> list[Any]:
    """한 클립 안에서 앞뒤가 맞는 판독만 — 이닝·초말 순서와 점수는 뒤로 가지 않는다.

    실측: 점수판 일부가 가려진 프레임을 '10회말 2-3 카운트 → 7회초 1-7'처럼 읽었다.
    """
    out: list[Any] = []
    for r in readings:
        if not (0 <= r.balls <= 3 and 0 <= r.strikes <= 2 and 0 <= r.outs <= 3 and r.inning >= 1):
            continue
        if out:
            p = out[-1]
            order = (r.inning, r.half == "bot") >= (p.inning, p.half == "bot")
            if not order or r.away < p.away or r.home < p.home or r.inning > p.inning + 1:
                continue
        out.append(r)
    return out


def endpoints(readings: list[Any]) -> list[Any]:
    """하이라이트는 대개 플레이 하나 — 처음 상태와 마지막 상태만 비교한다(중간 오독 무시)."""
    return readings if len(readings) <= 2 else [readings[0], readings[-1]]


CLIP_MODE = "endpoints"
DESC_CUES = True


def score_clip(label: dict[str, Any], snap: dict[str, Any]) -> dict[str, Any]:
    readings, cues = from_snapshot(snap)
    asr = [s for s in (snap.get("asr") or {}).get("segments") or [] if s.get("text")]
    if asr:  # 해설은 Whisper 받아쓰기로 (VLM 받아쓰기는 지어낸 문장이 많다) — 자막은 남긴다
        cues = [c for c in cues if c.kind != "speech" or c.text.startswith("[자막]")]
        cues = sorted(
            cues + [Cue(float(s["t"]), "speech", s["text"]) for s in asr], key=lambda c: c.t
        )
    crop = (snap.get("scorebug") or {}).get("scoreboard") or []
    if crop:  # 점수판 전용 판독이 있으면 그것을 쓴다 (영상 통째 판독보다 정확)
        readings, _ = from_snapshot({"scoreboard": crop})
    readings = consistent(readings)
    if CLIP_MODE == "endpoints":
        readings = endpoints(readings)
    if DESC_CUES:  # 장면 설명도 해설처럼 단서로 (클립에선 해설이 드물다)
        cues = sorted(
            cues + [Cue(c.t, "speech", c.text) for c in cues if c.kind != "speech" and c.text],
            key=lambda c: c.t,
        )
    # 클립: 점수판이 몇 번 안 나오고(min_repeat 1), 단서는 클립 어디에 있든 본다
    judgments = judge(readings, cues, min_repeat=1, window=(CLIP_WINDOW, CLIP_WINDOW))
    if all(j.code in GENERIC_CODES for j in judgments):  # 종류를 못 가렸으면 단서만으로
        extra = judge_from_cues(cues)
        if extra is not None:
            judgments.append(extra)
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
        # 정답이 아닌 구체 판정을 냈나 — 정답률만으론 지어낸 판정이 안 보인다
        "wrong": sorted({c for c in shown if c not in GENERIC_CODES and c != expect})
        if expect else [],
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
        "wrong": sum(1 for r in judged if r["wrong"]),
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
        f"추론 {meta.get('thinking', '?')} · {meta.get('scorebug', '')} · "
        "`python scripts/eval_clips.py`로 재생성. 정답은 영상 제목 기준(사람 확인 전).",
        "",
        "| 지표 | 값 |",
        "|---|---|",
        f"| 클립 | {s['clips']}개 |",
        f"| 판정 정답 (영상 판정이 정답 코드를 냄) | {pct(s['judge_hit'], s['judge_total'])} |",
        f"| 오판 (정답이 아닌 판정 코드를 냄) | {pct(s['wrong'], s['judge_total'])} |",
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


def build_reader(args: argparse.Namespace) -> Any:
    import os

    from app.adapters.video.cosmos import DEFAULT_BASE_URL, DEFAULT_MODEL
    from app.adapters.video.scorebug_reader import ScorebugReader

    return ScorebugReader(
        os.getenv("NVIDIA_API_KEY", "").strip(),
        args.base_url or DEFAULT_BASE_URL,
        args.model or DEFAULT_MODEL,
        max_reads=args.max_reads,
        ffmpeg=find_tool("ffmpeg", args.ffmpeg),
    )


def build_transcriber(args: argparse.Namespace) -> Any:
    import os

    from app.adapters.video.speech_asr import WhisperTranscriber

    return WhisperTranscriber(
        os.getenv("NVIDIA_API_KEY", "").strip(), ffmpeg=find_tool("ffmpeg", args.ffmpeg)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clips", type=Path, required=True, help="videos/·results/가 있는 비공개 폴더"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", default=None, help="이 문자열이 id에 든 클립만")
    parser.add_argument("--league", default="KBO", help="KBO | MLB | all (MVP는 KBO만)")
    parser.add_argument(
        "--report-only", action="store_true", help="API를 부르지 않고 캐시로 보고서만"
    )
    parser.add_argument(
        "--scorebug",
        action="store_true",
        help="점수판을 따로 잘라 읽는다 (app/adapters/video/scorebug_reader.py)",
    )
    parser.add_argument("--max-reads", type=int, default=8)
    parser.add_argument(
        "--asr", action="store_true", help="해설을 Whisper로 받아 적는다 ([asr] 설치)"
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
    if args.league != "all":
        labels = [c for c in labels if c["league"] == args.league]
    if args.only:
        labels = [c for c in labels if args.only in c["id"]]
    if args.limit:
        labels = labels[: args.limit]
    results_dir = args.clips / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    analyzer: Optional[Any] = None
    reader: Optional[Any] = None
    transcriber: Optional[Any] = None
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
            # 점수판 판독·해설 받아쓰기는 지시문과 무관 — 살린다
            kept = {k: (snap or {}).get(k) for k in ("scorebug", "asr") if (snap or {}).get(k)}
            snap = analyze_clip(video, analyzer, ffmpeg, ffprobe, args.clip_sec, args.height)
            snap.update(kept)
            snap["thinking"] = not args.no_think
            cache.write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
        if args.scorebug and (snap.get("scorebug") or {}).get("version") != SCOREBUG_VERSION:
            if args.report_only:
                print(f"[{i}] {label['id']} 점수판 판독 캐시 없음")
            else:
                video = args.clips / "videos" / f"{label['yt']}.mp4"
                if reader is None:
                    reader = build_reader(args)
                    ffprobe = find_tool("ffprobe", args.ffprobe)
                try:
                    sb = reader.read(video, probe_duration(ffprobe, video))
                except VideoAnalyzerError as exc:
                    print(f"[{i}] {label['id']} 점수판 판독 실패: {exc}")
                    sb = None
                if sb is not None:
                    snap["scorebug"] = {**sb, "version": SCOREBUG_VERSION}
                    cache.write_text(
                        json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8"
                    )
        if args.asr and (snap.get("asr") or {}).get("version") != ASR_VERSION:
            if args.report_only:
                print(f"[{i}] {label['id']} 해설 받아쓰기 캐시 없음")
            else:
                video = args.clips / "videos" / f"{label['yt']}.mp4"
                if transcriber is None:
                    transcriber = build_transcriber(args)
                    ffprobe = find_tool("ffprobe", args.ffprobe)
                segs = transcriber.transcribe(video, probe_duration(ffprobe, video))
                snap["asr"] = {"version": ASR_VERSION, "segments": segs}
                cache.write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
        meta = {
            "model": snap.get("model", "?"),
            "thinking": "켬" if snap.get("thinking") else "끔",
            "scorebug": ("점수판 전용 판독" if args.scorebug else "영상 통째 판독")
            + (" · 해설 Whisper" if args.asr else ""),
        }
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
