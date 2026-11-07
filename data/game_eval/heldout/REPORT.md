# 시험용 경기 채점(소리 없음, 영상만)

> 엔진 결과를 보기 전에 정답을 매긴 두 경기. 문동주(OvmkANKCcks)는 처음 측정 그대로, 안우진(zRqpvJdTybA)은 처음 측정 3/7 뒤 삼진 표시·실내 화면을 고친 결과(조정용이 됨) — `../../analysis/LOG.md`. 다시 만들기: `python scripts/eval_games.py --snapshots ../clips_private/heldout_final --out data/game_eval/heldout`

| 영상 | 채점 칸 | 맞음 | 틀림 | 놓침 | 헛판정/조용함 |
|---|---|---|---|---|---|
| zRqpvJdTybA (시험용 · KBO 삼성-키움(고척) 안우진 투구 모음 · 왼쪽 위 상자 점수판 (소리 없음)) | 7 | 6 | 0 | 1 | 0/15 |
| OvmkANKCcks (시험용 · KBO KT-한화(대전) 문동주 투구 모음 · MBC 스포츠+ (소리 없음)) | 12 | 10 | 2 | 0 | 3/39 |
| 합계 | 19 | 16 | 2 | 1 | 3/54 |

확실한 정답만: 17칸 중 맞음 15, 틀림 2, 놓침 0.

## 정답 종류별

| 정답 | 칸 | 맞음 | 틀림 | 놓침 |
|---|---|---|---|---|
| 아웃(인플레이) | 4 | 4 | 0 | 0 |
| 아웃(종류 모름) | 7 | 7 | 0 | 0 |
| 삼진 | 6 | 5 | 0 | 1 |
| 병살 | 1 | 0 | 1 | 0 |
| 도루 | 1 | 0 | 1 | 0 |

## 정답 × 엔진

| 정답 | 엔진 | 칸 |
|---|---|---|
| double_play | out | 1 |
| out | out | 3 |
| out | out reach | 1 |
| out_any | out | 2 |
| out_any | out out | 2 |
| out_any | strikeout | 3 |
| steal | balk | 1 |
| strikeout | (없음) | 1 |
| strikeout | strikeout | 4 |
| strikeout | strikeout reach reach | 1 |
