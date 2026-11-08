# 시험용 경기 채점(소리 없음, 영상만)

> 엔진 결과를 보기 전에 정답을 매긴 네 경기. 처음 측정은 안우진(zRqp) 3/7·문동주(Ovmk) 10/12·최민석(7TYZ) 4/7·안우진 11K(YiiG) 7/11 = 24/37. 아래 표는 처음 측정 뒤 고친 지금 엔진 — 문동주만 처음 측정 그대로이고 나머지는 조정용이 됐다(`../../analysis/LOG.md`). 다시 만들기: `python scripts/eval_games.py --snapshots ../clips_private/heldout_final --out data/game_eval/heldout`

| 영상 | 채점 칸 | 맞음 | 틀림 | 놓침 | 헛판정/조용함 |
|---|---|---|---|---|---|
| zRqpvJdTybA (시험용 · KBO 삼성-키움(고척) 안우진 투구 모음 · 왼쪽 위 상자 점수판 (소리 없음)) | 7 | 6 | 0 | 1 | 0/15 |
| OvmkANKCcks (시험용 · KBO KT-한화(대전) 문동주 투구 모음 · MBC 스포츠+ (소리 없음)) | 12 | 10 | 2 | 0 | 3/39 |
| 7TYZzeweCjk (시험용 · KBO 두산-KIA(광주) 최민석 투구 모음 · 왼쪽 위 상자 점수판 (소리 없음)) | 7 | 6 | 1 | 0 | 1/1 |
| YiiGl9UUV2E (시험용 · KBO LG-키움(고척) 안우진 삼진 모음 · 캐스터 띠 상자 점수판 (소리 없음)) | 11 | 10 | 1 | 0 | 0/6 |
| 합계 | 37 | 32 | 4 | 1 | 4/61 |

확실한 정답만: 33칸 중 맞음 29, 틀림 4, 놓침 0.

## 정답 종류별

| 정답 | 칸 | 맞음 | 틀림 | 놓침 |
|---|---|---|---|---|
| 아웃(인플레이) | 4 | 4 | 0 | 0 |
| 아웃(종류 모름) | 9 | 9 | 0 | 0 |
| 삼진 | 22 | 19 | 2 | 1 |
| 병살 | 1 | 0 | 1 | 0 |
| 도루 | 1 | 0 | 1 | 0 |

## 정답 × 엔진

| 정답 | 엔진 | 칸 |
|---|---|---|
| double_play | out | 1 |
| out | out | 3 |
| out | out reach | 1 |
| out_any | out | 3 |
| out_any | out out | 2 |
| out_any | strikeout | 4 |
| steal | balk | 1 |
| strikeout | (없음) | 1 |
| strikeout | out | 2 |
| strikeout | strikeout | 15 |
| strikeout | strikeout out | 3 |
| strikeout | strikeout reach reach | 1 |
