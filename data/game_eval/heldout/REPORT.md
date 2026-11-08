# 시험용 경기 채점(소리 없음, 영상만)

> 엔진 결과를 보기 전에 정답을 매긴 여덟 경기. 처음 측정 합 61칸 중 43(70%) — 안우진(zRqp) 3/7·문동주(Ovmk) 10/12·최민석(7TYZ) 4/7·안우진 11K(YiiG) 7/11·롯데(xtXE) 9/9·임찬규(89gN) 0/2·정우주(NVOf) 2/3·곽빈(p55O) 8/10. 아래 표는 처음 측정 뒤 고친 지금 엔진 — 고칠 때 쓴 경기는 조정용이 됐다(`../../analysis/LOG.md`). 다시 만들기: `python scripts/eval_games.py --snapshots ../clips_private/heldout_final --out data/game_eval/heldout`

| 영상 | 채점 칸 | 맞음 | 틀림 | 놓침 | 헛판정/조용함 |
|---|---|---|---|---|---|
| zRqpvJdTybA (시험용 · KBO 삼성-키움(고척) 안우진 투구 모음 · 왼쪽 위 상자 점수판 (소리 없음)) | 7 | 7 | 0 | 0 | 0/15 |
| OvmkANKCcks (시험용 · KBO KT-한화(대전) 문동주 투구 모음 · MBC 스포츠+ (소리 없음)) | 12 | 11 | 1 | 0 | 3/39 |
| 7TYZzeweCjk (시험용 · KBO 두산-KIA(광주) 최민석 투구 모음 · 왼쪽 위 상자 점수판 (소리 없음)) | 7 | 6 | 1 | 0 | 1/1 |
| YiiGl9UUV2E (시험용 · KBO LG-키움(고척) 안우진 삼진 모음 · 캐스터 띠 상자 점수판 (소리 없음)) | 11 | 11 | 0 | 0 | 0/6 |
| xtXEisvgTR0 (시험용 · KBO 롯데-LG(잠실) 롯데 투수 삼진 모음 · 캐스터 띠 상자 점수판 (소리 없음)) | 9 | 9 | 0 | 0 | 0/5 |
| 89gNJQjBRrQ (시험용 · KBO LG-키움(고척) 임찬규 투구 · SPOTV2 점수판 (소리 없음)) | 2 | 2 | 0 | 0 | 0/7 |
| NVOfYTDrqlU (시험용 · KBO 한화-키움(고척) 정우주 투구 · 오른쪽 아래 상자형 점수판 (소리 없음)) | 3 | 3 | 0 | 0 | 0/9 |
| p55O9v1xKRI (시험용 · KBO 한화-두산(잠실) 곽빈 삼진 모음 · 두 줄 가로 띠 점수판 (소리 없음)) | 10 | 9 | 1 | 0 | 1/16 |
| 합계 | 61 | 58 | 3 | 0 | 5/98 |

확실한 정답만: 57칸 중 맞음 54, 틀림 3, 놓침 0.

## 정답 종류별

| 정답 | 칸 | 맞음 | 틀림 | 놓침 |
|---|---|---|---|---|
| 아웃(인플레이) | 4 | 4 | 0 | 0 |
| 아웃(종류 모름) | 9 | 9 | 0 | 0 |
| 삼진 | 46 | 44 | 2 | 0 |
| 병살 | 1 | 0 | 1 | 0 |
| 도루 | 1 | 1 | 0 | 0 |

## 정답 × 엔진

| 정답 | 엔진 | 칸 |
|---|---|---|
| double_play | out | 1 |
| out | out | 3 |
| out | out reach | 1 |
| out_any | out | 3 |
| out_any | out out | 2 |
| out_any | strikeout | 4 |
| steal | steal | 1 |
| strikeout | out | 2 |
| strikeout | strikeout | 36 |
| strikeout | strikeout out | 7 |
| strikeout | strikeout reach reach | 1 |
