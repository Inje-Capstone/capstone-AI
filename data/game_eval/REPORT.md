# 1080p 경기 하이라이트 채점

> `python scripts/eval_games.py --snapshots <analyze_video 결과 폴더> [--ablation]`로 다시 만든다(조정용 3경기는 ../clips_private/game_eval_final). 정답은 `labels.json`(사람이 투구 화면마다 매김), 칸별 결과는 `plays.csv`.

| 영상 | 채점 칸 | 맞음 | 틀림 | 놓침 | 헛판정/조용함 |
|---|---|---|---|---|---|
| 3QYUP4AgNpI (2026 KBO 한화-삼성(대구) 하이라이트 · SBS Sports) | 37 | 36 | 1 | 0 | 0/4 |
| T93FGIYBM8E (2026 KBO 삼성-LG(잠실) 하이라이트 · MBC 스포츠+) | 47 | 38 | 4 | 5 | 2/4 |
| wyC8BnovUUA (2026 KBO 플레이오프 1차전 KIA-삼성(대구) 하이라이트 · TVING) | 35 | 30 | 2 | 3 | 0/4 |
| 합계 | 119 | 104 | 7 | 8 | 2/12 |

확실한 정답만: 101칸 중 맞음 88, 틀림 5, 놓침 8.

## 정답 종류별

| 정답 | 칸 | 맞음 | 틀림 | 놓침 |
|---|---|---|---|---|
| 안타·출루 | 35 | 31 | 3 | 1 |
| 아웃(인플레이) | 39 | 37 | 0 | 2 |
| 아웃(종류 모름) | 21 | 18 | 0 | 3 |
| 삼진 | 2 | 2 | 0 | 0 |
| 볼넷 | 7 | 5 | 1 | 1 |
| 몸에 맞는 공 | 2 | 1 | 0 | 1 |
| 홈런 | 6 | 5 | 1 | 0 |
| 희생플라이 | 2 | 2 | 0 | 0 |
| 병살 | 3 | 2 | 1 | 0 |
| 도루 | 2 | 1 | 1 | 0 |

## 정답 × 엔진

| 정답 | 엔진 | 칸 |
|---|---|---|
| double_play | double_play | 2 |
| double_play | out out | 1 |
| hbp | (없음) | 1 |
| hbp | hbp | 1 |
| homerun | homerun | 3 |
| homerun | homerun out | 2 |
| homerun | out | 1 |
| out | (없음) | 2 |
| out | dive_out | 1 |
| out | out | 31 |
| out | out balk | 1 |
| out | out out | 2 |
| out | out runner | 1 |
| out | out walk | 1 |
| out_any | (없음) | 3 |
| out_any | out | 8 |
| out_any | out walk | 1 |
| out_any | strikeout | 8 |
| out_any | strikeout reach reach | 1 |
| reach | (없음) | 1 |
| reach | out | 3 |
| reach | reach | 26 |
| reach | reach out | 3 |
| reach | reach reach | 2 |
| sac_fly | sac_fly | 2 |
| steal | reach | 1 |
| steal | steal | 1 |
| strikeout | strikeout | 2 |
| walk | (없음) | 1 |
| walk | hbp | 1 |
| walk | walk | 4 |
| walk | walk out | 1 |

## 입력별

| 입력 | 맞음 | 틀림 | 놓침 | 헛판정 |
|---|---|---|---|---|
| 점수판+관절+해설 | 104 | 7 | 8 | 2 |
| 점수판+관절(해설 없이) | 101 | 10 | 8 | 2 |
| 점수판+해설(관절 없이) | 94 | 14 | 11 | 3 |
| 점수판만 | 93 | 15 | 11 | 3 |
