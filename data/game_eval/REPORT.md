# 1080p 경기 하이라이트 채점

> `python scripts/eval_games.py --snapshots <analyze_video 결과 폴더> [--ablation]`로 다시 만든다(조정용 3경기는 ../clips_private/game_eval_final). 정답은 `labels.json`(사람이 투구 화면마다 매김), 칸별 결과는 `plays.csv`.

| 영상 | 채점 칸 | 맞음 | 틀림 | 놓침 | 헛판정/조용함 |
|---|---|---|---|---|---|
| 3QYUP4AgNpI (2026 KBO 한화-삼성(대구) 하이라이트 · SBS Sports) | 37 | 36 | 1 | 0 | 0/4 |
| T93FGIYBM8E (2026 KBO 삼성-LG(잠실) 하이라이트 · MBC 스포츠+) | 47 | 44 | 3 | 0 | 1/5 |
| wyC8BnovUUA (2026 KBO 플레이오프 1차전 KIA-삼성(대구) 하이라이트 · TVING) | 35 | 32 | 1 | 2 | 0/4 |
| 합계 | 119 | 112 | 5 | 2 | 1/13 |

확실한 정답만: 101칸 중 맞음 95, 틀림 4, 놓침 2.

## 정답 종류별

| 정답 | 칸 | 맞음 | 틀림 | 놓침 |
|---|---|---|---|---|
| 안타·출루 | 35 | 34 | 1 | 0 |
| 아웃(인플레이) | 39 | 38 | 0 | 1 |
| 아웃(종류 모름) | 21 | 20 | 0 | 1 |
| 삼진 | 2 | 2 | 0 | 0 |
| 볼넷 | 7 | 6 | 1 | 0 |
| 몸에 맞는 공 | 2 | 2 | 0 | 0 |
| 홈런 | 6 | 5 | 1 | 0 |
| 희생플라이 | 2 | 2 | 0 | 0 |
| 병살 | 3 | 2 | 1 | 0 |
| 도루 | 2 | 1 | 1 | 0 |

## 정답 × 엔진

| 정답 | 엔진 | 칸 |
|---|---|---|
| double_play | double_play | 2 |
| double_play | out out | 1 |
| hbp | hbp | 2 |
| homerun | homerun | 3 |
| homerun | homerun out | 2 |
| homerun | out | 1 |
| out | (없음) | 1 |
| out | dive_out | 1 |
| out | out | 33 |
| out | out balk | 1 |
| out | out out | 1 |
| out | out runner | 1 |
| out | out walk | 1 |
| out_any | (없음) | 1 |
| out_any | out | 9 |
| out_any | strikeout | 10 |
| out_any | strikeout reach | 1 |
| reach | out | 1 |
| reach | reach | 30 |
| reach | reach out | 2 |
| reach | reach reach | 2 |
| sac_fly | sac_fly | 2 |
| steal | reach | 1 |
| steal | steal | 1 |
| strikeout | strikeout | 2 |
| walk | hbp | 1 |
| walk | walk | 5 |
| walk | walk out | 1 |

## 입력별

| 입력 | 맞음 | 틀림 | 놓침 | 헛판정 |
|---|---|---|---|---|
| 점수판+관절+해설 | 112 | 5 | 2 | 1 |
| 점수판+관절(해설 없이) | 109 | 8 | 2 | 1 |
| 점수판+해설(관절 없이) | 96 | 13 | 10 | 3 |
| 점수판만 | 95 | 14 | 10 | 3 |
